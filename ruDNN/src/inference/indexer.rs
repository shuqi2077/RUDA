//! Multi-head-query attention scores and native score/top-k composition.
use super::{
    InferenceError, alloc, check, float,
    paged::{CacheLayout, PagedPlan},
    plane,
};
use ruda_core::tensor::DType;
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};

pub fn padded_seq_len(length: usize, alignment: usize) -> Result<usize, InferenceError> {
    if alignment == 0 {
        return Err(InferenceError("sequence alignment is zero"));
    }
    length
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or(InferenceError("padded sequence length overflow"))
}
pub fn min_block_table_width(max_length: usize, page_size: usize) -> Result<usize, InferenceError> {
    if page_size == 0 {
        return Err(InferenceError("page size is zero"));
    }
    Ok(max_length.div_ceil(page_size))
}
/// Sum over query heads of weight*ReLU(Q dot K). Unused logical keys receive
/// -infinity, so a subsequent top-k cannot select padded slots.
pub fn paged_mqa_logits<R: Runtime>(
    plan: &PagedPlan<R>,
    q: &RudaTensor<R>,
    keys: &RudaTensor<R>,
    head_weights: &RudaTensor<R>,
    layout: CacheLayout,
    padded_length: usize,
    scale: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    let q = into_contiguous(q.clone());
    let keys = into_contiguous(keys.clone());
    let weights = super::norm::cast(head_weights, DType::F32)?;
    for t in [&q, &keys, &weights, plan.device_metadata()] {
        check(t, &q)?;
    }
    float(&q)?;
    let host = plan.host();
    if q.meta.num_dims() != 3
        || keys.meta.num_dims() != 4
        || q.dtype != keys.dtype
        || q.meta.shape()[0] != host.query_count()
        || keys.meta.shape()[0] != host.page_count()
        || !scale.is_finite()
    {
        return Err(InferenceError("invalid MQA score operands"));
    }
    let (rows, heads, d) = (q.meta.shape()[0], q.meta.shape()[1], q.meta.shape()[2]);
    let kv_heads = keys.meta.shape()[if layout == CacheLayout::Nhd { 2 } else { 1 }];
    if heads == 0
        || kv_heads == 0
        || heads % kv_heads != 0
        || d == 0
        || keys.meta.shape()[3] != d
        || keys.meta.shape()[if layout == CacheLayout::Nhd { 1 } else { 2 }] != host.page_size()
        || weights.meta.shape()[..] != [rows, heads]
        || rows
            .checked_mul(padded_length)
            .filter(|&n| n <= u32::MAX as usize)
            .is_none()
    {
        return Err(InferenceError("MQA score dimensions mismatch"));
    }
    let lanes = plane(&q, rows, padded_length)?;
    let out = alloc(&q, [rows, padded_length], DType::F32);
    let words = &host.words;
    let maximum = words[2 * rows..2 * rows + host.sequence_count()]
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as usize;
    if padded_length < maximum {
        return Err(InferenceError("MQA padded length truncates a live history"));
    }
    if rows != 0 && padded_length != 0 {
        scores::launch::<R>(
            &q.client,
            RudaCount::Static(rows as u32, padded_length as u32, 1),
            RudaDim::new_1d(lanes),
            q.clone().into_array_arg(),
            keys.into_array_arg(),
            weights.into_array_arg(),
            plan.device_metadata().clone().into_array_arg(),
            out.clone().into_array_arg(),
            rows as u32,
            host.sequence_count() as u32,
            host.table_width as u32,
            host.page_size() as u32,
            heads as u32,
            kv_heads as u32,
            d as u32,
            padded_length as u32,
            scale,
            layout == CacheLayout::Hnd,
            q.dtype.into(),
        );
    }
    Ok(out)
}
pub fn fp8_paged_mqa_logits<R: Runtime>(
    plan: &PagedPlan<R>,
    q: &RudaTensor<R>,
    keys: &rublas::tensor_float_quant::BlockFloatTensor<R>,
    weights: &RudaTensor<R>,
    layout: CacheLayout,
    padded_length: usize,
    scale: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    let keys = keys.dequantize(q.dtype).map_err(|e| InferenceError(e.0))?;
    paged_mqa_logits(plan, q, &keys, weights, layout, padded_length, scale)
}
pub fn fp4_paged_mqa_logits<R: Runtime>(
    plan: &PagedPlan<R>,
    q: &RudaTensor<R>,
    keys: &rublas::tensor_float_quant::BlockFloatTensor<R>,
    weights: &RudaTensor<R>,
    layout: CacheLayout,
    padded_length: usize,
    scale: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    fp8_paged_mqa_logits(plan, q, keys, weights, layout, padded_length, scale)
}
pub fn paged_mqa_topk<R: Runtime>(
    plan: &PagedPlan<R>,
    q: &RudaTensor<R>,
    keys: &RudaTensor<R>,
    weights: &RudaTensor<R>,
    layout: CacheLayout,
    padded_length: usize,
    scale: f32,
    k: usize,
) -> Result<ruprim::topk::TopKOutput<R>, InferenceError> {
    let logits = paged_mqa_logits(plan, q, keys, weights, layout, padded_length, scale)?;
    ruprim::topk::top_k(&logits, k, ruprim::topk::TieBreak::SmallestIndex)
        .map_err(|e| InferenceError(e.0))
}
/// Retained FP8/FP4/BF16 score and top-k pipeline for a paged scheduler.
/// Q/K format decoding and head-weight casts stay in native device kernels.
pub struct PreparedMqaIndexer<'a, R: Runtime> {
    plan: &'a PagedPlan<R>,
    query: super::linear::PreparedMatrix<'a, R>,
    keys: super::linear::PreparedMatrix<'a, R>,
    weights: super::PreparedPack<R>,
    weights_f32: RudaTensor<R>,
    pub logits: RudaTensor<R>,
    lengths: RudaTensor<R>,
    topk: ruprim::topk::TopKPlan<R>,
    pub selected: ruprim::topk::TopKOutput<R>,
    layout: CacheLayout,
    lanes: u32,
}
impl<'a, R: Runtime> PreparedMqaIndexer<'a, R> {
    pub fn prepare(
        plan: &'a PagedPlan<R>,
        q: super::linear::Matrix<'a, R>,
        keys: super::linear::Matrix<'a, R>,
        head_weights: &RudaTensor<R>,
        layout: CacheLayout,
        padded_length: usize,
        k: usize,
    ) -> Result<Self, InferenceError> {
        let query = super::linear::PreparedMatrix::prepare(q)?;
        let keys = super::linear::PreparedMatrix::prepare(keys)?;
        let weights = super::PreparedPack::prepare(head_weights)?;
        let q = &query.output;
        let cache = &keys.output;
        for t in [cache, &weights.output, plan.device_metadata()] {
            check(t, q)?;
        }
        float(&weights.output)?;
        let h = plan.host();
        if q.meta.num_dims() != 3
            || cache.meta.num_dims() != 4
            || q.meta.shape()[0] != h.query_count()
            || cache.meta.shape()[0] != h.page_count()
            || q.meta.shape()[1] == 0
            || q.meta.shape()[2] == 0
            || cache.meta.shape()[3] != q.meta.shape()[2]
            || cache.meta.shape()[if layout == CacheLayout::Nhd { 1 } else { 2 }] != h.page_size()
        {
            return Err(InferenceError(
                "prepared MQA cache/query dimensions mismatch",
            ));
        }
        let heads = q.meta.shape()[1];
        let kh = cache.meta.shape()[if layout == CacheLayout::Nhd { 2 } else { 1 }];
        let rows = q.meta.shape()[0];
        if kh == 0
            || heads % kh != 0
            || weights.output.meta.shape()[..] != [rows, heads]
            || rows
                .checked_mul(padded_length)
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("prepared MQA weights/padding mismatch"));
        }
        let maximum = h.words[2 * rows..2 * rows + h.sequence_count()]
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize;
        if padded_length < maximum {
            return Err(InferenceError("MQA padding truncates live histories"));
        }
        let logits = alloc(q, [rows, padded_length], DType::F32);
        let lengths = alloc(q, [rows], DType::U32);
        let topk = ruprim::topk::TopKPlan::dense(&logits, k).map_err(|e| InferenceError(e.0))?;
        let selected = topk
            .allocate_output(&logits)
            .map_err(|e| InferenceError(e.0))?;
        let weights_f32 = alloc(q, [rows, heads], DType::F32);
        let lanes = plane(q, rows, padded_length)?;
        Ok(Self {
            plan,
            query,
            keys,
            weights,
            weights_f32,
            logits,
            lengths,
            topk,
            selected,
            layout,
            lanes,
        })
    }
    pub unsafe fn run(&self, scale: f32) -> Result<&ruprim::topk::TopKOutput<R>, InferenceError> {
        if !scale.is_finite() {
            return Err(InferenceError("MQA scale must be finite"));
        }
        unsafe {
            self.query.run()?;
            self.keys.run()?;
            self.weights.run();
            super::norm::cast_into(&self.weights.output, &self.weights_f32)?;
        }
        let q = &self.query.output;
        let keys = &self.keys.output;
        let h = self.plan.host();
        let rows = q.meta.shape()[0];
        let padded = self.logits.meta.shape()[1];
        if rows > 0 && padded > 0 {
            scores::launch::<R>(
                &q.client,
                RudaCount::Static(rows as u32, padded as u32, 1),
                RudaDim::new_1d(self.lanes),
                q.clone().into_array_arg(),
                keys.clone().into_array_arg(),
                self.weights_f32.clone().into_array_arg(),
                self.plan.device_metadata().clone().into_array_arg(),
                self.logits.clone().into_array_arg(),
                rows as u32,
                h.sequence_count() as u32,
                h.table_width as u32,
                h.page_size() as u32,
                q.meta.shape()[1] as u32,
                keys.meta.shape()[if self.layout == CacheLayout::Nhd {
                    2
                } else {
                    1
                }] as u32,
                q.meta.shape()[2] as u32,
                padded as u32,
                scale,
                self.layout == CacheLayout::Hnd,
                DType::F32.into(),
            );
        }
        if rows > 0 {
            let dim = RudaDim::new(q.client.properties(), rows);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, rows, dim);
            mqa_lengths::launch::<R>(
                &q.client,
                count,
                dim,
                self.plan.device_metadata().clone().into_array_arg(),
                self.lengths.clone().into_array_arg(),
                rows as u32,
                padded as u32,
            );
        }
        unsafe {
            self.topk
                .run_into(
                    &self.logits,
                    Some(&self.lengths),
                    None,
                    ruprim::topk::TieBreak::SmallestIndex,
                    ruprim::topk::TopKOutput {
                        values: self.selected.values.clone(),
                        indices: self.selected.indices.clone(),
                        counts: self.selected.counts.clone(),
                    },
                )
                .map_err(|e| InferenceError(e.0))?;
        }
        Ok(&self.selected)
    }
}
#[ruda(launch)]
pub fn scores<F: Float>(
    q: &Array<F>,
    keys: &Array<F>,
    weights: &Array<f32>,
    meta: &Array<u32>,
    out: &mut Array<f32>,
    queries: u32,
    sequences: u32,
    table_width: u32,
    page_size: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    padded: u32,
    scale: f32,
    #[comptime] hnd: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let token = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let seq = meta[row] as usize;
    let position = meta[queries as usize + row] as usize;
    let length = meta[2 * queries as usize + seq] as usize;
    let mut result = f32::NEG_INFINITY;
    if token < length && token <= position {
        let page = meta[2 * queries as usize
            + sequences as usize
            + seq * table_width as usize
            + token / page_size as usize] as usize;
        let offset = token % page_size as usize;
        result = 0.0;
        for head in 0..heads as usize {
            let kh = head / (heads / kv_heads) as usize;
            let mut cache = (page * page_size as usize + offset) * kv_heads as usize + kh;
            if comptime!(hnd) {
                cache = (page * kv_heads as usize + kh) * page_size as usize + offset;
            }
            let mut dot: f32 = 0.0f32;
            let mut col = lane;
            while col < width as usize {
                dot = fma(
                    f32::cast_from(q[(row * heads as usize + head) * width as usize + col]),
                    f32::cast_from(keys[cache * width as usize + col]),
                    dot,
                );
                col += PLANE_DIM as usize;
            }
            dot = plane_sum(dot);
            result += f32::max(dot * scale, 0.0) * weights[row * heads as usize + head];
        }
    }
    if lane == 0 {
        out[row * padded as usize + token] = result;
    }
}

#[ruda(launch)]
pub fn mqa_lengths(meta: &Array<u32>, out: &mut Array<u32>, rows: u32, padded: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < rows as usize {
        let seq = meta[i] as usize;
        out[i] = u32::min(
            padded,
            u32::min(meta[2 * rows as usize + seq], meta[rows as usize + i] + 1),
        );
    }
}
