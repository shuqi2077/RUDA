//! Planned paged decode/prefill with GQA, explicit positions and device LSE.
//! The generic kernel prioritizes coverage; the existing optimized paged path
//! remains available for its supported specialization set.
use super::cascade::{AttentionState, LogBase};
use super::{InferenceError, alloc, check, float, plane};
use crate::paged_attention::HostPlan;
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLayout {
    Nhd,
    Hnd,
}
#[derive(Clone, Copy, Debug)]
pub struct AttentionOptions {
    pub scale: f32,
    pub causal: bool,
    /// Number of preceding positions; -1 is unlimited.
    /// A value of zero attends the current position, not an empty window.
    pub window_left: i32,
    /// Zero disables c*tanh(dot*scale/c). ALiBi is added after capping.
    pub logits_soft_cap: f32,
    pub lse_base: LogBase,
}
/// Row-padded masks [queries, max_history] or [queries, ceil(max_history/8)].
/// Packed masks are little-endian (bit 0 is the first key). U8 zero means masked.
pub enum Mask<'a, R: Runtime> {
    Bytes(&'a RudaTensor<R>),
    Packed(&'a RudaTensor<R>),
}

pub struct PagedPlan<R: Runtime> {
    host: HostPlan,
    metadata: RudaTensor<R>,
    max_history: usize,
}
impl<R: Runtime> PagedPlan<R> {
    pub(crate) fn device_metadata(&self) -> &RudaTensor<R> {
        &self.metadata
    }
    pub fn upload(host: HostPlan, like: &RudaTensor<R>) -> Result<Self, InferenceError> {
        check(like, like)?;
        let max_history = host.words[2 * host.queries..2 * host.queries + host.sequences]
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let metadata = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([host.words.len()]),
            like.client
                .create_from_slice(bytemuck::cast_slice(&host.words)),
            DType::U32,
        );
        Ok(Self {
            host,
            metadata,
            max_history,
        })
    }
    pub fn host(&self) -> &HostPlan {
        &self.host
    }
    /// Gather the scheduled query positions from one cache into [queries,Hkv,D].
    /// Shared read pages are supported; empty or uncommitted positions are not.
    pub fn gather(
        &self,
        cache: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<RudaTensor<R>, InferenceError> {
        let (heads, width) = self.cache_shape(cache, layout)?;
        for row in 0..self.host.queries {
            let seq = self.host.words[row] as usize;
            if self.host.words[self.host.queries + row]
                >= self.host.words[2 * self.host.queries + seq]
            {
                return Err(InferenceError(
                    "gather position is outside committed history",
                ));
            }
        }
        let out = alloc(cache, [self.host.queries, heads, width], cache.dtype);
        self.transfer(&out, cache, layout, heads, width, false)?;
        Ok(out)
    }
    /// Append one K/V component. Return the updated cache, preserving aliases
    /// through device copy-on-write, just like the existing NHD append API.
    pub fn append(
        &self,
        tokens: &RudaTensor<R>,
        cache: RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<RudaTensor<R>, InferenceError> {
        self.host
            .validate_writes()
            .map_err(|e| InferenceError(e.0))?;
        let (heads, width) = self.cache_shape(&cache, layout)?;
        check(tokens, &cache)?;
        if tokens.dtype != cache.dtype
            || tokens.meta.shape()[..] != [self.host.queries, heads, width]
        {
            return Err(InferenceError(
                "append tokens must be [queries,Hkv,D] with cache dtype",
            ));
        }
        if self.host.queries == 0 {
            return Ok(cache);
        }
        let cache = if cache.can_mut() {
            cache
        } else {
            crate::paged_attention::copy_contiguous(&cache)
        };
        self.transfer(tokens, &cache, layout, heads, width, true)?;
        Ok(cache)
    }
    fn cache_shape(
        &self,
        cache: &RudaTensor<R>,
        layout: CacheLayout,
    ) -> Result<(usize, usize), InferenceError> {
        check(cache, &self.metadata)?;
        float(cache)?;
        if cache.meta.num_dims() != 4 {
            return Err(InferenceError("cache must have rank four"));
        }
        let heads = cache.meta.shape()[if layout == CacheLayout::Nhd { 2 } else { 1 }];
        let width = cache.meta.shape()[3];
        let shape = if layout == CacheLayout::Nhd {
            [self.host.pages, self.host.page_size, heads, width]
        } else {
            [self.host.pages, heads, self.host.page_size, width]
        };
        if cache.meta.shape()[..] != shape
            || heads == 0
            || width == 0
            || self
                .host
                .queries
                .checked_mul(heads)
                .and_then(|n| n.checked_mul(width))
                .filter(|&n| n <= u32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("invalid cache shape or transfer indexing"));
        }
        Ok((heads, width))
    }
    fn transfer(
        &self,
        tokens: &RudaTensor<R>,
        cache: &RudaTensor<R>,
        layout: CacheLayout,
        heads: usize,
        width: usize,
        append: bool,
    ) -> Result<(), InferenceError> {
        let lanes = plane(cache, self.host.queries, 1)?;
        if self.host.queries != 0 {
            transfer::launch::<R>(
                &cache.client,
                RudaCount::Static(self.host.queries as u32, 1, 1),
                RudaDim::new_1d(lanes),
                tokens.clone().into_array_arg(),
                cache.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),
                self.host.queries as u32,
                self.host.sequences as u32,
                self.host.table_width as u32,
                self.host.page_size as u32,
                heads as u32,
                width as u32,
                layout == CacheLayout::Hnd,
                append,
                cache.dtype.into(),
            );
        }
        Ok(())
    }
    /// Q=[queries,Hq,D]. NHD caches=[pages,page_size,Hkv,D], HND caches=
    /// [pages,Hkv,page_size,D]. V may have a different final width. Head groups
    /// must divide exactly. Only scheduler metadata is ever uploaded by this API.
    pub fn run(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        layout: CacheLayout,
        options: AttentionOptions,
        mask: Option<Mask<'_, R>>,
        alibi_slopes: Option<&RudaTensor<R>>,
    ) -> Result<AttentionState<R>, InferenceError> {
        for t in [q, k, v, &self.metadata] {
            check(t, q)?;
        }
        float(q)?;
        if q.meta.num_dims() != 3
            || k.meta.num_dims() != 4
            || v.meta.num_dims() != 4
            || q.dtype != k.dtype
            || q.dtype != v.dtype
        {
            return Err(InferenceError("invalid paged float ranks/dtypes"));
        }
        let (rows, heads, d) = (q.meta.shape()[0], q.meta.shape()[1], q.meta.shape()[2]);
        let kv_heads = k.meta.shape()[if layout == CacheLayout::Nhd { 2 } else { 1 }];
        let dv = v.meta.shape()[3];
        let expected = if layout == CacheLayout::Nhd {
            [self.host.pages, self.host.page_size, kv_heads]
        } else {
            [self.host.pages, kv_heads, self.host.page_size]
        };
        if rows != self.host.queries
            || heads == 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || d == 0
            || dv == 0
            || d > 1024
            || dv > 1024
            || rows
                .checked_mul(heads)
                .and_then(|n| n.checked_mul(dv))
                .filter(|&n| n <= u32::MAX as usize)
                .is_none()
            || k.meta.shape()[..3] != expected
            || v.meta.shape()[..3] != expected
            || k.meta.shape()[3] != d
            || !options.scale.is_finite()
            || !options.logits_soft_cap.is_finite()
            || options.logits_soft_cap < 0.0
            || options.window_left < -1
        {
            return Err(InferenceError("invalid paged shapes/options/head grouping"));
        }
        let (mask_tensor, mask_mode, mask_stride) = match mask {
            None => (&self.metadata, 0u32, 0usize),
            Some(Mask::Bytes(t)) => (t, 1, self.max_history),
            Some(Mask::Packed(t)) => (t, 2, self.max_history.div_ceil(8)),
        };
        if mask_mode != 0 {
            check(mask_tensor, q)?;
            if mask_tensor.dtype != DType::U8 || mask_tensor.meta.shape()[..] != [rows, mask_stride]
            {
                return Err(InferenceError(
                    "custom mask must be U8 [queries, row_stride]",
                ));
            }
        }
        let slopes = alibi_slopes.unwrap_or(q);
        if alibi_slopes.is_some() {
            check(slopes, q)?;
            if slopes.dtype != DType::F32 || slopes.meta.shape()[..] != [heads] {
                return Err(InferenceError("ALiBi slopes must be F32 [heads]"));
            }
        }
        let lanes = plane(q, rows, heads)?;
        let out = alloc(q, [rows, heads, dv], q.dtype);
        let lse = alloc(q, [rows, heads], DType::F32);
        if rows != 0 {
            attention::launch::<R>(
                &q.client,
                RudaCount::Static(rows as u32, heads as u32, 1),
                RudaDim::new_1d(lanes),
                q.clone().into_array_arg(),
                k.clone().into_array_arg(),
                v.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),
                mask_tensor.clone().into_array_arg(),
                slopes.clone().into_array_arg(),
                out.clone().into_array_arg(),
                lse.clone().into_array_arg(),
                options.scale,
                options.logits_soft_cap,
                options.window_left,
                rows as u32,
                self.host.sequences as u32,
                self.host.table_width as u32,
                self.host.page_size as u32,
                heads as u32,
                kv_heads as u32,
                mask_stride as u32,
                d,
                dv,
                lanes as usize,
                layout == CacheLayout::Hnd,
                options.causal,
                mask_mode,
                alibi_slopes.is_some(),
                options.lse_base == LogBase::Two,
                q.dtype.into(),
            );
        }
        Ok(AttentionState { values: out, lse })
    }
}

#[ruda(launch)]
pub fn transfer<F: Float>(
    tokens: &mut Array<F>,
    cache: &mut Array<F>,
    metadata: &Array<u32>,
    queries: u32,
    sequences: u32,
    table_width: u32,
    page_size: u32,
    heads: u32,
    width: u32,
    #[comptime] hnd: bool,
    #[comptime] append: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let seq = metadata[row] as usize;
    let pos = metadata[queries as usize + row] as usize;
    let page = metadata[2 * queries as usize
        + sequences as usize
        + seq * table_width as usize
        + pos / page_size as usize] as usize;
    let mut i = UNIT_POS as usize;
    while i < heads as usize * width as usize {
        let head = i / width as usize;
        let d = i % width as usize;
        let mut slot = ((page * page_size as usize + pos % page_size as usize) * heads as usize
            + head)
            * width as usize
            + d;
        if comptime!(hnd) {
            slot = ((page * heads as usize + head) * page_size as usize + pos % page_size as usize)
                * width as usize
                + d;
        }
        if comptime!(append) {
            cache[slot] = tokens[row * heads as usize * width as usize + i];
        } else {
            tokens[row * heads as usize * width as usize + i] = cache[slot];
        }
        i += RUDA_DIM as usize;
    }
}

#[ruda(launch)]
pub fn attention<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    metadata: &Array<u32>,
    mask: &Array<u8>,
    slopes: &Array<f32>,
    out: &mut Array<F>,
    lse: &mut Array<f32>,
    scale: f32,
    soft_cap: f32,
    window_left: i32,
    queries: u32,
    sequences: u32,
    table_width: u32,
    page_size: u32,
    heads: u32,
    kv_heads: u32,
    mask_stride: u32,
    #[comptime] d: usize,
    #[comptime] dv: usize,
    #[comptime] lanes: usize,
    #[comptime] hnd: bool,
    #[comptime] causal: bool,
    #[comptime] mask_mode: u32,
    #[comptime] alibi: bool,
    #[comptime] base_two: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let seq = metadata[row] as usize;
    let position = metadata[queries as usize + row] as usize;
    let length = metadata[2 * queries as usize + seq] as usize;
    let kv_head = head / (heads / kv_heads) as usize;
    let mut end = length;
    if comptime!(causal) {
        end = usize::min(end, position + 1);
    }
    let mut begin: usize = 0usize;
    if window_left >= 0 && position > window_left as usize {
        begin = usize::min(end, position - window_left as usize);
    }
    let mut query = Array::<f32>::new(((d + lanes - 1) / lanes));
    let mut acc = Array::<f32>::new(((dv + lanes - 1) / lanes));
    #[unroll]
    for j in 0usize..((d + lanes - 1) / lanes) {
        query[j] = 0.0;
        let col = lane + j * lanes;
        if col < d {
            query[j] = f32::cast_from(q[(row * heads as usize + head) * d + col]);
        }
    }
    #[unroll]
    for j in 0usize..((dv + lanes - 1) / lanes) {
        acc[j] = 0.0;
    }
    let mut maximum: f32 = 0.0f32;
    let mut denominator: f32 = 0.0f32;
    let mut bad = false;
    for token in begin..end {
        let mut visible = true;
        if comptime!(mask_mode == 1) {
            visible = mask[row * mask_stride as usize + token] != 0;
        }
        if comptime!(mask_mode == 2) {
            visible = (u32::cast_from(mask[row * mask_stride as usize + token / 8])
                & (1u32 << (token % 8) as u32))
                != 0;
        }
        if visible {
            let page = metadata[2 * queries as usize
                + sequences as usize
                + seq * table_width as usize
                + token / page_size as usize] as usize;
            let offset = token % page_size as usize;
            let mut cache = (page * page_size as usize + offset) * kv_heads as usize + kv_head;
            if comptime!(hnd) {
                cache = (page * kv_heads as usize + kv_head) * page_size as usize + offset;
            }
            let mut dot: f32 = 0.0f32;
            #[unroll]
            for j in 0usize..((d + lanes - 1) / lanes) {
                let col = lane + j * lanes;
                if col < d {
                    dot = fma(query[j], f32::cast_from(k[cache * d + col]), dot);
                }
            }
            let mut score = plane_sum(dot) * scale;
            if soft_cap > 0.0 {
                score = soft_cap * (score / soft_cap).tanh();
            }
            if comptime!(alibi) {
                // Subtract integer positions before converting, retaining local
                // offsets even for histories beyond F32's exact integer range.
                let mut relative: f32 = 0.0f32;
                if token >= position {
                    relative = (token - position) as f32;
                } else {
                    relative = -((position - token) as f32);
                }
                score += slopes[head] * relative;
            }
            if score != score || score == f32::INFINITY {
                bad = true;
            }
            if score != f32::NEG_INFINITY {
                let mut next = score;
                let mut old: f32 = 0.0f32;
                if denominator != 0.0 {
                    next = f32::max(maximum, score);
                    old = (maximum - next).exp();
                }
                let weight = (score - next).exp();
                denominator = fma(denominator, old, weight);
                #[unroll]
                for j in 0usize..((dv + lanes - 1) / lanes) {
                    let col = lane + j * lanes;
                    if col < dv {
                        acc[j] = fma(weight, f32::cast_from(v[cache * dv + col]), acc[j] * old);
                    }
                }
                maximum = next;
            }
        }
    }
    let index = row * heads as usize + head;
    #[unroll]
    for j in 0usize..((dv + lanes - 1) / lanes) {
        let col = lane + j * lanes;
        if col < dv {
            let mut value: f32 = 0.0f32;
            if denominator != 0.0 {
                value = acc[j] / denominator;
            }
            if bad {
                value = f32::NAN;
            }
            out[index * dv + col] = F::cast_from(value);
        }
    }
    if lane == 0 {
        let mut value = f32::NEG_INFINITY;
        if denominator != 0.0 {
            value = maximum + denominator.ln();
        }
        if bad {
            value = f32::NAN;
        }
        if comptime!(base_two) {
            value *= core::f32::consts::LOG2_E;
        }
        lse[index] = value;
    }
}
