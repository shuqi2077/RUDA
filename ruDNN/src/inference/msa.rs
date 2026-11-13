//! Minimax block proxy scoring, forced-boundary top-K and selected attention.
//! Flat native tensors, raw FP8 and block-scaled FP4 all feed this Rust path.
use super::{
    InferenceError, alloc,
    attention::{AttentionPlan, Options, RuntimeInputs},
    cascade::AttentionState,
    check,
    linear::Matrix,
};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
pub struct MsaPlan<R: Runtime> {
    pub q_indptr: RudaTensor<R>,
    pub kv_indptr: RudaTensor<R>,
    pub metadata: RudaTensor<R>,
    pub attention: AttentionPlan<R>,
    pub query_offsets: Option<RudaTensor<R>>,
    queries: usize,
    keys: usize,
    sequences: usize,
    blocks: usize,
    block_size: usize,
}
impl<R: Runtime> MsaPlan<R> {
    pub fn new(
        q_indptr: &RudaTensor<R>,
        kv_indptr: &RudaTensor<R>,
        queries: usize,
        keys: usize,
        max_history: usize,
        block_size: usize,
        query_offsets: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        for t in [q_indptr, kv_indptr] {
            check(t, q_indptr)?;
            if t.dtype != DType::I32 || t.meta.num_dims() != 1 {
                return Err(InferenceError("MSA requires I32 Q/K indptr vectors"));
            }
        }
        if q_indptr.meta.shape() != kv_indptr.meta.shape()
            || q_indptr.meta.shape()[0] < 2
            || keys > i32::MAX as usize
            || queries > i32::MAX as usize
            || max_history > i32::MAX as usize
            || block_size == 0
        {
            return Err(InferenceError("invalid MSA ragged capacities"));
        }
        let sequences = q_indptr.meta.shape()[0] - 1;
        if let Some(offsets) = query_offsets {
            check(offsets, q_indptr)?;
            if offsets.dtype != DType::I32
                || !(offsets.meta.num_elements() == 1 || offsets.meta.shape()[..] == [sequences])
            {
                return Err(InferenceError(
                    "MSA query offsets require I32 scalar/per-sequence",
                ));
            }
        }
        let metadata = alloc(q_indptr, [queries, 5], DType::U32);
        let attention = AttentionPlan::device(&metadata, keys, max_history)?;
        Ok(Self {
            q_indptr: q_indptr.clone(),
            kv_indptr: kv_indptr.clone(),
            metadata,
            attention,
            query_offsets: query_offsets.cloned(),
            queries,
            keys,
            sequences,
            blocks: max_history.div_ceil(block_size),
            block_size,
        })
    }
    pub fn refresh(&self) -> Result<(), InferenceError> {
        if self.queries > 0 {
            let dim = RudaDim::new(self.metadata.client.properties(), self.queries);
            let count =
                kernel_dsl::calculate_ruda_count_elemwise(&self.metadata.client, self.queries, dim);
            ragged_metadata::launch::<R>(
                &self.metadata.client,
                count,
                dim,
                self.q_indptr.clone().into_array_arg(),
                self.kv_indptr.clone().into_array_arg(),
                self.query_offsets
                    .as_ref()
                    .unwrap_or(&self.q_indptr)
                    .clone()
                    .into_array_arg(),
                self.metadata.clone().into_array_arg(),
                self.queries as u32,
                self.keys as u32,
                self.sequences as u32,
                self.query_offsets.is_some(),
                self.query_offsets
                    .as_ref()
                    .is_some_and(|t| t.meta.num_elements() == 1),
            );
        }
        Ok(())
    }
    /// Per-block maximum raw dot, shape [heads,blocks,queries]. Reducing heads
    /// uses max (not a sum), giving a common indexer selection for all heads.
    pub fn proxy_scores(
        &self,
        query: Matrix<'_, R>,
        key: Matrix<'_, R>,
        causal: bool,
        reduce_heads: bool,
    ) -> Result<RudaTensor<R>, InferenceError> {
        self.refresh()?;
        let q = query.decode()?;
        let k = key.decode()?;
        for t in [&k, &self.metadata] {
            check(t, &q)?;
        }
        if q.meta.num_dims() != 3
            || k.meta.num_dims() != 3
            || q.meta.shape()[0] != self.queries
            || k.meta.shape()[0] != self.keys
            || q.meta.shape()[2] != k.meta.shape()[2]
            || k.meta.shape()[1] == 0
            || q.meta.shape()[1] == 0
            || q.meta.shape()[1] % k.meta.shape()[1] != 0
        {
            return Err(InferenceError("MSA proxy Q/K dimensions mismatch"));
        }
        let h = q.meta.shape()[1];
        let kh = k.meta.shape()[1];
        let d = q.meta.shape()[2];
        let oh = if reduce_heads { 1 } else { h };
        let n = oh
            .checked_mul(self.blocks)
            .and_then(|n| n.checked_mul(self.queries))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("MSA score buffer exceeds U32"))?;
        let out = alloc(&q, [oh, self.blocks, self.queries], DType::F32);
        if n > 0 {
            let dim = RudaDim::new(q.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
            proxy::launch::<R>(
                &q.client,
                count,
                dim,
                q.clone().into_array_arg(),
                k.into_array_arg(),
                self.metadata.clone().into_array_arg(),
                out.clone().into_array_arg(),
                self.queries as u32,
                self.blocks as u32,
                self.block_size as u32,
                h as u32,
                kh as u32,
                d as u32,
                causal,
                reduce_heads,
            );
        }
        Ok(out)
    }
    /// Selected blocks [T,Hselect,K] are expanded to per-head token IDs. Blocks
    /// are local to each sequence; attention consumes the device row schedule.
    pub fn attend(
        &self,
        query: Matrix<'_, R>,
        key: Matrix<'_, R>,
        value: Matrix<'_, R>,
        blocks: &RudaTensor<R>,
        options: Options,
        output_dtype: DType,
    ) -> Result<AttentionState<R>, InferenceError> {
        self.refresh()?;
        let q = query.decode()?;
        let k = key.decode()?;
        let v = value.decode()?;
        check(blocks, &q)?;
        if blocks.dtype != DType::I32
            || blocks.meta.num_dims() != 3
            || blocks.meta.shape()[0] != self.queries
        {
            return Err(InferenceError(
                "MSA selected blocks must be I32 [T,Hselect,K]",
            ));
        }
        let heads = blocks.meta.shape()[1];
        let top_k = blocks.meta.shape()[2];
        if heads == 0 || q.meta.num_dims() != 3 || q.meta.shape()[1] % heads != 0 {
            return Err(InferenceError(
                "MSA selection heads must divide query heads",
            ));
        }
        let width = top_k
            .checked_mul(self.block_size)
            .ok_or(InferenceError("MSA selected width overflow"))?;
        let n = self
            .queries
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(width))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("MSA selected token buffer exceeds U32"))?;
        let ids = alloc(&q, [heads, self.queries, width], DType::I32);
        if n > 0 {
            let dim = RudaDim::new(q.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
            expand::launch::<R>(
                &q.client,
                count,
                dim,
                blocks.clone().into_array_arg(),
                ids.clone().into_array_arg(),
                self.queries as u32,
                heads as u32,
                top_k as u32,
                self.block_size as u32,
            );
        }
        let state = self.attention.run(
            &q,
            &k,
            &v,
            options,
            RuntimeInputs {
                indexed_keys: Some(&ids),
                ..Default::default()
            },
        )?;
        Ok(AttentionState {
            values: super::norm::cast(&state.values, output_dtype)?,
            lse: state.lse,
        })
    }
}
/// Full prepared score/select/attend path. Exposing the three stages allows
/// a native ICP publisher to consume selections without allocating new tensors.
pub struct PreparedMsa<'a, R: Runtime> {
    plan: &'a MsaPlan<R>,
    query: super::linear::PreparedMatrix<'a, R>,
    key: super::linear::PreparedMatrix<'a, R>,
    value: super::linear::PreparedMatrix<'a, R>,
    pub scores: RudaTensor<R>,
    pub blocks: RudaTensor<R>,
    pub indices: RudaTensor<R>,
    live_blocks: RudaTensor<R>,
    accumulated: AttentionState<R>,
    pub output: AttentionState<R>,
    k: usize,
    reduce_heads: bool,
    force_begin: usize,
    force_end: usize,
}
impl<'a, R: Runtime> PreparedMsa<'a, R> {
    pub fn prepare(
        plan: &'a MsaPlan<R>,
        q: Matrix<'a, R>,
        k: Matrix<'a, R>,
        v: Matrix<'a, R>,
        top_k: usize,
        reduce_heads: bool,
        force_begin: usize,
        force_end: usize,
        dtype: DType,
    ) -> Result<Self, InferenceError> {
        let query = super::linear::PreparedMatrix::prepare(q)?;
        let key = super::linear::PreparedMatrix::prepare(k)?;
        let value = super::linear::PreparedMatrix::prepare(v)?;
        let q = &query.output;
        let k = &key.output;
        let v = &value.output;
        for t in [k, v, &plan.metadata] {
            check(t, q)?;
        }
        if q.meta.num_dims() != 3
            || k.meta.num_dims() != 3
            || v.meta.num_dims() != 3
            || q.meta.shape()[0] != plan.queries
            || k.meta.shape()[0] != plan.keys
            || v.meta.shape()[0] != plan.keys
            || q.meta.shape()[1] == 0
            || k.meta.shape()[1] == 0
            || q.meta.shape()[1] % k.meta.shape()[1] != 0
            || k.meta.shape()[1] != v.meta.shape()[1]
            || q.meta.shape()[2] != k.meta.shape()[2]
            || top_k == 0
            || top_k > 1024
            || force_begin.checked_add(force_end).is_none_or(|n| n > top_k)
            || plan.blocks == 0
        {
            return Err(InferenceError("prepared MSA geometry/selection mismatch"));
        }
        let heads = q.meta.shape()[1];
        let selected_heads = if reduce_heads { 1 } else { heads };
        let width = top_k
            .checked_mul(plan.block_size)
            .ok_or(InferenceError("MSA token selection width overflow"))?;
        for shape in [
            [selected_heads, plan.blocks, plan.queries],
            [selected_heads, plan.queries, width],
        ] {
            if shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
            {
                return Err(InferenceError("MSA prepared buffers exceed U32"));
            }
        }
        let live_blocks = alloc(q, [plan.queries], DType::I32);
        let scores = alloc(q, [selected_heads, plan.blocks, plan.queries], DType::F32);
        let blocks = alloc(q, [plan.queries, selected_heads, top_k], DType::I32);
        let indices = alloc(q, [selected_heads, plan.queries, width], DType::I32);
        let lse = alloc(q, [plan.queries, heads], DType::F32);
        let accumulated = AttentionState {
            values: alloc(q, [plan.queries, heads, v.meta.shape()[2]], DType::F32),
            lse: lse.clone(),
        };
        let output = AttentionState {
            values: alloc(q, [plan.queries, heads, v.meta.shape()[2]], dtype),
            lse,
        };
        super::float(&output.values)?;
        Ok(Self {
            plan,
            query,
            key,
            value,
            scores,
            blocks,
            indices,
            live_blocks,
            accumulated,
            output,
            k: top_k,
            reduce_heads,
            force_begin,
            force_end,
        })
    }
    pub unsafe fn score(&self, causal: bool) -> Result<&RudaTensor<R>, InferenceError> {
        self.plan.refresh()?;
        unsafe {
            self.query.run()?;
            self.key.run()?;
        }
        let q = &self.query.output;
        let k = &self.key.output;
        let n = self.scores.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(q.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
            proxy::launch::<R>(
                &q.client,
                count,
                dim,
                q.clone().into_array_arg(),
                k.clone().into_array_arg(),
                self.plan.metadata.clone().into_array_arg(),
                self.scores.clone().into_array_arg(),
                self.plan.queries as u32,
                self.plan.blocks as u32,
                self.plan.block_size as u32,
                q.meta.shape()[1] as u32,
                k.meta.shape()[1] as u32,
                q.meta.shape()[2] as u32,
                causal,
                self.reduce_heads,
            );
        }
        let rows = self.plan.queries;
        if rows > 0 {
            let dim = RudaDim::new(q.client.properties(), rows);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, rows, dim);
            live_page_counts::launch::<R>(
                &q.client,
                count,
                dim,
                self.plan.metadata.clone().into_array_arg(),
                self.live_blocks.clone().into_array_arg(),
                rows as u32,
                self.plan.block_size as u32,
                causal,
            );
        }
        Ok(&self.scores)
    }
    pub unsafe fn select(
        &self,
        valid_pages: Option<&RudaTensor<R>>,
    ) -> Result<&RudaTensor<R>, InferenceError> {
        if let Some(p) = valid_pages {
            check(p, &self.scores)?;
            if p.dtype != DType::I32
                || !(p.meta.num_elements() == 1 || p.meta.shape()[..] == [self.plan.queries])
            {
                return Err(InferenceError(
                    "MSA valid page bounds require I32 scalar/[T]",
                ));
            }
        }
        let valid_pages = valid_pages.unwrap_or(&self.live_blocks);
        let h = self.scores.meta.shape()[0];
        let n = h * self.plan.queries;
        if n > 0 {
            let dim = RudaDim::new(self.scores.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.scores.client, n, dim);
            select::launch::<R>(
                &self.scores.client,
                count,
                dim,
                self.scores.clone().into_array_arg(),
                valid_pages.clone().into_array_arg(),
                self.blocks.clone().into_array_arg(),
                h as u32,
                self.plan.blocks as u32,
                self.plan.queries as u32,
                self.k,
                self.force_begin as u32,
                self.force_end as u32,
                true,
                valid_pages.meta.num_elements() == 1,
            );
        }
        Ok(&self.blocks)
    }
    /// Execute after score/select, or after another native device operator
    /// writes blocks. Bound Q/K decoders are refreshed by score; V by attend.
    pub unsafe fn attend(&self, options: Options) -> Result<&AttentionState<R>, InferenceError> {
        unsafe {
            self.value.run()?;
        }
        let n = self.indices.meta.num_elements();
        let h = self.blocks.meta.shape()[1];
        if n > 0 {
            let dim = RudaDim::new(self.indices.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.indices.client, n, dim);
            expand::launch::<R>(
                &self.indices.client,
                count,
                dim,
                self.blocks.clone().into_array_arg(),
                self.indices.clone().into_array_arg(),
                self.plan.queries as u32,
                h as u32,
                self.k as u32,
                self.plan.block_size as u32,
            );
        }
        unsafe {
            self.plan.attention.run_into(
                &self.query.output,
                &self.key.output,
                &self.value.output,
                options,
                RuntimeInputs {
                    indexed_keys: Some(&self.indices),
                    ..Default::default()
                },
                &self.accumulated,
            )?;
            super::norm::cast_into(&self.accumulated.values, &self.output.values)?;
        }
        Ok(&self.output)
    }
    pub unsafe fn run(
        &self,
        options: Options,
        valid_pages: Option<&RudaTensor<R>>,
    ) -> Result<&AttentionState<R>, InferenceError> {
        unsafe {
            self.score(options.causal)?;
            self.select(valid_pages)?;
            self.attend(options)
        }
    }
}
/// Fixed K selection with forced beginning/end blocks and lower-ID tie breaks.
/// IDs are returned ascending, padded by -1; NaN scores rank as -infinity.
pub fn topk<R: Runtime>(
    scores: &RudaTensor<R>,
    k: usize,
    valid_pages: Option<&RudaTensor<R>>,
    force_begin: usize,
    force_end: usize,
) -> Result<RudaTensor<R>, InferenceError> {
    check(scores, scores)?;
    if scores.dtype != DType::F32
        || scores.meta.num_dims() != 3
        || k == 0
        || k > 1024
        || force_begin + force_end > k
    {
        return Err(InferenceError(
            "MSA top-K needs FP32 [H,B,T] and valid forced counts",
        ));
    }
    let heads = scores.meta.shape()[0];
    let blocks = scores.meta.shape()[1];
    let tokens = scores.meta.shape()[2];
    if blocks == 0 || blocks > i32::MAX as usize {
        return Err(InferenceError("MSA block count invalid"));
    }
    if let Some(p) = valid_pages {
        check(p, scores)?;
        if p.dtype != DType::I32 || !(p.meta.num_elements() == 1 || p.meta.shape()[..] == [tokens])
        {
            return Err(InferenceError(
                "MSA valid pages require I32 scalar/per-token",
            ));
        }
    }
    let n = tokens
        .checked_mul(heads)
        .filter(|n| *n <= u32::MAX as usize / k)
        .ok_or(InferenceError("MSA top-K output exceeds U32"))?;
    let out = alloc(scores, [tokens, heads, k], DType::I32);
    if n > 0 {
        let dim = RudaDim::new(scores.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&scores.client, n, dim);
        select::launch::<R>(
            &scores.client,
            count,
            dim,
            scores.clone().into_array_arg(),
            valid_pages.unwrap_or(scores).clone().into_array_arg(),
            out.clone().into_array_arg(),
            heads as u32,
            blocks as u32,
            tokens as u32,
            k,
            force_begin as u32,
            force_end as u32,
            valid_pages.is_some(),
            valid_pages.is_some_and(|p| p.meta.num_elements() == 1),
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn ragged_metadata(
    qcu: &Array<i32>,
    kcu: &Array<i32>,
    offsets: &Array<i32>,
    out: &mut Array<u32>,
    queries: u32,
    keys: u32,
    sequences: u32,
    #[comptime] offset: bool,
    #[comptime] scalar: bool,
) {
    let row = ABSOLUTE_POS as usize;
    if row < queries as usize {
        let mut lo: usize = 0usize;
        let mut hi = sequences as usize;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if qcu[mid] <= row as i32 {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut start: i32 = 0i32;
        let mut length: i32 = 0i32;
        let mut position: i32 = -1i32;
        if lo < sequences as usize {
            start = i32::max(0, kcu[lo]);
            let end = i32::min(keys as i32, kcu[lo + 1]);
            length = i32::max(0, end - start);
            position = length - (qcu[lo + 1] - qcu[lo]) + row as i32 - qcu[lo];
            if comptime!(offset) {
                let mut oi = lo;
                if comptime!(scalar) {
                    oi = 0;
                }
                position += offsets[oi];
            }
        }
        out[row * 5] = start as u32;
        out[row * 5 + 1] = length as u32;
        out[row * 5 + 2] = position as u32;
        out[row * 5 + 3] = 0;
        out[row * 5 + 4] = length as u32;
    }
}
#[ruda(launch)]
pub fn proxy(
    q: &Array<f32>,
    k: &Array<f32>,
    metadata: &Array<u32>,
    out: &mut Array<f32>,
    queries: u32,
    blocks: u32,
    block_size: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    #[comptime] causal: bool,
    #[comptime] reduced: bool,
) {
    let i = ABSOLUTE_POS as usize;
    let mut oh = heads;
    if comptime!(reduced) {
        oh = 1;
    }
    if i < oh as usize * blocks as usize * queries as usize {
        let row = i % queries as usize;
        let block = i / queries as usize % blocks as usize;
        let head = i / (queries as usize * blocks as usize);
        let start = metadata[row * 5] as usize;
        let length = metadata[row * 5 + 1] as usize;
        let position = metadata[row * 5 + 2] as i32;
        let mut maximum = f32::NEG_INFINITY;
        let mut first_head = head;
        let mut end_head = head + 1;
        if comptime!(reduced) {
            first_head = 0;
            end_head = heads as usize;
        }
        for h in first_head..end_head {
            let kh = h / (heads / kv_heads) as usize;
            for offset in 0..block_size as usize {
                let token = block * block_size as usize + offset;
                let mut live = token < length;
                if comptime!(causal) {
                    live = live && token as i32 <= position;
                }
                if live {
                    let mut dot: f32 = 0.0f32;
                    for d in 0..width as usize {
                        dot = fma(
                            q[(row * heads as usize + h) * width as usize + d],
                            k[((start + token) * kv_heads as usize + kh) * width as usize + d],
                            dot,
                        );
                    }
                    maximum = f32::max(maximum, dot);
                }
            }
        }
        out[i] = maximum;
    }
}
#[ruda(launch)]
pub fn expand(
    blocks: &Array<i32>,
    out: &mut Array<i32>,
    tokens: u32,
    heads: u32,
    k: u32,
    block_size: u32,
) {
    let i = ABSOLUTE_POS as usize;
    let width = k as usize * block_size as usize;
    if i < tokens as usize * heads as usize * width {
        let offset = i % width;
        let token = i / width % tokens as usize;
        let head = i / (width * tokens as usize);
        let block =
            blocks[(token * heads as usize + head) * k as usize + offset / block_size as usize];
        let mut id: i32 = -1i32;
        if block >= 0 {
            id = block * block_size as i32 + (offset % block_size as usize) as i32;
        }
        out[i] = id;
    }
}
#[ruda(launch)]
pub fn select(
    scores: &Array<f32>,
    valid: &Array<i32>,
    out: &mut Array<i32>,
    heads: u32,
    blocks: u32,
    tokens: u32,
    #[comptime] k: usize,
    force_begin: u32,
    force_end: u32,
    #[comptime] bounded: bool,
    #[comptime] scalar: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * heads as usize {
        let token = i / heads as usize;
        let head = i % heads as usize;
        let mut count = blocks as usize;
        if comptime!(bounded) {
            let mut vi = token;
            if comptime!(scalar) {
                vi = 0;
            }
            count = usize::min(blocks as usize, i32::max(0, valid[vi]) as usize);
        }
        let mut ids = Array::<i32>::new(k);
        for rank in 0usize..k {
            let mut best: i32 = -1i32;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_forced = false;
            for block in 0..count {
                let mut used = false;
                for p in 0..rank {
                    used = used || ids[p] == block as i32;
                }
                if !used {
                    let mut score =
                        scores[(head * blocks as usize + block) * tokens as usize + token];
                    if score != score {
                        score = f32::NEG_INFINITY;
                    }
                    let forced =
                        block < force_begin as usize || block + force_end as usize >= count;
                    if best < 0
                        || (forced && !best_forced)
                        || (forced == best_forced
                            && (score > best_score
                                || (score == best_score && block < best as usize)))
                    {
                        best = block as i32;
                        best_score = score;
                        best_forced = forced;
                    }
                }
            }
            ids[rank] = best;
        }
        for rank in 0usize..k {
            let mut best = i32::MAX;
            let mut source: usize = 0usize;
            for p in 0usize..k {
                if ids[p] >= 0 && ids[p] < best {
                    best = ids[p];
                    source = p;
                }
            }
            let mut id: i32 = -1i32;
            if best != i32::MAX {
                id = best;
                ids[source] = -1;
            }
            out[i * k + rank] = id;
        }
    }
}

#[ruda(launch)]
pub fn live_page_counts(
    metadata: &Array<u32>,
    out: &mut Array<i32>,
    rows: u32,
    block_size: u32,
    #[comptime] causal: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < rows as usize {
        let mut n = metadata[i * 5 + 1];
        if comptime!(causal) {
            n = u32::min(n, i32::max(0, metadata[i * 5 + 2] as i32 + 1) as u32);
        }
        out[i] = ((n + block_size - 1) / block_size) as i32;
    }
}
