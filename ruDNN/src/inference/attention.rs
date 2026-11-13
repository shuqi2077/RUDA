//! Dense, packed-ragged, block/variable-block sparse and indexed attention share
//! one native online-softmax dataflow. CPU plans contain scheduling metadata
//! only; Q/K/V, masks, sparse selections, sinks and results stay on-device.
use super::{
    InferenceError, alloc,
    cascade::{AttentionState, LogBase},
    check, float, plane,
};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub scale: f32,
    pub causal: bool,
    pub window_left: i32,
    pub window_right: i32,
    pub soft_cap: f32,
    pub lse_base: LogBase,
}
pub struct RuntimeInputs<'a, R: Runtime> {
    pub byte_mask: Option<&'a RudaTensor<R>>,
    pub packed_mask: Option<&'a RudaTensor<R>>,
    pub indexed_keys: Option<&'a RudaTensor<R>>,
    pub indexed_positions: Option<&'a RudaTensor<R>>,
    pub alibi_slopes: Option<&'a RudaTensor<R>>,
    pub sinks: Option<&'a RudaTensor<R>>,
    pub bidirectional_ranges: Option<&'a RudaTensor<R>>,
}
impl<R: Runtime> Default for RuntimeInputs<'_, R> {
    fn default() -> Self {
        Self {
            byte_mask: None,
            packed_mask: None,
            indexed_keys: None,
            indexed_positions: None,
            alibi_slopes: None,
            sinks: None,
            bidirectional_ranges: None,
        }
    }
}
pub struct AttentionPlan<R: Runtime> {
    metadata: RudaTensor<R>,
    indices: RudaTensor<R>,
    queries: usize,
    keys: usize,
    max_history: usize,
    sparse: bool,
    q_indptr: Vec<u32>,
    kv_indptr: Vec<u32>,
}
impl<R: Runtime> AttentionPlan<R> {
    /// The same native online-softmax consumes decoded FP4/FP8 device values.
    pub fn run_quantized(
        &self,
        q: super::linear::Matrix<'_, R>,
        k: super::linear::Matrix<'_, R>,
        v: super::linear::Matrix<'_, R>,
        options: Options,
        extra: RuntimeInputs<'_, R>,
        output_dtype: DType,
    ) -> Result<AttentionState<R>, InferenceError> {
        let q = q.decode()?;
        let k = k.decode()?;
        let v = v.decode()?;
        let state = self.run(&q, &k, &v, options, extra)?;
        Ok(AttentionState {
            values: super::norm::cast(&state.values, output_dtype)?,
            lse: state.lse,
        })
    }
    /// Caller-owned device schedule [Q,5]: physical start, physical length,
    /// signed query position, sparse start, selected count. Values can change on
    /// replay, within the supplied physical/max-history capacities.
    pub fn device(
        metadata: &RudaTensor<R>,
        physical_keys: usize,
        max_history: usize,
    ) -> Result<Self, InferenceError> {
        check(metadata, metadata)?;
        if metadata.dtype != DType::U32
            || metadata.meta.num_dims() != 2
            || metadata.meta.shape()[1] != 5
            || physical_keys > i32::MAX as usize
            || max_history > i32::MAX as usize
        {
            return Err(InferenceError("invalid device attention schedule"));
        }
        let queries = metadata.meta.shape()[0];
        let indices = alloc(metadata, [0], DType::U32);
        Ok(Self {
            metadata: metadata.clone(),
            indices,
            queries,
            keys: physical_keys,
            max_history,
            sparse: false,
            q_indptr: Vec::new(),
            kv_indptr: Vec::new(),
        })
    }

    /// General read schedule. Each row is (physical key start, key length,
    /// query position relative to that interval). Intervals may overlap, which
    /// represents shared prefixes without duplicating any K/V payload.
    pub fn ranges(
        like: &RudaTensor<R>,
        physical_keys: usize,
        rows: &[(u32, u32, i32)],
    ) -> Result<Self, InferenceError> {
        if physical_keys > u32::MAX as usize || rows.len() > u32::MAX as usize / 5 {
            return Err(InferenceError("attention ranges exceed U32"));
        }
        let mut words = Vec::with_capacity(rows.len() * 5);
        let mut maximum: usize = 0usize;
        for &(start, len, pos) in rows {
            if len > i32::MAX as u32 || start as usize + len as usize > physical_keys {
                return Err(InferenceError("attention key interval out of range"));
            }
            maximum = maximum.max(len as usize);
            words.extend([start, len, pos as u32, 0, len]);
        }
        let metadata = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([rows.len(), 5]),
            like.client.create_from_slice(bytemuck::cast_slice(&words)),
            DType::U32,
        );
        let indices = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([0]),
            like.client.create_from_slice(&[]),
            DType::U32,
        );
        Ok(Self {
            metadata,
            indices,
            queries: rows.len(),
            keys: physical_keys,
            max_history: maximum,
            sparse: false,
            q_indptr: vec![0, rows.len() as u32],
            kv_indptr: vec![0, physical_keys as u32],
        })
    }
    pub fn ragged(
        like: &RudaTensor<R>,
        q_indptr: &[u32],
        kv_indptr: &[u32],
    ) -> Result<Self, InferenceError> {
        Self::build(like, q_indptr, kv_indptr, None)
    }
    /// Each query supplies local key positions in its own sequence. Lists may
    /// be empty; duplicates are rejected because they alter softmax semantics.
    pub fn indexed(
        like: &RudaTensor<R>,
        q_indptr: &[u32],
        kv_indptr: &[u32],
        query_keys: &[Vec<u32>],
    ) -> Result<Self, InferenceError> {
        Self::build(like, q_indptr, kv_indptr, Some(query_keys))
    }
    fn build(
        like: &RudaTensor<R>,
        q_indptr: &[u32],
        kv_indptr: &[u32],
        sparse: Option<&[Vec<u32>]>,
    ) -> Result<Self, InferenceError> {
        if q_indptr.len() < 2
            || q_indptr.len() != kv_indptr.len()
            || q_indptr[0] != 0
            || kv_indptr[0] != 0
            || q_indptr.windows(2).any(|p| p[0] > p[1])
            || kv_indptr.windows(2).any(|p| p[0] > p[1])
        {
            return Err(InferenceError("invalid ragged attention indptr"));
        }
        let queries = *q_indptr.last().unwrap() as usize;
        let keys = *kv_indptr.last().unwrap() as usize;
        if queries > u32::MAX as usize / 5 || sparse.is_some_and(|s| s.len() != queries) {
            return Err(InferenceError("ragged attention query metadata mismatch"));
        }
        let mut words = Vec::with_capacity(queries * 5);
        let mut indices = Vec::new();
        let mut max_history: usize = 0usize;
        for seq in 0..q_indptr.len() - 1 {
            let qs = q_indptr[seq] as usize;
            let qe = q_indptr[seq + 1] as usize;
            let ks = kv_indptr[seq] as usize;
            let ke = kv_indptr[seq + 1] as usize;
            let len = ke - ks;
            if len > i32::MAX as usize || qe - qs > i32::MAX as usize {
                return Err(InferenceError("attention position range exceeds I32"));
            }
            max_history = max_history.max(len);
            for row in qs..qe {
                let position = len as i64 - (qe - qs) as i64 + (row - qs) as i64;
                let mut begin: usize = 0usize;
                let mut count = len;
                if let Some(sparse) = sparse {
                    begin = indices.len();
                    let list = &sparse[row];
                    let mut unique = std::collections::BTreeSet::new();
                    for &id in list {
                        if id as usize >= len || !unique.insert(id) {
                            return Err(InferenceError("invalid/duplicate sparse attention key"));
                        }
                        indices.push(id);
                    }
                    count = list.len();
                    if indices.len() > u32::MAX as usize {
                        return Err(InferenceError("sparse attention index overflow"));
                    }
                }
                words.extend([
                    ks as u32,
                    len as u32,
                    position as i32 as u32,
                    begin as u32,
                    count as u32,
                ]);
            }
        }
        let metadata = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([queries, 5]),
            like.client.create_from_slice(bytemuck::cast_slice(&words)),
            DType::U32,
        );
        let indices = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([indices.len()]),
            like.client
                .create_from_slice(bytemuck::cast_slice(&indices)),
            DType::U32,
        );
        Ok(Self {
            metadata,
            indices,
            queries,
            keys,
            max_history,
            sparse: sparse.is_some(),
            q_indptr: q_indptr.to_vec(),
            kv_indptr: kv_indptr.to_vec(),
        })
    }
    /// BSR mask: row_ptr partitions block column IDs. A block may have a tail
    /// shorter than block_rows/block_cols; no padded Q/K/V data are gathered.
    pub fn block_sparse(
        like: &RudaTensor<R>,
        q_len: u32,
        kv_len: u32,
        row_ptr: &[u32],
        columns: &[u32],
        block_rows: usize,
        block_cols: usize,
    ) -> Result<Self, InferenceError> {
        if block_rows == 0
            || block_cols == 0
            || row_ptr.len() != (q_len as usize).div_ceil(block_rows) + 1
            || row_ptr.first() != Some(&0)
            || row_ptr.last().copied() != Some(columns.len() as u32)
            || row_ptr.windows(2).any(|p| p[0] > p[1])
        {
            return Err(InferenceError("invalid BSR schedule"));
        }
        let mut selections = Vec::with_capacity(q_len as usize);
        for row in 0..q_len as usize {
            let block = row / block_rows;
            let mut keys = Vec::new();
            for &col in &columns[row_ptr[block] as usize..row_ptr[block + 1] as usize] {
                let start = (col as usize)
                    .checked_mul(block_cols)
                    .ok_or(InferenceError("BSR column overflow"))?;
                if start >= kv_len as usize {
                    return Err(InferenceError("BSR column out of range"));
                }
                for token in start..(start + block_cols).min(kv_len as usize) {
                    keys.push(token as u32);
                }
            }
            selections.push(keys);
        }
        Self::indexed(like, &[0, q_len], &[0, kv_len], &selections)
    }
    /// Variable block boundaries, with block row CSR selecting key blocks.
    pub fn variable_block_sparse(
        like: &RudaTensor<R>,
        q_boundaries: &[u32],
        kv_boundaries: &[u32],
        row_ptr: &[u32],
        columns: &[u32],
    ) -> Result<Self, InferenceError> {
        if q_boundaries.len() < 2
            || kv_boundaries.len() < 2
            || q_boundaries[0] != 0
            || kv_boundaries[0] != 0
            || q_boundaries.windows(2).any(|p| p[0] >= p[1])
            || kv_boundaries.windows(2).any(|p| p[0] >= p[1])
            || row_ptr.len() != q_boundaries.len()
            || row_ptr.first() != Some(&0)
            || row_ptr.last().copied() != Some(columns.len() as u32)
            || row_ptr.windows(2).any(|p| p[0] > p[1])
        {
            return Err(InferenceError("invalid variable BSR schedule"));
        }
        let mut selections = Vec::new();
        for block in 0..q_boundaries.len() - 1 {
            let mut keys = Vec::new();
            for &col in &columns[row_ptr[block] as usize..row_ptr[block + 1] as usize] {
                let c = col as usize;
                if c + 1 >= kv_boundaries.len() {
                    return Err(InferenceError("variable BSR column out of range"));
                }
                keys.extend(kv_boundaries[c]..kv_boundaries[c + 1]);
            }
            for _ in q_boundaries[block]..q_boundaries[block + 1] {
                selections.push(keys.clone());
            }
        }
        Self::indexed(
            like,
            &[0, *q_boundaries.last().unwrap()],
            &[0, *kv_boundaries.last().unwrap()],
            &selections,
        )
    }
    pub fn query_indptr(&self) -> &[u32] {
        &self.q_indptr
    }
    pub fn kv_indptr(&self) -> &[u32] {
        &self.kv_indptr
    }
    pub fn run(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        options: Options,
        extra: RuntimeInputs<'_, R>,
    ) -> Result<AttentionState<R>, InferenceError> {
        self.execute(q, k, v, options, extra, None)
    }
    /// Fixed-address execution for graph capture/replay. Inputs and destinations
    /// must already be dense. The caller owns output aliasing and in-flight use.
    pub unsafe fn run_into(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        options: Options,
        extra: RuntimeInputs<'_, R>,
        output: &AttentionState<R>,
    ) -> Result<(), InferenceError> {
        for t in [q, k, v] {
            check(t, q)?;
        }
        self.execute(q, k, v, options, extra, Some(output))
            .map(|_| ())
    }
    fn execute(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        options: Options,
        extra: RuntimeInputs<'_, R>,
        destination: Option<&AttentionState<R>>,
    ) -> Result<AttentionState<R>, InferenceError> {
        let q = into_contiguous(q.clone());
        let k = into_contiguous(k.clone());
        let v = into_contiguous(v.clone());
        for t in [&q, &k, &v, &self.metadata] {
            check(t, &q)?;
        }
        float(&q)?;
        if q.meta.num_dims() != 3
            || k.meta.num_dims() != 3
            || v.meta.num_dims() != 3
            || q.dtype != k.dtype
            || q.dtype != v.dtype
            || q.meta.shape()[0] != self.queries
            || k.meta.shape()[0] != self.keys
            || v.meta.shape()[0] != self.keys
        {
            return Err(InferenceError("invalid packed attention shapes/dtypes"));
        }
        let heads = q.meta.shape()[1];
        let kv_heads = k.meta.shape()[1];
        let d = q.meta.shape()[2];
        let dv = v.meta.shape()[2];
        if heads == 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || d == 0
            || dv == 0
            || d > 1024
            || dv > 1024
            || k.meta.shape()[2] != d
            || v.meta.shape()[1] != kv_heads
            || !options.scale.is_finite()
            || !options.soft_cap.is_finite()
            || options.soft_cap < 0.0
            || options.window_left < -1
            || options.window_right < -1
            || self
                .queries
                .checked_mul(heads)
                .and_then(|n| n.checked_mul(dv))
                .filter(|&n| n <= u32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("invalid attention dimensions/options"));
        }
        if extra.byte_mask.is_some() && extra.packed_mask.is_some() {
            return Err(InferenceError("supply only one attention mask encoding"));
        }
        let mask = extra.byte_mask.or(extra.packed_mask);
        let mask_stride = if extra.packed_mask.is_some() {
            self.max_history.div_ceil(8)
        } else {
            self.max_history
        };
        if let Some(t) = mask {
            check(t, &q)?;
            if t.dtype != DType::U8 || t.meta.shape()[..] != [self.queries, mask_stride] {
                return Err(InferenceError(
                    "attention mask has the wrong row-padded extent",
                ));
            }
        }
        for t in extra.alibi_slopes.iter().chain(extra.sinks.iter()) {
            check(t, &q)?;
            if t.dtype != DType::F32 || t.meta.shape()[..] != [heads] {
                return Err(InferenceError("attention sinks/slopes must be F32 [heads]"));
            }
        }
        if let Some(t) = extra.bidirectional_ranges {
            check(t, &q)?;
            if t.dtype != DType::I32 || t.meta.shape()[..] != [self.queries, 2] {
                return Err(InferenceError(
                    "bidirectional ranges must be I32 [queries,2]",
                ));
            }
        }
        let mut selections: usize = 0usize;
        let mut selected_heads: usize = 1usize;
        if let Some(t) = extra.indexed_keys {
            check(t, &q)?;
            let rank = t.meta.num_dims();
            if t.dtype != DType::I32
                || !(rank == 2 || rank == 3)
                || t.meta.shape()[rank - 2] != self.queries
            {
                return Err(InferenceError(
                    "indexed keys must be I32 [queries,selections] or [selection_heads,queries,selections]",
                ));
            }
            selections = t.meta.shape()[rank - 1];
            if rank == 3 {
                selected_heads = t.meta.shape()[0];
            }
            if selected_heads == 0 || heads % selected_heads != 0 {
                return Err(InferenceError("selection heads must divide query heads"));
            }
        }
        if let Some(t) = extra.indexed_positions {
            check(t, &q)?;
            if extra.indexed_keys.is_none()
                || t.dtype != DType::I32
                || extra
                    .indexed_keys
                    .is_some_and(|k| k.meta.shape() != t.meta.shape())
            {
                return Err(InferenceError(
                    "indexed logical positions must match selected keys",
                ));
            }
        }
        let lanes = plane(&q, self.queries, heads)?;
        let (out, lse) = if let Some(state) = destination {
            check(&state.values, &q)?;
            check(&state.lse, &q)?;
            if state.values.dtype != q.dtype
                || state.values.meta.shape()[..] != [self.queries, heads, dv]
                || state.lse.dtype != DType::F32
                || state.lse.meta.shape()[..] != [self.queries, heads]
            {
                return Err(InferenceError("preallocated attention outputs mismatch"));
            }
            (state.values.clone(), state.lse.clone())
        } else {
            (
                alloc(&q, [self.queries, heads, dv], q.dtype),
                alloc(&q, [self.queries, heads], DType::F32),
            )
        };
        if self.queries != 0 {
            attend::launch::<R>(
                &q.client,
                RudaCount::Static(self.queries as u32, heads as u32, 1),
                RudaDim::new_1d(lanes),
                q.clone().into_array_arg(),
                k.into_array_arg(),
                v.into_array_arg(),
                self.metadata.clone().into_array_arg(),
                self.indices.clone().into_array_arg(),
                extra
                    .indexed_keys
                    .unwrap_or(&self.metadata)
                    .clone()
                    .into_array_arg(),
                extra
                    .indexed_positions
                    .unwrap_or(&self.metadata)
                    .clone()
                    .into_array_arg(),
                mask.unwrap_or(&self.metadata).clone().into_array_arg(),
                extra.alibi_slopes.unwrap_or(&q).clone().into_array_arg(),
                extra.sinks.unwrap_or(&q).clone().into_array_arg(),
                extra
                    .bidirectional_ranges
                    .unwrap_or(&self.metadata)
                    .clone()
                    .into_array_arg(),
                out.clone().into_array_arg(),
                lse.clone().into_array_arg(),
                heads as u32,
                kv_heads as u32,
                self.queries as u32,
                selected_heads as u32,
                mask_stride as u32,
                selections as u32,
                options.scale,
                options.soft_cap,
                options.window_left,
                options.window_right,
                d,
                dv,
                lanes as usize,
                self.sparse,
                extra.indexed_keys.is_some(),
                extra.indexed_positions.is_some(),
                options.causal,
                mask.is_some(),
                extra.packed_mask.is_some(),
                extra.alibi_slopes.is_some(),
                extra.sinks.is_some(),
                extra.bidirectional_ranges.is_some(),
                options.lse_base == LogBase::Two,
                q.dtype.into(),
            );
        }
        Ok(AttentionState { values: out, lse })
    }
}
/// Unbatched decode is a one-query prefill with right-aligned causal position.
/// Q may be [H,D] or [T,H,D]; keys and values are [K,Hkv,D/Dv].
pub fn single<R: Runtime>(
    query: &RudaTensor<R>,
    key: &RudaTensor<R>,
    value: &RudaTensor<R>,
    options: Options,
    extra: RuntimeInputs<'_, R>,
) -> Result<AttentionState<R>, InferenceError> {
    use ruda_kernel::tensor::reshape::reshape;
    let rank = query.meta.num_dims();
    if !(rank == 2 || rank == 3) || key.meta.num_dims() != 3 {
        return Err(InferenceError(
            "single attention needs Q[H,D]/[T,H,D], K/V[K,Hkv,D]",
        ));
    }
    let q = into_contiguous(query.clone());
    let q = if rank == 2 {
        reshape(
            q.clone(),
            Shape::from([1, q.meta.shape()[0], q.meta.shape()[1]]),
        )
    } else {
        q
    };
    let queries = q.meta.shape()[0];
    let keys = key.meta.shape()[0];
    if queries > u32::MAX as usize || keys > u32::MAX as usize {
        return Err(InferenceError("single attention exceeds U32"));
    }
    let plan = AttentionPlan::ragged(&q, &[0, queries as u32], &[0, keys as u32])?;
    let out = plan.run(&q, key, value, options, extra)?;
    if rank == 2 {
        let h = out.values.meta.shape()[1];
        let d = out.values.meta.shape()[2];
        Ok(AttentionState {
            values: reshape(out.values, Shape::from([h, d])),
            lse: reshape(out.lse, Shape::from([h])),
        })
    } else {
        Ok(out)
    }
}
/// Fixed-address dense/ragged/sparse attention with retained native format
/// decoders. Sinks/masks/selection tables may change on each graph replay.
pub struct PreparedAttention<'a, R: Runtime> {
    plan: &'a AttentionPlan<R>,
    query: super::linear::PreparedMatrix<'a, R>,
    key: super::linear::PreparedMatrix<'a, R>,
    value: super::linear::PreparedMatrix<'a, R>,
    accumulated: AttentionState<R>,
    pub output: AttentionState<R>,
}
impl<'a, R: Runtime> PreparedAttention<'a, R> {
    pub fn prepare(
        plan: &'a AttentionPlan<R>,
        q: super::linear::Matrix<'a, R>,
        k: super::linear::Matrix<'a, R>,
        v: super::linear::Matrix<'a, R>,
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
            || k.meta.shape()[1] == 0
            || q.meta.shape()[1] == 0
            || q.meta.shape()[1] % k.meta.shape()[1] != 0
            || k.meta.shape()[1] != v.meta.shape()[1]
            || q.meta.shape()[2] != k.meta.shape()[2]
            || q.meta.shape()[2] == 0
            || v.meta.shape()[2] == 0
        {
            return Err(InferenceError("prepared attention geometry mismatch"));
        }
        let shape = [plan.queries, q.meta.shape()[1], v.meta.shape()[2]];
        let lse = alloc(q, [plan.queries, q.meta.shape()[1]], DType::F32);
        let accumulated = AttentionState {
            values: alloc(q, shape, DType::F32),
            lse: lse.clone(),
        };
        let output = AttentionState {
            values: alloc(q, shape, dtype),
            lse,
        };
        float(&output.values)?;
        Ok(Self {
            plan,
            query,
            key,
            value,
            accumulated,
            output,
        })
    }
    pub unsafe fn run(
        &self,
        options: Options,
        extra: RuntimeInputs<'_, R>,
    ) -> Result<&AttentionState<R>, InferenceError> {
        unsafe {
            self.query.run()?;
            self.key.run()?;
            self.value.run()?;
            self.plan.run_into(
                &self.query.output,
                &self.key.output,
                &self.value.output,
                options,
                extra,
                &self.accumulated,
            )?;
            super::norm::cast_into(&self.accumulated.values, &self.output.values)?;
        }
        Ok(&self.output)
    }
}
/// Prepared MLA retains shared latent K/V rather than replicating it per head.
pub struct PreparedMla<'a, R: Runtime> {
    plan: &'a AttentionPlan<R>,
    query: super::linear::PreparedMatrix<'a, R>,
    query_rope: super::linear::PreparedMatrix<'a, R>,
    latent: super::linear::PreparedMatrix<'a, R>,
    cache_rope: super::linear::PreparedMatrix<'a, R>,
    combined_q: RudaTensor<R>,
    combined_k: RudaTensor<R>,
    value_view: RudaTensor<R>,
    accumulated: AttentionState<R>,
    pub output: AttentionState<R>,
}
impl<'a, R: Runtime> PreparedMla<'a, R> {
    pub fn prepare(
        plan: &'a AttentionPlan<R>,
        q: super::linear::Matrix<'a, R>,
        qp: super::linear::Matrix<'a, R>,
        latent: super::linear::Matrix<'a, R>,
        kp: super::linear::Matrix<'a, R>,
        dtype: DType,
    ) -> Result<Self, InferenceError> {
        let query = super::linear::PreparedMatrix::prepare(q)?;
        let query_rope = super::linear::PreparedMatrix::prepare(qp)?;
        let latent = super::linear::PreparedMatrix::prepare(latent)?;
        let cache_rope = super::linear::PreparedMatrix::prepare(kp)?;
        let q = &query.output;
        let qp = &query_rope.output;
        let c = &latent.output;
        let cp = &cache_rope.output;
        for t in [qp, c, cp, &plan.metadata] {
            check(t, q)?;
        }
        if q.meta.num_dims() != 3
            || qp.meta.num_dims() != 3
            || c.meta.num_dims() != 2
            || cp.meta.num_dims() != 2
            || q.meta.shape()[0] != plan.queries
            || c.meta.shape()[0] != plan.keys
            || q.meta.shape()[0..2] != qp.meta.shape()[0..2]
            || c.meta.shape()[0] != cp.meta.shape()[0]
            || q.meta.shape()[2] != c.meta.shape()[1]
            || qp.meta.shape()[2] != cp.meta.shape()[1]
        {
            return Err(InferenceError(
                "prepared MLA latent/rotary dimensions mismatch",
            ));
        }
        let (t, h, dc, dr) = (
            q.meta.shape()[0],
            q.meta.shape()[1],
            q.meta.shape()[2],
            qp.meta.shape()[2],
        );
        if h == 0 || dc == 0 || dr == 0 {
            return Err(InferenceError("prepared MLA axes must be positive"));
        }
        let combined_q = alloc(q, [t, h, dc + dr], DType::F32);
        let combined_k = alloc(q, [plan.keys, 1, dc + dr], DType::F32);
        let value_view =
            ruda_kernel::tensor::reshape::reshape(c.clone(), Shape::from([plan.keys, 1, dc]));
        let lse = alloc(q, [t, h], DType::F32);
        let accumulated = AttentionState {
            values: alloc(q, [t, h, dc], DType::F32),
            lse: lse.clone(),
        };
        let output = AttentionState {
            values: alloc(q, [t, h, dc], dtype),
            lse,
        };
        float(&output.values)?;
        Ok(Self {
            plan,
            query,
            query_rope,
            latent,
            cache_rope,
            combined_q,
            combined_k,
            value_view,
            accumulated,
            output,
        })
    }
    pub unsafe fn run(
        &self,
        options: Options,
        extra: RuntimeInputs<'_, R>,
    ) -> Result<&AttentionState<R>, InferenceError> {
        unsafe {
            self.query.run()?;
            self.query_rope.run()?;
            self.latent.run()?;
            self.cache_rope.run()?;
        }
        let q = &self.query.output;
        let dc = q.meta.shape()[2];
        let dr = self.query_rope.output.meta.shape()[2];
        let n = self
            .combined_q
            .meta
            .num_elements()
            .max(self.combined_k.meta.num_elements());
        if n > 0 {
            let dim = RudaDim::new(q.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
            concat_mla::launch::<R>(
                &q.client,
                count,
                dim,
                q.clone().into_array_arg(),
                self.query_rope.output.clone().into_array_arg(),
                self.latent.output.clone().into_array_arg(),
                self.cache_rope.output.clone().into_array_arg(),
                self.combined_q.clone().into_array_arg(),
                self.combined_k.clone().into_array_arg(),
                self.plan.queries as u32,
                q.meta.shape()[1] as u32,
                self.plan.keys as u32,
                dc as u32,
                dr as u32,
            );
        }
        unsafe {
            self.plan.run_into(
                &self.combined_q,
                &self.combined_k,
                &self.value_view,
                options,
                extra,
                &self.accumulated,
            )?;
            super::norm::cast_into(&self.accumulated.values, &self.output.values)?;
        }
        Ok(&self.output)
    }
}
/// MLA contracts the shared latent and positional components independently in
/// the dot product, then returns the latent V aggregate. No dense K/V expansion
/// into all query heads is performed.
pub fn mla<R: Runtime>(
    plan: &AttentionPlan<R>,
    query_latent: &RudaTensor<R>,
    query_rope: &RudaTensor<R>,
    cache_latent: &RudaTensor<R>,
    cache_rope: &RudaTensor<R>,
    options: Options,
    extra: RuntimeInputs<'_, R>,
) -> Result<AttentionState<R>, InferenceError> {
    use ruda_kernel::tensor::reshape::reshape;
    let q = super::norm::cast(query_latent, DType::F32)?;
    let qp = super::norm::cast(query_rope, DType::F32)?;
    let c = super::norm::cast(cache_latent, DType::F32)?;
    let cp = super::norm::cast(cache_rope, DType::F32)?;
    for t in [&qp, &c, &cp] {
        check(t, &q)?;
    }
    if q.meta.num_dims() != 3
        || qp.meta.num_dims() != 3
        || q.meta.shape()[0..2] != qp.meta.shape()[0..2]
        || c.meta.num_dims() != 2
        || cp.meta.num_dims() != 2
        || c.meta.shape()[0] != cp.meta.shape()[0]
        || q.meta.shape()[2] != c.meta.shape()[1]
        || qp.meta.shape()[2] != cp.meta.shape()[1]
    {
        return Err(InferenceError("MLA latent/positional shape mismatch"));
    }
    let queries = q.meta.shape()[0];
    let heads = q.meta.shape()[1];
    let keys = c.meta.shape()[0];
    let dc = c.meta.shape()[1];
    let dr = cp.meta.shape()[1];
    let combined_q = alloc(&q, [queries, heads, dc + dr], DType::F32);
    let combined_k = alloc(&q, [keys, 1, dc + dr], DType::F32);
    let n = combined_q
        .meta
        .num_elements()
        .max(combined_k.meta.num_elements());
    if n > u32::MAX as usize {
        return Err(InferenceError("MLA combined vectors exceed U32"));
    }
    if n > 0 {
        let dim = RudaDim::new(q.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
        concat_mla::launch::<R>(
            &q.client,
            count,
            dim,
            q.clone().into_array_arg(),
            qp.into_array_arg(),
            c.clone().into_array_arg(),
            cp.into_array_arg(),
            combined_q.clone().into_array_arg(),
            combined_k.clone().into_array_arg(),
            queries as u32,
            heads as u32,
            keys as u32,
            dc as u32,
            dr as u32,
        );
    }
    let v = reshape(c, Shape::from([keys, 1, dc]));
    let out = plan.run(&combined_q, &combined_k, &v, options, extra)?;
    Ok(AttentionState {
        values: super::norm::cast(&out.values, query_latent.dtype)?,
        lse: out.lse,
    })
}
#[ruda(launch)]
pub fn concat_mla(
    q: &Array<f32>,
    qp: &Array<f32>,
    c: &Array<f32>,
    cp: &Array<f32>,
    qo: &mut Array<f32>,
    ko: &mut Array<f32>,
    queries: u32,
    heads: u32,
    keys: u32,
    dc: u32,
    dr: u32,
) {
    let i = ABSOLUTE_POS as usize;
    let width = (dc + dr) as usize;
    let row = i / width;
    let col = i % width;
    if i < queries as usize * heads as usize * width {
        let mut value: f32 = 0.0f32;
        if col < dc as usize {
            value = q[row * dc as usize + col];
        } else {
            value = qp[row * dr as usize + col - dc as usize];
        }
        qo[i] = value;
    }
    if i < keys as usize * width {
        let mut value: f32 = 0.0f32;
        if col < dc as usize {
            value = c[row * dc as usize + col];
        } else {
            value = cp[row * dr as usize + col - dc as usize];
        }
        ko[i] = value;
    }
}
/// Simultaneous prefill and decode orchestration reuses independently planned
/// native paths and returns independent output lifetimes.
pub struct PodPlan<R: Runtime> {
    pub prefill: AttentionPlan<R>,
    pub decode: AttentionPlan<R>,
}
impl<R: Runtime> PodPlan<R> {
    pub fn run(
        &self,
        prefill: (&RudaTensor<R>, &RudaTensor<R>, &RudaTensor<R>),
        decode: (&RudaTensor<R>, &RudaTensor<R>, &RudaTensor<R>),
        prefill_options: Options,
        decode_options: Options,
    ) -> Result<(AttentionState<R>, AttentionState<R>), InferenceError> {
        let p = self.prefill.run(
            prefill.0,
            prefill.1,
            prefill.2,
            prefill_options,
            RuntimeInputs::default(),
        )?;
        let d = self.decode.run(
            decode.0,
            decode.1,
            decode.2,
            decode_options,
            RuntimeInputs::default(),
        )?;
        Ok((p, d))
    }
}

#[ruda(launch)]
pub fn attend<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    meta: &Array<u32>,
    sparse_keys: &Array<u32>,
    selected: &Array<i32>,
    logical_positions: &Array<i32>,
    mask: &Array<u8>,
    slopes: &Array<f32>,
    sinks: &Array<f32>,
    bidir: &Array<i32>,
    out: &mut Array<F>,
    lse: &mut Array<f32>,
    heads: u32,
    kv_heads: u32,
    queries: u32,
    selected_heads: u32,
    mask_stride: u32,
    selections: u32,
    scale: f32,
    cap: f32,
    window_left: i32,
    window_right: i32,
    #[comptime] d: usize,
    #[comptime] dv: usize,
    #[comptime] lanes: usize,
    #[comptime] sparse: bool,
    #[comptime] indexed: bool,
    #[comptime] logical: bool,
    #[comptime] causal: bool,
    #[comptime] masked: bool,
    #[comptime] packed: bool,
    #[comptime] alibi: bool,
    #[comptime] sink: bool,
    #[comptime] bidirectional: bool,
    #[comptime] base_two: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let kv_head = head / (heads / kv_heads) as usize;
    let start = meta[row * 5] as usize;
    let length = meta[row * 5 + 1] as usize;
    let position = meta[row * 5 + 2] as i32;
    let begin = meta[row * 5 + 3] as usize;
    let mut count = meta[row * 5 + 4] as usize;
    if comptime!(indexed) {
        count = selections as usize;
    }
    let mut query = Array::<f32>::new((d + lanes - 1) / lanes);
    let mut acc = Array::<f32>::new((dv + lanes - 1) / lanes);
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
    if comptime!(sink) {
        maximum = sinks[head];
        if maximum != f32::NEG_INFINITY {
            denominator = 1.0;
        }
        if maximum != maximum || maximum == f32::INFINITY {
            bad = true;
        }
    }
    for slot in 0..count {
        let mut token = slot as i32;
        if comptime!(sparse) {
            token = sparse_keys[begin + slot] as i32;
        }
        if comptime!(indexed) {
            token = selected[((head / (heads / selected_heads) as usize) * queries as usize + row)
                * selections as usize
                + slot];
        }
        let mut logical_token = token;
        if comptime!(logical) {
            logical_token = logical_positions[((head / (heads / selected_heads) as usize)
                * queries as usize
                + row)
                * selections as usize
                + slot];
        }
        let mut visible = token >= 0 && token < length as i32;
        if comptime!(causal) {
            let mut allowed = logical_token <= position;
            if comptime!(bidirectional) {
                allowed = allowed
                    || (logical_token >= bidir[row * 2] && logical_token < bidir[row * 2 + 1]);
            }
            visible = visible && allowed;
        }
        if window_left >= 0 {
            visible = visible && logical_token as i64 >= position as i64 - window_left as i64;
        }
        if window_right >= 0 {
            visible = visible && logical_token as i64 <= position as i64 + window_right as i64;
        }
        if visible {
            if comptime!(masked) {
                if comptime!(packed) {
                    visible = (u32::cast_from(
                        mask[row * mask_stride as usize + logical_token as usize / 8],
                    ) & (1u32 << (logical_token as u32 % 8)))
                        != 0;
                } else {
                    visible = mask[row * mask_stride as usize + logical_token as usize] != 0;
                }
            }
            if visible {
                let cache = (start + token as usize) * kv_heads as usize + kv_head;
                let mut dot: f32 = 0.0f32;
                #[unroll]
                for j in 0usize..((d + lanes - 1) / lanes) {
                    let col = lane + j * lanes;
                    if col < d {
                        dot = fma(query[j], f32::cast_from(k[cache * d + col]), dot);
                    }
                }
                let mut score = plane_sum(dot) * scale;
                if cap > 0.0 {
                    score = cap * (score / cap).tanh();
                }
                if comptime!(alibi) {
                    score += slopes[head] * (logical_token as i64 - position as i64) as f32;
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
    }
    let output = row * heads as usize + head;
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
            out[output * dv + col] = F::cast_from(value);
        }
    }
    if lane == 0 {
        let mut result = f32::NEG_INFINITY;
        if denominator != 0.0 {
            result = maximum + denominator.ln();
        }
        if bad {
            result = f32::NAN;
        }
        if comptime!(base_two) {
            result *= core::f32::consts::LOG2_E;
        }
        lse[output] = result;
    }
}

/// Compatibility sizing for external Split-KV scratch layouts. Native RUDA
/// plans own their own buffers and do not require these vendor ticket arrays.
pub fn get_dcp_spec_workspace_size_bytes(
    batch: usize,
    query_length: usize,
    query_heads: usize,
    splits: usize,
    head_dim: usize,
) -> Result<usize, InferenceError> {
    if batch == 0
        || query_length == 0
        || query_heads == 0
        || !(2..=16).contains(&splits)
        || !matches!(head_dim, 128 | 256)
    {
        return Err(InferenceError("invalid DCP split workspace geometry"));
    }
    batch
        .checked_mul(query_length)
        .and_then(|n| n.checked_mul(query_heads))
        .and_then(|n| n.checked_mul(splits))
        .and_then(|n| n.checked_mul(head_dim * 2 + 4))
        .ok_or(InferenceError("DCP scratch byte count overflow"))
}
pub fn get_dcp_spec_counter_bytes(
    batch: usize,
    query_length: usize,
    kv_heads: usize,
) -> Result<usize, InferenceError> {
    if batch == 0 || query_length == 0 || kv_heads == 0 {
        return Err(InferenceError("DCP counter axes must be positive"));
    }
    batch
        .checked_mul(query_length)
        .and_then(|n| n.checked_mul(kv_heads))
        .and_then(|n| n.checked_mul(4))
        .ok_or(InferenceError("DCP counter byte count overflow"))
}
