//! Native ICP index scoring and candidate carriers. No FP8/CUDA indexer binding.
use super::{InferenceError, alloc, check, linear::Matrix, msa::MsaPlan};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
pub fn score_columns(
    max_length: usize,
    page_size: usize,
    compact: bool,
) -> Result<usize, InferenceError> {
    if !matches!(page_size, 128 | 256) || compact && page_size != 256 {
        return Err(InferenceError("ICP page size/compact layout invalid"));
    }
    let blocks = max_length.div_ceil(128);
    Ok(if compact { blocks.div_ceil(2) } else { blocks })
}
pub fn decode_score_columns(max_length: usize, page_size: usize) -> Result<usize, InferenceError> {
    if !matches!(page_size, 128 | 256) {
        return Err(InferenceError("ICP page size must be 128/256"));
    }
    Ok(max_length
        .div_ceil(4 * page_size)
        .saturating_mul(4)
        .min(8192))
}
/// K is this rank's [pages,P/world,D] shard. Scores [T,H,C] contain
/// per-B128 maxima (or local P256 columns in compact mode), never softmax.
pub fn score<R: Runtime>(
    plan: &MsaPlan<R>,
    q: Matrix<'_, R>,
    k: Matrix<'_, R>,
    block_table: &RudaTensor<R>,
    rank: usize,
    world: usize,
    page_size: usize,
    columns: usize,
    compact: bool,
) -> Result<RudaTensor<R>, InferenceError> {
    plan.refresh()?;
    let q = q.decode()?;
    let k = k.decode()?;
    for t in [&k, block_table, &plan.q_indptr, &plan.metadata] {
        check(t, &q)?;
    }
    if !matches!(world, 2 | 4)
        || rank >= world
        || !matches!(page_size, 128 | 256)
        || compact && page_size != 256
        || q.meta.num_dims() != 3
        || k.meta.num_dims() != 3
        || k.meta.shape()[1] != page_size / world
        || k.meta.shape()[2] != q.meta.shape()[2]
        || block_table.dtype != DType::I32
        || block_table.meta.num_dims() != 2
        || block_table.meta.shape()[0] + 1 != plan.q_indptr.meta.num_elements()
        || columns == 0
    {
        return Err(InferenceError("ICP sharded geometry mismatch"));
    }
    let tokens = q.meta.shape()[0];
    let heads = q.meta.shape()[1];
    let width = q.meta.shape()[2];
    if plan.metadata.meta.shape()[..] != [tokens, 5] {
        return Err(InferenceError("ICP plan query capacity mismatch"));
    }
    let n = tokens
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(columns))
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(InferenceError("ICP scores exceed U32"))?;
    let out = alloc(&q, [tokens, heads, columns], DType::F32);
    if n > 0 {
        let dim = RudaDim::new(q.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
        shard_scores::launch::<R>(
            &q.client,
            count,
            dim,
            q.clone().into_array_arg(),
            k.into_array_arg(),
            block_table.clone().into_array_arg(),
            plan.q_indptr.clone().into_array_arg(),
            plan.metadata.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            tokens as u32,
            heads as u32,
            width as u32,
            columns as u32,
            (plan.q_indptr.meta.num_elements() - 1) as u32,
            block_table.meta.shape()[1] as u32,
            rank as u32,
            world as u32,
            page_size as u32,
            compact,
        );
    }
    Ok(out)
}
/// Local Top16 carrier [H,T,33]: raw score bits, ascending global block IDs,
/// ready word. Invalid/padding candidates retain (-infinity,-1).
pub fn select_carrier<R: Runtime>(
    scores: &RudaTensor<R>,
    valid_pages: &RudaTensor<R>,
    index_scale: usize,
    index_offset: usize,
    force_begin: usize,
    force_end: usize,
) -> Result<RudaTensor<R>, InferenceError> {
    check(scores, scores)?;
    check(valid_pages, scores)?;
    if scores.dtype != DType::F32
        || scores.meta.num_dims() != 3
        || valid_pages.dtype != DType::I32
        || valid_pages.meta.shape()[..] != [scores.meta.shape()[0]]
        || index_scale == 0
        || index_scale > i32::MAX as usize
        || index_offset > i32::MAX as usize
        || force_begin + force_end > 16
    {
        return Err(InferenceError("ICP carrier selection operands invalid"));
    }
    let tokens = scores.meta.shape()[0];
    let heads = scores.meta.shape()[1];
    let cols = scores.meta.shape()[2];
    let n = tokens * heads;
    let out = alloc(scores, [heads, tokens, 33], DType::I32);
    if out.meta.num_elements() > u32::MAX as usize {
        return Err(InferenceError("ICP carrier exceeds U32"));
    }
    if n > 0 {
        let dim = RudaDim::new(scores.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&scores.client, n, dim);
        carrier::launch::<R>(
            &scores.client,
            count,
            dim,
            scores.clone().into_array_arg(),
            valid_pages.clone().into_array_arg(),
            out.clone().into_array_arg(),
            tokens as u32,
            heads as u32,
            cols as u32,
            index_scale as u32,
            index_offset as u32,
            force_begin as u32,
            force_end as u32,
        );
    }
    Ok(out)
}
/// Merge exchanged carriers [world,owned_heads,capacity,33]. Duplicate IDs keep
/// their maximum score. Forced boundaries rank first, then score, then lower ID.
pub fn merge<R: Runtime>(
    received: &RudaTensor<R>,
    valid_pages: &RudaTensor<R>,
    tokens: usize,
    force_begin: usize,
    force_end: usize,
    prefill: bool,
) -> Result<RudaTensor<R>, InferenceError> {
    check(received, received)?;
    check(valid_pages, received)?;
    if received.dtype != DType::I32
        || received.meta.num_dims() != 4
        || received.meta.shape()[3] != 33
        || received.meta.shape()[2] < tokens
        || valid_pages.dtype != DType::I32
        || valid_pages.meta.shape()[..] != [tokens]
        || force_begin + force_end > 16
    {
        return Err(InferenceError("ICP received carriers invalid"));
    }
    let world = received.meta.shape()[0];
    let heads = received.meta.shape()[1];
    let out = alloc(received, [tokens, heads, 16], DType::I32);
    let n = tokens * heads;
    if n > 0 {
        let dim = RudaDim::new(received.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&received.client, n, dim);
        merge_carriers::launch::<R>(
            &received.client,
            count,
            dim,
            received.clone().into_array_arg(),
            valid_pages.clone().into_array_arg(),
            out.clone().into_array_arg(),
            tokens as u32,
            heads as u32,
            world as u32,
            received.meta.shape()[2] as u32,
            force_begin as u32,
            force_end as u32,
            prefill,
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn shard_scores(
    q: &Array<f32>,
    k: &Array<f32>,
    table: &Array<i32>,
    qcu: &Array<i32>,
    meta: &Array<u32>,
    out: &mut Array<f32>,
    n: u32,
    tokens: u32,
    heads: u32,
    width: u32,
    columns: u32,
    sequences: u32,
    table_width: u32,
    rank: u32,
    world: u32,
    page_size: u32,
    #[comptime] compact: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % columns as usize;
        let head = i / columns as usize % heads as usize;
        let token = i / (columns as usize * heads as usize);
        let mut seq: usize = 0usize;
        let mut hi = sequences as usize;
        while seq < hi {
            let mid = (seq + hi + 1) / 2;
            if qcu[mid] <= token as i32 {
                seq = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut maximum = f32::NEG_INFINITY;
        if seq < sequences as usize {
            let length = meta[token * 5 + 1] as usize;
            let position = meta[token * 5 + 2] as i32;
            let mut block = col;
            if comptime!(compact) {
                block = col * 2 + (rank * 2 / world) as usize;
            }
            for j in 0usize..128 {
                let logical = block * 128 + j;
                let local = logical % page_size as usize;
                let rank_begin = (rank * page_size / world) as usize;
                let rank_end = ((rank + 1) * page_size / world) as usize;
                let page = logical / page_size as usize;
                if logical < length
                    && logical as i32 <= position
                    && local >= rank_begin
                    && local < rank_end
                    && page < table_width as usize
                {
                    let physical = table[seq * table_width as usize + page];
                    if physical >= 0 {
                        let mut dot: f32 = 0.0f32;
                        for d in 0..width as usize {
                            dot = fma(
                                q[(token * heads as usize + head) * width as usize + d],
                                k[(physical as usize * (page_size / world) as usize + local
                                    - rank_begin)
                                    * width as usize
                                    + d],
                                dot,
                            );
                        }
                        maximum = f32::max(maximum, dot);
                    }
                }
            }
        }
        out[i] = maximum;
    }
}
#[ruda(launch)]
pub fn carrier(
    scores: &Array<f32>,
    valid: &Array<i32>,
    out: &mut Array<i32>,
    tokens: u32,
    heads: u32,
    columns: u32,
    index_scale: u32,
    index_offset: u32,
    force_begin: u32,
    force_end: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * heads as usize {
        let token = i / heads as usize;
        let head = i % heads as usize;
        let count = i32::min(i32::max(valid[token], 0), 8192);
        let mut chosen = Array::<i32>::new(16usize);
        let mut values = Array::<f32>::new(16usize);
        for rank in 0usize..16 {
            let mut best: i32 = -1i32;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_forced = false;
            for col in 0..columns as usize {
                let id = col as i64 * index_scale as i64 + index_offset as i64;
                let mut used = false;
                for p in 0..rank {
                    used = used || chosen[p] as i64 == id;
                }
                if !used && id < count as i64 {
                    let mut score =
                        scores[(token * heads as usize + head) * columns as usize + col];
                    if score != score {
                        score = f32::NEG_INFINITY;
                    }
                    let forced = id < force_begin as i64 || id + force_end as i64 >= count as i64;
                    if best < 0
                        || forced && !best_forced
                        || forced == best_forced
                            && (score > best_score || score == best_score && id < best as i64)
                    {
                        best = id as i32;
                        best_score = score;
                        best_forced = forced;
                    }
                }
            }
            chosen[rank] = best;
            values[rank] = best_score;
        }
        let base = (head * tokens as usize + token) * 33;
        for rank in 0usize..16 {
            let mut minimum = i32::MAX;
            let mut selected: usize = 0usize;
            for p in 0usize..16 {
                if chosen[p] >= 0 && chosen[p] < minimum {
                    minimum = chosen[p];
                    selected = p;
                }
            }
            let mut id: i32 = -1i32;
            let mut score = f32::NEG_INFINITY;
            if minimum != i32::MAX {
                id = minimum;
                score = values[selected];
                chosen[selected] = -1;
            }
            out[base + rank] = i32::reinterpret(score);
            out[base + 16 + rank] = id;
        }
        out[base + 32] = 1;
    }
}
#[ruda(launch)]
pub fn merge_carriers(
    received: &Array<i32>,
    valid: &Array<i32>,
    out: &mut Array<i32>,
    tokens: u32,
    heads: u32,
    world: u32,
    capacity: u32,
    force_begin: u32,
    force_end: u32,
    #[comptime] prefill: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * heads as usize {
        let token = i / heads as usize;
        let head = i % heads as usize;
        let count = i32::min(i32::max(valid[token], 0), 8192);
        let mut ids = Array::<i32>::new(16usize);
        for rank in 0usize..16 {
            let mut best: i32 = -1i32;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_forced = false;
            for peer in 0..world as usize {
                let base = ((peer * heads as usize + head) * capacity as usize + token) * 33;
                for p in 0usize..16 {
                    let id = received[base + 16 + p];
                    let mut used = false;
                    for prev in 0..rank {
                        used = used || ids[prev] == id;
                    }
                    if !used && id >= 0 && id < count {
                        let mut score = f32::reinterpret(received[base + p]);
                        if score != score {
                            score = f32::NEG_INFINITY;
                        }
                        let forced = id < force_begin as i32 || id + force_end as i32 >= count;
                        if best < 0
                            || forced && !best_forced
                            || forced == best_forced
                                && (score > best_score || score == best_score && id < best)
                        {
                            best = id;
                            best_score = score;
                            best_forced = forced;
                        }
                    }
                }
            }
            ids[rank] = best;
        }
        for rank in 0usize..16 {
            let mut minimum = i32::MAX;
            let mut selected: usize = 0usize;
            for p in 0usize..16 {
                if ids[p] >= 0 && ids[p] < minimum {
                    minimum = ids[p];
                    selected = p;
                }
            }
            let mut id: i32 = -1i32;
            if minimum != i32::MAX {
                id = minimum;
                ids[selected] = -1;
            }
            if comptime!(prefill) {
                if count == 0 && rank == 0 {
                    id = 0;
                }
            }
            out[i * 16 + rank] = id;
        }
    }
}

/// Complete retained ICP score/select/publish/receive/merge pipeline. Physical
/// page tables, valid-block counts and all candidate traffic stay on the GPU.
pub struct PreparedIcp<'a, R: Runtime> {
    plan: &'a MsaPlan<R>,
    query: super::linear::PreparedMatrix<'a, R>,
    key: super::linear::PreparedMatrix<'a, R>,
    table: super::metadata::PreparedIndices<R>,
    valid: super::metadata::PreparedIndices<R>,
    comm: &'a super::peer_candidates::CandidateComm<R>,
    pub scores: RudaTensor<R>,
    pub carrier: RudaTensor<R>,
    pub received: RudaTensor<R>,
    pub selected: RudaTensor<R>,
    rank: usize,
    world: usize,
    page_size: usize,
    compact: bool,
    index_scale: usize,
    index_offset: usize,
    force_begin: usize,
    force_end: usize,
}
impl<'a, R: Runtime> PreparedIcp<'a, R> {
    pub fn prepare(
        plan: &'a MsaPlan<R>,
        query: Matrix<'a, R>,
        key: Matrix<'a, R>,
        table: &RudaTensor<R>,
        valid_pages: &RudaTensor<R>,
        comm: &'a super::peer_candidates::CandidateComm<R>,
        rank: usize,
        world: usize,
        page_size: usize,
        columns: usize,
        compact: bool,
        index_scale: usize,
        index_offset: usize,
        force_begin: usize,
        force_end: usize,
    ) -> Result<Self, InferenceError> {
        let query = super::linear::PreparedMatrix::prepare(query)?;
        let key = super::linear::PreparedMatrix::prepare(key)?;
        let q = &query.output;
        let k = &key.output;
        let table = super::metadata::PreparedIndices::prepare(table, DType::I32)?;
        let valid = super::metadata::PreparedIndices::prepare(valid_pages, DType::I32)?;
        for t in [
            k,
            &table.output,
            &valid.output,
            &comm.addresses,
            &plan.metadata,
        ] {
            check(t, q)?;
        }
        if !matches!(world, 2 | 4)
            || rank >= world
            || !matches!(page_size, 128 | 256)
            || compact && page_size != 256
            || q.meta.num_dims() != 3
            || k.meta.num_dims() != 3
            || k.meta.shape()[1] != page_size / world
            || k.meta.shape()[2] != q.meta.shape()[2]
            || table.output.meta.num_dims() != 2
            || table.output.meta.shape()[0] + 1 != plan.q_indptr.meta.num_elements()
            || plan.metadata.meta.shape()[..] != [q.meta.shape()[0], 5]
            || valid.output.meta.shape()[..] != [q.meta.shape()[0]]
            || columns == 0
            || index_scale == 0
            || force_begin + force_end > 16
            || comm.layout.world != world
            || comm.layout.heads_per_rank * world != q.meta.shape()[1]
            || comm.layout.capacity < q.meta.shape()[0]
        {
            return Err(InferenceError(
                "prepared ICP geometry/communication mismatch",
            ));
        }
        let (t, h) = (q.meta.shape()[0], q.meta.shape()[1]);
        for shape in [[t, h, columns], [t, h, 33]] {
            if shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
            {
                return Err(InferenceError("ICP retained buffers exceed U32"));
            }
        }
        let scores = alloc(q, [t, h, columns], DType::F32);
        let carrier = alloc(q, [h, t, 33], DType::I32);
        let received = alloc(q, [world, comm.layout.heads_per_rank, t, 33], DType::I32);
        let selected = alloc(q, [t, comm.layout.heads_per_rank, 16], DType::I32);
        Ok(Self {
            plan,
            query,
            key,
            table,
            valid,
            comm,
            scores,
            carrier,
            received,
            selected,
            rank,
            world,
            page_size,
            compact,
            index_scale,
            index_offset,
            force_begin,
            force_end,
        })
    }
    pub unsafe fn score_select(&self) -> Result<&RudaTensor<R>, InferenceError> {
        self.plan.refresh()?;
        unsafe {
            self.query.run()?;
            self.key.run()?;
            self.table.run();
            self.valid.run();
        }
        let q = &self.query.output;
        let (t, h, d) = (q.meta.shape()[0], q.meta.shape()[1], q.meta.shape()[2]);
        let columns = self.scores.meta.shape()[2];
        let n = self.scores.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(q.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
            shard_scores::launch::<R>(
                &q.client,
                count,
                dim,
                q.clone().into_array_arg(),
                self.key.output.clone().into_array_arg(),
                self.table.output.clone().into_array_arg(),
                self.plan.q_indptr.clone().into_array_arg(),
                self.plan.metadata.clone().into_array_arg(),
                self.scores.clone().into_array_arg(),
                n as u32,
                t as u32,
                h as u32,
                d as u32,
                columns as u32,
                (self.plan.q_indptr.meta.num_elements() - 1) as u32,
                self.table.output.meta.shape()[1] as u32,
                self.rank as u32,
                self.world as u32,
                self.page_size as u32,
                self.compact,
            );
        }
        let n = t * h;
        if n > 0 {
            let dim = RudaDim::new(q.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
            carrier::launch::<R>(
                &q.client,
                count,
                dim,
                self.scores.clone().into_array_arg(),
                self.valid.output.clone().into_array_arg(),
                self.carrier.clone().into_array_arg(),
                t as u32,
                h as u32,
                columns as u32,
                self.index_scale as u32,
                self.index_offset as u32,
                self.force_begin as u32,
                self.force_end as u32,
            );
        }
        Ok(&self.carrier)
    }
    pub unsafe fn publish(
        &self,
        slot: usize,
        tagged: bool,
        spin_limit: u32,
    ) -> Result<(), InferenceError> {
        unsafe { self.comm.publish(&self.carrier, slot, tagged, spin_limit) }
    }
    pub unsafe fn receive_merge(
        &self,
        slot: usize,
        tagged: bool,
        spin_limit: u32,
        prefill: bool,
    ) -> Result<&RudaTensor<R>, InferenceError> {
        let t = self.scores.meta.shape()[0];
        let h = self.comm.layout.heads_per_rank;
        unsafe {
            self.comm
                .receive_into(t, slot, tagged, spin_limit, &self.received)?;
        }
        let n = t * h;
        if n > 0 {
            let dim = RudaDim::new(self.received.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.received.client, n, dim);
            merge_carriers::launch::<R>(
                &self.received.client,
                count,
                dim,
                self.received.clone().into_array_arg(),
                self.valid.output.clone().into_array_arg(),
                self.selected.clone().into_array_arg(),
                t as u32,
                h as u32,
                self.world as u32,
                t as u32,
                self.force_begin as u32,
                self.force_end as u32,
                prefill,
            );
        }
        Ok(&self.selected)
    }
}
