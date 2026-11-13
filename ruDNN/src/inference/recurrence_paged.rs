//! Page-indexed recurrent-state slots, aligned checkpoints and pool commits.
use super::{
    InferenceError, alloc, check, float,
    page::CsrPages,
    recurrence::{DeltaInputs, DeltaOptions, DeltaOutput, DeltaPlan, StateLayout},
};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
pub struct PrefillSlots<R: Runtime> {
    pub initial: RudaTensor<R>,
    pub final_state: RudaTensor<R>,
    pub checkpoint_starts: RudaTensor<R>,
    pub prefix_lengths: RudaTensor<R>,
}
pub struct DecodeSlots<R: Runtime> {
    pub initial: RudaTensor<R>,
    pub final_state: RudaTensor<R>,
}
impl<R: Runtime> CsrPages<R> {
    pub fn allocate_prefill_slots(&self) -> PrefillSlots<R> {
        PrefillSlots {
            initial: alloc(&self.indptr, [self.sequences], DType::I32),
            final_state: alloc(&self.indptr, [self.sequences], DType::I32),
            checkpoint_starts: alloc(&self.indptr, [self.sequences], DType::I32),
            prefix_lengths: alloc(&self.indptr, [self.sequences], DType::I32),
        }
    }
    pub fn prefill_slots(
        &self,
        cu_seqlens: &RudaTensor<R>,
    ) -> Result<PrefillSlots<R>, InferenceError> {
        let out = self.allocate_prefill_slots();
        unsafe {
            self.prefill_slots_into(cu_seqlens, &out)?;
        }
        Ok(out)
    }
    pub unsafe fn prefill_slots_into(
        &self,
        cu: &RudaTensor<R>,
        out: &PrefillSlots<R>,
    ) -> Result<(), InferenceError> {
        check(cu, &self.indptr)?;
        if !matches!(cu.dtype, DType::I32 | DType::U32)
            || cu.meta.shape()[..] != [self.sequences + 1]
        {
            return Err(InferenceError("paged recurrence needs device indptr[N+1]"));
        }
        for t in [
            &out.initial,
            &out.final_state,
            &out.checkpoint_starts,
            &out.prefix_lengths,
        ] {
            check(t, &self.indptr)?;
            if t.dtype != DType::I32 || t.meta.shape()[..] != [self.sequences] {
                return Err(InferenceError("paged prefill slot outputs mismatch"));
            }
        }
        if self.sequences > 0 {
            let dim = RudaDim::new(cu.client.properties(), self.sequences);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&cu.client, self.sequences, dim);
            prefill_slots::launch::<R>(
                &cu.client,
                count,
                dim,
                cu.clone().into_array_arg(),
                self.indptr.clone().into_array_arg(),
                self.indices.clone().into_array_arg(),
                self.last_page_len.clone().into_array_arg(),
                out.initial.clone().into_array_arg(),
                out.final_state.clone().into_array_arg(),
                out.checkpoint_starts.clone().into_array_arg(),
                out.prefix_lengths.clone().into_array_arg(),
                self.sequences as u32,
                self.page_size as u32,
            );
        }
        Ok(())
    }
    pub fn allocate_decode_slots(&self) -> DecodeSlots<R> {
        DecodeSlots {
            initial: alloc(&self.indptr, [self.sequences], DType::I32),
            final_state: alloc(&self.indptr, [self.sequences], DType::I32),
        }
    }
    pub fn decode_slots(&self) -> Result<DecodeSlots<R>, InferenceError> {
        let out = self.allocate_decode_slots();
        unsafe {
            self.decode_slots_into(&out)?;
        }
        Ok(out)
    }
    pub unsafe fn decode_slots_into(&self, out: &DecodeSlots<R>) -> Result<(), InferenceError> {
        for t in [&out.initial, &out.final_state] {
            check(t, &self.indptr)?;
            if t.dtype != DType::I32 || t.meta.shape()[..] != [self.sequences] {
                return Err(InferenceError("paged decode slot outputs mismatch"));
            }
        }
        if self.sequences > 0 {
            let dim = RudaDim::new(self.indptr.client.properties(), self.sequences);
            let count =
                kernel_dsl::calculate_ruda_count_elemwise(&self.indptr.client, self.sequences, dim);
            decode_slots::launch::<R>(
                &self.indptr.client,
                count,
                dim,
                self.indptr.clone().into_array_arg(),
                self.indices.clone().into_array_arg(),
                self.last_page_len.clone().into_array_arg(),
                out.initial.clone().into_array_arg(),
                out.final_state.clone().into_array_arg(),
                self.sequences as u32,
            );
        }
        Ok(())
    }
}
pub struct PagedDeltaOutput<R: Runtime> {
    pub result: DeltaOutput<R>,
    pub state_pool: RudaTensor<R>,
    pub slots: PrefillSlots<R>,
}
pub struct PagedDecodeOutput<R: Runtime> {
    pub result: DeltaOutput<R>,
    pub state_pool: RudaTensor<R>,
    pub slots: DecodeSlots<R>,
}
pub fn decode<R: Runtime>(
    plan: &DeltaPlan<R>,
    pages: &CsrPages<R>,
    q: &RudaTensor<R>,
    k: &RudaTensor<R>,
    v: &RudaTensor<R>,
    state_pool: &RudaTensor<R>,
    inputs: DeltaInputs<'_, R>,
    mut options: DeltaOptions,
) -> Result<PagedDecodeOutput<R>, InferenceError> {
    check(state_pool, q)?;
    float(state_pool)?;
    let slots = pages.decode_slots()?;
    let state = super::norm::cast(state_pool, DType::F32)?;
    options.null_state_is_zero = true;
    let result = plan.run(
        q,
        k,
        v,
        DeltaInputs {
            initial_state: Some(&state),
            state_indices: Some(&slots.initial),
            prefix_lengths: None,
            checkpoint_prefixes: None,
            ..inputs
        },
        options,
    )?;
    let pool = crate::paged_attention::copy_contiguous(state_pool);
    unsafe {
        commit_final(&pool, &result.final_state, &slots.final_state)?;
    }
    Ok(PagedDecodeOutput {
        result,
        state_pool: pool,
        slots,
    })
}
/// Recurrent prefill reads the state of the last prefix page, then publishes
/// completed-page checkpoints and the final partial-page state into a COW pool.
pub fn prefill<R: Runtime>(
    plan: &DeltaPlan<R>,
    pages: &CsrPages<R>,
    q: &RudaTensor<R>,
    k: &RudaTensor<R>,
    v: &RudaTensor<R>,
    state_pool: &RudaTensor<R>,
    inputs: DeltaInputs<'_, R>,
    mut options: DeltaOptions,
) -> Result<PagedDeltaOutput<R>, InferenceError> {
    if plan.checkpoint_stride() != pages.page_size {
        return Err(InferenceError(
            "paged recurrent plan checkpoint stride must equal page size",
        ));
    }
    check(state_pool, q)?;
    float(state_pool)?;
    let slots = pages.prefill_slots(plan.device_sequence_indptr())?;
    let state = super::norm::cast(state_pool, DType::F32)?;
    options.null_state_is_zero = true;
    let result = plan.run(
        q,
        k,
        v,
        DeltaInputs {
            initial_state: Some(&state),
            state_indices: Some(&slots.initial),
            prefix_lengths: None,
            checkpoint_prefixes: Some(&slots.prefix_lengths),
            ..inputs
        },
        options,
    )?;
    let pool = crate::paged_attention::copy_contiguous(state_pool);
    unsafe {
        commit_final(&pool, &result.final_state, &slots.final_state)?;
        commit_checkpoints(plan, pages, &slots, &result, &pool, options.state_layout)?;
    }
    Ok(PagedDeltaOutput {
        result,
        state_pool: pool,
        slots,
    })
}
pub unsafe fn commit_final<R: Runtime>(
    pool: &RudaTensor<R>,
    states: &RudaTensor<R>,
    rows: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    for t in [pool, states, rows] {
        check(t, pool)?;
    }
    float(pool)?;
    float(states)?;
    if pool.meta.num_dims() != 4
        || states.meta.num_dims() != 4
        || pool.meta.shape()[1..] != states.meta.shape()[1..]
        || rows.dtype != DType::I32
        || rows.meta.shape()[..] != [states.meta.shape()[0]]
    {
        return Err(InferenceError(
            "paged recurrent final-state commit mismatch",
        ));
    }
    let width: usize = states.meta.shape()[1..].iter().product();
    let n = states.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(pool.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&pool.client, n, dim);
        scatter_state::launch::<R>(
            &pool.client,
            count,
            dim,
            states.clone().into_array_arg(),
            rows.clone().into_array_arg(),
            pool.clone().into_array_arg(),
            n as u32,
            width as u32,
            pool.meta.shape()[0] as u32,
            states.dtype.into(),
            pool.dtype.into(),
        );
    }
    Ok(())
}
pub unsafe fn commit_checkpoints<R: Runtime>(
    plan: &DeltaPlan<R>,
    pages: &CsrPages<R>,
    slots: &PrefillSlots<R>,
    output: &DeltaOutput<R>,
    pool: &RudaTensor<R>,
    layout: StateLayout,
) -> Result<(), InferenceError> {
    check(pool, &output.checkpoints)?;
    float(pool)?;
    let c = &output.checkpoints;
    if c.dtype != DType::F32 || c.meta.num_dims() != 4 || pool.meta.num_dims() != 4 {
        return Err(InferenceError("paged checkpoint storage needs rank four"));
    }
    let heads = c.meta.shape()[1];
    let dk = c.meta.shape()[2];
    let dv = c.meta.shape()[3];
    let expected = if layout == StateLayout::KeyValue {
        vec![pool.meta.shape()[0], heads, dk, dv]
    } else {
        vec![pool.meta.shape()[0], heads, dv, dk]
    };
    if pool.meta.shape()[..] != expected {
        return Err(InferenceError("paged checkpoint pool dimensions mismatch"));
    }
    let n = pages
        .sequences
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(dk))
        .and_then(|n| n.checked_mul(dv))
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(InferenceError("paged checkpoint commit exceeds U32"))?;
    if n > 0 {
        let dim = RudaDim::new(pool.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&pool.client, n, dim);
        scatter_checkpoints::launch::<R>(
            &pool.client,
            count,
            dim,
            c.clone().into_array_arg(),
            plan.device_sequence_indptr().clone().into_array_arg(),
            plan.device_checkpoint_indptr().clone().into_array_arg(),
            slots.prefix_lengths.clone().into_array_arg(),
            slots.checkpoint_starts.clone().into_array_arg(),
            pages.indices.clone().into_array_arg(),
            pool.clone().into_array_arg(),
            n as u32,
            heads as u32,
            dk as u32,
            dv as u32,
            pages.page_size as u32,
            pool.meta.shape()[0] as u32,
            layout == StateLayout::ValueKey,
            pool.dtype.into(),
        );
    }
    Ok(())
}
#[ruda(launch)]
pub fn prefill_slots(
    cu: &Array<i32>,
    indptr: &Array<i32>,
    indices: &Array<i32>,
    last: &Array<i32>,
    read: &mut Array<i32>,
    write: &mut Array<i32>,
    checkpoint: &mut Array<i32>,
    prefix: &mut Array<i32>,
    n: u32,
    page_size: u32,
) {
    let seq = ABSOLUTE_POS as usize;
    if seq < n as usize {
        let length = cu[seq + 1] - cu[seq];
        let first = indptr[seq];
        let end = indptr[seq + 1];
        let total = i32::max(0, end - first - 1) * page_size as i32 + last[seq];
        let before = total - length;
        let mut initial: i32 = -1i32;
        if before > 0 {
            initial = indices[(first + (before - 1) / page_size as i32) as usize];
        }
        let mut final_row: i32 = -1i32;
        if length > 0 && total % page_size as i32 != 0 && end > first {
            final_row = indices[end as usize - 1];
        }
        read[seq] = initial;
        write[seq] = final_row;
        checkpoint[seq] = first + i32::max(0, before) / page_size as i32;
        prefix[seq] = before;
    }
}
#[ruda(launch)]
pub fn decode_slots(
    indptr: &Array<i32>,
    indices: &Array<i32>,
    last: &Array<i32>,
    read: &mut Array<i32>,
    write: &mut Array<i32>,
    n: u32,
) {
    let seq = ABSOLUTE_POS as usize;
    if seq < n as usize {
        let first = indptr[seq];
        let end = indptr[seq + 1];
        let mut initial: i32 = -1i32;
        let mut final_row: i32 = -1i32;
        if end > first {
            final_row = indices[end as usize - 1];
            initial = final_row;
            if last[seq] == 1 {
                let previous = i32::max(end - 2, 0);
                initial = indices[previous as usize];
            }
        }
        read[seq] = initial;
        write[seq] = final_row;
    }
}
#[ruda(launch)]
pub fn scatter_state<F: Float, O: Float>(
    states: &Array<F>,
    rows: &Array<i32>,
    pool: &mut Array<O>,
    n: u32,
    width: u32,
    slots: u32,
    #[define(F)] _f: StorageType,
    #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = rows[i / width as usize];
        if row >= 0 && row < slots as i32 {
            pool[row as usize * width as usize + i % width as usize] = O::cast_from(states[i]);
        }
    }
}
#[ruda(launch)]
pub fn scatter_checkpoints<O: Float>(
    states: &Array<f32>,
    cu: &Array<i32>,
    cp: &Array<u32>,
    prefix: &Array<i32>,
    starts: &Array<i32>,
    pages: &Array<i32>,
    pool: &mut Array<O>,
    n: u32,
    heads: u32,
    dk: u32,
    dv: u32,
    page_size: u32,
    slots: u32,
    #[comptime] vk: bool,
    #[define(O)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    let span = heads as usize * dk as usize * dv as usize;
    if i < n as usize {
        let seq = i / span;
        let local = i % span;
        let head = local / (dk as usize * dv as usize);
        let key = local / dv as usize % dk as usize;
        let value = local % dv as usize;
        let length = i32::max(0, cu[seq + 1] - cu[seq]) as usize;
        let offset = i32::max(0, prefix[seq]) as usize % page_size as usize;
        let boundaries = (offset + length) / page_size as usize;
        for b in 0..boundaries {
            let row = pages[starts[seq] as usize + b];
            if row >= 0 && row < slots as i32 {
                let source = (cp[seq] as usize + b + 1) * span + local;
                let mut dest = row as usize * span + local;
                if comptime!(vk) {
                    dest = ((row as usize * heads as usize + head) * dv as usize + value)
                        * dk as usize
                        + key;
                }
                pool[dest] = O::cast_from(states[source]);
            }
        }
    }
}

/// In-place graph-prepared paged GDN/KDA execution. The scheduler retains page
/// ownership; this operation writes only the selected final/checkpoint slots.
pub struct PreparedPagedDelta<'a, R: Runtime> {
    plan: &'a DeltaPlan<R>,
    pages: &'a CsrPages<R>,
    pool: super::PreparedPack<R>,
    delta: super::recurrence::PreparedDelta<'a, R>,
    prefill: Option<PrefillSlots<R>>,
    decode: Option<DecodeSlots<R>>,
    layout: StateLayout,
}
impl<'a, R: Runtime> PreparedPagedDelta<'a, R> {
    pub fn prepare(
        plan: &'a DeltaPlan<R>,
        pages: &'a CsrPages<R>,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        pool: &RudaTensor<R>,
        inputs: DeltaInputs<'_, R>,
        mut options: DeltaOptions,
        prefill: bool,
    ) -> Result<Self, InferenceError> {
        if prefill && plan.checkpoint_stride() != pages.page_size {
            return Err(InferenceError(
                "paged checkpoint stride must equal page size",
            ));
        }
        let pool = super::PreparedPack::prepare(pool)?;
        check(&pool.output, q)?;
        float(&pool.output)?;
        let (prefill_slots, decode_slots) = if prefill {
            (Some(pages.allocate_prefill_slots()), None)
        } else {
            (None, Some(pages.allocate_decode_slots()))
        };
        let initial = if let Some(slots) = &prefill_slots {
            &slots.initial
        } else {
            &decode_slots.as_ref().unwrap().initial
        };
        options.null_state_is_zero = true;
        let delta = super::recurrence::PreparedDelta::prepare(
            plan,
            q,
            k,
            v,
            DeltaInputs {
                initial_state: Some(&pool.output),
                state_indices: Some(initial),
                prefix_lengths: None,
                checkpoint_prefixes: prefill_slots.as_ref().map(|s| &s.prefix_lengths),
                ..inputs
            },
            options,
        )?;
        Ok(Self {
            plan,
            pages,
            pool,
            delta,
            prefill: prefill_slots,
            decode: decode_slots,
            layout: options.state_layout,
        })
    }
    pub fn output(&self) -> &DeltaOutput<R> {
        &self.delta.output
    }
    pub unsafe fn run(&self) -> Result<&DeltaOutput<R>, InferenceError> {
        unsafe {
            self.plan.refresh_schedule();
            self.pool.run();
            if let Some(s) = &self.prefill {
                self.pages
                    .prefill_slots_into(self.plan.device_sequence_indptr(), s)?;
            }
            if let Some(s) = &self.decode {
                self.pages.decode_slots_into(s)?;
            }
            self.delta.run()?;
            if let Some(s) = &self.prefill {
                commit_final(
                    &self.pool.output,
                    &self.delta.output.final_state,
                    &s.final_state,
                )?;
                commit_checkpoints(
                    self.plan,
                    self.pages,
                    s,
                    &self.delta.output,
                    &self.pool.output,
                    self.layout,
                )?;
            }
            if let Some(s) = &self.decode {
                commit_final(
                    &self.pool.output,
                    &self.delta.output.final_state,
                    &s.final_state,
                )?;
            }
            self.pool.write_back();
        }
        Ok(&self.delta.output)
    }
}
