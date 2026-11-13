//! Persistent selective-state update, speculative state caching and commit.
//! Quantized initial states are decoded only for the selected requests.
use super::{
    InferenceError, alloc, check, float,
    mamba::{SsmInputs, SsmOptions, SsmOutput, SsmPlan},
    state_storage::{QuantizedState, Rounding, StateFormat},
};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
pub enum StatePool<'a, R: Runtime> {
    Float(&'a RudaTensor<R>),
    Quantized(&'a QuantizedState<R>),
}
struct BoundState<R: Runtime> {
    data: RudaTensor<R>,
    quant: Option<QuantizedState<R>>,
    packs: Vec<super::PreparedPack<R>>,
}
impl<R: Runtime> BoundState<R> {
    fn prepare(
        pool: StatePool<'_, R>,
        rank: usize,
        like: &RudaTensor<R>,
    ) -> Result<Self, InferenceError> {
        let (source, q) = match pool {
            StatePool::Float(t) => {
                float(t)?;
                (t, None)
            }
            StatePool::Quantized(q) => (&q.data, Some(q)),
        };
        super::check_view(source, like)?;
        if source.meta.num_dims() != rank {
            return Err(InferenceError("state pool rank mismatch"));
        }
        let p = super::PreparedPack::prepare(source)?;
        let data = p.output.clone();
        let mut packs = vec![p];
        let quant = if let Some(q) = q {
            super::check_view(&q.scale, like)?;
            let sf = super::PreparedPack::prepare(&q.scale)?;
            let quant = QuantizedState {
                data: data.clone(),
                scale: sf.output.clone(),
                format: q.format,
            };
            quant.validate()?;
            packs.push(sf);
            Some(quant)
        } else {
            None
        };
        Ok(Self { data, quant, packs })
    }
    unsafe fn refresh(&self) {
        for p in &self.packs {
            unsafe {
                p.run();
            }
        }
    }
    unsafe fn commit(&self) {
        for p in &self.packs {
            unsafe {
                p.write_back();
            }
        }
    }
    fn format(&self) -> u32 {
        self.quant.as_ref().map_or(0, |q| match q.format {
            StateFormat::I16 => 1,
            StateFormat::I8 => 2,
            StateFormat::E4m3 => 3,
            StateFormat::E5m2 => 4,
        })
    }
    fn scale(&self) -> &RudaTensor<R> {
        self.quant.as_ref().map_or(&self.data, |q| &q.scale)
    }
}
/// Fixed bindings for graph recording/replay. `intermediate` is optional
/// [cacheSlots,cacheSteps,H,D,N] storage. Explicit destination tables may be
/// [batch] (final states) or [batch,steps] (per-token states).
pub struct PreparedSelectiveUpdate<'a, R: Runtime> {
    plan: &'a SsmPlan<R>,
    input: super::PreparedPack<R>,
    pool: BoundState<R>,
    intermediate: Option<BoundState<R>>,
    parameters: Vec<(super::PreparedPack<R>, RudaTensor<R>)>,
    dt: RudaTensor<R>,
    a: RudaTensor<R>,
    b: RudaTensor<R>,
    c: RudaTensor<R>,
    d: Option<RudaTensor<R>>,
    z: Option<RudaTensor<R>>,
    bias: Option<RudaTensor<R>>,
    source: Option<super::metadata::PreparedIndices<R>>,
    accepted: Option<super::metadata::PreparedIndices<R>>,
    destination: Option<super::metadata::PreparedIndices<R>>,
    history_slots: Option<super::metadata::PreparedIndices<R>>,
    boundaries: Option<super::metadata::PreparedIndices<R>>,
    selected: RudaTensor<R>,
    active: RudaTensor<R>,
    lengths: RudaTensor<R>,
    initial: RudaTensor<R>,
    final_destinations: RudaTensor<R>,
    history_destinations: RudaTensor<R>,
    final_sources: RudaTensor<R>,
    history_sources: RudaTensor<R>,
    encoded_final: Option<QuantizedState<R>>,
    encoded_history: Option<QuantizedState<R>>,
    pub scan: SsmOutput<R>,
    seed: Option<RudaTensor<R>>,
    rounding: Rounding,
    options: SsmOptions,
    disable_update: bool,
}
impl<'a, R: Runtime> PreparedSelectiveUpdate<'a, R> {
    pub fn prepare(
        plan: &'a SsmPlan<R>,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        pool: StatePool<'_, R>,
        destination: Option<&RudaTensor<R>>,
        intermediate: Option<StatePool<'_, R>>,
        intermediate_slots: Option<&RudaTensor<R>>,
        disable_update: bool,
        rounding: Rounding,
        seed: Option<&RudaTensor<R>>,
        mut options: SsmOptions,
    ) -> Result<Self, InferenceError> {
        let input = super::PreparedPack::prepare(input)?;
        let x = &input.output;
        float(x)?;
        let pool = BoundState::prepare(pool, 4, x)?;
        let shape = pool.data.meta.shape();
        let heads = shape[1];
        let dim = shape[2];
        let n = shape[3];
        if heads == 0
            || dim == 0
            || n == 0
            || n > 2048
            || x.meta.shape()[..] != [plan.tokens, heads, dim]
            || rounding.philox_rounds > 64
        {
            return Err(InferenceError(
                "selective update input/state geometry mismatch",
            ));
        }
        let index =
            |t: &RudaTensor<R>| -> Result<super::metadata::PreparedIndices<R>, InferenceError> {
                super::check_view(t, x)?;
                super::metadata::PreparedIndices::prepare(t, DType::I32)
            };
        let source = inputs.source_slots.map(&index).transpose()?;
        let accepted = inputs.accepted_tokens.map(&index).transpose()?;
        let destination = destination.map(&index).transpose()?;
        let history_slots = intermediate_slots.map(&index).transpose()?;
        let boundaries = inputs.checkpoint_boundaries.map(&index).transpose()?;
        for ids in source.iter().chain(destination.iter()) {
            let s = ids.output.meta.shape();
            if !(s[..] == [plan.sequences] || s.len() == 2 && s[0] == plan.sequences && s[1] > 0) {
                return Err(InferenceError(
                    "state indices need [batch] or [batch,steps]",
                ));
            }
        }
        for ids in accepted
            .iter()
            .chain(history_slots.iter())
            .chain(boundaries.iter())
        {
            if ids.output.meta.shape()[..] != [plan.sequences] {
                return Err(InferenceError("state counts/cache slots require [batch]"));
            }
        }
        if accepted.is_some()
            && source
                .as_ref()
                .is_none_or(|s| s.output.meta.num_dims() != 2)
        {
            return Err(InferenceError(
                "accepted counts require a two-dimensional source table",
            ));
        }
        if intermediate.is_some() && destination.is_some() {
            return Err(InferenceError(
                "intermediate state cache and explicit destinations are mutually exclusive",
            ));
        }
        let intermediate = intermediate
            .map(|p| BoundState::prepare(p, 5, x))
            .transpose()?;
        if intermediate.as_ref().is_some_and(|p| {
            p.data.meta.shape()[2..] != [heads, dim, n] || p.data.meta.shape()[1] == 0
        }) {
            return Err(InferenceError(
                "intermediate cache must be [slots,steps,H,D,N]",
            ));
        }
        let mut parameters = Vec::new();
        let mut parameter = |t: &RudaTensor<R>| -> Result<RudaTensor<R>, InferenceError> {
            super::check_view(t, x)?;
            float(t)?;
            let p = super::PreparedPack::prepare(t)?;
            let out = alloc(x, t.meta.shape().clone(), DType::F32);
            parameters.push((p, out.clone()));
            Ok(out)
        };
        let dt = parameter(inputs.dt)?;
        let a = parameter(inputs.a)?;
        let b = parameter(inputs.b)?;
        let c = parameter(inputs.c)?;
        let d = inputs.d.map(&mut parameter).transpose()?;
        let z = inputs.z.map(&mut parameter).transpose()?;
        let bias = inputs.dt_bias.map(&mut parameter).transpose()?;
        if b.meta.num_dims() != 3
            || b.meta.shape()[0] != plan.tokens
            || b.meta.shape()[2] != n
            || b.meta.shape()[1] == 0
            || heads % b.meta.shape()[1] != 0
            || c.meta.shape() != b.meta.shape()
            || (a.meta.shape()[..] != [heads]
                && a.meta.shape()[..] != [heads, n]
                && a.meta.shape()[..] != [heads, dim, n])
            || (dt.meta.shape()[..] != [plan.tokens, heads]
                && dt.meta.shape()[..] != [plan.tokens, heads, dim])
            || d.iter()
                .chain(bias.iter())
                .any(|p| p.meta.shape()[..] != [heads] && p.meta.shape()[..] != [heads, dim])
            || z.as_ref().is_some_and(|p| p.meta.shape() != x.meta.shape())
        {
            return Err(InferenceError("selective update parameter shapes mismatch"));
        }
        options.cache_intermediate |= intermediate.is_some()
            || destination
                .as_ref()
                .is_some_and(|d| d.output.meta.num_dims() == 2);
        let scan = plan.allocate_workspace(
            x,
            heads,
            dim,
            n,
            options.cache_intermediate,
            boundaries.is_some(),
        )?;
        let initial = alloc(x, [plan.sequences, heads, dim, n], DType::F32);
        let selected = alloc(x, [plan.sequences], DType::I32);
        let active = alloc(x, [plan.sequences], DType::I32);
        let lengths = alloc(x, [plan.sequences], DType::I32);
        let final_destinations = alloc(x, [plan.sequences], DType::I32);
        let history_destinations = alloc(x, [plan.tokens], DType::I32);
        let final_sources = alloc(x, [plan.sequences], DType::I32);
        let history_sources = alloc(x, [plan.tokens], DType::I32);
        let encoded_final = pool
            .quant
            .as_ref()
            .map(|q| QuantizedState::allocate(x, scan.final_states.meta.shape().clone(), q.format))
            .transpose()?;
        let history_format = intermediate
            .as_ref()
            .and_then(|p| p.quant.as_ref())
            .map(|q| q.format)
            .or_else(|| {
                if destination
                    .as_ref()
                    .is_some_and(|d| d.output.meta.num_dims() == 2)
                {
                    pool.quant.as_ref().map(|q| q.format)
                } else {
                    None
                }
            });
        let encoded_history = history_format
            .map(|f| QuantizedState::allocate(x, scan.intermediate_states.meta.shape().clone(), f))
            .transpose()?;
        if let Some(seed) = seed {
            check(seed, x)?;
            if !matches!(seed.dtype, DType::I64 | DType::U64) || seed.meta.num_elements() != 1 {
                return Err(InferenceError(
                    "selective rounding seed needs one I64/U64 element",
                ));
            }
        }
        Ok(Self {
            plan,
            input,
            pool,
            intermediate,
            parameters,
            dt,
            a,
            b,
            c,
            d,
            z,
            bias,
            source,
            accepted,
            destination,
            history_slots,
            boundaries,
            selected,
            active,
            lengths,
            initial,
            final_destinations,
            history_destinations,
            final_sources,
            history_sources,
            encoded_final,
            encoded_history,
            scan,
            seed: seed.cloned(),
            rounding,
            options,
            disable_update,
        })
    }
    /// Selected destination slots must be unique; all retained scratch and pool
    /// storage is exclusive until the queue completes. Inputs and parameters
    /// may be changed between replays, never while this operation is executing.
    pub unsafe fn run(&self) -> Result<&SsmOutput<R>, InferenceError> {
        unsafe {
            self.input.run();
            self.pool.refresh();
            if let Some(p) = &self.intermediate {
                p.refresh();
            }
            for ids in self
                .source
                .iter()
                .chain(self.accepted.iter())
                .chain(self.destination.iter())
                .chain(self.history_slots.iter())
                .chain(self.boundaries.iter())
            {
                ids.run();
            }
            for (p, out) in &self.parameters {
                p.run();
                super::norm::cast_into(&p.output, out)?;
            }
        }
        unsafe {
            self.plan.refresh_schedule();
        }
        let x = &self.input.output;
        let s = self.pool.data.meta.shape();
        let width = s[1] * s[2] * s[3];
        let sequences = self.plan.sequences;
        let dummy = &self.plan.cu;
        if sequences > 0 {
            let dim = RudaDim::new(x.client.properties(), sequences);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, sequences, dim);
            select_slots::launch::<R>(
                &x.client,
                count,
                dim,
                self.source
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(dummy)
                    .clone()
                    .into_array_arg(),
                self.accepted
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(dummy)
                    .clone()
                    .into_array_arg(),
                dummy.clone().into_array_arg(),
                self.selected.clone().into_array_arg(),
                self.active.clone().into_array_arg(),
                self.lengths.clone().into_array_arg(),
                self.final_sources.clone().into_array_arg(),
                sequences as u32,
                s[0] as u32,
                self.source.as_ref().map_or(1, |p| {
                    if p.output.meta.num_dims() == 2 {
                        p.output.meta.shape()[1]
                    } else {
                        1
                    }
                }) as u32,
                self.options.pad_slot,
                self.source.is_some(),
                self.accepted.is_some(),
            );
        }
        let n = self.initial.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            gather_state::launch::<R>(
                &x.client,
                count,
                dim,
                self.pool.data.clone().into_array_arg(),
                self.pool.scale().clone().into_array_arg(),
                self.selected.clone().into_array_arg(),
                self.initial.clone().into_array_arg(),
                n as u32,
                width as u32,
                s[3] as u32,
                self.pool.format(),
                self.pool.data.dtype.into(),
            );
        }
        unsafe {
            self.plan.run_into(
                x,
                SsmInputs {
                    dt: &self.dt,
                    a: &self.a,
                    b: &self.b,
                    c: &self.c,
                    d: self.d.as_ref(),
                    z: self.z.as_ref(),
                    dt_bias: self.bias.as_ref(),
                    checkpoint_boundaries: self.boundaries.as_ref().map(|p| &p.output),
                    state: Some(&self.initial),
                    source_slots: Some(&self.active),
                    accepted_tokens: None,
                },
                SsmOptions {
                    pad_slot: -1,
                    ..self.options
                },
                SsmOutput {
                    values: self.scan.values.clone(),
                    final_states: self.scan.final_states.clone(),
                    intermediate_states: self.scan.intermediate_states.clone(),
                    checkpoints: self.scan.checkpoints.clone(),
                },
            )?;
        }
        if sequences > 0 {
            let dim = RudaDim::new(x.client.properties(), sequences);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, sequences, dim);
            commit_maps::launch::<R>(
                &x.client,
                count,
                dim,
                self.selected.clone().into_array_arg(),
                self.destination
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(dummy)
                    .clone()
                    .into_array_arg(),
                self.history_slots
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(dummy)
                    .clone()
                    .into_array_arg(),
                dummy.clone().into_array_arg(),
                self.final_destinations.clone().into_array_arg(),
                self.history_destinations.clone().into_array_arg(),
                self.history_sources.clone().into_array_arg(),
                sequences as u32,
                self.plan.tokens as u32,
                self.destination.as_ref().map_or(1, |p| {
                    if p.output.meta.num_dims() == 2 {
                        p.output.meta.shape()[1]
                    } else {
                        1
                    }
                }) as u32,
                self.intermediate
                    .as_ref()
                    .map_or(0, |p| p.data.meta.shape()[1]) as u32,
                self.options.pad_slot,
                self.destination.is_some(),
                self.destination
                    .as_ref()
                    .is_some_and(|p| p.output.meta.num_dims() == 2),
                self.history_slots.is_some(),
                self.intermediate.is_some(),
            );
        }
        if !self.disable_update
            && self
                .destination
                .as_ref()
                .is_none_or(|p| p.output.meta.num_dims() == 1)
        {
            if let (Some(q), Some(encoded)) = (&self.pool.quant, &self.encoded_final) {
                unsafe {
                    encoded.encode_into(
                        &self.scan.final_states,
                        self.rounding,
                        self.seed.as_ref(),
                    )?;
                    super::state_storage::copy_selected_rows(
                        q,
                        &self.selected,
                        encoded,
                        &self.final_sources,
                        Some(&self.lengths),
                        true,
                    )?;
                    super::state_storage::copy_selected_rows(
                        encoded,
                        &self.final_sources,
                        q,
                        &self.final_destinations,
                        None,
                        false,
                    )?;
                }
            } else {
                unsafe {
                    self.write_states(
                        &self.pool,
                        &self.scan.final_states,
                        None,
                        &self.final_sources,
                        &self.final_destinations,
                    )?;
                }
            }
        }

        let history_target = if let Some(p) = &self.intermediate {
            Some(p)
        } else if !self.disable_update
            && self
                .destination
                .as_ref()
                .is_some_and(|p| p.output.meta.num_dims() == 2)
        {
            Some(&self.pool)
        } else {
            None
        };
        if let Some(target) = history_target {
            unsafe {
                self.write_states(
                    target,
                    &self.scan.intermediate_states,
                    self.encoded_history.as_ref(),
                    &self.history_sources,
                    &self.history_destinations,
                )?;
                target.commit();
            }
        }
        if !self.disable_update {
            unsafe {
                self.pool.commit();
            }
        }
        Ok(&self.scan)
    }
    unsafe fn write_states(
        &self,
        target: &BoundState<R>,
        values: &RudaTensor<R>,
        encoded: Option<&QuantizedState<R>>,
        source_ids: &RudaTensor<R>,
        dest_ids: &RudaTensor<R>,
    ) -> Result<(), InferenceError> {
        let s = self.pool.data.meta.shape();
        let width = s[1] * s[2] * s[3];
        if let (Some(q), Some(encoded)) = (&target.quant, encoded) {
            let flat = if target.data.meta.num_dims() == 5 {
                let t = target.data.meta.shape();
                QuantizedState {
                    data: ruda_kernel::tensor::reshape::reshape(
                        q.data.clone(),
                        Shape::from([t[0] * t[1], t[2], t[3], t[4]]),
                    ),
                    scale: ruda_kernel::tensor::reshape::reshape(
                        q.scale.clone(),
                        Shape::from([t[0] * t[1], t[2], t[3]]),
                    ),
                    format: q.format,
                }
            } else {
                QuantizedState {
                    data: q.data.clone(),
                    scale: q.scale.clone(),
                    format: q.format,
                }
            };
            unsafe {
                encoded.encode_into(values, self.rounding, self.seed.as_ref())?;
                super::state_storage::copy_selected_rows(
                    encoded, source_ids, &flat, dest_ids, None, false,
                )?;
            }
        } else {
            let n = values.meta.num_elements();
            if n > 0 {
                let dim = RudaDim::new(values.client.properties(), n);
                let count = kernel_dsl::calculate_ruda_count_elemwise(&values.client, n, dim);
                scatter_state::launch::<R>(
                    &values.client,
                    count,
                    dim,
                    values.clone().into_array_arg(),
                    dest_ids.clone().into_array_arg(),
                    target.data.clone().into_array_arg(),
                    n as u32,
                    width as u32,
                    (target.data.meta.num_elements() / width) as u32,
                    target.data.dtype.into(),
                );
            }
        }
        Ok(())
    }
}
#[ruda(launch)]
pub fn select_slots(
    ids: &Array<i32>,
    accepted: &Array<i32>,
    cu: &Array<i32>,
    selected: &mut Array<i32>,
    active: &mut Array<i32>,
    lengths: &mut Array<i32>,
    source_ids: &mut Array<i32>,
    batch: u32,
    slots: u32,
    steps: u32,
    pad: i32,
    #[comptime] indexed: bool,
    #[comptime] has_accepted: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < batch as usize {
        let mut slot = i as i32;
        let mut valid = true;
        if comptime!(indexed) {
            let mut column: usize = 0usize;
            if comptime!(has_accepted) {
                let count = accepted[i];
                valid = count > 0 && count <= steps as i32;
                column = i32::max(0, i32::min(count - 1, steps as i32 - 1)) as usize;
            }
            slot = ids[i * steps as usize + column];
        }
        valid = valid && slot != pad && slot >= 0 && slot < slots as i32;
        selected[i] = -1;
        active[i] = -1;
        if valid {
            selected[i] = slot;
            active[i] = i as i32;
        }
        lengths[i] = i32::max(0, cu[i + 1] - cu[i]);
        source_ids[i] = i as i32;
    }
}
#[ruda(launch)]
pub fn gather_state<S: Numeric>(
    source: &Array<S>,
    sf: &Array<f32>,
    ids: &Array<i32>,
    out: &mut Array<f32>,
    n: u32,
    width: u32,
    state_dim: u32,
    #[comptime] format: u32,
    #[define(S)] _s: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / width as usize;
        let col = i % width as usize;
        let slot = ids[row];
        let mut value: f32 = 0.0f32;
        if slot >= 0 {
            let offset = slot as usize * width as usize + col;
            value = f32::cast_from(source[offset]);
            if comptime!(format == 3 || format == 4) {
                value = rublas::tensor_float_quant::fp8_decode(
                    u32::cast_from(source[offset]),
                    format == 4,
                );
            }
            if comptime!(format != 0) {
                value *= sf[offset / state_dim as usize];
            }
        }
        out[i] = value;
    }
}
#[ruda(launch)]
pub fn commit_maps(
    selected: &Array<i32>,
    destination: &Array<i32>,
    history_slots: &Array<i32>,
    cu: &Array<i32>,
    final_ids: &mut Array<i32>,
    history_ids: &mut Array<i32>,
    source_ids: &mut Array<i32>,
    batch: u32,
    tokens: u32,
    dest_steps: u32,
    cache_steps: u32,
    pad: i32,
    #[comptime] explicit: bool,
    #[comptime] per_token: bool,
    #[comptime] indexed_history: bool,
    #[comptime] cache: bool,
) {
    let seq = ABSOLUTE_POS as usize;
    if seq < batch as usize {
        let valid = selected[seq] >= 0;
        let mut final_slot = selected[seq];
        if comptime!(explicit) {
            final_slot = destination[seq * dest_steps as usize];
        }
        if !valid || final_slot == pad {
            final_slot = -1;
        }
        final_ids[seq] = final_slot;
        let begin = i32::max(0, cu[seq]) as usize;
        let end = usize::min(i32::max(0, cu[seq + 1]) as usize, tokens as usize);
        for token in begin..end {
            let step = token - begin;
            let mut slot: i32 = -1i32;
            if valid {
                if comptime!(per_token) {
                    if step < dest_steps as usize {
                        slot = destination[seq * dest_steps as usize + step];
                        if slot == pad {
                            slot = -1;
                        }
                    }
                }
                if comptime!(cache) {
                    let mut base = seq as i32;
                    if comptime!(indexed_history) {
                        base = history_slots[seq];
                    }
                    if base >= 0 && base != pad && step < cache_steps as usize {
                        slot = base * cache_steps as i32 + step as i32;
                    }
                }
            }
            history_ids[token] = slot;
            source_ids[token] = token as i32;
        }
    }
}
#[ruda(launch)]
pub fn scatter_state<S: Float>(
    values: &Array<f32>,
    ids: &Array<i32>,
    out: &mut Array<S>,
    n: u32,
    width: u32,
    slots: u32,
    #[define(S)] _s: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let slot = ids[i / width as usize];
        if slot >= 0 && slot < slots as i32 {
            out[slot as usize * width as usize + i % width as usize] = S::cast_from(values[i]);
        }
    }
}
