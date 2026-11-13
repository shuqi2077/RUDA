//! Native ReplaySSM ring reconstruction and checkpoint flush.
//! Ring cursors and acceptance counters are caller-owned, read-only metadata.
//! Numerical states and replay rings never leave the device.
use super::{
    InferenceError, alloc, check,
    mamba::{SsmInputs, SsmOptions, SsmOutput, SsmPlan},
    state_storage::{QuantizedState, Rounding, StateFormat},
};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
pub struct ReplayCache<R: Runtime> {
    pub state: RudaTensor<R>,
    pub x: RudaTensor<R>,
    pub b: RudaTensor<R>,
    pub dt: RudaTensor<R>,
    pub ring_start: RudaTensor<R>,
    pub accepted: RudaTensor<R>,
}
pub struct QuantizedReplayCache<R: Runtime> {
    pub state: QuantizedState<R>,
    pub x: RudaTensor<R>,
    pub b: RudaTensor<R>,
    pub dt: RudaTensor<R>,
    pub ring_start: RudaTensor<R>,
    pub accepted: RudaTensor<R>,
}
impl<R: Runtime> QuantizedReplayCache<R> {
    fn raw(&self) -> ReplayCache<R> {
        ReplayCache {
            state: self.state.data.clone(),
            x: self.x.clone(),
            b: self.b.clone(),
            dt: self.dt.clone(),
            ring_start: self.ring_start.clone(),
            accepted: self.accepted.clone(),
        }
    }
}
pub struct ReplayOutput<R: Runtime> {
    pub scan: SsmOutput<R>,
    pub cache: ReplayCache<R>,
    pub flushed: RudaTensor<R>,
}
pub struct QuantizedReplayOutput<R: Runtime> {
    pub scan: SsmOutput<R>,
    pub cache: QuantizedReplayCache<R>,
    pub flushed: RudaTensor<R>,
}
struct Geometry {
    slots: usize,
    heads: usize,
    dim: usize,
    n: usize,
    groups: usize,
    ring: usize,
}
fn format_code(f: StateFormat) -> u32 {
    match f {
        StateFormat::I16 => 1,
        StateFormat::I8 => 2,
        StateFormat::E4m3 => 3,
        StateFormat::E5m2 => 4,
    }
}
fn geometry<R: Runtime>(cache: &ReplayCache<R>) -> Result<Geometry, InferenceError> {
    for t in [
        &cache.state,
        &cache.x,
        &cache.b,
        &cache.dt,
        &cache.ring_start,
        &cache.accepted,
    ] {
        check(t, &cache.state)?;
    }
    if !matches!(
        cache.state.dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::I16 | DType::I8 | DType::U8
    ) || cache.state.meta.num_dims() != 4
        || cache.x.meta.num_dims() != 4
        || cache.b.meta.num_dims() != 4
        || cache.dt.dtype != DType::F32
        || cache.dt.meta.num_dims() != 3
        || cache.ring_start.dtype != DType::I32
        || cache.accepted.dtype != DType::I32
    {
        return Err(InferenceError("invalid replay cache storage"));
    }
    let s = cache.state.meta.shape();
    let g = Geometry {
        slots: s[0],
        heads: s[1],
        dim: s[2],
        n: s[3],
        groups: cache.b.meta.shape()[1],
        ring: cache.x.meta.shape()[2],
    };
    if g.heads == 0
        || g.dim == 0
        || g.n == 0
        || g.n > 2048
        || g.groups == 0
        || g.heads % g.groups != 0
        || g.ring == 0
        || cache.x.meta.shape()[..] != [g.slots, g.heads, g.ring, g.dim]
        || cache.b.meta.shape()[..] != [g.slots, g.groups, g.ring, g.n]
        || cache.dt.meta.shape()[..] != [g.slots, g.heads, g.ring]
        || cache.ring_start.meta.shape()[..] != [g.slots]
        || cache.accepted.meta.shape()[..] != [g.slots]
    {
        return Err(InferenceError("replay cache dimensions mismatch"));
    }
    super::float(&cache.x)?;
    super::float(&cache.b)?;
    Ok(g)
}
fn clone_scan<R: Runtime>(s: &SsmOutput<R>) -> SsmOutput<R> {
    SsmOutput {
        values: s.values.clone(),
        final_states: s.final_states.clone(),
        intermediate_states: s.intermediate_states.clone(),
        checkpoints: s.checkpoints.clone(),
    }
}
fn iota<R: Runtime>(like: &RudaTensor<R>, n: usize) -> Result<RudaTensor<R>, InferenceError> {
    if n > i32::MAX as usize {
        return Err(InferenceError("replay request indices exceed I32"));
    }
    let out = alloc(like, [n], DType::I32);
    if n > 0 {
        let dim = RudaDim::new(like.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&like.client, n, dim);
        super::pre_attention::positions::launch::<R>(
            &like.client,
            count,
            dim,
            out.clone().into_array_arg(),
            n as u32,
        );
    }
    Ok(out)
}
/// Reconstruct one FP32 state per request. Cache entries are read on demand;
/// no full-pool casts or intermediate token-state history are required.
pub fn materialize<R: Runtime>(
    cache: &ReplayCache<R>,
    a: &RudaTensor<R>,
    source_slots: &RudaTensor<R>,
    starts: &RudaTensor<R>,
    prefix: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    super::float(&cache.state)?;
    let g = geometry(cache)?;
    let a = super::norm::cast(a, DType::F32)?;
    let out = alloc(
        &cache.state,
        [source_slots.meta.num_elements(), g.heads, g.dim, g.n],
        DType::F32,
    );
    unsafe {
        materialize_into(cache, &a, source_slots, starts, prefix, None, 0, &out)?;
    }
    Ok(out)
}
/// Reconstruct and encode per-request states. A zero replay prefix uses an
/// exact payload/scale copy rather than a lossy decode/encode round trip.
pub fn materialize_quantized<R: Runtime>(
    cache: &QuantizedReplayCache<R>,
    a: &RudaTensor<R>,
    source_slots: &RudaTensor<R>,
    starts: &RudaTensor<R>,
    prefix: &RudaTensor<R>,
    rounding: Rounding,
    seed: Option<&RudaTensor<R>>,
) -> Result<QuantizedState<R>, InferenceError> {
    cache.state.validate()?;
    let raw = cache.raw();
    let g = geometry(&raw)?;
    let a = super::norm::cast(a, DType::F32)?;
    let decoded = alloc(
        &raw.state,
        [source_slots.meta.num_elements(), g.heads, g.dim, g.n],
        DType::F32,
    );
    unsafe {
        materialize_into(
            &raw,
            &a,
            source_slots,
            starts,
            prefix,
            Some(&cache.state.scale),
            format_code(cache.state.format),
            &decoded,
        )?;
    }
    let out = QuantizedState::allocate(&decoded, decoded.meta.shape().clone(), cache.state.format)?;
    let ids = iota(&decoded, source_slots.meta.num_elements())?;
    unsafe {
        out.encode_into(&decoded, rounding, seed)?;
        super::state_storage::copy_selected_rows(
            &cache.state,
            source_slots,
            &out,
            &ids,
            Some(prefix),
            true,
        )?;
    }
    Ok(out)
}
unsafe fn materialize_into<R: Runtime>(
    cache: &ReplayCache<R>,
    a: &RudaTensor<R>,
    source_slots: &RudaTensor<R>,
    starts: &RudaTensor<R>,
    prefix: &RudaTensor<R>,
    scales: Option<&RudaTensor<R>>,
    format: u32,
    out: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    let g = geometry(cache)?;
    for t in [a, source_slots, starts, prefix, out] {
        check(t, &cache.state)?;
    }
    if a.dtype != DType::F32
        || a.meta.shape()[..] != [g.heads]
        || source_slots.dtype != DType::I32
        || source_slots.meta.num_dims() != 1
        || starts.dtype != DType::I32
        || prefix.dtype != DType::I32
        || starts.meta.shape() != source_slots.meta.shape()
        || prefix.meta.shape() != source_slots.meta.shape()
        || out.dtype != DType::F32
        || out.meta.shape()[..] != [source_slots.meta.num_elements(), g.heads, g.dim, g.n]
    {
        return Err(InferenceError("replay materialization operands mismatch"));
    }
    if let Some(sf) = scales {
        check(sf, &cache.state)?;
        if sf.dtype != DType::F32 || sf.meta.shape()[..] != [g.slots, g.heads, g.dim] {
            return Err(InferenceError("replay state scale dimensions mismatch"));
        }
    } else if format != 0 {
        return Err(InferenceError("quantized replay requires state scales"));
    }
    let n = out.meta.num_elements() / g.n;
    if n > 0 {
        let dim = RudaDim::new(out.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
        reconstruct::launch::<R>(
            &out.client,
            count,
            dim,
            cache.state.clone().into_array_arg(),
            scales.unwrap_or(&cache.dt).clone().into_array_arg(),
            cache.x.clone().into_array_arg(),
            cache.b.clone().into_array_arg(),
            cache.dt.clone().into_array_arg(),
            a.clone().into_array_arg(),
            source_slots.clone().into_array_arg(),
            starts.clone().into_array_arg(),
            prefix.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            g.slots as u32,
            g.heads as u32,
            g.dim as u32,
            g.groups as u32,
            g.ring as u32,
            g.n,
            format,
            cache.state.dtype.into(),
            cache.x.dtype.into(),
            cache.b.dtype.into(),
        );
    }
    Ok(())
}
/// Layers retain typed ownership anchors instead of untyped pointer tables.
pub fn materialize_layers<R: Runtime>(
    layers: &[(&ReplayCache<R>, &RudaTensor<R>)],
    source_slots: &[&RudaTensor<R>],
    starts: &RudaTensor<R>,
    prefix: &RudaTensor<R>,
) -> Result<Vec<RudaTensor<R>>, InferenceError> {
    if layers.len() != source_slots.len() {
        return Err(InferenceError("replay layer/source-table count mismatch"));
    }
    layers
        .iter()
        .zip(source_slots)
        .map(|((c, a), ids)| materialize(c, a, ids, starts, prefix))
        .collect()
}
/// Prepared in-place checkpointing SSU. Only selected request states are
/// decoded/encoded. Scratch is O(batch*H*D*N), independent of cache capacity.
/// Strided cache views use retained packed storage and explicit write-back.
pub struct PreparedReplay<'a, R: Runtime> {
    plan: &'a SsmPlan<R>,
    pack: super::PreparedPack<R>,
    cache_packs: Vec<super::PreparedPack<R>>,
    cache: ReplayCache<R>,
    parameters: Vec<(super::PreparedPack<R>, RudaTensor<R>)>,
    dt: RudaTensor<R>,
    a: RudaTensor<R>,
    b: RudaTensor<R>,
    c: RudaTensor<R>,
    d: Option<RudaTensor<R>>,
    z: Option<RudaTensor<R>>,
    bias: Option<RudaTensor<R>>,
    boundaries: Option<super::metadata::PreparedIndices<R>>,
    ids: super::metadata::PreparedIndices<R>,
    effective_ids: RudaTensor<R>,
    active: RudaTensor<R>,
    starts: RudaTensor<R>,
    prefix: RudaTensor<R>,
    pub flushed: RudaTensor<R>,
    initial: RudaTensor<R>,
    pub scan: SsmOutput<R>,
    quantized: Option<QuantizedState<R>>,
    encoded: Option<QuantizedState<R>>,
    request_ids: RudaTensor<R>,
    seed: Option<RudaTensor<R>>,
    rounding: Rounding,
    options: SsmOptions,
    predicted: usize,
}
impl<'a, R: Runtime> PreparedReplay<'a, R> {
    pub fn prepare(
        plan: &'a SsmPlan<R>,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        cache: &ReplayCache<R>,
        source_slots: &RudaTensor<R>,
        max_predicted: usize,
        options: SsmOptions,
    ) -> Result<Self, InferenceError> {
        super::float(&cache.state)?;
        Self::bind(
            plan,
            input,
            inputs,
            cache,
            None,
            source_slots,
            max_predicted,
            Rounding::default(),
            None,
            options,
        )
    }
    pub fn prepare_quantized(
        plan: &'a SsmPlan<R>,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        cache: &QuantizedReplayCache<R>,
        source_slots: &RudaTensor<R>,
        max_predicted: usize,
        rounding: Rounding,
        seed: Option<&RudaTensor<R>>,
        options: SsmOptions,
    ) -> Result<Self, InferenceError> {
        Self::bind(
            plan,
            input,
            inputs,
            &cache.raw(),
            Some(&cache.state),
            source_slots,
            max_predicted,
            rounding,
            seed,
            options,
        )
    }
    fn bind(
        plan: &'a SsmPlan<R>,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        cache: &ReplayCache<R>,
        quant: Option<&QuantizedState<R>>,
        source_slots: &RudaTensor<R>,
        predicted: usize,
        rounding: Rounding,
        seed: Option<&RudaTensor<R>>,
        options: SsmOptions,
    ) -> Result<Self, InferenceError> {
        let pack = super::PreparedPack::prepare(input)?;
        let input = &pack.output;
        super::float(input)?;
        let mut cache_packs = Vec::new();
        for t in [
            &cache.state,
            &cache.x,
            &cache.b,
            &cache.dt,
            &cache.ring_start,
            &cache.accepted,
        ] {
            super::check_view(t, input)?;
            cache_packs.push(super::PreparedPack::prepare(t)?);
        }
        let cache = ReplayCache {
            state: cache_packs[0].output.clone(),
            x: cache_packs[1].output.clone(),
            b: cache_packs[2].output.clone(),
            dt: cache_packs[3].output.clone(),
            ring_start: cache_packs[4].output.clone(),
            accepted: cache_packs[5].output.clone(),
        };
        let g = geometry(&cache)?;
        if predicted == 0
            || predicted >= g.ring
            || input.meta.shape()[..] != [plan.tokens, g.heads, g.dim]
            || source_slots.meta.shape()[..] != [plan.sequences]
            || rounding.philox_rounds > 64
        {
            return Err(InferenceError(
                "checkpoint SSU predicted window/slots mismatch",
            ));
        }
        let ids = super::metadata::PreparedIndices::prepare(source_slots, DType::I32)?;
        let mut parameters = Vec::new();
        let mut parameter = |t: &RudaTensor<R>| -> Result<RudaTensor<R>, InferenceError> {
            super::check_view(t, input)?;
            super::float(t)?;
            let p = super::PreparedPack::prepare(t)?;
            let out = alloc(input, t.meta.shape().clone(), DType::F32);
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
        if a.meta.shape()[..] != [g.heads]
            || (dt.meta.shape()[..] != [plan.tokens, g.heads]
                && dt.meta.shape()[..] != [plan.tokens, g.heads, g.dim])
            || b.meta.shape()[..] != [plan.tokens, g.groups, g.n]
            || c.meta.shape() != b.meta.shape()
            || z.as_ref()
                .is_some_and(|t| t.meta.shape() != input.meta.shape())
            || d.iter()
                .chain(bias.iter())
                .any(|t| t.meta.shape()[..] != [g.heads] && t.meta.shape()[..] != [g.heads, g.dim])
        {
            return Err(InferenceError(
                "ReplaySSM parameters require head-tied dt/A and compatible B/C",
            ));
        }
        let boundaries = inputs
            .checkpoint_boundaries
            .map(|t| super::metadata::PreparedIndices::prepare(t, DType::I32))
            .transpose()?;
        if boundaries
            .as_ref()
            .is_some_and(|t| t.output.meta.shape()[..] != [plan.sequences])
        {
            return Err(InferenceError("ReplaySSM checkpoint boundaries mismatch"));
        }
        let quantized = if let Some(q) = quant {
            let p = super::PreparedPack::prepare(&q.scale)?;
            let sf = p.output.clone();
            cache_packs.push(p);
            {
                let state = QuantizedState {
                    data: cache.state.clone(),
                    scale: sf,
                    format: q.format,
                };
                state.validate()?;
                Some(state)
            }
        } else {
            None
        };
        if let Some(seed) = seed {
            check(seed, input)?;
            if !matches!(seed.dtype, DType::I64 | DType::U64) || seed.meta.num_elements() != 1 {
                return Err(InferenceError("ReplaySSM seed must be an I64/U64 scalar"));
            }
        }
        let encoded = quant
            .map(|q| {
                QuantizedState::allocate(
                    input,
                    ruda_core::tensor::Shape::from([plan.sequences, g.heads, g.dim, g.n]),
                    q.format,
                )
            })
            .transpose()?;
        let initial = alloc(input, [plan.sequences, g.heads, g.dim, g.n], DType::F32);
        let scan = plan.allocate_workspace(
            input,
            g.heads,
            g.dim,
            g.n,
            options.cache_intermediate,
            boundaries.is_some(),
        )?;
        let effective_ids = alloc(input, [plan.sequences], DType::I32);
        let active = alloc(input, [plan.sequences], DType::I32);
        let starts = alloc(input, [plan.sequences], DType::I32);
        let prefix = alloc(input, [plan.sequences], DType::I32);
        let flushed = alloc(input, [plan.sequences], DType::I32);
        let request_ids = iota(input, plan.sequences)?;
        Ok(Self {
            plan,
            pack,
            cache_packs,
            cache,
            parameters,
            dt,
            a,
            b,
            c,
            d,
            z,
            bias,
            boundaries,
            ids,
            effective_ids,
            active,
            starts,
            prefix,
            flushed,
            initial,
            scan,
            quantized,
            encoded,
            request_ids,
            seed: seed.cloned(),
            rounding,
            options,
            predicted,
        })
    }
    /// The input, selected cache slots, scale payloads and all scratch must be
    /// exclusively owned on this queue. Selected slots must be unique. dt and
    /// dt_bias feature axes, when present, must be head-tied (broadcast values).
    /// Ring cursors and accepted counters are read-only; caller advances them.
    pub unsafe fn run(&self) -> Result<&SsmOutput<R>, InferenceError> {
        unsafe {
            self.pack.run();
            for p in &self.cache_packs {
                p.run();
            }
            self.ids.run();
            if let Some(b) = &self.boundaries {
                b.run();
            }
            for (source, out) in &self.parameters {
                source.run();
                super::norm::cast_into(&source.output, out)?;
            }
        }
        let input = &self.pack.output;
        let g = geometry(&self.cache)?;
        let n = self.plan.sequences;
        if n > 0 {
            let dim = RudaDim::new(input.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
            request_metadata::launch::<R>(
                &input.client,
                count,
                dim,
                self.ids.output.clone().into_array_arg(),
                self.cache.ring_start.clone().into_array_arg(),
                self.cache.accepted.clone().into_array_arg(),
                self.starts.clone().into_array_arg(),
                self.prefix.clone().into_array_arg(),
                self.flushed.clone().into_array_arg(),
                self.effective_ids.clone().into_array_arg(),
                self.active.clone().into_array_arg(),
                n as u32,
                g.slots as u32,
                g.ring as u32,
                self.predicted as u32,
                self.options.pad_slot,
            );
        }
        unsafe {
            materialize_into(
                &self.cache,
                &self.a,
                &self.effective_ids,
                &self.starts,
                &self.prefix,
                self.quantized.as_ref().map(|q| &q.scale),
                self.quantized.as_ref().map_or(0, |q| format_code(q.format)),
                &self.initial,
            )?;
            self.plan.run_into(
                input,
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
                clone_scan(&self.scan),
            )?;
        }
        let work = n
            .checked_mul(g.heads)
            .and_then(|n| n.checked_mul(g.dim.max(g.n)))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("replay ring update exceeds U32"))?;
        if work > 0 {
            let dim = RudaDim::new(input.client.properties(), work);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, work, dim);
            record::launch::<R>(
                &input.client,
                count,
                dim,
                input.clone().into_array_arg(),
                self.b.clone().into_array_arg(),
                self.dt.clone().into_array_arg(),
                self.bias
                    .as_ref()
                    .unwrap_or(&self.dt)
                    .clone()
                    .into_array_arg(),
                self.initial.clone().into_array_arg(),
                self.effective_ids.clone().into_array_arg(),
                self.starts.clone().into_array_arg(),
                self.prefix.clone().into_array_arg(),
                self.flushed.clone().into_array_arg(),
                self.plan.cu.clone().into_array_arg(),
                self.cache.state.clone().into_array_arg(),
                self.cache.x.clone().into_array_arg(),
                self.cache.b.clone().into_array_arg(),
                self.cache.dt.clone().into_array_arg(),
                n as u32,
                g.slots as u32,
                g.heads as u32,
                g.dim as u32,
                g.n as u32,
                g.groups as u32,
                g.ring as u32,
                self.options.dt_min,
                self.options.dt_max,
                self.options.dt_softplus,
                self.bias.is_some(),
                self.dt.meta.num_dims() == 3,
                self.bias.as_ref().is_some_and(|t| t.meta.num_dims() == 2),
                self.quantized.is_none(),
                input.dtype.into(),
                self.cache.x.dtype.into(),
                self.cache.b.dtype.into(),
                self.cache.state.dtype.into(),
            );
        }
        if let (Some(q), Some(encoded)) = (&self.quantized, &self.encoded) {
            unsafe {
                encoded.encode_into(&self.initial, self.rounding, self.seed.as_ref())?;
                super::state_storage::copy_selected_rows(
                    encoded,
                    &self.request_ids,
                    q,
                    &self.effective_ids,
                    Some(&self.flushed),
                    false,
                )?;
            }
        }
        unsafe {
            for (i, p) in self.cache_packs.iter().enumerate() {
                if i < 4 || i == 6 {
                    p.write_back();
                }
            }
        }
        Ok(&self.scan)
    }
}
/// Prepared prefix-cache materialization into selected slots of the same
/// layer. All source requests are snapshotted before any destination write,
/// so overlapping source/destination tables are well-defined.
pub struct PreparedMaterialize<R: Runtime> {
    packs: Vec<super::PreparedPack<R>>,
    cache: ReplayCache<R>,
    a_pack: super::PreparedPack<R>,
    a: RudaTensor<R>,
    source: super::metadata::PreparedIndices<R>,
    destination: super::metadata::PreparedIndices<R>,
    starts: super::metadata::PreparedIndices<R>,
    prefix: super::metadata::PreparedIndices<R>,
    active: Option<super::metadata::PreparedIndices<R>>,
    source_ids: RudaTensor<R>,
    destination_ids: RudaTensor<R>,
    request_ids: RudaTensor<R>,
    decoded: RudaTensor<R>,
    encoded: RudaTensor<R>,
    quantized: Option<QuantizedState<R>>,
    encoded_quantized: Option<QuantizedState<R>>,
    rounding: Rounding,
    seed: Option<RudaTensor<R>>,
}
impl<R: Runtime> PreparedMaterialize<R> {
    pub fn prepare(
        cache: &ReplayCache<R>,
        a: &RudaTensor<R>,
        source: &RudaTensor<R>,
        destination: &RudaTensor<R>,
        starts: &RudaTensor<R>,
        prefix: &RudaTensor<R>,
        active: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        super::float(&cache.state)?;
        Self::bind(
            cache,
            None,
            a,
            source,
            destination,
            starts,
            prefix,
            active,
            Rounding::default(),
            None,
        )
    }
    pub fn prepare_quantized(
        cache: &QuantizedReplayCache<R>,
        a: &RudaTensor<R>,
        source: &RudaTensor<R>,
        destination: &RudaTensor<R>,
        starts: &RudaTensor<R>,
        prefix: &RudaTensor<R>,
        active: Option<&RudaTensor<R>>,
        rounding: Rounding,
        seed: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        Self::bind(
            &cache.raw(),
            Some(&cache.state),
            a,
            source,
            destination,
            starts,
            prefix,
            active,
            rounding,
            seed,
        )
    }
    fn bind(
        cache: &ReplayCache<R>,
        quant: Option<&QuantizedState<R>>,
        a: &RudaTensor<R>,
        source: &RudaTensor<R>,
        destination: &RudaTensor<R>,
        starts: &RudaTensor<R>,
        prefix: &RudaTensor<R>,
        active: Option<&RudaTensor<R>>,
        rounding: Rounding,
        seed: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        let mut packs = Vec::new();
        for t in [
            &cache.state,
            &cache.x,
            &cache.b,
            &cache.dt,
            &cache.ring_start,
            &cache.accepted,
        ] {
            packs.push(super::PreparedPack::prepare(t)?);
        }
        let cache = ReplayCache {
            state: packs[0].output.clone(),
            x: packs[1].output.clone(),
            b: packs[2].output.clone(),
            dt: packs[3].output.clone(),
            ring_start: packs[4].output.clone(),
            accepted: packs[5].output.clone(),
        };
        let g = geometry(&cache)?;
        let a_pack = super::PreparedPack::prepare(a)?;
        check(&a_pack.output, &cache.state)?;
        super::float(&a_pack.output)?;
        if a.meta.shape()[..] != [g.heads] || rounding.philox_rounds > 64 {
            return Err(InferenceError("materialization A/rounding mismatch"));
        }
        let a = alloc(&cache.state, [g.heads], DType::F32);
        let index =
            |t: &RudaTensor<R>| -> Result<super::metadata::PreparedIndices<R>, InferenceError> {
                super::check_view(t, &cache.state)?;
                super::metadata::PreparedIndices::prepare(t, DType::I32)
            };
        let source = index(source)?;
        let destination = index(destination)?;
        let starts = index(starts)?;
        let prefix = index(prefix)?;
        let active = active.map(index).transpose()?;
        let requests = source.output.meta.num_elements();
        if source.output.meta.num_dims() != 1
            || [&destination.output, &starts.output, &prefix.output]
                .iter()
                .any(|p| p.meta.shape() != source.output.meta.shape())
            || active
                .as_ref()
                .is_some_and(|p| p.output.meta.shape() != source.output.meta.shape())
        {
            return Err(InferenceError(
                "materialization slot/metadata vectors mismatch",
            ));
        }
        let quantized = if let Some(q) = quant {
            let p = super::PreparedPack::prepare(&q.scale)?;
            let scale = p.output.clone();
            packs.push(p);
            let q = QuantizedState {
                data: cache.state.clone(),
                scale,
                format: q.format,
            };
            q.validate()?;
            Some(q)
        } else {
            None
        };
        let shape = ruda_core::tensor::Shape::from([requests, g.heads, g.dim, g.n]);
        let decoded = alloc(&cache.state, shape.clone(), DType::F32);
        let encoded_quantized = quantized
            .as_ref()
            .map(|q| QuantizedState::allocate(&cache.state, shape.clone(), q.format))
            .transpose()?;
        let encoded = encoded_quantized.as_ref().map_or_else(
            || alloc(&cache.state, shape.clone(), cache.state.dtype),
            |q| q.data.clone(),
        );
        let source_ids = alloc(&cache.state, [requests], DType::I32);
        let destination_ids = alloc(&cache.state, [requests], DType::I32);
        let request_ids = iota(&cache.state, requests)?;
        if let Some(seed) = seed {
            check(seed, &cache.state)?;
            if !matches!(seed.dtype, DType::I64 | DType::U64) || seed.meta.num_elements() != 1 {
                return Err(InferenceError(
                    "materialization seed requires I64/U64 scalar",
                ));
            }
        }
        Ok(Self {
            packs,
            cache,
            a_pack,
            a,
            source,
            destination,
            starts,
            prefix,
            active,
            source_ids,
            destination_ids,
            request_ids,
            decoded,
            encoded,
            quantized,
            encoded_quantized,
            rounding,
            seed: seed.cloned(),
        })
    }
    /// Active destination slots must be unique. `active`, if present, is a
    /// permutation prefix of request indices followed by -1 entries. Caller
    /// retains exclusive access to selected destinations until queue completion.
    pub unsafe fn run(&self) -> Result<(), InferenceError> {
        unsafe {
            for p in &self.packs {
                p.run();
            }
            self.a_pack.run();
            super::norm::cast_into(&self.a_pack.output, &self.a)?;
            for p in [&self.source, &self.destination, &self.starts, &self.prefix] {
                p.run();
            }
            if let Some(p) = &self.active {
                p.run();
            }
        }
        let g = geometry(&self.cache)?;
        let n = self.source_ids.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(self.a.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.a.client, n, dim);
            materialize_requests::launch::<R>(
                &self.a.client,
                count,
                dim,
                self.source.output.clone().into_array_arg(),
                self.destination.output.clone().into_array_arg(),
                self.active
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(&self.source.output)
                    .clone()
                    .into_array_arg(),
                self.source_ids.clone().into_array_arg(),
                self.destination_ids.clone().into_array_arg(),
                n as u32,
                g.slots as u32,
                self.active.is_some(),
            );
        }
        unsafe {
            materialize_into(
                &self.cache,
                &self.a,
                &self.source_ids,
                &self.starts.output,
                &self.prefix.output,
                self.quantized.as_ref().map(|q| &q.scale),
                self.quantized.as_ref().map_or(0, |q| format_code(q.format)),
                &self.decoded,
            )?;
        }
        if let (Some(q), Some(encoded)) = (&self.quantized, &self.encoded_quantized) {
            unsafe {
                encoded.encode_into(&self.decoded, self.rounding, self.seed.as_ref())?;
                super::state_storage::copy_selected_rows(
                    q,
                    &self.source_ids,
                    encoded,
                    &self.request_ids,
                    Some(&self.prefix.output),
                    true,
                )?;
                super::state_storage::copy_selected_rows(
                    encoded,
                    &self.request_ids,
                    q,
                    &self.destination_ids,
                    None,
                    false,
                )?;
            }
        } else {
            unsafe {
                super::norm::cast_into(&self.decoded, &self.encoded)?;
                self.raw_copy(
                    &self.cache.state,
                    &self.source_ids,
                    &self.encoded,
                    &self.request_ids,
                    true,
                )?;
                self.raw_copy(
                    &self.encoded,
                    &self.request_ids,
                    &self.cache.state,
                    &self.destination_ids,
                    false,
                )?;
            }
        }
        unsafe {
            self.packs[0].write_back();
            if let Some(p) = self.packs.get(6) {
                p.write_back();
            }
        }
        Ok(())
    }
    unsafe fn raw_copy(
        &self,
        source: &RudaTensor<R>,
        source_ids: &RudaTensor<R>,
        destination: &RudaTensor<R>,
        dest_ids: &RudaTensor<R>,
        zero_prefix: bool,
    ) -> Result<(), InferenceError> {
        let g = geometry(&self.cache)?;
        let width = g.heads * g.dim * g.n;
        let n = source_ids
            .meta
            .num_elements()
            .checked_mul(width)
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("materialization copy exceeds U32"))?;
        if n > 0 {
            let dim = RudaDim::new(source.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&source.client, n, dim);
            copy_state_bits::launch::<R>(
                &source.client,
                count,
                dim,
                source.clone().into_array_arg(),
                source_ids.clone().into_array_arg(),
                dest_ids.clone().into_array_arg(),
                self.prefix.output.clone().into_array_arg(),
                destination.clone().into_array_arg(),
                n as u32,
                width as u32,
                (source.meta.num_elements() / width) as u32,
                (destination.meta.num_elements() / width) as u32,
                zero_prefix,
                source.dtype.into(),
            );
        }
        Ok(())
    }
}
#[ruda(launch)]
pub fn materialize_requests(
    source: &Array<i32>,
    destination: &Array<i32>,
    active: &Array<i32>,
    src: &mut Array<i32>,
    dst: &mut Array<i32>,
    n: u32,
    slots: u32,
    #[comptime] indexed: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut enabled = true;
        if comptime!(indexed) {
            enabled = false;
            let mut j: usize = 0usize;
            while j < n as usize {
                let request = active[j];
                if request < 0 {
                    break;
                }
                if request == i as i32 {
                    enabled = true;
                    break;
                }
                j += 1;
            }
        }
        let s = source[i];
        let d = destination[i];
        enabled = enabled && s >= 0 && s < slots as i32 && d >= 0 && d < slots as i32;
        src[i] = -1;
        dst[i] = -1;
        if enabled {
            src[i] = s;
            dst[i] = d;
        }
    }
}
#[ruda(launch)]
pub fn copy_state_bits<S: Numeric>(
    source: &Array<S>,
    src: &Array<i32>,
    dst: &Array<i32>,
    prefix: &Array<i32>,
    out: &mut Array<S>,
    n: u32,
    width: u32,
    source_slots: u32,
    destination_slots: u32,
    #[comptime] zero_only: bool,
    #[define(S)] _s: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let request = i / width as usize;
        let col = i % width as usize;
        let s = src[request];
        let d = dst[request];
        let mut enabled =
            s >= 0 && s < source_slots as i32 && d >= 0 && d < destination_slots as i32;
        if comptime!(zero_only) {
            enabled = enabled && prefix[request] == 0;
        }
        if enabled {
            out[d as usize * width as usize + col] = source[s as usize * width as usize + col];
        }
    }
}
impl<R: Runtime> SsmPlan<R> {
    /// Functional copy-on-write convenience. Use PreparedReplay for graph-safe
    /// in-place execution; it does not copy an entire cache per invocation.
    pub fn checkpointing_ssu(
        &self,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        cache: &ReplayCache<R>,
        source_slots: &RudaTensor<R>,
        max_predicted: usize,
        options: SsmOptions,
    ) -> Result<ReplayOutput<R>, InferenceError> {
        let copy = |t: &RudaTensor<R>| crate::paged_attention::copy_contiguous(t);
        let next = ReplayCache {
            state: copy(&cache.state),
            x: copy(&cache.x),
            b: copy(&cache.b),
            dt: copy(&cache.dt),
            ring_start: copy(&cache.ring_start),
            accepted: copy(&cache.accepted),
        };
        let plan = PreparedReplay::prepare(
            self,
            input,
            inputs,
            &next,
            source_slots,
            max_predicted,
            options,
        )?;
        unsafe {
            plan.run()?;
        }
        Ok(ReplayOutput {
            scan: clone_scan(&plan.scan),
            cache: next,
            flushed: plan.flushed.clone(),
        })
    }
    pub fn checkpointing_ssu_quantized(
        &self,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        cache: &QuantizedReplayCache<R>,
        source_slots: &RudaTensor<R>,
        max_predicted: usize,
        rounding: Rounding,
        seed: Option<&RudaTensor<R>>,
        options: SsmOptions,
    ) -> Result<QuantizedReplayOutput<R>, InferenceError> {
        let copy = |t: &RudaTensor<R>| crate::paged_attention::copy_contiguous(t);
        let next = QuantizedReplayCache {
            state: cache.state.copy(),
            x: copy(&cache.x),
            b: copy(&cache.b),
            dt: copy(&cache.dt),
            ring_start: copy(&cache.ring_start),
            accepted: copy(&cache.accepted),
        };
        let plan = PreparedReplay::prepare_quantized(
            self,
            input,
            inputs,
            &next,
            source_slots,
            max_predicted,
            rounding,
            seed,
            options,
        )?;
        unsafe {
            plan.run()?;
        }
        Ok(QuantizedReplayOutput {
            scan: clone_scan(&plan.scan),
            cache: next,
            flushed: plan.flushed.clone(),
        })
    }
}
#[ruda(launch)]
pub fn request_metadata(
    ids: &Array<i32>,
    starts: &Array<i32>,
    counts: &Array<i32>,
    out_start: &mut Array<i32>,
    out_count: &mut Array<i32>,
    flush: &mut Array<i32>,
    effective: &mut Array<i32>,
    active: &mut Array<i32>,
    n: u32,
    slots: u32,
    ring: u32,
    predicted: u32,
    pad: i32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let slot = ids[i];
        effective[i] = -1;
        active[i] = -1;
        let valid = slot >= 0 && slot < slots as i32 && slot != pad;
        if valid {
            effective[i] = slot;
            active[i] = i as i32;
        }
        let mut start: i32 = 0i32;
        let mut count: i32 = 0i32;
        if valid {
            start = starts[slot as usize];
            count = counts[slot as usize];
        }
        out_start[i] = start;
        out_count[i] = count;
        let mut flag: i32 = 0i32;
        if valid && count >= 0 && count as i64 + predicted as i64 * 2 > ring as i64 {
            flag = 1;
        }
        flush[i] = flag;
    }
}
#[ruda(launch)]
pub fn reconstruct<S: Numeric, X: Float, B: Float>(
    initial: &Array<S>,
    state_scales: &Array<f32>,
    x: &Array<X>,
    b: &Array<B>,
    dt: &Array<f32>,
    a: &Array<f32>,
    slots_in: &Array<i32>,
    starts: &Array<i32>,
    prefix: &Array<i32>,
    out: &mut Array<f32>,
    units: u32,
    slots: u32,
    heads: u32,
    dim: u32,
    groups: u32,
    ring: u32,
    #[comptime] n: usize,
    #[comptime] format: u32,
    #[define(S)] _s: StorageType,
    #[define(X)] _x: StorageType,
    #[define(B)] _b: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < units as usize {
        let channel = i % dim as usize;
        let head = i / dim as usize % heads as usize;
        let request = i / (dim as usize * heads as usize);
        let slot = slots_in[request];
        let start = starts[request];
        let count = prefix[request];
        let valid = slot >= 0
            && slot < slots as i32
            && start >= 0
            && start < ring as i32
            && count >= 0
            && count <= ring as i32;
        let mut state = Array::<f32>::new(n);
        for j in 0usize..n {
            state[j] = 0.0;
            if valid {
                let vector = (slot as usize * heads as usize + head) * dim as usize + channel;
                let mut value = f32::cast_from(initial[vector * n + j]);
                if comptime!(format == 3 || format == 4) {
                    value = rublas::tensor_float_quant::fp8_decode(
                        u32::cast_from(initial[vector * n + j]),
                        format == 4,
                    );
                }
                if comptime!(format != 0) {
                    value *= state_scales[vector];
                }
                state[j] = value;
            }
        }
        if valid {
            let group = head / (heads / groups) as usize;
            for token in 0..count as usize {
                let row = (start as usize + token) % ring as usize;
                let step = dt[(slot as usize * heads as usize + head) * ring as usize + row];
                let decay = (step * a[head]).exp();
                let value = f32::cast_from(
                    x[((slot as usize * heads as usize + head) * ring as usize + row)
                        * dim as usize
                        + channel],
                );
                for j in 0usize..n {
                    state[j] = fma(
                        state[j],
                        decay,
                        step * value
                            * f32::cast_from(
                                b[((slot as usize * groups as usize + group) * ring as usize
                                    + row)
                                    * n
                                    + j],
                            ),
                    );
                }
            }
        }
        for j in 0usize..n {
            out[i * n + j] = state[j];
        }
    }
}
#[ruda(launch)]
pub fn record<F: Float, X: Float, B: Float, S: Numeric>(
    x: &Array<F>,
    b: &Array<f32>,
    dt: &Array<f32>,
    bias: &Array<f32>,
    replayed: &Array<f32>,
    ids: &Array<i32>,
    starts: &Array<i32>,
    prefix: &Array<i32>,
    flush: &Array<i32>,
    cu: &Array<i32>,
    state: &mut Array<S>,
    x_ring: &mut Array<X>,
    b_ring: &mut Array<B>,
    dt_ring: &mut Array<f32>,
    sequences: u32,
    slots: u32,
    heads: u32,
    dim: u32,
    n: u32,
    groups: u32,
    ring: u32,
    dt_min: f32,
    dt_max: f32,
    #[comptime] softplus: bool,
    #[comptime] biased: bool,
    #[comptime] dt_full: bool,
    #[comptime] bias_full: bool,
    #[comptime] save_state: bool,
    #[define(F)] _f: StorageType,
    #[define(X)] _x: StorageType,
    #[define(B)] _b: StorageType,
    #[define(S)] _s: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    let width = u32::max(dim, n) as usize;
    if i < sequences as usize * heads as usize * width {
        let channel = i % width;
        let head = i / width % heads as usize;
        let seq = i / (width * heads as usize);
        let slot = ids[seq];
        if slot >= 0 && slot < slots as i32 {
            let start = starts[seq] as usize;
            let count = prefix[seq] as usize;
            let write_start = (start + count) % ring as usize;
            let flushed = flush[seq] != 0;
            if comptime!(save_state) {
                if flushed && channel < dim as usize {
                    for j in 0..n as usize {
                        state[((slot as usize * heads as usize + head) * dim as usize + channel)
                            * n as usize
                            + j] = S::cast_from(
                            replayed[((seq * heads as usize + head) * dim as usize + channel)
                                * n as usize
                                + j],
                        );
                    }
                }
            }
            for token in cu[seq] as usize..cu[seq + 1] as usize {
                let row = (write_start + token - cu[seq] as usize) % ring as usize;
                if channel < dim as usize {
                    x_ring[((slot as usize * heads as usize + head) * ring as usize + row)
                        * dim as usize
                        + channel] =
                        X::cast_from(x[(token * heads as usize + head) * dim as usize + channel]);
                }
                if head % (heads / groups) as usize == 0 && channel < n as usize {
                    let group = head / (heads / groups) as usize;
                    b_ring[((slot as usize * groups as usize + group) * ring as usize + row)
                        * n as usize
                        + channel] =
                        B::cast_from(b[(token * groups as usize + group) * n as usize + channel]);
                }
                if channel == 0 {
                    let mut di = token * heads as usize + head;
                    if comptime!(dt_full) {
                        di *= dim as usize;
                    }
                    let mut step = dt[di];
                    if comptime!(biased) {
                        let mut bi = head;
                        if comptime!(bias_full) {
                            bi *= dim as usize;
                        }
                        step += bias[bi];
                    }
                    if comptime!(softplus) {
                        if step < 20.0 {
                            step = (1.0 + step.exp()).ln();
                        }
                    }
                    step = f32::min(f32::max(step, dt_min), dt_max);
                    dt_ring[(slot as usize * heads as usize + head) * ring as usize + row] = step;
                }
            }
        }
    }
}
