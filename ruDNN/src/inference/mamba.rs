//! Native selective-state scans, SSD and raw-history causal convolution.
//! Plans retain device sequence boundaries: replay can change lengths and slots
//! without copying numerical tensors or scheduling metadata back to the CPU.
use super::{InferenceError, alloc, check, float};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};

#[derive(Clone, Copy, Debug)]
pub struct SsmOptions {
    pub cache_intermediate: bool,
    pub dt_softplus: bool,
    pub dt_min: f32,
    pub dt_max: f32,
    pub pad_slot: i32,
}
impl Default for SsmOptions {
    fn default() -> Self {
        Self {
            cache_intermediate: false,
            dt_softplus: false,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
            pad_slot: -1,
        }
    }
}
/// A is [H], [H,N] or [H,D,N]; dt is [T,H] or [T,H,D].
/// B/C are [T,G,N], with H divisible by G. State is FP32 [slots,H,D,N].
pub struct SsmInputs<'a, R: Runtime> {
    pub dt: &'a RudaTensor<R>,
    pub a: &'a RudaTensor<R>,
    pub b: &'a RudaTensor<R>,
    pub c: &'a RudaTensor<R>,
    pub d: Option<&'a RudaTensor<R>>,
    pub z: Option<&'a RudaTensor<R>>,
    pub dt_bias: Option<&'a RudaTensor<R>>,
    pub checkpoint_boundaries: Option<&'a RudaTensor<R>>,
    pub state: Option<&'a RudaTensor<R>>,
    pub source_slots: Option<&'a RudaTensor<R>>,
    pub accepted_tokens: Option<&'a RudaTensor<R>>,
}
pub struct SsmOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub final_states: RudaTensor<R>,
    pub intermediate_states: RudaTensor<R>,
    pub checkpoints: RudaTensor<R>,
}
pub struct SsmPlan<R: Runtime> {
    cu_cast: Option<super::metadata::PreparedIndices<R>>,
    pub(crate) cu: RudaTensor<R>,
    pub(crate) sequences: usize,
    pub(crate) tokens: usize,
}
impl<R: Runtime> SsmPlan<R> {
    pub(crate) unsafe fn refresh_schedule(&self) {
        if let Some(p) = &self.cu_cast {
            unsafe {
                p.run();
            }
        }
    }
    pub fn allocate_output(
        &self,
        like: &RudaTensor<R>,
        heads: usize,
        dim: usize,
        state_dim: usize,
    ) -> Result<SsmOutput<R>, InferenceError> {
        self.allocate_workspace(like, heads, dim, state_dim, false, false)
    }
    pub fn allocate_workspace(
        &self,
        like: &RudaTensor<R>,
        heads: usize,
        dim: usize,
        state_dim: usize,
        history: bool,
        checkpoints: bool,
    ) -> Result<SsmOutput<R>, InferenceError> {
        check(like, &self.cu)?;
        float(like)?;
        if heads == 0 || dim == 0 || state_dim == 0 {
            return Err(InferenceError("SSM workspace axes must be positive"));
        }
        for s in [
            vec![self.tokens, heads, dim],
            vec![self.sequences, heads, dim, state_dim],
            vec![if history { self.tokens } else { 0 }, heads, dim, state_dim],
        ] {
            if s.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
            {
                return Err(InferenceError("SSM workspace exceeds U32"));
            }
        }
        Ok(SsmOutput {
            values: alloc(like, [self.tokens, heads, dim], like.dtype),
            final_states: alloc(like, [self.sequences, heads, dim, state_dim], DType::F32),
            intermediate_states: alloc(
                like,
                [if history { self.tokens } else { 0 }, heads, dim, state_dim],
                DType::F32,
            ),
            checkpoints: alloc(
                like,
                [
                    if checkpoints { self.sequences } else { 0 },
                    heads,
                    dim,
                    state_dim,
                ],
                DType::F32,
            ),
        })
    }
    /// Construct packed boundaries from monotonically nondecreasing device
    /// sequence IDs. Empty sequences are represented without a host readback.
    pub fn from_sequence_ids(
        ids: &RudaTensor<R>,
        sequences: usize,
    ) -> Result<Self, InferenceError> {
        check(ids, ids)?;
        if !matches!(ids.dtype, DType::I32 | DType::I64)
            || ids.meta.num_dims() != 1
            || sequences > i32::MAX as usize
        {
            return Err(InferenceError("SSM sequence IDs need I32 [tokens]"));
        }
        let cu = alloc(ids, [sequences + 1], DType::I32);
        let n = sequences + 1;
        let dim = RudaDim::new(ids.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&ids.client, n, dim);
        ids_to_boundaries::launch::<R>(
            &ids.client,
            count,
            dim,
            ids.clone().into_array_arg(),
            cu.clone().into_array_arg(),
            ids.meta.num_elements() as u32,
            sequences as u32,
            ids.dtype.into(),
        );
        Self::packed(&cu, ids.meta.num_elements())
    }
    pub fn run_quantized(
        &self,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        state: &super::state_storage::QuantizedState<R>,
        rounding: super::state_storage::Rounding,
        options: SsmOptions,
    ) -> Result<QuantizedSsmOutput<R>, InferenceError> {
        let initial = state.decode()?;
        let scan = self.run(
            input,
            SsmInputs {
                state: Some(&initial),
                ..inputs
            },
            options,
        )?;
        let final_states =
            super::state_storage::encode(&scan.final_states, state.format, rounding)?;
        let intermediate_states =
            super::state_storage::encode(&scan.intermediate_states, state.format, rounding)?;
        let checkpoints = super::state_storage::encode(&scan.checkpoints, state.format, rounding)?;
        Ok(QuantizedSsmOutput {
            values: scan.values,
            final_states,
            intermediate_states,
            checkpoints,
        })
    }
    pub fn packed(cu_seqlens: &RudaTensor<R>, tokens: usize) -> Result<Self, InferenceError> {
        check(cu_seqlens, cu_seqlens)?;
        if !matches!(cu_seqlens.dtype, DType::I32 | DType::I64 | DType::U32)
            || cu_seqlens.meta.num_dims() != 1
            || cu_seqlens.meta.shape()[0] < 2
            || tokens > i32::MAX as usize
        {
            return Err(InferenceError(
                "SSM schedule needs I32 [N+1] offsets and I32-indexed tokens",
            ));
        }
        let cu_cast = if cu_seqlens.dtype != DType::I32 {
            Some(super::metadata::PreparedIndices::prepare(
                cu_seqlens,
                DType::I32,
            )?)
        } else {
            None
        };
        let cu = cu_cast
            .as_ref()
            .map_or_else(|| cu_seqlens.clone(), |p| p.output.clone());
        Ok(Self {
            cu_cast,
            cu,
            sequences: cu_seqlens.meta.shape()[0] - 1,
            tokens,
        })
    }
    pub fn uniform(
        like: &RudaTensor<R>,
        batch: usize,
        length: usize,
    ) -> Result<Self, InferenceError> {
        let tokens = batch
            .checked_mul(length)
            .filter(|n| *n <= i32::MAX as usize)
            .ok_or(InferenceError("SSM schedule exceeds I32"))?;
        let prefix: Vec<i32> = (0..=batch).map(|n| (n * length) as i32).collect();
        let cu = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([prefix.len()]),
            like.client.create_from_slice(bytemuck::cast_slice(&prefix)),
            DType::I32,
        );
        Self::packed(&cu, tokens)
    }
    pub fn run_float_state(
        &self,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        options: SsmOptions,
    ) -> Result<SsmOutput<R>, InferenceError> {
        let dtype = inputs.state.map_or(DType::F32, |s| s.dtype);
        let state = inputs
            .state
            .map(|s| super::norm::cast(s, DType::F32))
            .transpose()?;
        let mut output = self.run(
            input,
            SsmInputs {
                state: state.as_ref(),
                ..inputs
            },
            options,
        )?;
        output.final_states = super::norm::cast(&output.final_states, dtype)?;
        output.intermediate_states = super::norm::cast(&output.intermediate_states, dtype)?;
        output.checkpoints = super::norm::cast(&output.checkpoints, dtype)?;
        Ok(output)
    }
    pub fn run(
        &self,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        options: SsmOptions,
    ) -> Result<SsmOutput<R>, InferenceError> {
        self.execute(input, inputs, options, None)
    }
    /// Allocation-free launch after operands and FP32 parameters are prepared.
    /// Outputs must be exclusive and must not alias inputs or one another.
    pub unsafe fn run_into(
        &self,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        options: SsmOptions,
        output: SsmOutput<R>,
    ) -> Result<SsmOutput<R>, InferenceError> {
        self.execute(input, inputs, options, Some(output))
    }
    fn execute(
        &self,
        input: &RudaTensor<R>,
        inputs: SsmInputs<'_, R>,
        options: SsmOptions,
        destination: Option<SsmOutput<R>>,
    ) -> Result<SsmOutput<R>, InferenceError> {
        if let Some(p) = &self.cu_cast {
            unsafe {
                p.run();
            }
        }
        let capture = destination.is_some();
        let x = if capture {
            input.clone()
        } else {
            into_contiguous(input.clone())
        };
        check(&x, &x)?;
        check(&self.cu, &x)?;
        float(&x)?;
        if x.meta.num_dims() != 3
            || x.meta.shape()[0] != self.tokens
            || options.dt_min.is_nan()
            || options.dt_max.is_nan()
            || options.dt_min > options.dt_max
        {
            return Err(InferenceError(
                "SSM needs [tokens,heads,dim] and ordered dt limits",
            ));
        }
        let heads = x.meta.shape()[1];
        let dim = x.meta.shape()[2];
        if heads == 0 || dim == 0 {
            return Err(InferenceError("SSM head axes must be nonempty"));
        }
        let prepare = |t: &RudaTensor<R>| -> Result<RudaTensor<R>, InferenceError> {
            if capture {
                check(t, &x)?;
                if t.dtype != DType::F32 {
                    return Err(InferenceError(
                        "SSM capture parameters must already be FP32",
                    ));
                }
                Ok(t.clone())
            } else {
                let t = super::norm::cast(t, DType::F32)?;
                check(&t, &x)?;
                Ok(t)
            }
        };
        let dt = prepare(inputs.dt)?;
        let a = prepare(inputs.a)?;
        let b = prepare(inputs.b)?;
        let c = prepare(inputs.c)?;
        if b.meta.num_dims() != 3
            || b.meta.shape()[0] != self.tokens
            || b.meta.shape() != c.meta.shape()
        {
            return Err(InferenceError("SSM B/C must be [tokens,groups,state_dim]"));
        }
        let groups = b.meta.shape()[1];
        let state_dim = b.meta.shape()[2];
        if groups == 0 || heads % groups != 0 || state_dim == 0 || state_dim > 2048 {
            return Err(InferenceError("SSM group/state dimensions invalid"));
        }
        let a_mode = if a.meta.shape()[..] == [heads] {
            0
        } else if a.meta.shape()[..] == [heads, state_dim] {
            1
        } else if a.meta.shape()[..] == [heads, dim, state_dim] {
            2
        } else {
            return Err(InferenceError("SSM A must be [H], [H,N] or [H,D,N]"));
        };
        let dt_full = dt.meta.shape()[..] == [self.tokens, heads, dim];
        if !dt_full && dt.meta.shape()[..] != [self.tokens, heads] {
            return Err(InferenceError("SSM dt must be [T,H] or [T,H,D]"));
        }
        let d = inputs.d.map(prepare).transpose()?;
        let z = inputs.z.map(prepare).transpose()?;
        let bias = inputs.dt_bias.map(prepare).transpose()?;
        for t in d.iter().chain(bias.iter()) {
            if t.meta.shape()[..] != [heads] && t.meta.shape()[..] != [heads, dim] {
                return Err(InferenceError("SSM skip/dt_bias must be [H] or [H,D]"));
            }
        }
        if z.as_ref().is_some_and(|z| z.meta.shape() != x.meta.shape()) {
            return Err(InferenceError("SSM gate shape mismatch"));
        }
        let mut slots: usize = 0usize;
        if let Some(s) = inputs.state {
            check(s, &x)?;
            if s.dtype != DType::F32
                || s.meta.num_dims() != 4
                || s.meta.shape()[1..] != [heads, dim, state_dim]
            {
                return Err(InferenceError("SSM state must be FP32 [slots,H,D,N]"));
            }
            slots = s.meta.shape()[0];
            if inputs.source_slots.is_none() && slots < self.sequences {
                return Err(InferenceError("SSM state has insufficient slots"));
            }
        }
        let mut index_width: usize = 1usize;
        if let Some(ids) = inputs.source_slots {
            check(ids, &x)?;
            if ids.dtype != DType::I32
                || ids.meta.shape()[0] != self.sequences
                || !(ids.meta.num_dims() == 1 || ids.meta.num_dims() == 2)
            {
                return Err(InferenceError(
                    "SSM source slots need I32 [sequences] or [sequences,steps]",
                ));
            }
            if ids.meta.num_dims() == 2 {
                index_width = ids.meta.shape()[1];
                if index_width == 0 {
                    return Err(InferenceError("empty SSM state index width"));
                }
            }
        }
        if let Some(t) = inputs.accepted_tokens {
            check(t, &x)?;
            if t.dtype != DType::I32 || t.meta.shape()[..] != [self.sequences] {
                return Err(InferenceError("SSM accepted counts need I32 [sequences]"));
            }
        }
        let state_shape = [self.sequences, heads, dim, state_dim];
        let history_shape = [
            if options.cache_intermediate {
                self.tokens
            } else {
                0
            },
            heads,
            dim,
            state_dim,
        ];
        let checkpoint_shape = [
            if inputs.checkpoint_boundaries.is_some() {
                self.sequences
            } else {
                0
            },
            heads,
            dim,
            state_dim,
        ];
        if let Some(t) = inputs.checkpoint_boundaries {
            check(t, &x)?;
            if t.dtype != DType::I32 || t.meta.shape()[..] != [self.sequences] {
                return Err(InferenceError("SSM checkpoint boundaries require I32 [N]"));
            }
        }
        for shape in [state_shape, history_shape] {
            if shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
            {
                return Err(InferenceError("SSM state/history exceeds U32"));
            }
        }
        let output = destination.unwrap_or_else(|| SsmOutput {
            values: alloc(&x, x.meta.shape().clone(), x.dtype),
            final_states: alloc(&x, state_shape, DType::F32),
            intermediate_states: alloc(&x, history_shape, DType::F32),
            checkpoints: alloc(&x, checkpoint_shape, DType::F32),
        });
        for (t, dtype, shape) in [
            (&output.values, x.dtype, x.meta.shape().to_vec()),
            (&output.final_states, DType::F32, state_shape.to_vec()),
            (
                &output.intermediate_states,
                DType::F32,
                history_shape.to_vec(),
            ),
            (&output.checkpoints, DType::F32, checkpoint_shape.to_vec()),
        ] {
            check(t, &x)?;
            if t.dtype != dtype || t.meta.shape()[..] != shape {
                return Err(InferenceError("SSM output shape/dtype mismatch"));
            }
        }
        let hw = &x.client.properties().hardware;
        let units = 128u32.min(hw.max_ruda_dim.0).min(hw.max_units_per_ruda);
        if self.sequences > hw.max_ruda_count.0 as usize
            || heads > hw.max_ruda_count.1 as usize
            || dim.div_ceil(units as usize) > hw.max_ruda_count.2 as usize
        {
            return Err(InferenceError("SSM grid exceeds device limits"));
        }
        if self.sequences != 0 {
            selective_scan::launch::<R>(
                &x.client,
                RudaCount::Static(
                    self.sequences as u32,
                    heads as u32,
                    dim.div_ceil(units as usize) as u32,
                ),
                RudaDim::new_1d(units),
                x.clone().into_array_arg(),
                dt.clone().into_array_arg(),
                a.into_array_arg(),
                b.into_array_arg(),
                c.into_array_arg(),
                d.as_ref().unwrap_or(&dt).clone().into_array_arg(),
                z.as_ref().unwrap_or(&dt).clone().into_array_arg(),
                bias.as_ref().unwrap_or(&dt).clone().into_array_arg(),
                inputs.state.unwrap_or(&dt).clone().into_array_arg(),
                self.cu.clone().into_array_arg(),
                inputs
                    .source_slots
                    .unwrap_or(&self.cu)
                    .clone()
                    .into_array_arg(),
                inputs
                    .accepted_tokens
                    .unwrap_or(&self.cu)
                    .clone()
                    .into_array_arg(),
                inputs
                    .checkpoint_boundaries
                    .unwrap_or(&self.cu)
                    .clone()
                    .into_array_arg(),
                output.values.clone().into_array_arg(),
                output.final_states.clone().into_array_arg(),
                output.intermediate_states.clone().into_array_arg(),
                output.checkpoints.clone().into_array_arg(),
                self.tokens as u32,
                heads as u32,
                dim as u32,
                groups as u32,
                slots as u32,
                index_width as u32,
                options.pad_slot,
                options.dt_min,
                options.dt_max,
                state_dim,
                a_mode,
                dt_full,
                d.is_some(),
                d.as_ref()
                    .is_some_and(|t| t.meta.num_elements() == heads * dim),
                z.is_some(),
                bias.is_some(),
                bias.as_ref()
                    .is_some_and(|t| t.meta.num_elements() == heads * dim),
                inputs.state.is_some(),
                inputs.source_slots.is_some(),
                inputs.accepted_tokens.is_some(),
                options.dt_softplus,
                options.cache_intermediate,
                inputs.checkpoint_boundaries.is_some(),
                x.dtype.into(),
            );
        }
        Ok(output)
    }
}
pub struct QuantizedSsmOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub final_states: super::state_storage::QuantizedState<R>,
    pub intermediate_states: super::state_storage::QuantizedState<R>,
    pub checkpoints: super::state_storage::QuantizedState<R>,
}
pub struct SsdOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub final_states: RudaTensor<R>,
    pub intermediate_states: RudaTensor<R>,
    pub checkpoints: RudaTensor<R>,
}
/// Dense batched SSD is the same selective scan with head-tied dt and A.
/// This interface accepts [B,L,H,D] and restores that shape in the result.
pub fn ssd_combined<R: Runtime>(
    input: &RudaTensor<R>,
    dt: &RudaTensor<R>,
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    c: &RudaTensor<R>,
    d: Option<&RudaTensor<R>>,
    z: Option<&RudaTensor<R>>,
    dt_bias: Option<&RudaTensor<R>>,
    initial: Option<&RudaTensor<R>>,
    options: SsmOptions,
) -> Result<SsdOutput<R>, InferenceError> {
    ssd_combined_impl(input, dt, a, b, c, d, z, dt_bias, initial, None, options)
}
pub fn ssd_combined_checkpointed<R: Runtime>(
    input: &RudaTensor<R>,
    dt: &RudaTensor<R>,
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    c: &RudaTensor<R>,
    d: Option<&RudaTensor<R>>,
    z: Option<&RudaTensor<R>>,
    dt_bias: Option<&RudaTensor<R>>,
    initial: Option<&RudaTensor<R>>,
    boundaries: &RudaTensor<R>,
    options: SsmOptions,
) -> Result<SsdOutput<R>, InferenceError> {
    ssd_combined_impl(
        input,
        dt,
        a,
        b,
        c,
        d,
        z,
        dt_bias,
        initial,
        Some(boundaries),
        options,
    )
}
pub fn ssd_combined_impl<R: Runtime>(
    input: &RudaTensor<R>,
    dt: &RudaTensor<R>,
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    c: &RudaTensor<R>,
    d: Option<&RudaTensor<R>>,
    z: Option<&RudaTensor<R>>,
    dt_bias: Option<&RudaTensor<R>>,
    initial: Option<&RudaTensor<R>>,
    boundaries: Option<&RudaTensor<R>>,
    options: SsmOptions,
) -> Result<SsdOutput<R>, InferenceError> {
    use ruda_kernel::tensor::reshape::reshape;
    let x = into_contiguous(input.clone());
    check(&x, &x)?;
    if x.meta.num_dims() != 4
        || b.meta.num_dims() != 4
        || b.meta.shape()[0..2] != x.meta.shape()[0..2]
        || c.meta.shape() != b.meta.shape()
    {
        return Err(InferenceError("SSD expects X[B,L,H,D], B/C[B,L,G,N]"));
    }
    let batch = x.meta.shape()[0];
    let length = x.meta.shape()[1];
    let heads = x.meta.shape()[2];
    let dim = x.meta.shape()[3];
    let groups = b.meta.shape()[2];
    let n = b.meta.shape()[3];
    let tokens = batch
        .checked_mul(length)
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(InferenceError("SSD token count overflow"))?;
    if dt.meta.shape()[..] != [batch, length, heads] {
        return Err(InferenceError("SSD dt must be [B,L,H]"));
    }
    let xf = reshape(x.clone(), Shape::from([tokens, heads, dim]));
    let dt = reshape(into_contiguous(dt.clone()), Shape::from([tokens, heads]));
    let b = reshape(into_contiguous(b.clone()), Shape::from([tokens, groups, n]));
    let c = reshape(into_contiguous(c.clone()), Shape::from([tokens, groups, n]));
    let gate = z
        .map(|z| {
            if z.meta.shape() != x.meta.shape() {
                return Err(InferenceError("SSD gate shape mismatch"));
            }
            Ok(reshape(
                into_contiguous(z.clone()),
                Shape::from([tokens, heads, dim]),
            ))
        })
        .transpose()?;
    let initial_dtype = initial.map_or(DType::F32, |s| s.dtype);
    let initial = initial
        .map(|s| super::norm::cast(s, DType::F32))
        .transpose()?;
    let plan = SsmPlan::uniform(&x, batch, length)?;
    let out = plan.run(
        &xf,
        SsmInputs {
            dt: &dt,
            a,
            b: &b,
            c: &c,
            d,
            z: gate.as_ref(),
            dt_bias,
            checkpoint_boundaries: boundaries,
            state: initial.as_ref(),
            source_slots: None,
            accepted_tokens: None,
        },
        options,
    )?;
    Ok(SsdOutput {
        values: reshape(out.values, x.meta.shape().clone()),
        final_states: super::norm::cast(&out.final_states, initial_dtype)?,
        intermediate_states: super::norm::cast(&out.intermediate_states, initial_dtype)?,
        checkpoints: super::norm::cast(&out.checkpoints, initial_dtype)?,
    })
}
/// Select exclusive token-boundary checkpoints, including boundary zero.
/// A negative boundary writes an all-zero record. Per-sequence boundaries are
/// relative to its packed interval, not global token IDs.
pub fn checkpoint_states<R: Runtime>(
    plan: &SsmPlan<R>,
    output: &SsmOutput<R>,
    initial: Option<&RudaTensor<R>>,
    boundaries: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    let h = &output.intermediate_states;
    check(h, h)?;
    check(boundaries, h)?;
    if h.dtype != DType::F32
        || h.meta.num_dims() != 4
        || h.meta.shape()[0] != plan.tokens
        || boundaries.dtype != DType::I32
        || boundaries.meta.shape()[..] != [plan.sequences]
    {
        return Err(InferenceError("SSM checkpoint operands mismatch"));
    }
    let width: usize = h.meta.shape()[1..].iter().product();
    if let Some(s) = initial {
        check(s, h)?;
        if s.dtype != DType::F32
            || s.meta.shape()[..]
                != [
                    plan.sequences,
                    h.meta.shape()[1],
                    h.meta.shape()[2],
                    h.meta.shape()[3],
                ]
        {
            return Err(InferenceError(
                "SSM checkpoint initial state shape mismatch",
            ));
        }
    }
    let out = alloc(
        h,
        [
            plan.sequences,
            h.meta.shape()[1],
            h.meta.shape()[2],
            h.meta.shape()[3],
        ],
        DType::F32,
    );
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(h.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&h.client, n, dim);
        capture_states::launch::<R>(
            &h.client,
            count,
            dim,
            h.clone().into_array_arg(),
            initial.unwrap_or(h).clone().into_array_arg(),
            plan.cu.clone().into_array_arg(),
            boundaries.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            width as u32,
            initial.is_some(),
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn ids_to_boundaries<I: Int>(
    ids: &Array<I>,
    out: &mut Array<i32>,
    tokens: u32,
    sequences: u32,
    #[define(I)] _dtype: StorageType,
) {
    let seq = ABSOLUTE_POS as u32;
    if seq <= sequences {
        let mut lo: u32 = 0u32;
        let mut hi = tokens;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if i64::cast_from(ids[mid as usize]) < seq as i64 {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[seq as usize] = lo as i32;
    }
}
#[ruda(launch)]
pub fn capture_states(
    history: &Array<f32>,
    initial: &Array<f32>,
    cu: &Array<i32>,
    boundaries: &Array<i32>,
    out: &mut Array<f32>,
    n: u32,
    width: u32,
    #[comptime] has_initial: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let seq = i / width as usize;
        let d = i % width as usize;
        let boundary = boundaries[seq];
        let length = cu[seq + 1] - cu[seq];
        let mut value: f32 = 0.0f32;
        if boundary == 0 {
            if comptime!(has_initial) {
                value = initial[i];
            }
        }
        if boundary > 0 && boundary <= length {
            value = history[(cu[seq] + boundary - 1) as usize * width as usize + d];
        }
        out[i] = value;
    }
}

#[ruda(launch)]
pub fn selective_scan<F: Float>(
    x: &Array<F>,
    dt: &Array<f32>,
    a: &Array<f32>,
    b: &Array<f32>,
    c: &Array<f32>,
    d: &Array<f32>,
    z: &Array<f32>,
    bias: &Array<f32>,
    initial: &Array<f32>,
    cu: &Array<i32>,
    indices: &Array<i32>,
    accepted: &Array<i32>,
    checkpoint_boundaries: &Array<i32>,
    out: &mut Array<F>,
    final_state: &mut Array<f32>,
    history: &mut Array<f32>,
    checkpoints: &mut Array<f32>,
    tokens: u32,
    heads: u32,
    dim: u32,
    groups: u32,
    slots: u32,
    index_width: u32,
    pad: i32,
    dt_min: f32,
    dt_max: f32,
    #[comptime] n: usize,
    #[comptime] a_mode: u32,
    #[comptime] dt_full: bool,
    #[comptime] has_d: bool,
    #[comptime] d_full: bool,
    #[comptime] has_z: bool,
    #[comptime] has_bias: bool,
    #[comptime] bias_full: bool,
    #[comptime] has_initial: bool,
    #[comptime] indexed: bool,
    #[comptime] has_accepted: bool,
    #[comptime] softplus: bool,
    #[comptime] save_history: bool,
    #[comptime] save_checkpoint: bool,
    #[define(F)] _dtype: StorageType,
) {
    let seq = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let channel = RUDA_POS_Z as usize * RUDA_DIM as usize + UNIT_POS as usize;
    if channel < dim as usize {
        let mut source = seq as i32;
        let mut valid = true;
        if comptime!(indexed) {
            let mut column: usize = 0usize;
            if comptime!(has_accepted) {
                let count = accepted[seq];
                valid = count > 0;
                column = usize::min(i32::max(count - 1, 0) as usize, index_width as usize - 1);
            }
            source = indices[seq * index_width as usize + column];
            valid = valid && source != pad && source >= 0 && source < slots as i32;
        }
        let mut state = Array::<f32>::new(n);
        let final_base = ((seq * heads as usize + head) * dim as usize + channel) * n;
        for j in 0usize..n {
            state[j] = 0.0;
            if comptime!(has_initial) {
                if valid && source >= 0 && source < slots as i32 {
                    state[j] = initial[((source as usize * heads as usize + head) * dim as usize
                        + channel)
                        * n
                        + j];
                }
            }
        }
        if comptime!(save_checkpoint) {
            for j in 0usize..n {
                let mut value: f32 = 0.0f32;
                if checkpoint_boundaries[seq] == 0 {
                    value = state[j];
                }
                checkpoints[final_base + j] = value;
            }
        }
        let begin = i32::max(0, cu[seq]) as usize;
        let end = usize::min(i32::max(0, cu[seq + 1]) as usize, tokens as usize);
        let group = head / (heads / groups) as usize;
        for token in begin..end {
            let xi = (token * heads as usize + head) * dim as usize + channel;
            let mut value: f32 = 0.0f32;
            if valid {
                let mut step_index = token * heads as usize + head;
                if comptime!(dt_full) {
                    step_index = xi;
                }
                let mut step = dt[step_index];
                if comptime!(has_bias) {
                    let mut bi = head;
                    if comptime!(bias_full) {
                        bi = head * dim as usize + channel;
                    }
                    step += bias[bi];
                }
                if comptime!(softplus) {
                    if step < 20.0 {
                        step = (1.0 + step.exp()).ln();
                    }
                }
                step = f32::min(f32::max(step, dt_min), dt_max);
                let xv = f32::cast_from(x[xi]);
                for j in 0usize..n {
                    let mut ai = head;
                    if comptime!(a_mode == 1) {
                        ai = head * n + j;
                    }
                    if comptime!(a_mode == 2) {
                        ai = (head * dim as usize + channel) * n + j;
                    }
                    let projection = (token * groups as usize + group) * n + j;
                    state[j] = fma(state[j], (step * a[ai]).exp(), step * b[projection] * xv);
                    value += state[j] * c[projection];
                }
                if comptime!(has_d) {
                    let mut di = head;
                    if comptime!(d_full) {
                        di = head * dim as usize + channel;
                    }
                    value = fma(d[di], xv, value);
                }
                if comptime!(has_z) {
                    let gate = z[xi];
                    value *= gate / (1.0 + (-gate).exp());
                }
            }
            out[xi] = F::cast_from(value);
            if comptime!(save_history) {
                for j in 0usize..n {
                    history[xi * n + j] = state[j];
                }
            }
            if comptime!(save_checkpoint) {
                if token + 1 - begin == checkpoint_boundaries[seq] as usize {
                    for j in 0usize..n {
                        checkpoints[final_base + j] = state[j];
                    }
                }
            }
        }
        for j in 0usize..n {
            final_state[final_base + j] = state[j];
        }
    }
}

/// Commit a final scan state or every intermediate state to a caller-owned pool.
/// IDs [N] select final states; IDs [T] select token states. Negative IDs skip.
/// Live destination IDs must be unique; concurrent writers must be disjoint.
pub unsafe fn commit_states<R: Runtime>(
    pool: &mut RudaTensor<R>,
    states: &RudaTensor<R>,
    destination_slots: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    for t in [&*pool, states, destination_slots] {
        check(t, pool)?;
    }
    if pool.dtype != DType::F32
        || states.dtype != DType::F32
        || pool.meta.num_dims() != 4
        || states.meta.num_dims() != 4
        || pool.meta.shape()[1..] != states.meta.shape()[1..]
        || destination_slots.dtype != DType::I32
        || destination_slots.meta.shape()[..] != [states.meta.shape()[0]]
    {
        return Err(InferenceError("state commit dimensions mismatch"));
    }
    let width: usize = pool.meta.shape()[1..].iter().product();
    let n = states.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(pool.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&pool.client, n, dim);
        state_scatter::launch::<R>(
            &pool.client,
            count,
            dim,
            states.clone().into_array_arg(),
            destination_slots.clone().into_array_arg(),
            pool.clone().into_array_arg(),
            n as u32,
            width as u32,
            pool.meta.shape()[0] as u32,
        );
    }
    Ok(())
}
#[ruda(launch)]
pub fn state_scatter(
    source: &Array<f32>,
    ids: &Array<i32>,
    pool: &mut Array<f32>,
    n: u32,
    width: u32,
    slots: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / width as usize;
        let slot = ids[row];
        if slot >= 0 && slot < slots as i32 {
            pool[slot as usize * width as usize + i % width as usize] = source[i];
        }
    }
}

/// Functional causal convolution retaining unmodified raw-input history.
pub struct ConvOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub state: RudaTensor<R>,
}
pub struct ConvInputs<'a, R: Runtime> {
    pub weight: &'a RudaTensor<R>,
    pub bias: Option<&'a RudaTensor<R>>,
    pub state: &'a RudaTensor<R>,
    pub slots: &'a RudaTensor<R>,
    pub has_initial: Option<&'a RudaTensor<R>>,
    pub accepted_tokens: Option<&'a RudaTensor<R>>,
    pub speculative: bool,
    pub silu: bool,
}
impl<R: Runtime> SsmPlan<R> {
    pub fn causal_conv1d(
        &self,
        input: &RudaTensor<R>,
        inputs: ConvInputs<'_, R>,
    ) -> Result<ConvOutput<R>, InferenceError> {
        let state = super::super::paged_attention::copy_contiguous(inputs.state);
        let values = alloc(input, input.meta.shape().clone(), input.dtype);
        unsafe {
            self.causal_conv1d_into(input, inputs, &values, &state)?;
        }
        Ok(ConvOutput { values, state })
    }
    /// In-place state history plus preallocated output. No live state-slot aliasing.
    pub unsafe fn causal_conv1d_into(
        &self,
        input: &RudaTensor<R>,
        inputs: ConvInputs<'_, R>,
        output: &RudaTensor<R>,
        state_output: &RudaTensor<R>,
    ) -> Result<(), InferenceError> {
        if let Some(p) = &self.cu_cast {
            unsafe {
                p.run();
            }
        }
        for t in [
            input,
            inputs.state,
            inputs.slots,
            inputs.weight,
            output,
            state_output,
            &self.cu,
        ] {
            check(t, input)?;
        }
        float(input)?;
        if input.meta.num_dims() != 2
            || input.meta.shape()[0] != self.tokens
            || inputs.state.meta.num_dims() != 3
            || inputs.state.dtype != input.dtype
            || inputs.weight.dtype != DType::F32
            || output.dtype != input.dtype
            || output.meta.shape() != input.meta.shape()
            || state_output.dtype != input.dtype
            || state_output.meta.shape() != inputs.state.meta.shape()
        {
            return Err(InferenceError(
                "causal conv expects [T,C], FP32 [C,taps], state [slots,C,L]",
            ));
        }
        let channels = input.meta.shape()[1];
        let slots = inputs.state.meta.shape()[0];
        let history = inputs.state.meta.shape()[2];
        if channels == 0
            || inputs.state.meta.shape()[1] != channels
            || inputs.weight.meta.num_dims() != 2
            || inputs.weight.meta.shape()[0] != channels
            || inputs.weight.meta.shape()[1] < 2
            || history < inputs.weight.meta.shape()[1] - 1
            || inputs.slots.dtype != DType::I32
            || inputs.slots.meta.shape()[..] != [self.sequences]
        {
            return Err(InferenceError(
                "causal convolution history/weight/slots mismatch",
            ));
        }
        let taps = inputs.weight.meta.shape()[1];
        if taps > 128 {
            return Err(InferenceError("causal convolution supports up to 128 taps"));
        }
        if let Some(b) = inputs.bias {
            check(b, input)?;
            if b.dtype != DType::F32 || b.meta.shape()[..] != [channels] {
                return Err(InferenceError("causal convolution bias must be FP32 [C]"));
            }
        }
        for t in [inputs.has_initial, inputs.accepted_tokens]
            .into_iter()
            .flatten()
        {
            check(t, input)?;
            if t.dtype != DType::I32 || t.meta.shape()[..] != [self.sequences] {
                return Err(InferenceError(
                    "causal convolution controls must be I32 [N]",
                ));
            }
        }
        let hw = &input.client.properties().hardware;
        let units = 128u32.min(hw.max_ruda_dim.0).min(hw.max_units_per_ruda);
        if self.sequences > hw.max_ruda_count.0 as usize
            || channels.div_ceil(units as usize) > hw.max_ruda_count.1 as usize
        {
            return Err(InferenceError(
                "causal convolution grid exceeds device limit",
            ));
        }
        if self.sequences > 0 {
            conv_history::launch::<R>(
                &input.client,
                RudaCount::Static(
                    self.sequences as u32,
                    channels.div_ceil(units as usize) as u32,
                    1,
                ),
                RudaDim::new_1d(units),
                input.clone().into_array_arg(),
                inputs.weight.clone().into_array_arg(),
                inputs
                    .bias
                    .unwrap_or(inputs.weight)
                    .clone()
                    .into_array_arg(),
                inputs.state.clone().into_array_arg(),
                inputs.slots.clone().into_array_arg(),
                inputs
                    .has_initial
                    .unwrap_or(&self.cu)
                    .clone()
                    .into_array_arg(),
                inputs
                    .accepted_tokens
                    .unwrap_or(&self.cu)
                    .clone()
                    .into_array_arg(),
                self.cu.clone().into_array_arg(),
                output.clone().into_array_arg(),
                state_output.clone().into_array_arg(),
                self.tokens as u32,
                channels as u32,
                slots as u32,
                history as u32,
                taps,
                inputs.bias.is_some(),
                inputs.has_initial.is_some(),
                inputs.accepted_tokens.is_some(),
                inputs.speculative,
                inputs.silu,
                input.dtype.into(),
            );
        }
        Ok(())
    }
}
#[ruda(launch)]
pub fn conv_history<F: Float>(
    x: &Array<F>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    old: &Array<F>,
    slots: &Array<i32>,
    initial: &Array<i32>,
    accepted: &Array<i32>,
    cu: &Array<i32>,
    out: &mut Array<F>,
    next: &mut Array<F>,
    tokens: u32,
    channels: u32,
    pool: u32,
    length: u32,
    #[comptime] taps: usize,
    #[comptime] biased: bool,
    #[comptime] has_initial: bool,
    #[comptime] has_accepted: bool,
    #[comptime] speculative: bool,
    #[comptime] silu: bool,
    #[define(F)] _dtype: StorageType,
) {
    let seq = RUDA_POS_X as usize;
    let channel = RUDA_POS_Y as usize * RUDA_DIM as usize + UNIT_POS as usize;
    if channel < channels as usize {
        let slot = slots[seq];
        let mut valid = slot >= 0 && slot < pool as i32;
        let mut shift: usize = 0usize;
        if comptime!(has_accepted) {
            let a = accepted[seq];
            valid = valid && a > 0 && a <= length as i32 - (taps as i32 - 2);
            shift = i32::max(0, a - 1) as usize;
        }
        let mut h = Array::<f32>::new(taps);
        for j in 0usize..taps - 1 {
            h[j] = 0.0;
            if valid {
                let mut read = true;
                if comptime!(has_initial) {
                    read = initial[seq] != 0;
                }
                if read {
                    h[j] = f32::cast_from(
                        old[(slot as usize * channels as usize + channel) * length as usize
                            + shift
                            + j],
                    );
                }
            }
        }
        let begin = i32::max(0, cu[seq]) as usize;
        let end = usize::min(i32::max(0, cu[seq + 1]) as usize, tokens as usize);
        if comptime!(speculative) {
            if valid && begin < end {
                for j in 0usize..taps - 2 {
                    next[(slot as usize * channels as usize + channel) * length as usize + j] =
                        F::cast_from(h[j + 1]);
                }
            }
        }
        for token in begin..end {
            let mut result: f32 = 0.0f32;
            let value = f32::cast_from(x[token * channels as usize + channel]);
            if valid {
                if comptime!(biased) {
                    result = bias[channel];
                }
                for j in 0usize..taps - 1 {
                    result = fma(h[j], weight[channel * taps + j], result);
                }
                result = fma(value, weight[channel * taps + taps - 1], result);
                if comptime!(silu) {
                    result /= 1.0 + (-result).exp();
                }
                for j in 0usize..taps - 2 {
                    h[j] = h[j + 1];
                }
                h[taps - 2] = value;
                if comptime!(speculative) {
                    let position = token - begin + taps - 2;
                    if position < length as usize {
                        next[(slot as usize * channels as usize + channel) * length as usize
                            + position] = F::cast_from(value);
                    }
                }
            }
            out[token * channels as usize + channel] = F::cast_from(result);
        }
        if comptime!(!speculative) {
            if valid && begin < end {
                for j in 0usize..taps - 1 {
                    next[(slot as usize * channels as usize + channel) * length as usize + j] =
                        F::cast_from(h[j]);
                }
            }
        }
    }
}
