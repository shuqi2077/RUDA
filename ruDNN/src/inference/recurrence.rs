//! Packed GDN, GDP, channel-gated KDA/GDN-2 and indexed state/checkpoint I/O.
//! A unit owns one value channel's entire recurrent key vector. Reductions and
//! update ordering are local, deterministic, and need no floating atomics.
use super::{InferenceError, alloc, check, float};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateLayout {
    KeyValue,
    ValueKey,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateTransform {
    Linear,
    Log,
    Softplus,
    BoundedSigmoid,
}
#[derive(Clone, Copy, Debug)]
pub struct DeltaOptions {
    pub null_state_is_zero: bool,
    pub scale: f32,
    pub householder: usize,
    pub channel_decay: bool,
    pub channel_erase: bool,
    pub beta_is_logit: bool,
    pub normalize_qk: bool,
    pub normalization_epsilon: f32,
    pub gate_transform: GateTransform,
    pub lower_bound: f32,
    pub state_layout: StateLayout,
}
impl DeltaOptions {
    pub fn gdn(scale: f32) -> Self {
        Self {
            null_state_is_zero: false,
            scale,
            householder: 1,
            channel_decay: false,
            channel_erase: false,
            beta_is_logit: false,
            normalize_qk: false,
            normalization_epsilon: 1e-6,
            gate_transform: GateTransform::Log,
            lower_bound: -5.0,
            state_layout: StateLayout::KeyValue,
        }
    }
    pub fn gdp(scale: f32, householder: usize) -> Self {
        Self {
            householder,
            ..Self::gdn(scale)
        }
    }
    pub fn kda(scale: f32) -> Self {
        Self {
            channel_decay: true,
            ..Self::gdn(scale)
        }
    }
    pub fn gdn2(scale: f32) -> Self {
        Self {
            channel_decay: true,
            channel_erase: true,
            ..Self::gdn(scale)
        }
    }
}
pub struct DeltaInputs<'a, R: Runtime> {
    pub gate: Option<&'a RudaTensor<R>>,
    pub beta: Option<&'a RudaTensor<R>>,
    pub write_gate: Option<&'a RudaTensor<R>>,
    pub a_log: Option<&'a RudaTensor<R>>,
    pub dt_bias: Option<&'a RudaTensor<R>>,
    pub initial_state: Option<&'a RudaTensor<R>>,
    pub state_indices: Option<&'a RudaTensor<R>>,
    pub sequence_order: Option<&'a RudaTensor<R>>,
    pub prefix_lengths: Option<&'a RudaTensor<R>>,
    pub checkpoint_prefixes: Option<&'a RudaTensor<R>>,
    pub accepted_tokens: Option<&'a RudaTensor<R>>,
}
impl<R: Runtime> Default for DeltaInputs<'_, R> {
    fn default() -> Self {
        Self {
            gate: None,
            beta: None,
            write_gate: None,
            a_log: None,
            dt_bias: None,
            initial_state: None,
            state_indices: None,
            sequence_order: None,
            prefix_lengths: None,
            checkpoint_prefixes: None,
            accepted_tokens: None,
        }
    }
}
pub struct DeltaOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub final_state: RudaTensor<R>,
    pub checkpoints: RudaTensor<R>,
    pub checkpoint_counts: RudaTensor<R>,
    pub corrections: RudaTensor<R>,
    pub checkpoint_indptr: Vec<u32>,
}
pub struct DeltaPlan<R: Runtime> {
    prefix_cast: Option<super::metadata::PreparedIndices<R>>,
    prefix: RudaTensor<R>,
    checkpoint_offsets: RudaTensor<R>,
    host_checkpoint_offsets: Vec<u32>,
    sequences: usize,
    tokens: usize,
    checkpoint_stride: usize,
}
impl<R: Runtime> DeltaPlan<R> {
    pub(crate) unsafe fn refresh_schedule(&self) {
        if let Some(p) = &self.prefix_cast {
            unsafe {
                p.run();
            }
        }
    }
    pub fn device_sequence_indptr(&self) -> &RudaTensor<R> {
        &self.prefix
    }
    pub fn device_checkpoint_indptr(&self) -> &RudaTensor<R> {
        &self.checkpoint_offsets
    }
    pub fn checkpoint_stride(&self) -> usize {
        self.checkpoint_stride
    }
    pub fn allocate_output(
        &self,
        like: &RudaTensor<R>,
        heads: usize,
        key_width: usize,
        value_width: usize,
        householder: usize,
        layout: StateLayout,
    ) -> Result<DeltaOutput<R>, InferenceError> {
        check(like, &self.prefix)?;
        float(like)?;
        if heads == 0 || key_width == 0 || value_width == 0 || householder == 0 {
            return Err(InferenceError("delta workspace axes must be positive"));
        }
        let state = if layout == StateLayout::KeyValue {
            vec![self.sequences, heads, key_width, value_width]
        } else {
            vec![self.sequences, heads, value_width, key_width]
        };
        let cp = vec![
            *self.host_checkpoint_offsets.last().unwrap() as usize,
            heads,
            key_width,
            value_width,
        ];
        let corrections = vec![
            self.tokens
                .checked_mul(householder)
                .ok_or(InferenceError("delta workspace token overflow"))?,
            heads,
            value_width,
        ];
        for s in [
            vec![self.tokens, heads, value_width],
            state.clone(),
            cp.clone(),
            corrections.clone(),
        ] {
            if s.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
            {
                return Err(InferenceError("delta workspace exceeds U32"));
            }
        }
        Ok(DeltaOutput {
            values: alloc(like, [self.tokens, heads, value_width], like.dtype),
            final_state: alloc(like, Shape::from(state), DType::F32),
            checkpoints: alloc(like, Shape::from(cp), DType::F32),
            checkpoint_counts: alloc(like, [self.sequences], DType::U32),
            corrections: alloc(like, Shape::from(corrections), DType::F32),
            checkpoint_indptr: self.host_checkpoint_offsets.clone(),
        })
    }
    /// Device ragged plan. Fixed checkpoint regions reserve the maximum length;
    /// checkpoint_counts reports each live prefix, and unused records are zero.
    /// GPU indptr values must be ordered, end at tokens and obey max_seq_len.
    pub fn from_device(
        cu_seqlens: &RudaTensor<R>,
        tokens: usize,
        max_seq_len: usize,
        checkpoint_stride: usize,
    ) -> Result<Self, InferenceError> {
        check(cu_seqlens, cu_seqlens)?;
        if !matches!(
            cu_seqlens.dtype,
            DType::I32 | DType::U32 | DType::I64 | DType::U64
        ) || cu_seqlens.meta.num_dims() != 1
            || cu_seqlens.meta.num_elements() < 2
            || tokens > i32::MAX as usize
            || max_seq_len > i32::MAX as usize
            || checkpoint_stride > u32::MAX as usize
        {
            return Err(InferenceError("invalid device delta schedule capacities"));
        }
        let sequences = cu_seqlens.meta.num_elements() - 1;
        let count = if checkpoint_stride == 0 {
            0
        } else {
            max_seq_len.div_ceil(checkpoint_stride) + 2
        };
        let mut offsets = Vec::with_capacity(sequences + 1);
        for seq in 0..=sequences {
            offsets.push(
                seq.checked_mul(count)
                    .filter(|n| *n <= u32::MAX as usize)
                    .ok_or(InferenceError("device checkpoint capacity overflow"))?
                    as u32,
            );
        }
        let checkpoint_offsets = RudaTensor::new_contiguous(
            cu_seqlens.client.clone(),
            cu_seqlens.device.clone(),
            Shape::from([offsets.len()]),
            cu_seqlens
                .client
                .create_from_slice(bytemuck::cast_slice(&offsets)),
            DType::U32,
        );
        let prefix_cast = if matches!(cu_seqlens.dtype, DType::I64 | DType::U64) {
            Some(super::metadata::PreparedIndices::prepare(
                cu_seqlens,
                DType::U32,
            )?)
        } else {
            None
        };
        let prefix = prefix_cast
            .as_ref()
            .map_or_else(|| cu_seqlens.clone(), |p| p.output.clone());
        Ok(Self {
            prefix_cast,
            prefix,
            checkpoint_offsets,
            host_checkpoint_offsets: offsets,
            sequences,
            tokens,
            checkpoint_stride,
        })
    }
    pub fn new(
        like: &RudaTensor<R>,
        cu_seqlens: &[u32],
        checkpoint_stride: usize,
    ) -> Result<Self, InferenceError> {
        if cu_seqlens.len() < 2
            || cu_seqlens[0] != 0
            || cu_seqlens.windows(2).any(|v| v[0] > v[1])
            || cu_seqlens.last().copied().unwrap() > i32::MAX as u32
            || checkpoint_stride > u32::MAX as usize
        {
            return Err(InferenceError(
                "invalid recurrent sequence/checkpoint schedule",
            ));
        }
        let sequences = cu_seqlens.len() - 1;
        let tokens = *cu_seqlens.last().unwrap() as usize;
        let mut offsets = vec![0u32];
        for v in cu_seqlens.windows(2) {
            let length = (v[1] - v[0]) as usize;
            let count = if checkpoint_stride == 0 {
                0
            } else {
                length.div_ceil(checkpoint_stride) + 2
            };
            offsets.push(
                offsets
                    .last()
                    .unwrap()
                    .checked_add(count as u32)
                    .ok_or(InferenceError("checkpoint offsets overflow"))?,
            );
        }
        let prefix = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([cu_seqlens.len()]),
            like.client
                .create_from_slice(bytemuck::cast_slice(cu_seqlens)),
            DType::U32,
        );
        let checkpoint_offsets = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([offsets.len()]),
            like.client
                .create_from_slice(bytemuck::cast_slice(&offsets)),
            DType::U32,
        );
        Ok(Self {
            prefix_cast: None,
            prefix,
            checkpoint_offsets,
            host_checkpoint_offsets: offsets,
            sequences,
            tokens,
            checkpoint_stride,
        })
    }
    pub fn run(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        inputs: DeltaInputs<'_, R>,
        options: DeltaOptions,
    ) -> Result<DeltaOutput<R>, InferenceError> {
        self.execute(q, k, v, inputs, options, None)
    }
    /// Inputs/gates are already packed; all optional numeric parameters are FP32.
    /// Output and checkpoint buffers must be exclusively owned during replay.
    pub unsafe fn run_into(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        inputs: DeltaInputs<'_, R>,
        options: DeltaOptions,
        output: &DeltaOutput<R>,
    ) -> Result<(), InferenceError> {
        for t in [q, k, v] {
            check(t, q)?;
        }
        self.execute(q, k, v, inputs, options, Some(output))
            .map(|_| ())
    }
    fn execute(
        &self,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        inputs: DeltaInputs<'_, R>,
        options: DeltaOptions,
        destination: Option<&DeltaOutput<R>>,
    ) -> Result<DeltaOutput<R>, InferenceError> {
        unsafe {
            self.refresh_schedule();
        }
        let capture = destination.is_some();
        let q = into_contiguous(q.clone());
        let k = into_contiguous(k.clone());
        let v = into_contiguous(v.clone());
        for t in [&q, &k, &v, &self.prefix] {
            check(t, &q)?;
        }
        float(&q)?;
        if q.meta.num_dims() != 3
            || k.meta.num_dims() != 3
            || v.meta.num_dims() != 3
            || q.dtype != k.dtype
            || q.dtype != v.dtype
            || q.meta.shape()[0] != self.tokens
            || options.householder == 0
            || !options.scale.is_finite()
            || !options.normalization_epsilon.is_finite()
            || options.normalization_epsilon <= 0.0
        {
            return Err(InferenceError("invalid delta inputs/options"));
        }
        let qh = q.meta.shape()[1];
        let kh = k.meta.shape()[1];
        let vh = v.meta.shape()[1];
        let heads = qh.max(vh);
        let dk = q.meta.shape()[2];
        let dv = v.meta.shape()[2];
        if qh == 0
            || kh == 0
            || vh == 0
            || heads % qh != 0
            || heads % kh != 0
            || heads % vh != 0
            || dk == 0
            || dk > 1024
            || dv == 0
            || k.meta.shape()[2] != dk
            || k.meta.shape()[0] != self.tokens * options.householder
            || v.meta.shape()[0] != self.tokens * options.householder
        {
            return Err(InferenceError("delta head/householder dimensions mismatch"));
        }
        let f32_param = |t: Option<&RudaTensor<R>>| {
            t.map(|v| {
                if capture {
                    check(v, &q)?;
                    if v.dtype != DType::F32 {
                        return Err(InferenceError("prepared delta parameters must be F32"));
                    }
                    Ok(v.clone())
                } else {
                    super::norm::cast(v, DType::F32)
                }
            })
            .transpose()
        };
        let gate = f32_param(inputs.gate)?;
        let beta = f32_param(inputs.beta)?;
        let write = f32_param(inputs.write_gate)?;
        let a = f32_param(inputs.a_log)?;
        let bias = f32_param(inputs.dt_bias)?;
        let gate_width = if options.channel_decay { dk } else { 1 };
        let beta_width = if options.channel_erase { dk } else { 1 };
        for (t, shape) in [
            (gate.as_ref(), vec![self.tokens, heads, gate_width]),
            (
                beta.as_ref(),
                vec![self.tokens * options.householder, heads, beta_width],
            ),
            (
                write.as_ref(),
                vec![self.tokens * options.householder, heads, dv],
            ),
        ] {
            if let Some(t) = t {
                check(t, &q)?;
                let n: usize = shape.iter().product();
                if t.meta.num_elements() != n {
                    return Err(InferenceError("delta gate shape mismatch"));
                }
            }
        }
        for t in a.iter().chain(bias.iter()) {
            check(t, &q)?;
            if t.meta.num_elements() != heads && t.meta.num_elements() != heads * gate_width {
                return Err(InferenceError("delta A_log/dt_bias shape mismatch"));
            }
        }
        for t in [
            inputs.state_indices,
            inputs.sequence_order,
            inputs.prefix_lengths,
            inputs.checkpoint_prefixes,
            inputs.accepted_tokens,
        ]
        .into_iter()
        .flatten()
        {
            check(t, &q)?;
            if t.dtype != DType::I32 || t.meta.shape()[..] != [self.sequences] {
                return Err(InferenceError(
                    "delta sequence controls must be I32 [sequences]",
                ));
            }
        }
        let mut pool: usize = 0usize;
        if let Some(s) = inputs.initial_state {
            check(s, &q)?;
            if s.dtype != DType::F32 || s.meta.num_dims() != 4 {
                return Err(InferenceError("delta initial state must be rank-four F32"));
            }
            pool = s.meta.shape()[0];
            let expected = if options.state_layout == StateLayout::KeyValue {
                vec![pool, heads, dk, dv]
            } else {
                vec![pool, heads, dv, dk]
            };
            if s.meta.shape()[..] != expected
                || (inputs.state_indices.is_none() && pool != self.sequences)
            {
                return Err(InferenceError("delta initial-state dimensions mismatch"));
            }
        }
        for dims in [
            vec![self.tokens, heads, dv],
            vec![self.sequences, heads, dk, dv],
            vec![
                *self.host_checkpoint_offsets.last().unwrap() as usize,
                heads,
                dk,
                dv,
            ],
        ] {
            if dims
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n <= u32::MAX as usize)
                .is_none()
            {
                return Err(InferenceError("delta output/state exceeds U32"));
            }
        }
        let state_shape = if options.state_layout == StateLayout::KeyValue {
            vec![self.sequences, heads, dk, dv]
        } else {
            vec![self.sequences, heads, dv, dk]
        };
        let cp_shape = vec![
            *self.host_checkpoint_offsets.last().unwrap() as usize,
            heads,
            dk,
            dv,
        ];
        let (values, final_state, checkpoints, checkpoint_counts, corrections) =
            if let Some(out) = destination {
                for (t, dtype, shape) in [
                    (&out.values, q.dtype, vec![self.tokens, heads, dv]),
                    (&out.final_state, DType::F32, state_shape.clone()),
                    (&out.checkpoints, DType::F32, cp_shape.clone()),
                    (&out.checkpoint_counts, DType::U32, vec![self.sequences]),
                    (
                        &out.corrections,
                        DType::F32,
                        vec![self.tokens * options.householder, heads, dv],
                    ),
                ] {
                    check(t, &q)?;
                    if t.dtype != dtype || t.meta.shape()[..] != shape {
                        return Err(InferenceError("prepared delta output shape mismatch"));
                    }
                }
                if out.checkpoint_indptr != self.host_checkpoint_offsets {
                    return Err(InferenceError(
                        "prepared delta checkpoint capacity mismatch",
                    ));
                }
                (
                    out.values.clone(),
                    out.final_state.clone(),
                    out.checkpoints.clone(),
                    out.checkpoint_counts.clone(),
                    out.corrections.clone(),
                )
            } else {
                (
                    alloc(&q, [self.tokens, heads, dv], q.dtype),
                    alloc(&q, Shape::from(state_shape), DType::F32),
                    alloc(&q, Shape::from(cp_shape), DType::F32),
                    alloc(&q, [self.sequences], DType::U32),
                    alloc(
                        &q,
                        [self.tokens * options.householder, heads, dv],
                        DType::F32,
                    ),
                )
            };
        let dummy = if capture {
            final_state.clone()
        } else {
            super::norm::cast(&q, DType::F32)?
        };
        let hw = &q.client.properties().hardware;
        let units = 128u32.min(hw.max_ruda_dim.0).min(hw.max_units_per_ruda);
        if self.sequences > hw.max_ruda_count.0 as usize
            || heads > hw.max_ruda_count.1 as usize
            || dv.div_ceil(units as usize) > hw.max_ruda_count.2 as usize
        {
            return Err(InferenceError("delta grid exceeds device limit"));
        }
        if self.sequences != 0 {
            delta::launch::<R>(
                &q.client,
                RudaCount::Static(
                    self.sequences as u32,
                    heads as u32,
                    dv.div_ceil(units as usize) as u32,
                ),
                RudaDim::new_1d(units),
                q.clone().into_array_arg(),
                k.into_array_arg(),
                v.into_array_arg(),
                gate.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
                beta.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
                write.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
                a.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
                bias.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
                inputs
                    .initial_state
                    .unwrap_or(&dummy)
                    .clone()
                    .into_array_arg(),
                self.prefix.clone().into_array_arg(),
                self.checkpoint_offsets.clone().into_array_arg(),
                inputs
                    .state_indices
                    .unwrap_or(&self.prefix)
                    .clone()
                    .into_array_arg(),
                inputs
                    .sequence_order
                    .unwrap_or(&self.prefix)
                    .clone()
                    .into_array_arg(),
                inputs
                    .prefix_lengths
                    .unwrap_or(&self.prefix)
                    .clone()
                    .into_array_arg(),
                inputs
                    .checkpoint_prefixes
                    .unwrap_or(&self.prefix)
                    .clone()
                    .into_array_arg(),
                inputs
                    .accepted_tokens
                    .unwrap_or(&self.prefix)
                    .clone()
                    .into_array_arg(),
                values.clone().into_array_arg(),
                final_state.clone().into_array_arg(),
                checkpoints.clone().into_array_arg(),
                checkpoint_counts.clone().into_array_arg(),
                corrections.clone().into_array_arg(),
                self.sequences as u32,
                pool as u32,
                qh as u32,
                kh as u32,
                vh as u32,
                heads as u32,
                dv as u32,
                options.householder as u32,
                self.checkpoint_stride as u32,
                options.scale,
                options.normalization_epsilon,
                options.lower_bound,
                dk,
                options.channel_decay,
                options.channel_erase,
                options.beta_is_logit,
                options.normalize_qk,
                match options.gate_transform {
                    GateTransform::Linear => 0,
                    GateTransform::Log => 1,
                    GateTransform::Softplus => 2,
                    GateTransform::BoundedSigmoid => 3,
                },
                options.state_layout == StateLayout::ValueKey,
                gate.is_some(),
                beta.is_some(),
                write.is_some(),
                a.is_some(),
                a.as_ref().is_some_and(|t| t.meta.num_elements() == heads),
                bias.is_some(),
                bias.as_ref()
                    .is_some_and(|t| t.meta.num_elements() == heads),
                inputs.initial_state.is_some(),
                inputs.state_indices.is_some(),
                inputs.sequence_order.is_some(),
                inputs.prefix_lengths.is_some(),
                inputs.checkpoint_prefixes.is_some(),
                inputs.accepted_tokens.is_some(),
                options.null_state_is_zero,
                q.dtype.into(),
            );
        }
        Ok(DeltaOutput {
            values,
            final_state,
            checkpoints,
            checkpoint_counts,
            corrections,
            checkpoint_indptr: self.host_checkpoint_offsets.clone(),
        })
    }
}
/// Resolve prior accepted speculative states into one cache slot per sequence.
/// Counts include the bonus token; zero/negative counts or -1 slots skip a row.
pub fn select_state_slots<R: Runtime>(
    history_slots: &RudaTensor<R>,
    accepted: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    for t in [history_slots, accepted] {
        check(t, history_slots)?;
    }
    if history_slots.dtype != DType::I32
        || history_slots.meta.num_dims() != 2
        || history_slots.meta.shape()[1] == 0
        || accepted.dtype != DType::I32
        || accepted.meta.shape()[..] != [history_slots.meta.shape()[0]]
    {
        return Err(InferenceError(
            "state slot selection needs I32 [N,steps] and accepted[N]",
        ));
    }
    let n = accepted.meta.num_elements();
    let out = alloc(history_slots, [n], DType::I32);
    if n > 0 {
        let dim = RudaDim::new(history_slots.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&history_slots.client, n, dim);
        state_slots::launch::<R>(
            &history_slots.client,
            count,
            dim,
            history_slots.clone().into_array_arg(),
            accepted.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            history_slots.meta.shape()[1] as u32,
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn state_slots(
    history: &Array<i32>,
    accepted: &Array<i32>,
    out: &mut Array<i32>,
    n: u32,
    steps: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let count = accepted[i];
        let mut slot: i32 = -1i32;
        if count > 0 && count <= steps as i32 {
            slot = history[i * steps as usize + count as usize - 1];
        }
        out[i] = slot;
    }
}

#[ruda]
fn state_index(
    seq: usize,
    head: usize,
    key: usize,
    value: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    #[comptime] vk: bool,
) -> usize {
    let mut index = ((seq * heads + head) * dk + key) * dv + value;
    if comptime!(vk) {
        index = ((seq * heads + head) * dv + value) * dk + key;
    }
    index
}
#[ruda]
fn decay_value(x: f32, a: f32, bias: f32, lower: f32, #[comptime] kind: u32) -> f32 {
    let mut value = x;
    if comptime!(kind == 1) {
        value = x.exp();
    }
    if comptime!(kind == 2) {
        let z = x + bias;
        let mut softplus = z;
        if z < 20.0 {
            softplus = (1.0 + z.exp()).ln();
        }
        value = (-a.exp() * softplus).exp();
    }
    if comptime!(kind == 3) {
        value = (lower / (1.0 + (-a.exp() * (x + bias)).exp())).exp();
    }
    value
}
#[ruda(launch)]
pub fn delta<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    gate: &Array<f32>,
    beta: &Array<f32>,
    write: &Array<f32>,
    a: &Array<f32>,
    bias: &Array<f32>,
    initial: &Array<f32>,
    prefix: &Array<u32>,
    cp_offsets: &Array<u32>,
    state_ids: &Array<i32>,
    order: &Array<i32>,
    prefix_lengths: &Array<i32>,
    checkpoint_prefixes: &Array<i32>,
    accepted: &Array<i32>,
    out: &mut Array<F>,
    final_state: &mut Array<f32>,
    checkpoints: &mut Array<f32>,
    cp_counts: &mut Array<u32>,
    corrections: &mut Array<f32>,
    sequences: u32,
    pool: u32,
    qh: u32,
    kh: u32,
    vh: u32,
    heads: u32,
    dv: u32,
    householder: u32,
    checkpoint_stride: u32,
    query_scale: f32,
    epsilon: f32,
    lower: f32,
    #[comptime] dk: usize,
    #[comptime] channel_decay: bool,
    #[comptime] channel_erase: bool,
    #[comptime] beta_logit: bool,
    #[comptime] normalize: bool,
    #[comptime] transform: u32,
    #[comptime] vk: bool,
    #[comptime] has_gate: bool,
    #[comptime] has_beta: bool,
    #[comptime] has_write: bool,
    #[comptime] has_a: bool,
    #[comptime] scalar_a: bool,
    #[comptime] has_bias: bool,
    #[comptime] scalar_bias: bool,
    #[comptime] has_initial: bool,
    #[comptime] indexed: bool,
    #[comptime] reordered: bool,
    #[comptime] prefixed: bool,
    #[comptime] aligned_checkpoints: bool,
    #[comptime] speculative: bool,
    #[comptime] null_zero: bool,
    #[define(F)] _dtype: StorageType,
) {
    let mut seq = RUDA_POS_X as usize;
    if comptime!(reordered) {
        let requested = order[seq];
        if requested >= 0 && requested < sequences as i32 {
            seq = requested as usize;
        }
    }
    let head = RUDA_POS_Y as usize;
    let value = RUDA_POS_Z as usize * RUDA_DIM as usize + UNIT_POS as usize;
    if value < dv as usize {
        let mut state = Array::<f32>::new(dk);
        let mut source = seq as i32;
        if comptime!(indexed) {
            source = state_ids[seq];
        }
        for j in 0usize..dk {
            state[j] = 0.0;
            if comptime!(has_initial) {
                if source >= 0 && source < pool as i32 {
                    state[j] = initial[state_index(
                        source as usize,
                        head,
                        j,
                        value,
                        heads as usize,
                        dk,
                        dv as usize,
                        vk,
                    )];
                }
            }
        }
        let checkpoint_begin = cp_offsets[seq] as usize;
        let checkpoint_end = cp_offsets[seq + 1] as usize;
        let begin = prefix[seq] as usize;
        let end = prefix[seq + 1] as usize;
        let mut slot_valid = true;
        if comptime!(indexed) {
            slot_valid = source >= 0 && source < pool as i32;
            if comptime!(null_zero) {
                if source == -1 {
                    slot_valid = true;
                }
            }
        }
        let mut skip: usize = 0usize;
        if comptime!(prefixed) {
            skip = usize::min(
                (end - begin),
                usize::cast_from(i32::max(0, prefix_lengths[seq])),
            );
        }
        let live_begin = begin + skip;
        let mut cp_prefix: usize = 0usize;
        if comptime!(aligned_checkpoints) {
            if checkpoint_stride > 0 {
                cp_prefix =
                    i32::max(0, checkpoint_prefixes[seq]) as usize % checkpoint_stride as usize;
            }
        }
        if head == 0 && value == 0 {
            let mut count: u32 = 0u32;
            if checkpoint_stride > 0 {
                count = 1u32;
                if end > live_begin {
                    count = ((cp_prefix + end - live_begin + checkpoint_stride as usize - 1)
                        / checkpoint_stride as usize
                        + 1) as u32;
                }
            }
            cp_counts[seq] = count;
        }
        if checkpoint_stride > 0 {
            for cp in checkpoint_begin..checkpoint_end {
                for j in 0usize..dk {
                    checkpoints
                        [state_index(cp, head, j, value, heads as usize, dk, dv as usize, false)] =
                        0.0;
                }
            }
        }
        let mut save_after = end - live_begin;
        if comptime!(speculative) {
            save_after = usize::min(save_after, usize::cast_from(i32::max(0, accepted[seq])));
        }
        for j in 0usize..dk {
            final_state[state_index(seq, head, j, value, heads as usize, dk, dv as usize, vk)] =
                state[j];
        }
        if checkpoint_stride > 0 {
            for j in 0usize..dk {
                checkpoints[state_index(
                    checkpoint_begin,
                    head,
                    j,
                    value,
                    heads as usize,
                    dk,
                    dv as usize,
                    false,
                )] = state[j];
            }
        }
        let q_head = head / (heads / qh) as usize;
        let k_head = head / (heads / kh) as usize;
        let v_head = head / (heads / vh) as usize;
        for token in begin..end {
            if token < live_begin || !slot_valid {
                out[(token * heads as usize + head) * dv as usize + value] = F::cast_from(0.0f32);
                for sub in 0..householder as usize {
                    corrections[((token * householder as usize + sub) * heads as usize + head)
                        * dv as usize
                        + value] = 0.0;
                }
            } else {
                for j in 0usize..dk {
                    let mut alpha: f32 = 1.0f32;
                    if comptime!(has_gate) {
                        let mut gi = token * heads as usize + head;
                        if comptime!(channel_decay) {
                            gi = gi * dk + j;
                        }
                        let mut ai = head;
                        let mut bi = head;
                        if comptime!(!scalar_a) {
                            ai = head * dk + j;
                        }
                        if comptime!(!scalar_bias) {
                            bi = head * dk + j;
                        }
                        let mut av: f32 = 0.0f32;
                        let mut bv: f32 = 0.0f32;
                        if comptime!(has_a) {
                            av = a[ai];
                        }
                        if comptime!(has_bias) {
                            bv = bias[bi];
                        }
                        alpha = decay_value(gate[gi], av, bv, lower, transform);
                    }
                    state[j] *= alpha;
                }
                for sub in 0..householder as usize {
                    let kt = token * householder as usize + sub;
                    let mut kn: f32 = 1.0f32;
                    if comptime!(normalize) {
                        let mut square: f32 = 0.0f32;
                        for j in 0usize..dk {
                            let x = f32::cast_from(k[(kt * kh as usize + k_head) * dk + j]);
                            square += x * x;
                        }
                        kn = 1.0 / (square + epsilon).sqrt();
                    }
                    let mut prediction: f32 = 0.0f32;
                    for j in 0usize..dk {
                        let key = f32::cast_from(k[(kt * kh as usize + k_head) * dk + j]) * kn;
                        let mut erase: f32 = 1.0f32;
                        if comptime!(channel_erase && has_beta) {
                            erase = beta[(kt * heads as usize + head) * dk + j];
                            if comptime!(beta_logit) {
                                erase = 1.0 / (1.0 + (-erase).exp());
                            }
                        }
                        prediction += state[j] * key * erase;
                    }
                    let incoming =
                        f32::cast_from(v[(kt * vh as usize + v_head) * dv as usize + value]);
                    let mut update = incoming - prediction;
                    if comptime!(channel_erase) {
                        let mut wg: f32 = 1.0f32;
                        if comptime!(has_write) {
                            wg = write[(kt * heads as usize + head) * dv as usize + value];
                        }
                        update = wg * incoming - prediction;
                    } else {
                        if comptime!(has_beta) {
                            let mut b = beta[kt * heads as usize + head];
                            if comptime!(beta_logit) {
                                b = 1.0 / (1.0 + (-b).exp());
                            }
                            update *= b;
                        }
                    }
                    corrections[(kt * heads as usize + head) * dv as usize + value] = update;
                    for j in 0usize..dk {
                        let key = f32::cast_from(k[(kt * kh as usize + k_head) * dk + j]) * kn;
                        state[j] = fma(key, update, state[j]);
                    }
                }
                let mut result: f32 = 0.0f32;
                let mut qn: f32 = 1.0f32;
                if comptime!(normalize) {
                    let mut square: f32 = 0.0f32;
                    for j in 0usize..dk {
                        let x = f32::cast_from(q[(token * qh as usize + q_head) * dk + j]);
                        square += x * x;
                    }
                    qn = 1.0 / (square + epsilon).sqrt();
                }
                for j in 0usize..dk {
                    result +=
                        state[j] * f32::cast_from(q[(token * qh as usize + q_head) * dk + j]) * qn;
                }
                out[(token * heads as usize + head) * dv as usize + value] =
                    F::cast_from(result * query_scale);
                let processed = token - live_begin + 1;
                if processed == save_after {
                    for j in 0usize..dk {
                        final_state[state_index(
                            seq,
                            head,
                            j,
                            value,
                            heads as usize,
                            dk,
                            dv as usize,
                            vk,
                        )] = state[j];
                    }
                }
                if checkpoint_stride > 0
                    && (((cp_prefix + processed) % checkpoint_stride as usize) == 0
                        || token + 1 == end)
                {
                    let cp = checkpoint_begin
                        + (cp_prefix + processed + checkpoint_stride as usize - 1)
                            / checkpoint_stride as usize;
                    for j in 0usize..dk {
                        checkpoints[state_index(
                            cp,
                            head,
                            j,
                            value,
                            heads as usize,
                            dk,
                            dv as usize,
                            false,
                        )] = state[j];
                    }
                }
            }
        }
    }
}

/// Packed native delta-rule workspace shared by GDN/GDP/KDA/GDN2. All device
/// control tables and floating parameters are refreshed from their bindings.
pub struct PreparedDelta<'a, R: Runtime> {
    plan: &'a DeltaPlan<R>,
    q: super::PreparedPack<R>,
    k: super::PreparedPack<R>,
    v: super::PreparedPack<R>,
    parameters: Vec<Option<(super::PreparedPack<R>, RudaTensor<R>)>>,
    controls: Vec<Option<super::metadata::PreparedIndices<R>>>,
    state: Option<super::PreparedPack<R>>,
    state_f32: Option<RudaTensor<R>>,
    pub output: DeltaOutput<R>,
    options: DeltaOptions,
}
impl<'a, R: Runtime> PreparedDelta<'a, R> {
    pub fn prepare(
        plan: &'a DeltaPlan<R>,
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        inputs: DeltaInputs<'_, R>,
        options: DeltaOptions,
    ) -> Result<Self, InferenceError> {
        let q = super::PreparedPack::prepare(q)?;
        let k = super::PreparedPack::prepare(k)?;
        let v = super::PreparedPack::prepare(v)?;
        let x = &q.output;
        for t in [x, &k.output, &v.output] {
            check(t, x)?;
            float(t)?;
        }
        if x.meta.num_dims() != 3
            || k.output.meta.num_dims() != 3
            || v.output.meta.num_dims() != 3
            || x.dtype != k.output.dtype
            || x.dtype != v.output.dtype
            || x.meta.shape()[0] != plan.tokens
            || options.householder == 0
            || k.output.meta.shape()[0] != plan.tokens * options.householder
            || v.output.meta.shape()[0] != plan.tokens * options.householder
            || x.meta.shape()[2] != k.output.meta.shape()[2]
        {
            return Err(InferenceError("prepared delta Q/K/V mismatch"));
        }
        let heads = x.meta.shape()[1].max(v.output.meta.shape()[1]);
        let dk = x.meta.shape()[2];
        let dv = v.output.meta.shape()[2];
        if heads == 0
            || x.meta.shape()[1] == 0
            || k.output.meta.shape()[1] == 0
            || v.output.meta.shape()[1] == 0
            || heads % x.meta.shape()[1] != 0
            || heads % k.output.meta.shape()[1] != 0
            || heads % v.output.meta.shape()[1] != 0
            || dk == 0
            || dk > 1024
            || dv == 0
        {
            return Err(InferenceError("prepared delta head axes invalid"));
        }
        let mut parameters = Vec::new();
        for p in [
            inputs.gate,
            inputs.beta,
            inputs.write_gate,
            inputs.a_log,
            inputs.dt_bias,
        ] {
            parameters.push(
                p.map(|t| {
                    super::check_view(t, x)?;
                    float(t)?;
                    let pack = super::PreparedPack::prepare(t)?;
                    let f32 = alloc(x, t.meta.shape().clone(), DType::F32);
                    Ok::<_, InferenceError>((pack, f32))
                })
                .transpose()?,
            );
        }
        let mut controls = Vec::new();
        for p in [
            inputs.state_indices,
            inputs.sequence_order,
            inputs.prefix_lengths,
            inputs.checkpoint_prefixes,
            inputs.accepted_tokens,
        ] {
            controls.push(
                p.map(|t| {
                    super::check_view(t, x)?;
                    if t.meta.shape()[..] != [plan.sequences] {
                        return Err(InferenceError(
                            "prepared delta controls require [sequences]",
                        ));
                    }
                    super::metadata::PreparedIndices::prepare(t, DType::I32)
                })
                .transpose()?,
            );
        }
        let state = inputs
            .initial_state
            .map(super::PreparedPack::prepare)
            .transpose()?;
        let state_f32 = state
            .as_ref()
            .map(|p| {
                let s = &p.output;
                check(s, x)?;
                float(s)?;
                if s.meta.num_dims() != 4
                    || s.meta.shape()[1] != heads
                    || (options.state_layout == StateLayout::KeyValue
                        && s.meta.shape()[2..] != [dk, dv])
                    || (options.state_layout == StateLayout::ValueKey
                        && s.meta.shape()[2..] != [dv, dk])
                {
                    return Err(InferenceError("prepared delta state layout mismatch"));
                }
                Ok(if s.dtype == DType::F32 {
                    s.clone()
                } else {
                    alloc(s, s.meta.shape().clone(), DType::F32)
                })
            })
            .transpose()?;
        let output =
            plan.allocate_output(x, heads, dk, dv, options.householder, options.state_layout)?;
        Ok(Self {
            plan,
            q,
            k,
            v,
            parameters,
            controls,
            state,
            state_f32,
            output,
            options,
        })
    }
    pub unsafe fn run(&self) -> Result<&DeltaOutput<R>, InferenceError> {
        unsafe {
            self.q.run();
            self.k.run();
            self.v.run();
            for p in self.parameters.iter().flatten() {
                p.0.run();
                super::norm::cast_into(&p.0.output, &p.1)?;
            }
            for p in self.controls.iter().flatten() {
                p.run();
            }
            if let (Some(p), Some(f32)) = (&self.state, &self.state_f32) {
                p.run();
                if p.output.dtype != DType::F32 {
                    super::norm::cast_into(&p.output, f32)?;
                }
            }
        }
        let parameter = |i: usize| self.parameters[i].as_ref().map(|p| &p.1);
        let control = |i: usize| self.controls[i].as_ref().map(|p| &p.output);
        unsafe {
            self.plan.run_into(
                &self.q.output,
                &self.k.output,
                &self.v.output,
                DeltaInputs {
                    gate: parameter(0),
                    beta: parameter(1),
                    write_gate: parameter(2),
                    a_log: parameter(3),
                    dt_bias: parameter(4),
                    initial_state: self.state_f32.as_ref(),
                    state_indices: control(0),
                    sequence_order: control(1),
                    prefix_lengths: control(2),
                    checkpoint_prefixes: control(3),
                    accepted_tokens: control(4),
                },
                self.options,
                &self.output,
            )?;
        }
        Ok(&self.output)
    }
}
