//! General inference normalization and residual/gating compositions. All
//! intermediates, broadcasting and casts are native device kernels.
use super::activation::{self, Activation};
use super::{InferenceError, alloc, check, float, plane};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    Layer,
}
#[derive(Clone, Copy, Debug)]
pub struct NormOptions {
    pub kind: NormKind,
    pub epsilon: f32,
    pub gain_center: f32,
    pub weight_before_reduction: bool,
    pub activation: Activation,
    pub output_dtype: Option<DType>,
}
impl Default for NormOptions {
    fn default() -> Self {
        Self {
            kind: NormKind::Rms,
            epsilon: 1e-6,
            gain_center: 0.0,
            weight_before_reduction: false,
            activation: Activation::Identity,
            output_dtype: None,
        }
    }
}
pub struct NormOutput<R: Runtime> {
    pub values: RudaTensor<R>,
    pub residual: Option<RudaTensor<R>>,
}
/// Fixed-shape, fixed-address normalization launch. Parameters are already FP32;
/// residual and gate have the complete input shape (broadcasting is a separate
/// producer operation). Preparation allocates once; run only submits a kernel.
pub struct PreparedNorm<R: Runtime> {
    pack: super::PreparedPack<R>,
    input: RudaTensor<R>,
    weight: Option<RudaTensor<R>>,
    bias: Option<RudaTensor<R>>,
    residual: Option<RudaTensor<R>>,
    gate: Option<RudaTensor<R>>,
    pub output: RudaTensor<R>,
    pub residual_output: RudaTensor<R>,
    options: NormOptions,
    rows: usize,
    width: usize,
    lanes: u32,
}
impl<R: Runtime> PreparedNorm<R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        weight: Option<&RudaTensor<R>>,
        bias: Option<&RudaTensor<R>>,
        residual: Option<&RudaTensor<R>>,
        gate: Option<&RudaTensor<R>>,
        options: NormOptions,
    ) -> Result<Self, InferenceError> {
        let pack = super::PreparedPack::prepare(input)?;
        let input = &pack.output;
        check(input, input)?;
        float(input)?;
        let width = *input
            .meta
            .shape()
            .last()
            .ok_or(InferenceError("prepared norm needs an axis"))?;
        if width == 0
            || !options.epsilon.is_finite()
            || options.epsilon <= 0.0
            || !options.gain_center.is_finite()
        {
            return Err(InferenceError("invalid prepared norm options"));
        }
        for t in weight.iter().chain(bias.iter()) {
            check(t, input)?;
            if t.dtype != DType::F32 || t.meta.shape()[..] != [width] {
                return Err(InferenceError(
                    "prepared norm affine parameters must be F32 [D]",
                ));
            }
        }
        for t in residual.iter().chain(gate.iter()) {
            check(t, input)?;
            if t.dtype != DType::F32 || t.meta.shape() != input.meta.shape() {
                return Err(InferenceError(
                    "prepared residual/gate must be full-shape F32",
                ));
            }
        }
        let dtype = options.output_dtype.unwrap_or(input.dtype);
        let output = alloc(input, input.meta.shape().clone(), dtype);
        float(&output)?;
        let residual_output = alloc(input, input.meta.shape().clone(), input.dtype);
        let rows = input.meta.num_elements() / width;
        let lanes = plane(input, rows, 1)?;
        let input = input.clone();
        Ok(Self {
            pack,
            input,
            weight: weight.cloned(),
            bias: bias.cloned(),
            residual: residual.cloned(),
            gate: gate.cloned(),
            output,
            residual_output,
            options,
            rows,
            width,
            lanes,
        })
    }
    /// Mutating returned fixed buffers or replaying concurrently requires the
    /// caller to enforce exclusive access and stream dependencies.
    pub unsafe fn run(&self) -> NormOutput<R> {
        unsafe {
            self.pack.run();
        }
        let x = &self.input;
        let o = self.options;
        if self.rows > 0 {
            norm_kernel::launch::<R>(
                &x.client,
                RudaCount::Static(self.rows as u32, 1, 1),
                RudaDim::new_1d(self.lanes),
                x.clone().into_array_arg(),
                self.weight.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.bias.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.residual.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.gate.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                self.residual_output.clone().into_array_arg(),
                self.width as u32,
                o.epsilon,
                o.gain_center,
                o.kind == NormKind::Layer,
                self.weight.is_some(),
                self.bias.is_some(),
                self.residual.is_some(),
                self.gate.is_some(),
                o.weight_before_reduction,
                o.activation.code(),
                x.dtype.into(),
                self.output.dtype.into(),
            );
        }
        NormOutput {
            values: self.output.clone(),
            residual: self.residual.as_ref().map(|_| self.residual_output.clone()),
        }
    }
}

pub fn cast<R: Runtime>(x: &RudaTensor<R>, dtype: DType) -> Result<RudaTensor<R>, InferenceError> {
    float(x)?;
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(InferenceError("invalid inference float cast"));
    }
    let x = into_contiguous(x.clone());
    check(&x, &x)?;
    if x.dtype == dtype {
        return Ok(x);
    }
    let out = alloc(&x, x.meta.shape().clone(), dtype);
    let n = x.meta.num_elements();
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        cast_values::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            x.dtype.into(),
            dtype.into(),
        );
    }
    Ok(out)
}
pub unsafe fn cast_into<R: Runtime>(
    x: &RudaTensor<R>,
    out: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    for t in [x, out] {
        check(t, x)?;
        float(t)?;
    }
    if x.meta.shape() != out.meta.shape() {
        return Err(InferenceError("cast destination shape mismatch"));
    }
    let n = x.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        cast_values::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            x.dtype.into(),
            out.dtype.into(),
        );
    }
    Ok(())
}
pub struct PreparedBroadcast<R: Runtime> {
    pack: super::PreparedPack<R>,
    source: RudaTensor<R>,
    metadata: RudaTensor<R>,
    pub output: RudaTensor<R>,
    rank: usize,
}
impl<R: Runtime> PreparedBroadcast<R> {
    pub fn prepare(source: &RudaTensor<R>, like: &RudaTensor<R>) -> Result<Self, InferenceError> {
        super::check_view(source, like)?;
        check(like, like)?;
        float(source)?;
        let pack = super::PreparedPack::prepare(source)?;
        let source = &pack.output;
        let target = like.meta.shape();
        let shape = source.meta.shape();
        if shape.len() > target.len() {
            return Err(InferenceError(
                "prepared broadcast source rank exceeds target",
            ));
        }
        let mut words = Vec::with_capacity(target.len() * 3);
        let mut stride: usize = 1usize;
        for axis in (0..target.len()).rev() {
            let src = axis.checked_sub(target.len() - shape.len());
            let extent = src.map_or(1, |i| shape[i]);
            if extent != 1 && extent != target[axis] {
                return Err(InferenceError("prepared broadcast axes incompatible"));
            }
            let input_stride = src.map_or(0, |i| {
                if extent == 1 {
                    0
                } else {
                    source.meta.strides()[i]
                }
            });
            words.extend([target[axis] as u32, stride as u32, input_stride as u32]);
            stride = stride
                .checked_mul(target[axis])
                .ok_or(InferenceError("prepared broadcast size overflow"))?;
        }
        let output = alloc(like, target.clone(), DType::F32);
        let metadata = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([words.len()]),
            like.client.create_from_slice(bytemuck::cast_slice(&words)),
            DType::U32,
        );
        let source = source.clone();
        Ok(Self {
            pack,
            source,
            metadata,
            output,
            rank: target.len(),
        })
    }
    pub unsafe fn run(&self) {
        unsafe {
            self.pack.run();
        }
        let n = self.output.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(self.output.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.output.client, n, dim);
            broadcast_numeric::launch::<R>(
                &self.output.client,
                count,
                dim,
                self.source.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                n as u32,
                self.rank as u32,
                self.source.dtype.into(),
            );
        }
    }
}
#[ruda(launch)]
pub fn broadcast_numeric<F: Float>(
    x: &Array<F>,
    metadata: &Array<u32>,
    out: &mut Array<f32>,
    n: u32,
    rank: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut source: usize = 0usize;
        for axis in 0..rank as usize {
            source += (i / metadata[axis * 3 + 1] as usize % metadata[axis * 3] as usize)
                * metadata[axis * 3 + 2] as usize;
        }
        out[i] = f32::cast_from(x[source]);
    }
}
/// NumPy-style right-aligned broadcasting, performed on-device. Metadata are
/// generated from shapes/strides only; tensor values are never read on the CPU.
pub fn broadcast_f32<R: Runtime>(
    value: &RudaTensor<R>,
    like: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    let value = cast(value, DType::F32)?;
    check(&value, like)?;
    let target = like.meta.shape();
    let source = value.meta.shape();
    if source.len() > target.len() {
        return Err(InferenceError("broadcast source rank exceeds target"));
    }
    if source == target {
        return Ok(value);
    }
    let mut metadata = Vec::<u32>::with_capacity(target.len() * 3);
    let mut stride: usize = 1usize;
    for axis in (0..target.len()).rev() {
        let src = axis.checked_sub(target.len() - source.len());
        let extent = src.map_or(1, |i| source[i]);
        if extent != 1 && extent != target[axis] {
            return Err(InferenceError("incompatible broadcast axes"));
        }
        let input_stride = src.map_or(0, |i| {
            if extent == 1 {
                0
            } else {
                value.meta.strides()[i]
            }
        });
        metadata.extend([target[axis] as u32, stride as u32, input_stride as u32]);
        stride = stride
            .checked_mul(target[axis])
            .ok_or(InferenceError("broadcast size overflow"))?;
    }
    check(like, like)?;
    let out = alloc(like, target.clone(), DType::F32);
    if stride != 0 {
        let meta = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([metadata.len()]),
            like.client
                .create_from_slice(bytemuck::cast_slice(&metadata)),
            DType::U32,
        );
        let dim = RudaDim::new(like.client.properties(), stride);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&like.client, stride, dim);
        broadcast::launch::<R>(
            &like.client,
            count,
            dim,
            value.into_array_arg(),
            meta.into_array_arg(),
            out.clone().into_array_arg(),
            stride as u32,
            target.len() as u32,
        );
    }
    Ok(out)
}

/// Last-axis normalization. Optional residual is summed in FP32 before the
/// reduction; optional sigmoid gate is applied after affine normalization.
pub fn normalize<R: Runtime>(
    input: &RudaTensor<R>,
    weight: Option<&RudaTensor<R>>,
    bias: Option<&RudaTensor<R>>,
    residual: Option<&RudaTensor<R>>,
    sigmoid_gate: Option<&RudaTensor<R>>,
    options: NormOptions,
) -> Result<NormOutput<R>, InferenceError> {
    let x = into_contiguous(input.clone());
    check(&x, &x)?;
    float(&x)?;
    let width = *x
        .meta
        .shape()
        .last()
        .ok_or(InferenceError("normalization needs a final axis"))?;
    if width == 0
        || !options.epsilon.is_finite()
        || options.epsilon <= 0.0
        || !options.gain_center.is_finite()
    {
        return Err(InferenceError("invalid normalization width/epsilon/gain"));
    }
    let dtype = options.output_dtype.unwrap_or(x.dtype);
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(InferenceError("invalid normalization output dtype"));
    }
    let w = weight.map(|t| cast(t, DType::F32)).transpose()?;
    let b = bias.map(|t| cast(t, DType::F32)).transpose()?;
    for t in w.iter().chain(b.iter()) {
        check(t, &x)?;
        if t.meta.shape()[..] != [width] {
            return Err(InferenceError(
                "normalization affine tensors must be [width]",
            ));
        }
    }
    let r = residual.map(|t| broadcast_f32(t, &x)).transpose()?;
    let g = sigmoid_gate.map(|t| broadcast_f32(t, &x)).transpose()?;
    let dummy = cast(&x, DType::F32)?;
    let rows = x.meta.num_elements() / width;
    let lanes = plane(&x, rows, 1)?;
    let out = alloc(&x, x.meta.shape().clone(), dtype);
    let residual_out = alloc(&x, x.meta.shape().clone(), x.dtype);
    if rows != 0 {
        norm_kernel::launch::<R>(
            &x.client,
            RudaCount::Static(rows as u32, 1, 1),
            RudaDim::new_1d(lanes),
            x.clone().into_array_arg(),
            w.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
            b.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
            r.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
            g.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
            out.clone().into_array_arg(),
            residual_out.clone().into_array_arg(),
            width as u32,
            options.epsilon,
            options.gain_center,
            options.kind == NormKind::Layer,
            w.is_some(),
            b.is_some(),
            r.is_some(),
            g.is_some(),
            options.weight_before_reduction,
            options.activation.code(),
            x.dtype.into(),
            dtype.into(),
        );
    }
    Ok(NormOutput {
        values: out,
        residual: residual.map(|_| residual_out),
    })
}
pub fn rmsnorm<R: Runtime>(
    x: &RudaTensor<R>,
    w: &RudaTensor<R>,
    eps: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    Ok(normalize(
        x,
        Some(w),
        None,
        None,
        None,
        NormOptions {
            epsilon: eps,
            ..Default::default()
        },
    )?
    .values)
}
pub fn gemma_rmsnorm<R: Runtime>(
    x: &RudaTensor<R>,
    w: &RudaTensor<R>,
    eps: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    Ok(normalize(
        x,
        Some(w),
        None,
        None,
        None,
        NormOptions {
            epsilon: eps,
            gain_center: 1.0,
            ..Default::default()
        },
    )?
    .values)
}
pub fn layernorm<R: Runtime>(
    x: &RudaTensor<R>,
    w: Option<&RudaTensor<R>>,
    b: Option<&RudaTensor<R>>,
    eps: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    Ok(normalize(
        x,
        w,
        b,
        None,
        None,
        NormOptions {
            kind: NormKind::Layer,
            epsilon: eps,
            ..Default::default()
        },
    )?
    .values)
}
pub fn fused_add_rmsnorm<R: Runtime>(
    x: &RudaTensor<R>,
    r: &RudaTensor<R>,
    w: &RudaTensor<R>,
    eps: f32,
    gemma: bool,
) -> Result<NormOutput<R>, InferenceError> {
    normalize(
        x,
        Some(w),
        None,
        Some(r),
        None,
        NormOptions {
            epsilon: eps,
            gain_center: if gemma { 1.0 } else { 0.0 },
            ..Default::default()
        },
    )
}
pub fn fused_rmsnorm_silu<R: Runtime>(
    x: &RudaTensor<R>,
    w: &RudaTensor<R>,
    eps: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    Ok(normalize(
        x,
        Some(w),
        None,
        None,
        None,
        NormOptions {
            epsilon: eps,
            activation: Activation::Silu,
            ..Default::default()
        },
    )?
    .values)
}
pub fn rmsnorm_sigmoid_gate<R: Runtime>(
    x: &RudaTensor<R>,
    gate: &RudaTensor<R>,
    eps: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    Ok(normalize(
        x,
        None,
        None,
        None,
        Some(gate),
        NormOptions {
            epsilon: eps,
            ..Default::default()
        },
    )?
    .values)
}

/// alpha*residual + beta*RMSNorm(branch*(weight+gain_center)), retaining FP32
/// until the returned stream or a subsequent normalization is rounded.
pub fn gated_residual_rmsnorm<R: Runtime>(
    branch: &RudaTensor<R>,
    residual: &RudaTensor<R>,
    weight: Option<&RudaTensor<R>>,
    alpha: &RudaTensor<R>,
    beta: &RudaTensor<R>,
    eps: f32,
    gain_center: f32,
) -> Result<RudaTensor<R>, InferenceError> {
    let normalized = normalize(
        branch,
        weight,
        None,
        None,
        None,
        NormOptions {
            epsilon: eps,
            gain_center,
            weight_before_reduction: true,
            output_dtype: Some(DType::F32),
            ..Default::default()
        },
    )?
    .values;
    let residual = broadcast_f32(residual, &normalized)?;
    let alpha = broadcast_f32(alpha, &normalized)?;
    let beta = broadcast_f32(beta, &normalized)?;
    blend(&normalized, &residual, &beta, &alpha, None)
}
pub struct ResidualNext<R: Runtime> {
    pub residual: RudaTensor<R>,
    pub ffn_input: RudaTensor<R>,
}
pub fn gated_residual_next_rmsnorm<R: Runtime>(
    branch: &RudaTensor<R>,
    residual: &RudaTensor<R>,
    post_weight: Option<&RudaTensor<R>>,
    next_weight: &RudaTensor<R>,
    alpha: &RudaTensor<R>,
    beta: &RudaTensor<R>,
    post_eps: f32,
    next_eps: f32,
    post_gain: f32,
    next_gain: f32,
) -> Result<ResidualNext<R>, InferenceError> {
    let stream = gated_residual_rmsnorm(
        branch,
        residual,
        post_weight,
        alpha,
        beta,
        post_eps,
        post_gain,
    )?;
    let ffn_input = normalize(
        &stream,
        Some(next_weight),
        None,
        None,
        None,
        NormOptions {
            epsilon: next_eps,
            gain_center: next_gain,
            output_dtype: Some(branch.dtype),
            ..Default::default()
        },
    )?
    .values;
    Ok(ResidualNext {
        residual: stream,
        ffn_input,
    })
}
pub struct DitOutput<R: Runtime> {
    pub residual: RudaTensor<R>,
    pub normalized: RudaTensor<R>,
}
pub struct DitParameters<'a, R: Runtime> {
    pub residual: Option<&'a RudaTensor<R>>,
    pub gate: Option<&'a RudaTensor<R>>,
    pub gate_bias: Option<&'a RudaTensor<R>>,
    pub scale: Option<&'a RudaTensor<R>>,
    pub scale_bias: Option<&'a RudaTensor<R>>,
    pub shift: Option<&'a RudaTensor<R>>,
    pub shift_bias: Option<&'a RudaTensor<R>>,
    pub gamma: Option<&'a RudaTensor<R>>,
    pub beta: Option<&'a RudaTensor<R>>,
    pub epsilon: f32,
}
pub fn dit_layernorm<R: Runtime>(
    input: &RudaTensor<R>,
    p: DitParameters<'_, R>,
) -> Result<DitOutput<R>, InferenceError> {
    let x = cast(input, DType::F32)?;
    let n = x.meta.num_elements();
    let prepare = |t: Option<&RudaTensor<R>>| t.map(|v| broadcast_f32(v, &x)).transpose();
    let r = prepare(p.residual)?;
    let gate = prepare(p.gate)?;
    let gb = prepare(p.gate_bias)?;
    let scale = prepare(p.scale)?;
    let sb = prepare(p.scale_bias)?;
    let shift = prepare(p.shift)?;
    let shb = prepare(p.shift_bias)?;
    let stream = alloc(&x, x.meta.shape().clone(), DType::F32);
    let out = alloc(&x, x.meta.shape().clone(), DType::F32);
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        dit_stream::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            r.as_ref().unwrap_or(&x).clone().into_array_arg(),
            gate.as_ref().unwrap_or(&x).clone().into_array_arg(),
            gb.as_ref().unwrap_or(&x).clone().into_array_arg(),
            stream.clone().into_array_arg(),
            n as u32,
            r.is_some(),
            gate.is_some(),
            gb.is_some(),
        );
    }
    let normalized = layernorm(&stream, p.gamma, p.beta, p.epsilon)?;
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        dit_affine::launch::<R>(
            &x.client,
            count,
            dim,
            normalized.into_array_arg(),
            scale.as_ref().unwrap_or(&x).clone().into_array_arg(),
            sb.as_ref().unwrap_or(&x).clone().into_array_arg(),
            shift.as_ref().unwrap_or(&x).clone().into_array_arg(),
            shb.as_ref().unwrap_or(&x).clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            scale.is_some(),
            sb.is_some(),
            shift.is_some(),
            shb.is_some(),
        );
    }
    Ok(DitOutput {
        residual: cast(&stream, input.dtype)?,
        normalized: cast(&out, input.dtype)?,
    })
}
fn blend<R: Runtime>(
    x: &RudaTensor<R>,
    r: &RudaTensor<R>,
    a: &RudaTensor<R>,
    b: &RudaTensor<R>,
    bias: Option<&RudaTensor<R>>,
) -> Result<RudaTensor<R>, InferenceError> {
    let out = alloc(x, x.meta.shape().clone(), DType::F32);
    let n = x.meta.num_elements();
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        blend_kernel::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            r.clone().into_array_arg(),
            a.clone().into_array_arg(),
            b.clone().into_array_arg(),
            bias.unwrap_or(x).clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            bias.is_some(),
        );
    }
    Ok(out)
}

#[ruda(launch)]
pub fn cast_values<F: Float, O: Float>(
    x: &Array<F>,
    out: &mut Array<O>,
    n: u32,
    #[define(F)] _in: StorageType,
    #[define(O)] _out: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = O::cast_from(f32::cast_from(x[i]));
    }
}
#[ruda(launch)]
pub fn broadcast(x: &Array<f32>, metadata: &Array<u32>, out: &mut Array<f32>, n: u32, rank: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut src: usize = 0usize;
        for axis in 0..rank as usize {
            src += (i / metadata[axis * 3 + 1] as usize % metadata[axis * 3] as usize)
                * metadata[axis * 3 + 2] as usize;
        }
        out[i] = x[src];
    }
}
#[ruda(launch)]
pub fn norm_kernel<F: Float, O: Float>(
    x: &Array<F>,
    w: &Array<f32>,
    b: &Array<f32>,
    residual: &Array<f32>,
    gate: &Array<f32>,
    out: &mut Array<O>,
    residual_out: &mut Array<F>,
    width: u32,
    epsilon: f32,
    gain: f32,
    #[comptime] layer: bool,
    #[comptime] weighted: bool,
    #[comptime] biased: bool,
    #[comptime] added: bool,
    #[comptime] gated: bool,
    #[comptime] preweight: bool,
    #[comptime] act: u32,
    #[define(F)] _in: StorageType,
    #[define(O)] _out: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let lanes = PLANE_DIM as usize;
    let mut sum: f32 = 0.0f32;
    let mut sq: f32 = 0.0f32;
    let mut col = lane;
    while col < width as usize {
        let i = row * width as usize + col;
        let mut v = f32::cast_from(x[i]);
        if comptime!(added) {
            v += residual[i];
        }
        if comptime!(weighted && preweight) {
            v *= w[col] + gain;
        }
        sum += v;
        sq += v * v;
        col += lanes;
    }
    sum = plane_sum(sum);
    sq = plane_sum(sq);
    let mut mean: f32 = 0.0f32;
    if comptime!(layer) {
        mean = sum / width as f32;
        let mut variance: f32 = 0.0f32;
        col = lane;
        while col < width as usize {
            let i = row * width as usize + col;
            let mut v = f32::cast_from(x[i]);
            if comptime!(added) {
                v += residual[i];
            }
            if comptime!(weighted && preweight) {
                v *= w[col] + gain;
            }
            let centered = v - mean;
            variance += centered * centered;
            col += lanes;
        }
        sq = plane_sum(variance);
    }
    let inverse = 1.0 / (sq / width as f32 + epsilon).sqrt();
    col = lane;
    while col < width as usize {
        let i = row * width as usize + col;
        let mut v = f32::cast_from(x[i]);
        if comptime!(added) {
            v += residual[i];
            residual_out[i] = F::cast_from(v);
        }
        if comptime!(weighted && preweight) {
            v *= w[col] + gain;
        }
        let mut y = (v - mean) * inverse;
        if comptime!(weighted && !preweight) {
            y *= w[col] + gain;
        }
        if comptime!(biased) {
            y += b[col];
        }
        if comptime!(gated) {
            y /= 1.0 + (-gate[i]).exp();
        }
        out[i] = O::cast_from(activation::apply(y, act));
        col += lanes;
    }
}
#[ruda(launch)]
pub fn blend_kernel(
    x: &Array<f32>,
    r: &Array<f32>,
    a: &Array<f32>,
    b: &Array<f32>,
    bias: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    #[comptime] biased: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut y = x[i] * a[i] + r[i] * b[i];
        if comptime!(biased) {
            y += bias[i];
        }
        out[i] = y;
    }
}
#[ruda(launch)]
pub fn dit_stream(
    x: &Array<f32>,
    r: &Array<f32>,
    gate: &Array<f32>,
    gb: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    #[comptime] residual: bool,
    #[comptime] gated: bool,
    #[comptime] gate_bias: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut g: f32 = 1.0f32;
        if comptime!(gated) {
            g = gate[i];
        }
        if comptime!(gate_bias) {
            g += gb[i];
        }
        let mut v = x[i] * g;
        if comptime!(residual) {
            v += r[i];
        }
        out[i] = v;
    }
}
#[ruda(launch)]
pub fn dit_affine(
    x: &Array<f32>,
    scale: &Array<f32>,
    sb: &Array<f32>,
    shift: &Array<f32>,
    shb: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    #[comptime] scaled: bool,
    #[comptime] scale_bias: bool,
    #[comptime] shifted: bool,
    #[comptime] shift_bias: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut s: f32 = 1.0f32;
        let mut b: f32 = 0.0f32;
        if comptime!(scaled) {
            s += scale[i];
        }
        if comptime!(scale_bias) {
            s += sb[i];
        }
        if comptime!(shifted) {
            b += shift[i];
        }
        if comptime!(shift_bias) {
            b += shb[i];
        }
        out[i] = x[i] * s + b;
    }
}
