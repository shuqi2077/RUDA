//! Native RMSNorm/SiLU-gate training and packed causal depthwise convolution.
//! Gradient reductions have one owner per parameter and a fixed row order.
use super::{InferenceError, alloc, check, float, plane};
use ruda_core::tensor::DType;
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};
pub struct RmsTraining<R: Runtime> {
    pub output: RudaTensor<R>,
    pub source: RudaTensor<R>,
    pub reciprocal_std: RudaTensor<R>,
    pub residual: Option<RudaTensor<R>>,
}
pub struct RmsGradients<R: Runtime> {
    pub input: RudaTensor<R>,
    pub weight: RudaTensor<R>,
    pub gate: Option<RudaTensor<R>>,
}
/// Persistent forward/backward buffers for the deterministic and ordinary RMS
/// training APIs. Both use the same fixed-owner native gradient reductions.
pub struct PreparedRmsTraining<R: Runtime> {
    pack: super::PreparedPack<R>,
    weight_pack: super::PreparedPack<R>,
    input: RudaTensor<R>,
    weight_source: RudaTensor<R>,
    weight: RudaTensor<R>,
    residual: Option<super::norm::PreparedBroadcast<R>>,
    gate: Option<super::norm::PreparedBroadcast<R>>,
    pub saved: RmsTraining<R>,
    pub gradients: RmsGradients<R>,
    dy: RudaTensor<R>,
    gr: RudaTensor<R>,
    dx: RudaTensor<R>,
    dw: RudaTensor<R>,
    dz: RudaTensor<R>,
    rows: usize,
    width: usize,
    lanes: u32,
    epsilon: f32,
}
impl<R: Runtime> PreparedRmsTraining<R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        weight: &RudaTensor<R>,
        residual: Option<&RudaTensor<R>>,
        gate: Option<&RudaTensor<R>>,
        epsilon: f32,
    ) -> Result<Self, InferenceError> {
        let pack = super::PreparedPack::prepare(input)?;
        let weight_pack = super::PreparedPack::prepare(weight)?;
        let input = &pack.output;
        let weight = &weight_pack.output;
        for t in [input, weight] {
            check(t, input)?;
            float(t)?;
        }
        let width = *input
            .meta
            .shape()
            .last()
            .ok_or(InferenceError("prepared training RMS needs an axis"))?;
        if width == 0
            || weight.meta.shape()[..] != [width]
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(InferenceError(
                "prepared training RMS weight/options invalid",
            ));
        }
        for t in residual.iter().chain(gate.iter()) {
            super::check_view(t, input)?;
            float(t)?;
            if t.meta.shape() != input.meta.shape() {
                return Err(InferenceError(
                    "training residual/gate must have the input shape",
                ));
            }
        }
        let rows = input.meta.num_elements() / width;
        let lanes = plane(input, rows, 1)?;
        let weight_f32 = alloc(input, [width], DType::F32);
        let saved = RmsTraining {
            output: alloc(input, input.meta.shape().clone(), input.dtype),
            source: alloc(input, input.meta.shape().clone(), DType::F32),
            reciprocal_std: alloc(input, [rows], DType::F32),
            residual: residual.map(|_| alloc(input, input.meta.shape().clone(), input.dtype)),
        };
        let gradients = RmsGradients {
            input: alloc(input, input.meta.shape().clone(), input.dtype),
            weight: alloc(input, [width], weight.dtype),
            gate: gate.map(|g| alloc(input, g.meta.shape().clone(), g.dtype)),
        };
        let input_value = input.clone();
        let weight_source = weight.clone();
        let residual_plan = residual
            .map(|r| super::norm::PreparedBroadcast::prepare(r, input))
            .transpose()?;
        let gate_plan = gate
            .map(|g| super::norm::PreparedBroadcast::prepare(g, input))
            .transpose()?;
        let dy = alloc(input, input.meta.shape().clone(), DType::F32);
        let gr = alloc(input, input.meta.shape().clone(), DType::F32);
        let dx = alloc(input, input.meta.shape().clone(), DType::F32);
        let dw = alloc(input, [width], DType::F32);
        let dz = alloc(input, input.meta.shape().clone(), DType::F32);
        Ok(Self {
            pack,
            weight_pack,
            input: input_value,
            weight_source,
            weight: weight_f32,
            residual: residual_plan,
            gate: gate_plan,
            saved,
            gradients,
            dy,
            gr,
            dx,
            dw,
            dz,
            rows,
            width,
            lanes,
            epsilon,
        })
    }
    pub unsafe fn forward(&self) -> Result<&RmsTraining<R>, InferenceError> {
        unsafe {
            self.pack.run();
            self.weight_pack.run();
            super::norm::cast_into(&self.weight_source, &self.weight)?;
            if let Some(r) = &self.residual {
                r.run();
            }
            if let Some(z) = &self.gate {
                z.run();
            }
        }
        let x = &self.input;
        if self.rows > 0 {
            rms_train::launch::<R>(
                &x.client,
                RudaCount::Static(self.rows as u32, 1, 1),
                RudaDim::new_1d(self.lanes),
                x.clone().into_array_arg(),
                self.weight.clone().into_array_arg(),
                self.residual
                    .as_ref()
                    .map(|r| &r.output)
                    .unwrap_or(&self.dy)
                    .clone()
                    .into_array_arg(),
                self.gate
                    .as_ref()
                    .map(|g| &g.output)
                    .unwrap_or(&self.dy)
                    .clone()
                    .into_array_arg(),
                self.saved.output.clone().into_array_arg(),
                self.saved.source.clone().into_array_arg(),
                self.saved.reciprocal_std.clone().into_array_arg(),
                self.width as u32,
                self.epsilon,
                self.residual.is_some(),
                self.gate.is_some(),
                x.dtype.into(),
            );
        }
        if let Some(r) = &self.saved.residual {
            unsafe {
                super::norm::cast_into(&self.saved.source, r)?;
            }
        }
        Ok(&self.saved)
    }
    /// Call after the corresponding forward and before changing its saved input,
    /// parameters or gates. Gradient destinations are exclusive scratch.
    pub unsafe fn backward(
        &self,
        grad: &RudaTensor<R>,
        grad_residual: Option<&RudaTensor<R>>,
    ) -> Result<&RmsGradients<R>, InferenceError> {
        unsafe {
            super::norm::cast_into(grad, &self.dy)?;
            if let Some(g) = grad_residual {
                super::norm::cast_into(g, &self.gr)?;
            }
        }
        let x = &self.input;
        let source = &self.saved.source;
        let z = self.gate.as_ref().map(|g| &g.output).unwrap_or(source);
        if self.rows > 0 {
            rms_dx::launch::<R>(
                &x.client,
                RudaCount::Static(self.rows as u32, 1, 1),
                RudaDim::new_1d(self.lanes),
                source.clone().into_array_arg(),
                self.weight.clone().into_array_arg(),
                self.dy.clone().into_array_arg(),
                self.saved.reciprocal_std.clone().into_array_arg(),
                z.clone().into_array_arg(),
                self.gr.clone().into_array_arg(),
                self.dx.clone().into_array_arg(),
                self.dz.clone().into_array_arg(),
                self.width as u32,
                self.gate.is_some(),
                grad_residual.is_some(),
            );
        }
        let dim = RudaDim::new(x.client.properties(), self.width);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, self.width, dim);
        rms_dw::launch::<R>(
            &x.client,
            count.clone(),
            dim,
            source.clone().into_array_arg(),
            self.dy.clone().into_array_arg(),
            self.saved.reciprocal_std.clone().into_array_arg(),
            z.clone().into_array_arg(),
            self.dw.clone().into_array_arg(),
            self.rows as u32,
            self.width as u32,
            self.gate.is_some(),
        );
        unsafe {
            super::norm::cast_into(&self.dx, &self.gradients.input)?;
            super::norm::cast_into(&self.dw, &self.gradients.weight)?;
            if let Some(g) = &self.gradients.gate {
                super::norm::cast_into(&self.dz, g)?;
            }
        }
        Ok(&self.gradients)
    }
}
pub fn rms_forward<R: Runtime>(
    input: &RudaTensor<R>,
    weight: &RudaTensor<R>,
    residual: Option<&RudaTensor<R>>,
    silu_gate: Option<&RudaTensor<R>>,
    epsilon: f32,
) -> Result<RmsTraining<R>, InferenceError> {
    let x = into_contiguous(input.clone());
    check(&x, &x)?;
    float(&x)?;
    let width = *x
        .meta
        .shape()
        .last()
        .ok_or(InferenceError("training RMS needs an axis"))?;
    if width == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(InferenceError("invalid training RMS epsilon/width"));
    }
    let w = super::norm::cast(weight, DType::F32)?;
    check(&w, &x)?;
    if w.meta.shape()[..] != [width] {
        return Err(InferenceError("training RMS weight must be [width]"));
    }
    let r = residual
        .map(|t| super::norm::broadcast_f32(t, &x))
        .transpose()?;
    let z = silu_gate
        .map(|t| super::norm::broadcast_f32(t, &x))
        .transpose()?;
    let source = alloc(&x, x.meta.shape().clone(), DType::F32);
    let output = alloc(&x, x.meta.shape().clone(), x.dtype);
    let rows = x.meta.num_elements() / width;
    let rstd = alloc(&x, [rows], DType::F32);
    let dummy = super::norm::cast(&x, DType::F32)?;
    let lanes = plane(&x, rows, 1)?;
    if rows != 0 {
        rms_train::launch::<R>(
            &x.client,
            RudaCount::Static(rows as u32, 1, 1),
            RudaDim::new_1d(lanes),
            x.clone().into_array_arg(),
            w.into_array_arg(),
            r.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
            z.as_ref().unwrap_or(&dummy).clone().into_array_arg(),
            output.clone().into_array_arg(),
            source.clone().into_array_arg(),
            rstd.clone().into_array_arg(),
            width as u32,
            epsilon,
            r.is_some(),
            z.is_some(),
            x.dtype.into(),
        );
    }
    let residual = if residual.is_some() {
        Some(super::norm::cast(&source, input.dtype)?)
    } else {
        None
    };
    Ok(RmsTraining {
        output,
        source,
        reciprocal_std: rstd,
        residual,
    })
}
pub fn rms_backward<R: Runtime>(
    saved: &RmsTraining<R>,
    weight: &RudaTensor<R>,
    grad: &RudaTensor<R>,
    silu_gate: Option<&RudaTensor<R>>,
    grad_residual: Option<&RudaTensor<R>>,
) -> Result<RmsGradients<R>, InferenceError> {
    let source = &saved.source;
    check(source, source)?;
    let width = *source
        .meta
        .shape()
        .last()
        .ok_or(InferenceError("invalid saved RMS source"))?;
    let rows = source.meta.num_elements() / width;
    let w = super::norm::cast(weight, DType::F32)?;
    let g = super::norm::cast(grad, DType::F32)?;
    let z = silu_gate
        .map(|t| super::norm::broadcast_f32(t, source))
        .transpose()?;
    let gr = grad_residual
        .map(|t| super::norm::broadcast_f32(t, source))
        .transpose()?;
    for t in [&w, &g, &saved.reciprocal_std] {
        check(t, source)?;
    }
    if w.meta.shape()[..] != [width]
        || g.meta.shape() != source.meta.shape()
        || saved.reciprocal_std.meta.shape()[..] != [rows]
    {
        return Err(InferenceError("RMS backward shape mismatch"));
    }
    let dx = alloc(source, source.meta.shape().clone(), DType::F32);
    let dz = alloc(source, source.meta.shape().clone(), DType::F32);
    let dw = alloc(source, [width], DType::F32);
    let lanes = plane(source, rows, 1)?;
    if rows != 0 {
        rms_dx::launch::<R>(
            &source.client,
            RudaCount::Static(rows as u32, 1, 1),
            RudaDim::new_1d(lanes),
            source.clone().into_array_arg(),
            w.into_array_arg(),
            g.clone().into_array_arg(),
            saved.reciprocal_std.clone().into_array_arg(),
            z.as_ref().unwrap_or(source).clone().into_array_arg(),
            gr.as_ref().unwrap_or(source).clone().into_array_arg(),
            dx.clone().into_array_arg(),
            dz.clone().into_array_arg(),
            width as u32,
            z.is_some(),
            gr.is_some(),
        );
    }
    let dim = RudaDim::new(source.client.properties(), width);
    let count = kernel_dsl::calculate_ruda_count_elemwise(&source.client, width, dim);
    rms_dw::launch::<R>(
        &source.client,
        count.clone(),
        dim,
        source.clone().into_array_arg(),
        g.into_array_arg(),
        saved.reciprocal_std.clone().into_array_arg(),
        z.as_ref().unwrap_or(source).clone().into_array_arg(),
        dw.clone().into_array_arg(),
        rows as u32,
        width as u32,
        z.is_some(),
    );
    Ok(RmsGradients {
        input: super::norm::cast(&dx, saved.output.dtype)?,
        weight: super::norm::cast(&dw, weight.dtype)?,
        gate: if z.is_some() {
            Some(super::norm::cast(&dz, silu_gate.unwrap().dtype)?)
        } else {
            None
        },
    })
}
pub struct CausalConvOutput<R: Runtime> {
    pub output: RudaTensor<R>,
    pub preactivation: RudaTensor<R>,
}
pub struct CausalConvGradients<R: Runtime> {
    pub input: RudaTensor<R>,
    pub weight: RudaTensor<R>,
    pub bias: Option<RudaTensor<R>>,
}
/// Retained native convolution training buffers for packed documents. Forward
/// refreshes inputs and parameters; backward consumes exactly that snapshot.
pub struct PreparedCausalConv<R: Runtime> {
    input: super::PreparedPack<R>,
    weight: super::PreparedPack<R>,
    bias: Option<super::PreparedPack<R>>,
    cu: super::metadata::PreparedIndices<R>,
    weight_f32: RudaTensor<R>,
    bias_f32: RudaTensor<R>,
    grad_f32: RudaTensor<R>,
    dz: RudaTensor<R>,
    dw: RudaTensor<R>,
    db: RudaTensor<R>,
    pub saved: CausalConvOutput<R>,
    pub gradients: CausalConvGradients<R>,
    tokens: usize,
    channels: usize,
    taps: usize,
    documents: usize,
    silu: bool,
    round_preactivation: bool,
}
impl<R: Runtime> PreparedCausalConv<R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        weight: &RudaTensor<R>,
        cu: &RudaTensor<R>,
        bias: Option<&RudaTensor<R>>,
        silu: bool,
        round_preactivation: bool,
    ) -> Result<Self, InferenceError> {
        let input = super::PreparedPack::prepare(input)?;
        let weight = super::PreparedPack::prepare(weight)?;
        let x = &input.output;
        float(x)?;
        check(&weight.output, x)?;
        float(&weight.output)?;
        let cu = super::metadata::PreparedIndices::prepare(cu, DType::I32)?;
        check(&cu.output, x)?;
        if x.meta.num_dims() != 2
            || weight.output.meta.num_dims() != 2
            || x.meta.shape()[1] == 0
            || weight.output.meta.shape()[0] != x.meta.shape()[1]
            || weight.output.meta.shape()[1] == 0
            || cu.output.meta.num_dims() != 1
            || cu.output.meta.shape()[0] < 2
        {
            return Err(InferenceError(
                "prepared convolution needs X[T,C], W[C,taps], document offsets[N+1]",
            ));
        }
        let tokens = x.meta.shape()[0];
        let channels = x.meta.shape()[1];
        let taps = weight.output.meta.shape()[1];
        let documents = cu.output.meta.shape()[0] - 1;
        let bias = bias.map(super::PreparedPack::prepare).transpose()?;
        if let Some(b) = &bias {
            check(&b.output, x)?;
            float(&b.output)?;
            if b.output.meta.shape()[..] != [channels] {
                return Err(InferenceError("prepared convolution bias must be [C]"));
            }
        }
        let weight_f32 = alloc(x, [channels, taps], DType::F32);
        let bias_f32 = alloc(x, [channels], DType::F32);
        let grad_f32 = alloc(x, x.meta.shape().clone(), DType::F32);
        let dz = alloc(x, x.meta.shape().clone(), DType::F32);
        let dw = alloc(x, [channels, taps], DType::F32);
        let db = alloc(x, [channels], DType::F32);
        let saved = CausalConvOutput {
            output: alloc(x, x.meta.shape().clone(), x.dtype),
            preactivation: alloc(x, x.meta.shape().clone(), DType::F32),
        };
        let gradients = CausalConvGradients {
            input: alloc(x, x.meta.shape().clone(), x.dtype),
            weight: alloc(x, [channels, taps], weight.output.dtype),
            bias: bias.as_ref().map(|b| alloc(x, [channels], b.output.dtype)),
        };
        Ok(Self {
            input,
            weight,
            bias,
            cu,
            weight_f32,
            bias_f32,
            grad_f32,
            dz,
            dw,
            db,
            saved,
            gradients,
            tokens,
            channels,
            taps,
            documents,
            silu,
            round_preactivation,
        })
    }
    /// Retained destinations and original inputs must not be concurrently used
    /// by another stream. All device document boundaries are caller contracts.
    pub unsafe fn forward(&self) -> Result<&CausalConvOutput<R>, InferenceError> {
        unsafe {
            self.input.run();
            self.weight.run();
            self.cu.run();
            super::norm::cast_into(&self.weight.output, &self.weight_f32)?;
            if let Some(b) = &self.bias {
                b.run();
                super::norm::cast_into(&b.output, &self.bias_f32)?;
            }
        }
        let x = &self.input.output;
        let n = self.tokens * self.channels;
        if n > 0 {
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            conv_forward::launch::<R>(
                &x.client,
                count.clone(),
                dim,
                x.clone().into_array_arg(),
                self.weight_f32.clone().into_array_arg(),
                self.bias_f32.clone().into_array_arg(),
                self.cu.output.clone().into_array_arg(),
                self.saved.output.clone().into_array_arg(),
                self.saved.preactivation.clone().into_array_arg(),
                self.tokens as u32,
                self.channels as u32,
                self.taps as u32,
                self.documents as u32,
                self.bias.is_some(),
                self.silu,
                x.dtype.into(),
            );
        }
        Ok(&self.saved)
    }
    /// `grad` is tight contiguous and shape-equal to the saved output. Neither
    /// forward nor bound parameters may change until this backward completes.
    pub unsafe fn backward(
        &self,
        grad: &RudaTensor<R>,
    ) -> Result<&CausalConvGradients<R>, InferenceError> {
        unsafe {
            super::norm::cast_into(grad, &self.grad_f32)?;
        }
        let x = &self.input.output;
        let n = self.tokens * self.channels;
        if n > 0 {
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            conv_activation_backward::launch::<R>(
                &x.client,
                count.clone(),
                dim,
                self.saved.preactivation.clone().into_array_arg(),
                self.grad_f32.clone().into_array_arg(),
                self.dz.clone().into_array_arg(),
                n as u32,
                self.silu,
                self.round_preactivation,
                x.dtype.into(),
            );
            conv_dx::launch::<R>(
                &x.client,
                count.clone(),
                dim,
                self.dz.clone().into_array_arg(),
                self.weight_f32.clone().into_array_arg(),
                self.cu.output.clone().into_array_arg(),
                self.gradients.input.clone().into_array_arg(),
                self.tokens as u32,
                self.channels as u32,
                self.taps as u32,
                self.documents as u32,
                x.dtype.into(),
            );
        }
        let n = self.channels * self.taps;
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        conv_dw::launch::<R>(
            &x.client,
            count.clone(),
            dim,
            x.clone().into_array_arg(),
            self.dz.clone().into_array_arg(),
            self.cu.output.clone().into_array_arg(),
            self.dw.clone().into_array_arg(),
            self.db.clone().into_array_arg(),
            self.tokens as u32,
            self.channels as u32,
            self.taps as u32,
            self.documents as u32,
            x.dtype.into(),
        );
        unsafe {
            super::norm::cast_into(&self.dw, &self.gradients.weight)?;
            if let Some(b) = &self.gradients.bias {
                super::norm::cast_into(&self.db, b)?;
            }
        }
        Ok(&self.gradients)
    }
}
fn conv_inputs<R: Runtime>(
    x: &RudaTensor<R>,
    weight: &RudaTensor<R>,
    cu: &RudaTensor<R>,
    bias: Option<&RudaTensor<R>>,
) -> Result<
    (
        RudaTensor<R>,
        RudaTensor<R>,
        Option<RudaTensor<R>>,
        usize,
        usize,
        usize,
    ),
    InferenceError,
> {
    let x = into_contiguous(x.clone());
    float(&x)?;
    check(&x, &x)?;
    check(cu, &x)?;
    if x.meta.num_dims() != 2
        || cu.dtype != DType::I32
        || cu.meta.num_dims() != 1
        || cu.meta.shape()[0] < 2
    {
        return Err(InferenceError(
            "causal convolution needs [tokens,channels] and I32 document indptr",
        ));
    }
    let channels = x.meta.shape()[1];
    let w = super::norm::cast(weight, DType::F32)?;
    check(&w, &x)?;
    if channels == 0
        || w.meta.num_dims() != 2
        || w.meta.shape()[0] != channels
        || w.meta.shape()[1] == 0
    {
        return Err(InferenceError(
            "causal convolution weight must be [channels,taps]",
        ));
    }
    let taps = w.meta.shape()[1];
    let b = bias.map(|t| super::norm::cast(t, DType::F32)).transpose()?;
    if let Some(b) = &b {
        check(b, &x)?;
        if b.meta.shape()[..] != [channels] {
            return Err(InferenceError("causal convolution bias must be [channels]"));
        }
    }
    let tokens = x.meta.shape()[0];
    Ok((x, w, b, tokens, channels, taps))
}
pub fn causal_conv1d<R: Runtime>(
    input: &RudaTensor<R>,
    weight: &RudaTensor<R>,
    cu_seqlens: &RudaTensor<R>,
    bias: Option<&RudaTensor<R>>,
    silu: bool,
) -> Result<CausalConvOutput<R>, InferenceError> {
    let (x, w, b, tokens, channels, taps) = conv_inputs(input, weight, cu_seqlens, bias)?;
    let out = alloc(&x, x.meta.shape().clone(), x.dtype);
    let pre = alloc(&x, x.meta.shape().clone(), DType::F32);
    let n = tokens * channels;
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        conv_forward::launch::<R>(
            &x.client,
            count.clone(),
            dim,
            x.clone().into_array_arg(),
            w.into_array_arg(),
            b.as_ref().unwrap_or(&pre).clone().into_array_arg(),
            cu_seqlens.clone().into_array_arg(),
            out.clone().into_array_arg(),
            pre.clone().into_array_arg(),
            tokens as u32,
            channels as u32,
            taps as u32,
            (cu_seqlens.meta.shape()[0] - 1) as u32,
            b.is_some(),
            silu,
            x.dtype.into(),
        );
    }
    Ok(CausalConvOutput {
        output: out,
        preactivation: pre,
    })
}
pub fn causal_conv1d_backward<R: Runtime>(
    input: &RudaTensor<R>,
    weight: &RudaTensor<R>,
    cu_seqlens: &RudaTensor<R>,
    bias: Option<&RudaTensor<R>>,
    grad: &RudaTensor<R>,
    saved: &CausalConvOutput<R>,
    silu: bool,
    round_preactivation: bool,
) -> Result<CausalConvGradients<R>, InferenceError> {
    let (x, w, b, tokens, channels, taps) = conv_inputs(input, weight, cu_seqlens, bias)?;
    let g = super::norm::cast(grad, DType::F32)?;
    for t in [&g, &saved.preactivation] {
        check(t, &x)?;
        if t.meta.shape() != x.meta.shape() {
            return Err(InferenceError("convolution backward shape mismatch"));
        }
    }
    let dz = alloc(&x, x.meta.shape().clone(), DType::F32);
    let dx = alloc(&x, x.meta.shape().clone(), x.dtype);
    let dw = alloc(&x, [channels, taps], DType::F32);
    let db = alloc(&x, [channels], DType::F32);
    let n = tokens * channels;
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        conv_activation_backward::launch::<R>(
            &x.client,
            count.clone(),
            dim,
            saved.preactivation.clone().into_array_arg(),
            g.into_array_arg(),
            dz.clone().into_array_arg(),
            n as u32,
            silu,
            round_preactivation,
            x.dtype.into(),
        );
        conv_dx::launch::<R>(
            &x.client,
            count.clone(),
            dim,
            dz.clone().into_array_arg(),
            w.into_array_arg(),
            cu_seqlens.clone().into_array_arg(),
            dx.clone().into_array_arg(),
            tokens as u32,
            channels as u32,
            taps as u32,
            (cu_seqlens.meta.shape()[0] - 1) as u32,
            x.dtype.into(),
        );
    }
    let parameter_count = channels * taps;
    let dim = RudaDim::new(x.client.properties(), parameter_count);
    let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, parameter_count, dim);
    conv_dw::launch::<R>(
        &x.client,
        count.clone(),
        dim,
        x.clone().into_array_arg(),
        dz.into_array_arg(),
        cu_seqlens.clone().into_array_arg(),
        dw.clone().into_array_arg(),
        db.clone().into_array_arg(),
        tokens as u32,
        channels as u32,
        taps as u32,
        (cu_seqlens.meta.shape()[0] - 1) as u32,
        x.dtype.into(),
    );
    Ok(CausalConvGradients {
        input: dx,
        weight: super::norm::cast(&dw, weight.dtype)?,
        bias: if b.is_some() {
            Some(super::norm::cast(&db, bias.unwrap().dtype)?)
        } else {
            None
        },
    })
}

#[ruda(launch)]
pub fn rms_train<F: Float>(
    x: &Array<F>,
    w: &Array<f32>,
    residual: &Array<f32>,
    z: &Array<f32>,
    out: &mut Array<F>,
    source: &mut Array<f32>,
    rstd: &mut Array<f32>,
    width: u32,
    eps: f32,
    #[comptime] added: bool,
    #[comptime] gated: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut col = lane;
    let mut sum: f32 = 0.0f32;
    while col < width as usize {
        let i = row * width as usize + col;
        let mut value = f32::cast_from(x[i]);
        if comptime!(added) {
            value += residual[i];
        }
        source[i] = value;
        sum += value * value;
        col += PLANE_DIM as usize;
    }
    let inv = 1.0 / (plane_sum(sum) / width as f32 + eps).sqrt();
    col = lane;
    while col < width as usize {
        let i = row * width as usize + col;
        let mut value = source[i] * inv * w[col];
        if comptime!(gated) {
            value *= z[i] / (1.0 + (-z[i]).exp());
        }
        out[i] = F::cast_from(value);
        col += PLANE_DIM as usize;
    }
    if lane == 0 {
        rstd[row] = inv;
    }
}
#[ruda(launch)]
pub fn rms_dx(
    x: &Array<f32>,
    w: &Array<f32>,
    g: &Array<f32>,
    rstd: &Array<f32>,
    z: &Array<f32>,
    gr: &Array<f32>,
    dx: &mut Array<f32>,
    dz: &mut Array<f32>,
    width: u32,
    #[comptime] gated: bool,
    #[comptime] residual_gradient: bool,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let inv = rstd[row];
    let mut dot: f32 = 0.0f32;
    let mut col = lane;
    while col < width as usize {
        let i = row * width as usize + col;
        let mut dy = g[i];
        if comptime!(gated) {
            dy *= z[i] / (1.0 + (-z[i]).exp());
        }
        dot += dy * w[col] * x[i];
        col += PLANE_DIM as usize;
    }
    dot = plane_sum(dot) * inv * inv / width as f32;
    col = lane;
    while col < width as usize {
        let i = row * width as usize + col;
        let mut dy = g[i];
        if comptime!(gated) {
            let sigmoid = 1.0 / (1.0 + (-z[i]).exp());
            dz[i] = g[i] * (x[i] * inv * w[col]) * sigmoid * (1.0 + z[i] * (1.0 - sigmoid));
            dy *= z[i] * sigmoid;
        }
        let mut value = inv * (dy * w[col] - x[i] * dot);
        if comptime!(residual_gradient) {
            value += gr[i];
        }
        dx[i] = value;
        col += PLANE_DIM as usize;
    }
}
#[ruda(launch)]
pub fn rms_dw(
    x: &Array<f32>,
    g: &Array<f32>,
    rstd: &Array<f32>,
    z: &Array<f32>,
    dw: &mut Array<f32>,
    rows: u32,
    width: u32,
    #[comptime] gated: bool,
) {
    let col = ABSOLUTE_POS as usize;
    if col < width as usize {
        let mut sum: f32 = 0.0f32;
        for row in 0..rows as usize {
            let i = row * width as usize + col;
            let mut dy = g[i];
            if comptime!(gated) {
                dy *= z[i] / (1.0 + (-z[i]).exp());
            }
            sum += dy * x[i] * rstd[row];
        }
        dw[col] = sum;
    }
}
#[ruda]
fn document(cu: &Array<i32>, token: usize, documents: u32) -> usize {
    let mut lo: usize = 0usize;
    let mut hi = documents as usize;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if cu[mid] <= token as i32 {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    usize::min(lo, documents as usize - 1)
}
#[ruda(launch)]
pub fn conv_forward<F: Float>(
    x: &Array<F>,
    w: &Array<f32>,
    bias: &Array<f32>,
    cu: &Array<i32>,
    out: &mut Array<F>,
    pre: &mut Array<f32>,
    tokens: u32,
    channels: u32,
    taps: u32,
    documents: u32,
    #[comptime] biased: bool,
    #[comptime] silu: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * channels as usize {
        let token = i / channels as usize;
        let channel = i % channels as usize;
        let doc = document(cu, token, documents);
        let start = cu[doc] as i64;
        let mut sum: f32 = 0.0f32;
        for tap in 0..taps as usize {
            let source = token as i64 + tap as i64 - (taps as i64 - 1);
            if source >= start && source >= 0 && source < tokens as i64 {
                sum = fma(
                    f32::cast_from(x[source as usize * channels as usize + channel]),
                    w[channel * taps as usize + tap],
                    sum,
                );
            }
        }
        if comptime!(biased) {
            sum += bias[channel];
        }
        pre[i] = sum;
        let mut value = sum;
        if comptime!(silu) {
            value = sum / (1.0 + (-sum).exp());
        }
        out[i] = F::cast_from(value);
    }
}
#[ruda(launch)]
pub fn conv_activation_backward<F: Float>(
    pre: &Array<f32>,
    grad: &Array<f32>,
    dz: &mut Array<f32>,
    n: u32,
    #[comptime] silu: bool,
    #[comptime] rounded: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut derivative: f32 = 1.0f32;
        if comptime!(silu) {
            let mut value = pre[i];
            if comptime!(rounded) {
                value = f32::cast_from(F::cast_from(value));
            }
            let sigmoid = 1.0 / (1.0 + (-value).exp());
            derivative = sigmoid * (1.0 + value * (1.0 - sigmoid));
        }
        dz[i] = grad[i] * derivative;
    }
}
#[ruda(launch)]
pub fn conv_dx<F: Float>(
    dz: &Array<f32>,
    w: &Array<f32>,
    cu: &Array<i32>,
    out: &mut Array<F>,
    tokens: u32,
    channels: u32,
    taps: u32,
    documents: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * channels as usize {
        let token = i / channels as usize;
        let channel = i % channels as usize;
        let doc = document(cu, token, documents);
        let end = cu[doc + 1] as usize;
        let mut sum: f32 = 0.0f32;
        for tap in 0..taps as usize {
            let target = token + taps as usize - 1 - tap;
            if target < end && target < tokens as usize {
                sum = fma(
                    dz[target * channels as usize + channel],
                    w[channel * taps as usize + tap],
                    sum,
                );
            }
        }
        out[i] = F::cast_from(sum);
    }
}
#[ruda(launch)]
pub fn conv_dw<F: Float>(
    x: &Array<F>,
    dz: &Array<f32>,
    cu: &Array<i32>,
    dw: &mut Array<f32>,
    db: &mut Array<f32>,
    tokens: u32,
    channels: u32,
    taps: u32,
    documents: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < channels as usize * taps as usize {
        let channel = i / taps as usize;
        let tap = i % taps as usize;
        let mut sum: f32 = 0.0f32;
        let mut bias: f32 = 0.0f32;
        for token in 0..tokens as usize {
            let doc = document(cu, token, documents);
            let source = token as i64 + tap as i64 - (taps as i64 - 1);
            let grad = dz[token * channels as usize + channel];
            if source >= cu[doc] as i64 && source >= 0 {
                sum += grad * f32::cast_from(x[source as usize * channels as usize + channel]);
            }
            if tap == 0 {
                bias += grad;
            }
        }
        dw[i] = sum;
        if tap == 0 {
            db[channel] = bias;
        }
    }
}
