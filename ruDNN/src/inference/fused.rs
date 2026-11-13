//! Native inference/training compositions, retaining the unrounded FP32 value
//! until the final requested activation/normalization quantization boundary.
use super::{InferenceError, alloc, check, float};
use rublas::tensor_float_quant::{BlockFloatTensor, QuantOptions};
use ruda_core::tensor::DType;
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};
pub struct QuantizedNorm<R: Runtime> {
    pub values: BlockFloatTensor<R>,
    pub residual: Option<RudaTensor<R>>,
}
pub fn norm_block_quantize<R: Runtime>(
    input: &RudaTensor<R>,
    weight: Option<&RudaTensor<R>>,
    bias: Option<&RudaTensor<R>>,
    residual: Option<&RudaTensor<R>>,
    options: super::norm::NormOptions,
    recipe: QuantOptions,
    global: Option<&RudaTensor<R>>,
) -> Result<QuantizedNorm<R>, InferenceError> {
    let source = super::norm::normalize(
        input,
        weight,
        bias,
        residual,
        None,
        super::norm::NormOptions {
            output_dtype: Some(DType::F32),
            ..options
        },
    )?;
    let values = rublas::tensor_float_quant::quantize(&source.values, global, None, recipe)
        .map_err(|e| InferenceError(e.0))?;
    Ok(QuantizedNorm {
        values,
        residual: source.residual,
    })
}
pub fn norm_fp8_quantize<R: Runtime>(
    input: &RudaTensor<R>,
    weight: Option<&RudaTensor<R>>,
    bias: Option<&RudaTensor<R>>,
    residual: Option<&RudaTensor<R>>,
    options: super::norm::NormOptions,
    decoder_scale: &RudaTensor<R>,
) -> Result<(RudaTensor<R>, Option<RudaTensor<R>>), InferenceError> {
    let source = super::norm::normalize(
        input,
        weight,
        bias,
        residual,
        None,
        super::norm::NormOptions {
            output_dtype: Some(DType::F32),
            ..options
        },
    )?;
    Ok((
        super::linear::encode_fp8(
            &source.values,
            decoder_scale,
            super::linear::FloatFormat::E4m3,
        )?,
        source.residual,
    ))
}
pub struct OrientedQuantized<R: Runtime> {
    pub rowwise: Option<BlockFloatTensor<R>>,
    pub colwise: Option<BlockFloatTensor<R>>,
}
pub fn gated_experts_quantize<R: Runtime>(
    input: &RudaTensor<R>,
    valid_rows: &RudaTensor<R>,
    global: Option<&RudaTensor<R>>,
    recipe: QuantOptions,
) -> Result<BlockFloatTensor<R>, InferenceError> {
    if input.meta.num_dims() != 3 {
        return Err(InferenceError("expert gated quantization expects [E,M,2K]"));
    }
    let x = super::norm::cast(input, DType::F32)?;
    let activated = super::activation::silu_and_mul(&x)?;
    rublas::tensor_float_quant::quantize(&activated, global, Some(valid_rows), recipe)
        .map_err(|e| InferenceError(e.0))
}
/// Row orientation groups adjacent columns; column orientation groups adjacent
/// rows. Both are independently stored, with no rounded BF16 intermediate.
pub fn gated_quantize<R: Runtime>(
    input: &RudaTensor<R>,
    grad: Option<&RudaTensor<R>>,
    rowwise: bool,
    colwise: bool,
    recipe: QuantOptions,
    global: Option<&RudaTensor<R>>,
) -> Result<OrientedQuantized<R>, InferenceError> {
    let input = super::norm::cast(input, DType::F32)?;
    if input.meta.num_dims() != 2 || input.meta.shape()[1] == 0 || input.meta.shape()[1] % 2 != 0 {
        return Err(InferenceError("gated quantization expects [M,2K]"));
    }
    let source = if let Some(dy) = grad {
        gated_backward(&input, dy, super::activation::Activation::Silu, DType::F32)?
    } else {
        super::activation::silu_and_mul(&input)?
    };
    let rows = if rowwise {
        Some(
            rublas::tensor_float_quant::quantize(&source, global, None, recipe)
                .map_err(|e| InferenceError(e.0))?,
        )
    } else {
        None
    };
    let cols = if colwise {
        let transposed = transpose(&source)?;
        Some(
            rublas::tensor_float_quant::quantize(&transposed, global, None, recipe)
                .map_err(|e| InferenceError(e.0))?,
        )
    } else {
        None
    };
    Ok(OrientedQuantized {
        rowwise: rows,
        colwise: cols,
    })
}
pub fn transpose<R: Runtime>(input: &RudaTensor<R>) -> Result<RudaTensor<R>, InferenceError> {
    let x = into_contiguous(input.clone());
    check(&x, &x)?;
    float(&x)?;
    if x.meta.num_dims() != 2 {
        return Err(InferenceError("transpose needs a matrix"));
    }
    let rows = x.meta.shape()[0];
    let cols = x.meta.shape()[1];
    let out = alloc(&x, [cols, rows], x.dtype);
    let n = x.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        transpose_kernel::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            out.clone().into_array_arg(),
            rows as u32,
            cols as u32,
            x.dtype.into(),
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn transpose_kernel<F: Float>(
    x: &Array<F>,
    out: &mut Array<F>,
    rows: u32,
    cols: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < rows as usize * cols as usize {
        out[i % cols as usize * rows as usize + i / cols as usize] = x[i];
    }
}
pub fn gated_backward<R: Runtime>(
    input: &RudaTensor<R>,
    grad: &RudaTensor<R>,
    activation: super::activation::Activation,
    dtype: DType,
) -> Result<RudaTensor<R>, InferenceError> {
    let x = super::norm::cast(input, DType::F32)?;
    let dy = super::norm::cast(grad, DType::F32)?;
    check(&dy, &x)?;
    if x.meta.num_dims() != 2
        || x.meta.shape()[1] == 0
        || x.meta.shape()[1] % 2 != 0
        || dy.meta.shape()[..] != [x.meta.shape()[0], x.meta.shape()[1] / 2]
    {
        return Err(InferenceError("gated backward shape mismatch"));
    }
    let out = alloc(&x, x.meta.shape().clone(), dtype);
    float(&out)?;
    let n = dy.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        gate_gradient::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            dy.into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            (x.meta.shape()[1] / 2) as u32,
            activation.code(),
            dtype.into(),
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn gate_gradient<O: Float>(
    x: &Array<f32>,
    dy: &Array<f32>,
    out: &mut Array<O>,
    n: u32,
    width: u32,
    #[comptime] activation: u32,
    #[define(O)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / width as usize;
        let col = i % width as usize;
        let a = x[row * width as usize * 2 + col];
        let b = x[row * width as usize * 2 + width as usize + col];
        let mut derivative: f32 = 1.0f32;
        if comptime!(activation == 1) {
            let s = 1.0 / (1.0 + (-a).exp());
            derivative = s * (1.0 + a * (1.0 - s));
        }
        if comptime!(activation == 2) {
            let activated = crate::inference::activation::apply(a, 2u32);
            derivative = 0.5;
            if a != 0.0 {
                derivative = activated / a;
            }
            derivative += a * 0.3989422804014327 * (-0.5 * a * a).exp();
        }
        if comptime!(activation == 3) {
            let t = (0.7978845608028654 * (a + 0.044715 * a * a * a)).tanh();
            derivative = 0.5 * (1.0 + t)
                + 0.5 * a * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * a * a);
        }
        if comptime!(activation == 4) {
            derivative = 0.0;
            if a > 0.0 {
                derivative = 1.0;
            }
        }
        if comptime!(activation == 5) {
            derivative = 2.0 * f32::max(a, 0.0);
        }
        out[row * width as usize * 2 + col] = O::cast_from(dy[i] * b * derivative);
        out[row * width as usize * 2 + width as usize + col] =
            O::cast_from(dy[i] * crate::inference::activation::apply(a, activation));
    }
}

pub struct Qkv<R: Runtime> {
    pub query: RudaTensor<R>,
    pub key: RudaTensor<R>,
    pub value: RudaTensor<R>,
}
/// Q/K head-wise RMSNorm and RoPE; V is packed independently, never rotated.
pub fn normalize_rotary_qkv<R: Runtime>(
    q: &RudaTensor<R>,
    k: &RudaTensor<R>,
    v: &RudaTensor<R>,
    q_weight: &RudaTensor<R>,
    k_weight: &RudaTensor<R>,
    positions: &RudaTensor<R>,
    epsilon: f32,
    rope: super::rope::RopeOptions,
) -> Result<Qkv<R>, InferenceError> {
    let q = super::norm::rmsnorm(q, q_weight, epsilon)?;
    let k = super::norm::rmsnorm(k, k_weight, epsilon)?;
    let pair = super::rope::apply_rope(&q, &k, positions, rope)?;
    Ok(Qkv {
        query: pair.query,
        key: pair.key,
        value: into_contiguous(v.clone()),
    })
}
/// Shared MLA positional keys are broadcast over heads during FP8 packing.
pub fn concat_mla_kv_fp8<R: Runtime>(
    kv_nope: &RudaTensor<R>,
    k_pe: &RudaTensor<R>,
    nope_width: usize,
) -> Result<(RudaTensor<R>, RudaTensor<R>), InferenceError> {
    let kv = super::norm::cast(kv_nope, DType::F32)?;
    let pe = super::norm::cast(k_pe, DType::F32)?;
    check(&pe, &kv)?;
    if kv.meta.num_dims() != 3
        || pe.meta.num_dims() != 2
        || pe.meta.shape()[0] != kv.meta.shape()[0]
        || nope_width == 0
        || nope_width >= kv.meta.shape()[2]
    {
        return Err(InferenceError(
            "MLA pack expects [T,H,K+V] and shared [T,R]",
        ));
    }
    let tokens = kv.meta.shape()[0];
    let heads = kv.meta.shape()[1];
    let vwidth = kv.meta.shape()[2] - nope_width;
    let r = pe.meta.shape()[1];
    let key = alloc(&kv, [tokens, heads, nope_width + r], DType::U8);
    let value = alloc(&kv, [tokens, heads, vwidth], DType::U8);
    let n = key.meta.num_elements().max(value.meta.num_elements());
    if n > u32::MAX as usize {
        return Err(InferenceError("MLA packed output exceeds U32"));
    }
    if n > 0 {
        let dim = RudaDim::new(kv.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&kv.client, n, dim);
        mla_pack::launch::<R>(
            &kv.client,
            count,
            dim,
            kv.clone().into_array_arg(),
            pe.into_array_arg(),
            key.clone().into_array_arg(),
            value.clone().into_array_arg(),
            tokens as u32,
            heads as u32,
            nope_width as u32,
            r as u32,
            vwidth as u32,
        );
    }
    Ok((key, value))
}
#[ruda(launch)]
pub fn mla_pack(
    kv: &Array<f32>,
    pe: &Array<f32>,
    key: &mut Array<u8>,
    value: &mut Array<u8>,
    tokens: u32,
    heads: u32,
    nope: u32,
    rope: u32,
    vwidth: u32,
) {
    let i = ABSOLUTE_POS as usize;
    let kw = (nope + rope) as usize;
    let source = (nope + vwidth) as usize;
    if i < tokens as usize * heads as usize * kw {
        let row = i / kw;
        let col = i % kw;
        let mut x: f32 = 0.0f32;
        if col < nope as usize {
            x = kv[row * source + col];
        } else {
            x = pe[row / heads as usize * rope as usize + col - nope as usize];
        }
        key[i] = u8::cast_from(rublas::tensor_float_quant::fp8_encode(x, false));
    }
    if i < tokens as usize * heads as usize * vwidth as usize {
        let row = i / vwidth as usize;
        let col = i % vwidth as usize;
        value[i] = u8::cast_from(rublas::tensor_float_quant::fp8_encode(
            kv[row * source + nope as usize + col],
            false,
        ));
    }
}

/// Retained RMS/Layer/add-residual normalization and block quantization. All
/// weight/residual producers are native kernels; no rounded BF16 intermediate
/// is inserted between normalization and the final quantization boundary.
pub struct PreparedNormQuant<R: Runtime> {
    input: super::PreparedPack<R>,
    parameters: Vec<(super::PreparedPack<R>, RudaTensor<R>)>,
    residual: Option<super::norm::PreparedBroadcast<R>>,
    norm: super::norm::PreparedNorm<R>,
    pub output: QuantizedNorm<R>,
}
impl<R: Runtime> PreparedNormQuant<R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        weight: Option<&RudaTensor<R>>,
        bias: Option<&RudaTensor<R>>,
        residual: Option<&RudaTensor<R>>,
        options: super::norm::NormOptions,
        recipe: QuantOptions,
        global: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        let input_pack = super::PreparedPack::prepare(input)?;
        let input = &input_pack.output;
        let mut parameters = Vec::new();
        let mut parameter = |p: &RudaTensor<R>| -> Result<RudaTensor<R>, InferenceError> {
            super::check_view(p, input)?;
            float(p)?;
            let pack = super::PreparedPack::prepare(p)?;
            let out = alloc(input, p.meta.shape().clone(), DType::F32);
            parameters.push((pack, out.clone()));
            Ok(out)
        };
        let weight = weight.map(&mut parameter).transpose()?;
        let bias = bias.map(&mut parameter).transpose()?;
        let residual = residual
            .map(|p| super::norm::PreparedBroadcast::prepare(p, input))
            .transpose()?;
        let norm = super::norm::PreparedNorm::prepare(
            input,
            weight.as_ref(),
            bias.as_ref(),
            residual.as_ref().map(|p| &p.output),
            None,
            super::norm::NormOptions {
                output_dtype: Some(DType::F32),
                ..options
            },
        )?;
        let values = rublas::tensor_float_quant::allocate_quantized(&norm.output, global, recipe)
            .map_err(|e| InferenceError(e.0))?;
        let output = QuantizedNorm {
            values,
            residual: residual.as_ref().map(|_| norm.residual_output.clone()),
        };
        Ok(Self {
            input: input_pack,
            parameters,
            residual,
            norm,
            output,
        })
    }
    pub unsafe fn run(&self) -> Result<&QuantizedNorm<R>, InferenceError> {
        unsafe {
            self.input.run();
            for (p, out) in &self.parameters {
                p.run();
                super::norm::cast_into(&p.output, out)?;
            }
            if let Some(p) = &self.residual {
                p.run();
            }
            self.norm.run();
            self.output
                .values
                .quantize_into(&self.norm.output, None)
                .map_err(|e| InferenceError(e.0))?;
        }
        Ok(&self.output)
    }
}
/// Retained SiLU/GELU gated forward or backward and row/column block codecs.
pub struct PreparedGatedQuant<R: Runtime> {
    input: super::PreparedPack<R>,
    input_f32: RudaTensor<R>,
    grad: Option<super::PreparedPack<R>>,
    grad_f32: Option<RudaTensor<R>>,
    source: RudaTensor<R>,
    transposed: Option<RudaTensor<R>>,
    pub output: OrientedQuantized<R>,
    activation: super::activation::Activation,
}
impl<R: Runtime> PreparedGatedQuant<R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        grad: Option<&RudaTensor<R>>,
        activation: super::activation::Activation,
        rowwise: bool,
        colwise: bool,
        recipe: QuantOptions,
        global: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        let input = super::PreparedPack::prepare(input)?;
        let x = &input.output;
        float(x)?;
        if x.meta.num_dims() != 2
            || x.meta.shape()[1] == 0
            || x.meta.shape()[1] % 2 != 0
            || !rowwise && !colwise
        {
            return Err(InferenceError(
                "gated quantization needs [M,2K] and an orientation",
            ));
        }
        let (m, k) = (x.meta.shape()[0], x.meta.shape()[1] / 2);
        let grad = grad.map(super::PreparedPack::prepare).transpose()?;
        let grad_f32 = if let Some(g) = &grad {
            check(&g.output, x)?;
            float(&g.output)?;
            if g.output.meta.shape()[..] != [m, k] {
                return Err(InferenceError("gated backward gradient must be [M,K]"));
            }
            Some(alloc(x, [m, k], DType::F32))
        } else {
            None
        };
        let input_f32 = alloc(x, x.meta.shape().clone(), DType::F32);
        let source = alloc(x, [m, if grad.is_some() { 2 * k } else { k }], DType::F32);
        let transposed = if colwise {
            Some(alloc(x, [source.meta.shape()[1], m], DType::F32))
        } else {
            None
        };
        let rowwise = if rowwise {
            Some(
                rublas::tensor_float_quant::allocate_quantized(&source, global, recipe)
                    .map_err(|e| InferenceError(e.0))?,
            )
        } else {
            None
        };
        let colwise = transposed
            .as_ref()
            .map(|p| {
                rublas::tensor_float_quant::allocate_quantized(p, global, recipe)
                    .map_err(|e| InferenceError(e.0))
            })
            .transpose()?;
        Ok(Self {
            input,
            input_f32,
            grad,
            grad_f32,
            source,
            transposed,
            output: OrientedQuantized { rowwise, colwise },
            activation,
        })
    }
    pub unsafe fn run(&self) -> Result<&OrientedQuantized<R>, InferenceError> {
        unsafe {
            self.input.run();
            super::norm::cast_into(&self.input.output, &self.input_f32)?;
        }
        let x = &self.input_f32;
        let m = x.meta.shape()[0];
        let k = x.meta.shape()[1] / 2;
        if let (Some(g), Some(dy)) = (&self.grad, &self.grad_f32) {
            unsafe {
                g.run();
                super::norm::cast_into(&g.output, dy)?;
            }
            let n = m * k;
            if n > 0 {
                let dim = RudaDim::new(x.client.properties(), n);
                let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
                gate_gradient::launch::<R>(
                    &x.client,
                    count,
                    dim,
                    x.clone().into_array_arg(),
                    dy.clone().into_array_arg(),
                    self.source.clone().into_array_arg(),
                    n as u32,
                    k as u32,
                    self.activation.code(),
                    DType::F32.into(),
                );
            }
        } else {
            unsafe {
                super::activation::gated_into(x, self.activation, &self.source)?;
            }
        }
        if let Some(q) = &self.output.rowwise {
            unsafe {
                q.quantize_into(&self.source, None)
                    .map_err(|e| InferenceError(e.0))?;
            }
        }
        if let (Some(t), Some(q)) = (&self.transposed, &self.output.colwise) {
            let n = self.source.meta.num_elements();
            if n > 0 {
                let dim = RudaDim::new(x.client.properties(), n);
                let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
                transpose_kernel::launch::<R>(
                    &x.client,
                    count,
                    dim,
                    self.source.clone().into_array_arg(),
                    t.clone().into_array_arg(),
                    self.source.meta.shape()[0] as u32,
                    self.source.meta.shape()[1] as u32,
                    DType::F32.into(),
                );
            }
            unsafe {
                q.quantize_into(t, None).map_err(|e| InferenceError(e.0))?;
            }
        }
        Ok(&self.output)
    }
}
