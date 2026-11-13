//! Native AdaLN/quantized-QKV/head-norm/RoPE/destination-pack pipeline.
use super::{
    InferenceError, alloc, check, float,
    fused::Qkv,
    linear::{LinearOptions, Matrix},
};
use rublas::tensor_float_quant::{BlockFloatTensor, QuantOptions};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionOrder {
    HeadKindFeature,
    KindHeadFeature,
}
pub struct PreAttentionParameters<'a, R: Runtime> {
    pub input_weight: &'a RudaTensor<R>,
    pub adaln_scale: &'a RudaTensor<R>,
    pub adaln_shift: &'a RudaTensor<R>,
    pub adaln_indices: &'a RudaTensor<R>,
    pub projection: Matrix<'a, R>,
    pub query_weight: &'a RudaTensor<R>,
    pub key_weight: &'a RudaTensor<R>,
    pub cos_sin: &'a RudaTensor<R>,
    pub positions: Option<&'a RudaTensor<R>>,
    pub qk_epsilon: Option<f32>,
    pub heads: usize,
    pub head_dim: usize,
    pub epsilon: f32,
    pub order: ProjectionOrder,
    pub activation_fp8: Option<super::linear::Fp8Recipe>,
    pub activation_recipe: Option<QuantOptions>,
    pub activation_decoder_scale: Option<&'a RudaTensor<R>>,
}
pub struct PreAttentionOutput<R: Runtime> {
    pub qkv: Qkv<R>,
    pub activation: Option<BlockFloatTensor<R>>,
    pub activation_fp8: Option<super::linear::ScaledFp8<R>>,
}
/// Caller-retained buffers for allocation-free graph recording/replay. Weight
/// decoders and normalization parameters reread their bound buffers each run.
pub struct PreparedPreAttention<'a, R: Runtime> {
    input_norm: super::norm::PreparedNorm<R>,
    query_norm: super::norm::PreparedNorm<R>,
    key_norm: super::norm::PreparedNorm<R>,
    parameters: Vec<(RudaTensor<R>, RudaTensor<R>)>,
    scale: RudaTensor<R>,
    shift: RudaTensor<R>,
    indices: RudaTensor<R>,
    modulated: RudaTensor<R>,
    pub activation: Option<BlockFloatTensor<R>>,
    pub activation_fp8: Option<super::linear::ScaledFp8<R>>,
    activation_f32: RudaTensor<R>,
    projection: super::linear::PreparedMatrix<'a, R>,
    projected_f32: RudaTensor<R>,
    projected: RudaTensor<R>,
    split: Qkv<R>,
    rotated: Qkv<R>,
    table: RudaTensor<R>,
    positions: RudaTensor<R>,
    positions_pack: Option<super::metadata::PreparedIndices<R>>,
    packed: RudaTensor<R>,
    pub output: Option<BlockFloatTensor<R>>,
    destinations: usize,
    order: ProjectionOrder,
}
impl<'a, R: Runtime> PreparedPreAttention<'a, R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        p: PreAttentionParameters<'a, R>,
        destinations: usize,
        output_recipe: QuantOptions,
        output_decoder: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        Self::bind(input, p, destinations, Some(output_recipe), output_decoder)
    }
    pub fn prepare_dense(
        input: &RudaTensor<R>,
        p: PreAttentionParameters<'a, R>,
        destinations: usize,
    ) -> Result<Self, InferenceError> {
        Self::bind(input, p, destinations, None, None)
    }
    fn bind(
        input: &RudaTensor<R>,
        p: PreAttentionParameters<'a, R>,
        destinations: usize,
        output_recipe: Option<QuantOptions>,
        output_decoder: Option<&RudaTensor<R>>,
    ) -> Result<Self, InferenceError> {
        check(input, input)?;
        float(input)?;
        if input.meta.num_dims() != 2
            || p.heads == 0
            || p.head_dim == 0
            || destinations == 0
            || p.heads % destinations != 0
        {
            return Err(InferenceError("prepared pre-attention geometry invalid"));
        }
        let tokens = input.meta.shape()[0];
        let hidden = input.meta.shape()[1];
        let n = p
            .heads
            .checked_mul(3)
            .and_then(|n| n.checked_mul(p.head_dim))
            .ok_or(InferenceError("QKV projected width overflow"))?;
        let mut parameters = Vec::new();
        let mut parameter = |source: &RudaTensor<R>| -> Result<RudaTensor<R>, InferenceError> {
            check(source, input)?;
            float(source)?;
            let out = alloc(input, source.meta.shape().clone(), DType::F32);
            parameters.push((source.clone(), out.clone()));
            Ok(out)
        };
        let input_weight = parameter(p.input_weight)?;
        let qw = parameter(p.query_weight)?;
        let kw = parameter(p.key_weight)?;
        let scale = parameter(p.adaln_scale)?;
        let shift = parameter(p.adaln_shift)?;
        let table = parameter(p.cos_sin)?;
        if input_weight.meta.shape()[..] != [hidden]
            || qw.meta.shape()[..] != [p.head_dim]
            || kw.meta.shape()[..] != [p.head_dim]
            || scale.meta.num_dims() != 2
            || scale.meta.shape()[1] != hidden
            || shift.meta.shape() != scale.meta.shape()
            || p.adaln_indices.dtype != DType::I32
            || p.adaln_indices.meta.shape()[..] != [tokens]
            || table.meta.num_dims() != 2
            || (p.positions.is_none() && table.meta.shape()[0] < tokens)
            || table.meta.shape()[1] == 0
            || table.meta.shape()[1] > p.head_dim
            || table.meta.shape()[1] % 2 != 0
        {
            return Err(InferenceError("prepared pre-attention parameters mismatch"));
        }
        check(p.adaln_indices, input)?;
        let input_norm = super::norm::PreparedNorm::prepare(
            input,
            Some(&input_weight),
            None,
            None,
            None,
            super::norm::NormOptions {
                epsilon: p.epsilon,
                ..Default::default()
            },
        )?;
        let modulated = alloc(input, input.meta.shape().clone(), input.dtype);
        if p.activation_recipe.is_some() && p.activation_fp8.is_some() {
            return Err(InferenceError(
                "activation quantization recipes are mutually exclusive",
            ));
        }
        let activation_fp8 = p
            .activation_fp8
            .map(|r| {
                super::linear::ScaledFp8::allocate(&modulated, r.block_rows, r.block_cols, r.format)
            })
            .transpose()?;
        let activation = p
            .activation_recipe
            .map(|recipe| {
                rublas::tensor_float_quant::allocate_quantized(
                    &modulated,
                    p.activation_decoder_scale,
                    recipe,
                )
                .map_err(|e| InferenceError(e.0))
            })
            .transpose()?;
        let activation_f32 = alloc(input, input.meta.shape().clone(), DType::F32);
        let projection = super::linear::PreparedMatrix::prepare(p.projection)?;
        if projection.output.meta.shape()[..] != [n, hidden] {
            return Err(InferenceError(
                "prepared QKV projection weight must be [3*H*D,K]",
            ));
        }
        let projected_f32 = alloc(input, [tokens, n], DType::F32);
        let projected = alloc(input, [tokens, n], input.dtype);
        let split = Qkv {
            query: alloc(input, [tokens, p.heads, p.head_dim], input.dtype),
            key: alloc(input, [tokens, p.heads, p.head_dim], input.dtype),
            value: alloc(input, [tokens, p.heads, p.head_dim], input.dtype),
        };
        let query_norm = super::norm::PreparedNorm::prepare(
            &split.query,
            Some(&qw),
            None,
            None,
            None,
            super::norm::NormOptions {
                epsilon: p.qk_epsilon.unwrap_or(p.epsilon),
                ..Default::default()
            },
        )?;
        let key_norm = super::norm::PreparedNorm::prepare(
            &split.key,
            Some(&kw),
            None,
            None,
            None,
            super::norm::NormOptions {
                epsilon: p.qk_epsilon.unwrap_or(p.epsilon),
                ..Default::default()
            },
        )?;
        let rotated = Qkv {
            query: alloc(input, [tokens, p.heads, p.head_dim], input.dtype),
            key: alloc(input, [tokens, p.heads, p.head_dim], input.dtype),
            value: split.value.clone(),
        };
        if tokens > i32::MAX as usize {
            return Err(InferenceError("prepared RoPE positions exceed I32"));
        }
        let positions_pack = p
            .positions
            .map(|t| super::metadata::PreparedIndices::prepare(t, DType::I32))
            .transpose()?;
        let positions = if let Some(plan) = &positions_pack {
            check(&plan.output, input)?;
            if plan.output.meta.shape()[..] != [tokens] {
                return Err(InferenceError("prepared RoPE position IDs must be [T]"));
            }
            plan.output.clone()
        } else {
            let ids: Vec<i32> = (0..tokens as i32).collect();
            RudaTensor::new_contiguous(
                input.client.clone(),
                input.device.clone(),
                ruda_core::tensor::Shape::from([tokens]),
                input.client.create_from_slice(bytemuck::cast_slice(&ids)),
                DType::I32,
            )
        };
        let packed = alloc(
            input,
            [
                destinations,
                tokens * (p.heads / destinations) * 3,
                p.head_dim,
            ],
            DType::F32,
        );
        let output = output_recipe
            .map(|recipe| {
                rublas::tensor_float_quant::allocate_quantized(&packed, output_decoder, recipe)
                    .map_err(|e| InferenceError(e.0))
            })
            .transpose()?;

        Ok(Self {
            input_norm,
            query_norm,
            key_norm,
            parameters,
            scale,
            shift,
            indices: p.adaln_indices.clone(),
            modulated,
            activation,
            activation_fp8,
            activation_f32,
            projection,
            projected_f32,
            projected,
            split,
            rotated,
            table,
            positions,
            positions_pack,
            packed,
            output,
            destinations,
            order: p.order,
        })
    }
    /// Every output/workspace is overwritten. Bound parameters, inputs and
    /// outputs must not be accessed concurrently on another stream.
    pub unsafe fn run_qkv(&self) -> Result<&Qkv<R>, InferenceError> {
        for (source, destination) in &self.parameters {
            unsafe {
                super::norm::cast_into(source, destination)?;
            }
        }
        unsafe {
            self.input_norm.run();
            adaln_into(
                &self.input_norm.output,
                &self.scale,
                &self.shift,
                &self.indices,
                &self.modulated,
            )?;
            if let Some(a) = &self.activation {
                a.quantize_into(&self.modulated, None)
                    .map_err(|e| InferenceError(e.0))?;
                a.dequantize_into(&self.activation_f32)
                    .map_err(|e| InferenceError(e.0))?;
            } else {
                super::norm::cast_into(&self.modulated, &self.activation_f32)?;
                if let Some(a) = &self.activation_fp8 {
                    a.quantize_into(&self.activation_f32)?;
                    a.decode_into(&self.activation_f32)?;
                }
            }
            if let Some(p) = &self.positions_pack {
                p.run();
            }
            self.projection.run()?;
        }
        let weight =
            ruda_kernel::tensor::permutation::swap_dims(self.projection.output.clone(), 0, 1);
        rublas::tensor_matmul::matmul(
            self.activation_f32.clone(),
            weight,
            Some(self.projected_f32.clone()),
            rublas::tensor_matmul::MatmulStrategy::Naive,
            DType::F32,
        )
        .map_err(|_| InferenceError("prepared native QKV GEMM failed"))?;
        unsafe {
            super::norm::cast_into(&self.projected_f32, &self.projected)?;
            split_qkv_into(&self.projected, &self.split, self.order)?;
            self.query_norm.run();
            self.key_norm.run();
        }
        let rope = super::rope::RopeOptions {
            rotary_dim: self.table.meta.shape()[1],
            interleaved: false,
            theta: 10000.0,
            scale: 1.0,
            llama31: None,
        };
        unsafe {
            super::rope::apply_one_into(
                &self.query_norm.output,
                &self.positions,
                rope,
                Some(&self.table),
                &self.rotated.query,
            )?;
            super::rope::apply_one_into(
                &self.key_norm.output,
                &self.positions,
                rope,
                Some(&self.table),
                &self.rotated.key,
            )?;
        }
        Ok(&self.rotated)
    }
    pub unsafe fn run(&self) -> Result<&BlockFloatTensor<R>, InferenceError> {
        let out = self.output.as_ref().ok_or(InferenceError(
            "dense prepared pipeline has no block-quantized output",
        ))?;
        unsafe {
            self.run_qkv()?;
            pack_into(&self.rotated, self.destinations, &self.packed)?;
            out.quantize_into(&self.packed, None)
                .map_err(|e| InferenceError(e.0))?;
        }
        Ok(out)
    }
    /// BF16/F16/F32 destination-major dense send buffer. No quantization is
    /// introduced when activation_recipe is None.
    pub unsafe fn run_dense_into(&self, out: &RudaTensor<R>) -> Result<(), InferenceError> {
        unsafe {
            self.run_qkv()?;
            pack_into(&self.rotated, self.destinations, &self.packed)?;
            super::norm::cast_into(&self.packed, out)?;
        }
        Ok(())
    }
}
/// AdaLN retains the documented BF16 rounding boundaries before quantization.
pub fn pre_attention<R: Runtime>(
    input: &RudaTensor<R>,
    p: PreAttentionParameters<'_, R>,
) -> Result<PreAttentionOutput<R>, InferenceError> {
    check(input, input)?;
    float(input)?;
    if input.meta.num_dims() != 2 || p.heads == 0 || p.head_dim == 0 {
        return Err(InferenceError("pre-attention input/heads invalid"));
    }
    let norm = super::norm::rmsnorm(input, p.input_weight, p.epsilon)?;
    let modulated = adaln(&norm, p.adaln_scale, p.adaln_shift, p.adaln_indices)?;
    if p.activation_recipe.is_some() && p.activation_fp8.is_some() {
        return Err(InferenceError(
            "activation quantization recipes are mutually exclusive",
        ));
    }
    let activation = p
        .activation_recipe
        .map(|recipe| {
            rublas::tensor_float_quant::quantize(
                &modulated,
                p.activation_decoder_scale,
                None,
                recipe,
            )
            .map_err(|e| InferenceError(e.0))
        })
        .transpose()?;
    let activation_fp8 = p
        .activation_fp8
        .map(|r| {
            super::linear::quantize_fp8_blocks(&modulated, r.block_rows, r.block_cols, r.format)
        })
        .transpose()?;
    let matrix = if let Some(a) = &activation {
        Matrix::BlockFloat(a)
    } else if let Some(a) = &activation_fp8 {
        a.matrix()
    } else {
        Matrix::Dense(&modulated)
    };
    let projected = super::linear::matmul(
        matrix,
        p.projection,
        None,
        LinearOptions {
            output_dtype: input.dtype,
            ..Default::default()
        },
    )?;
    let qkv = split_qkv(&projected, p.heads, p.head_dim, p.order)?;
    let q = super::norm::rmsnorm(
        &qkv.query,
        p.query_weight,
        p.qk_epsilon.unwrap_or(p.epsilon),
    )?;
    let k = super::norm::rmsnorm(&qkv.key, p.key_weight, p.qk_epsilon.unwrap_or(p.epsilon))?;
    let table = super::norm::cast(p.cos_sin, DType::F32)?;
    if table.meta.num_dims() != 2
        || (p.positions.is_none() && table.meta.shape()[0] < input.meta.shape()[0])
        || table.meta.shape()[1] == 0
        || table.meta.shape()[1] > p.head_dim
        || table.meta.shape()[1] % 2 != 0
    {
        return Err(InferenceError(
            "pre-attention RoPE cache must be [T,rotary_dim]",
        ));
    }
    let tokens = input.meta.shape()[0];
    let ids = if let Some(ids) = p.positions {
        let ids = super::metadata::cast_indices(ids, DType::I32)?;
        if ids.meta.shape()[..] != [tokens] {
            return Err(InferenceError("RoPE position IDs must be [T]"));
        }
        ids
    } else {
        let ids = alloc(input, [tokens], DType::I32);
        if tokens > 0 {
            let dim = RudaDim::new(input.client.properties(), tokens);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, tokens, dim);
            positions::launch::<R>(
                &input.client,
                count,
                dim,
                ids.clone().into_array_arg(),
                tokens as u32,
            );
        }
        ids
    };
    let rotated = super::rope::apply_rope_with_cos_sin_cache(
        &q,
        &k,
        &ids,
        &table,
        table.meta.shape()[1],
        false,
    )?;
    Ok(PreAttentionOutput {
        qkv: Qkv {
            query: rotated.query,
            key: rotated.key,
            value: qkv.value,
        },
        activation,
        activation_fp8,
    })
}
pub fn adaln<R: Runtime>(
    normalized: &RudaTensor<R>,
    scale_table: &RudaTensor<R>,
    shift_table: &RudaTensor<R>,
    indices: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    check(normalized, normalized)?;
    float(normalized)?;
    let scale = super::norm::cast(scale_table, DType::F32)?;
    let shift = super::norm::cast(shift_table, DType::F32)?;
    for t in [&scale, &shift, indices] {
        check(t, normalized)?;
    }
    if normalized.meta.num_dims() != 2
        || scale.meta.num_dims() != 2
        || scale.meta.shape() != shift.meta.shape()
        || scale.meta.shape()[1] != normalized.meta.shape()[1]
        || indices.dtype != DType::I32
        || indices.meta.shape()[..] != [normalized.meta.shape()[0]]
    {
        return Err(InferenceError(
            "AdaLN requires X[T,D], scale/shift[rows,D], I32 indices[T]",
        ));
    }
    let out = alloc(
        normalized,
        normalized.meta.shape().clone(),
        normalized.dtype,
    );
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(out.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
        adaln_values::launch::<R>(
            &out.client,
            count,
            dim,
            normalized.clone().into_array_arg(),
            scale.into_array_arg(),
            shift.into_array_arg(),
            indices.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            normalized.meta.shape()[1] as u32,
            scale_table.meta.shape()[0] as u32,
            normalized.dtype.into(),
        );
    }
    Ok(out)
}
pub fn split_qkv<R: Runtime>(
    projected: &RudaTensor<R>,
    heads: usize,
    width: usize,
    order: ProjectionOrder,
) -> Result<Qkv<R>, InferenceError> {
    check(projected, projected)?;
    float(projected)?;
    if projected.meta.num_dims() != 2
        || heads == 0
        || width == 0
        || projected.meta.shape()[1] != heads * width * 3
    {
        return Err(InferenceError("QKV split expects [T,3*H*D]"));
    }
    let tokens = projected.meta.shape()[0];
    let q = alloc(projected, [tokens, heads, width], projected.dtype);
    let k = alloc(projected, [tokens, heads, width], projected.dtype);
    let v = alloc(projected, [tokens, heads, width], projected.dtype);
    let n = q.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(projected.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&projected.client, n, dim);
        split::launch::<R>(
            &projected.client,
            count,
            dim,
            projected.clone().into_array_arg(),
            q.clone().into_array_arg(),
            k.clone().into_array_arg(),
            v.clone().into_array_arg(),
            n as u32,
            heads as u32,
            width as u32,
            order == ProjectionOrder::HeadKindFeature,
            projected.dtype.into(),
        );
    }
    Ok(Qkv {
        query: q,
        key: k,
        value: v,
    })
}
/// Destination-major [P,T,H/P,3,D], with one independent scale matrix per P.
/// The BlockFloatTensor's logical matrix is [P,T*(H/P)*3,D].
pub fn quantize_pack<R: Runtime>(
    qkv: &Qkv<R>,
    destinations: usize,
    recipe: QuantOptions,
    global_decoder: Option<&RudaTensor<R>>,
) -> Result<BlockFloatTensor<R>, InferenceError> {
    let packed = pack(qkv, destinations)?;
    rublas::tensor_float_quant::quantize(&packed, global_decoder, None, recipe)
        .map_err(|e| InferenceError(e.0))
}
pub fn pack<R: Runtime>(
    qkv: &Qkv<R>,
    destinations: usize,
) -> Result<RudaTensor<R>, InferenceError> {
    let q = super::norm::cast(&qkv.query, DType::F32)?;
    let k = super::norm::cast(&qkv.key, DType::F32)?;
    let v = super::norm::cast(&qkv.value, DType::F32)?;
    for t in [&k, &v] {
        check(t, &q)?;
    }
    if q.meta.num_dims() != 3
        || q.meta.shape() != k.meta.shape()
        || q.meta.shape() != v.meta.shape()
        || destinations == 0
        || q.meta.shape()[1] % destinations != 0
    {
        return Err(InferenceError(
            "QKV destination pack needs equal [T,H,D], P dividing H",
        ));
    }
    let tokens = q.meta.shape()[0];
    let heads = q.meta.shape()[1];
    let width = q.meta.shape()[2];
    let per = heads / destinations;
    let out = alloc(&q, [destinations, tokens * per * 3, width], DType::F32);
    let n = out.meta.num_elements();
    if n > u32::MAX as usize {
        return Err(InferenceError("QKV send buffer exceeds U32"));
    }
    if n > 0 {
        let dim = RudaDim::new(q.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
        destination_pack::launch::<R>(
            &q.client,
            count,
            dim,
            q.clone().into_array_arg(),
            k.into_array_arg(),
            v.into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            tokens as u32,
            heads as u32,
            width as u32,
            per as u32,
        );
    }
    Ok(out)
}
pub unsafe fn adaln_into<R: Runtime>(
    x: &RudaTensor<R>,
    scale: &RudaTensor<R>,
    shift: &RudaTensor<R>,
    indices: &RudaTensor<R>,
    out: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    for t in [x, scale, shift, indices, out] {
        check(t, x)?;
    }
    float(x)?;
    if x.meta.num_dims() != 2
        || scale.dtype != DType::F32
        || shift.dtype != DType::F32
        || scale.meta.num_dims() != 2
        || scale.meta.shape() != shift.meta.shape()
        || scale.meta.shape()[1] != x.meta.shape()[1]
        || indices.dtype != DType::I32
        || indices.meta.shape()[..] != [x.meta.shape()[0]]
        || out.dtype != x.dtype
        || out.meta.shape() != x.meta.shape()
    {
        return Err(InferenceError("preallocated AdaLN operands mismatch"));
    }
    let n = x.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        adaln_values::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            scale.clone().into_array_arg(),
            shift.clone().into_array_arg(),
            indices.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            x.meta.shape()[1] as u32,
            scale.meta.shape()[0] as u32,
            x.dtype.into(),
        );
    }
    Ok(())
}
pub unsafe fn split_qkv_into<R: Runtime>(
    input: &RudaTensor<R>,
    out: &Qkv<R>,
    order: ProjectionOrder,
) -> Result<(), InferenceError> {
    for t in [input, &out.query, &out.key, &out.value] {
        check(t, input)?;
        float(t)?;
    }
    if input.meta.num_dims() != 2
        || out.query.meta.num_dims() != 3
        || out.query.meta.shape() != out.key.meta.shape()
        || out.query.meta.shape() != out.value.meta.shape()
        || out.query.dtype != input.dtype
        || out.key.dtype != input.dtype
        || out.value.dtype != input.dtype
        || out.query.meta.shape()[0] != input.meta.shape()[0]
        || out.query.meta.shape()[1] * out.query.meta.shape()[2] * 3 != input.meta.shape()[1]
    {
        return Err(InferenceError("preallocated QKV split shape mismatch"));
    }
    let n = out.query.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(input.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
        split::launch::<R>(
            &input.client,
            count,
            dim,
            input.clone().into_array_arg(),
            out.query.clone().into_array_arg(),
            out.key.clone().into_array_arg(),
            out.value.clone().into_array_arg(),
            n as u32,
            out.query.meta.shape()[1] as u32,
            out.query.meta.shape()[2] as u32,
            order == ProjectionOrder::HeadKindFeature,
            input.dtype.into(),
        );
    }
    Ok(())
}
pub unsafe fn pack_into<R: Runtime>(
    qkv: &Qkv<R>,
    destinations: usize,
    out: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    let q = &qkv.query;
    let k = &qkv.key;
    let v = &qkv.value;
    for t in [q, k, v, out] {
        check(t, q)?;
    }
    float(q)?;
    if q.meta.num_dims() != 3
        || q.meta.shape() != k.meta.shape()
        || q.meta.shape() != v.meta.shape()
        || q.dtype != k.dtype
        || q.dtype != v.dtype
        || out.dtype != DType::F32
        || destinations == 0
        || q.meta.shape()[1] % destinations != 0
        || out.meta.shape()[..]
            != [
                destinations,
                q.meta.shape()[0] * (q.meta.shape()[1] / destinations) * 3,
                q.meta.shape()[2],
            ]
    {
        return Err(InferenceError("preallocated destination QKV pack mismatch"));
    }
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(q.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
        destination_pack_generic::launch::<R>(
            &q.client,
            count,
            dim,
            q.clone().into_array_arg(),
            k.clone().into_array_arg(),
            v.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            q.meta.shape()[0] as u32,
            q.meta.shape()[1] as u32,
            q.meta.shape()[2] as u32,
            (q.meta.shape()[1] / destinations) as u32,
            q.dtype.into(),
        );
    }
    Ok(())
}
#[ruda(launch)]
pub fn positions(out: &mut Array<i32>, n: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = i as i32;
    }
}
#[ruda(launch)]
pub fn adaln_values<F: Float>(
    x: &Array<F>,
    scale: &Array<f32>,
    shift: &Array<f32>,
    indices: &Array<i32>,
    out: &mut Array<F>,
    n: u32,
    width: u32,
    table_rows: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let token = i / width as usize;
        let d = i % width as usize;
        let row = indices[token];
        let mut value: f32 = 0.0f32;
        if row >= 0 && row < table_rows as i32 {
            let offset = row as usize * width as usize + d;
            let gain = f32::cast_from(F::cast_from(1.0 + scale[offset]));
            value = fma(f32::cast_from(x[i]), gain, shift[offset]);
        }
        out[i] = F::cast_from(value);
    }
}
#[ruda(launch)]
pub fn split<F: Float>(
    input: &Array<F>,
    q: &mut Array<F>,
    k: &mut Array<F>,
    v: &mut Array<F>,
    n: u32,
    heads: u32,
    width: u32,
    #[comptime] head_major: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let token = i / (heads as usize * width as usize);
        let head = i / width as usize % heads as usize;
        let d = i % width as usize;
        let mut source = token * heads as usize * 3 * width as usize + head * width as usize + d;
        let mut stride = heads as usize * width as usize;
        if comptime!(head_major) {
            source = ((token * heads as usize + head) * 3) * width as usize + d;
            stride = width as usize;
        }
        q[i] = input[source];
        k[i] = input[source + stride];
        v[i] = input[source + stride * 2];
    }
}
#[ruda(launch)]
pub fn destination_pack(
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    tokens: u32,
    heads: u32,
    width: u32,
    per: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let d = i % width as usize;
        let kind = i / width as usize % 3;
        let head = i / (width as usize * 3) % per as usize;
        let token = i / (width as usize * 3 * per as usize) % tokens as usize;
        let destination = i / (width as usize * 3 * per as usize * tokens as usize);
        let source =
            (token * heads as usize + destination * per as usize + head) * width as usize + d;
        let mut value = q[source];
        if kind == 1 {
            value = k[source];
        }
        if kind == 2 {
            value = v[source];
        }
        out[i] = value;
    }
}
#[ruda(launch)]
pub fn destination_pack_generic<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    out: &mut Array<f32>,
    n: u32,
    tokens: u32,
    heads: u32,
    width: u32,
    per: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let d = i % width as usize;
        let kind = i / width as usize % 3;
        let head = i / (width as usize * 3) % per as usize;
        let token = i / (width as usize * 3 * per as usize) % tokens as usize;
        let destination = i / (width as usize * 3 * per as usize * tokens as usize);
        let source =
            (token * heads as usize + destination * per as usize + head) * width as usize + d;
        let mut value = q[source];
        if kind == 1 {
            value = k[source];
        }
        if kind == 2 {
            value = v[source];
        }
        out[i] = f32::cast_from(value);
    }
}

/// Per-kind tensor-scaled E4M3/E5M2 destination-major Q/K/V send buffer.
/// Decoder scales [3] are read from the device on every replay.
pub struct PreparedTensorFp8Pack<R: Runtime> {
    q: super::PreparedPack<R>,
    k: super::PreparedPack<R>,
    v: super::PreparedPack<R>,
    scales: super::PreparedPack<R>,
    scales_f32: RudaTensor<R>,
    pub output: RudaTensor<R>,
    destinations: usize,
    format: super::linear::FloatFormat,
}
impl<R: Runtime> PreparedTensorFp8Pack<R> {
    pub fn prepare(
        qkv: &Qkv<R>,
        destinations: usize,
        decoder_scales: &RudaTensor<R>,
        format: super::linear::FloatFormat,
    ) -> Result<Self, InferenceError> {
        let q = super::PreparedPack::prepare(&qkv.query)?;
        let k = super::PreparedPack::prepare(&qkv.key)?;
        let v = super::PreparedPack::prepare(&qkv.value)?;
        let x = &q.output;
        float(x)?;
        for t in [&k.output, &v.output] {
            check(t, x)?;
            if t.dtype != x.dtype || t.meta.shape() != x.meta.shape() {
                return Err(InferenceError("tensor FP8 pack Q/K/V mismatch"));
            }
        }
        if x.meta.num_dims() != 3
            || destinations == 0
            || x.meta.shape()[1] % destinations != 0
            || format == super::linear::FloatFormat::E2m1
        {
            return Err(InferenceError(
                "tensor FP8 destination geometry/format invalid",
            ));
        }
        let scales = super::PreparedPack::prepare(decoder_scales)?;
        check(&scales.output, x)?;
        float(&scales.output)?;
        if scales.output.meta.shape()[..] != [3] {
            return Err(InferenceError(
                "tensor FP8 Q/K/V decoder scales must be [3]",
            ));
        }
        let scales_f32 = alloc(x, [3], DType::F32);
        let output = alloc(
            x,
            [
                destinations,
                x.meta.shape()[0] * (x.meta.shape()[1] / destinations) * 3,
                x.meta.shape()[2],
            ],
            DType::U8,
        );
        Ok(Self {
            q,
            k,
            v,
            scales,
            scales_f32,
            output,
            destinations,
            format,
        })
    }
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        unsafe {
            self.q.run();
            self.k.run();
            self.v.run();
            self.scales.run();
            super::norm::cast_into(&self.scales.output, &self.scales_f32)?;
        }
        let q = &self.q.output;
        let n = self.output.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(q.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&q.client, n, dim);
            destination_fp8::launch::<R>(
                &q.client,
                count,
                dim,
                q.clone().into_array_arg(),
                self.k.output.clone().into_array_arg(),
                self.v.output.clone().into_array_arg(),
                self.scales_f32.clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                n as u32,
                q.meta.shape()[0] as u32,
                q.meta.shape()[1] as u32,
                q.meta.shape()[2] as u32,
                (q.meta.shape()[1] / self.destinations) as u32,
                self.format == super::linear::FloatFormat::E5m2,
                q.dtype.into(),
            );
        }
        Ok(&self.output)
    }
}
#[ruda(launch)]
pub fn destination_fp8<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    scale: &Array<f32>,
    out: &mut Array<u8>,
    n: u32,
    tokens: u32,
    heads: u32,
    width: u32,
    per: u32,
    #[comptime] e5m2: bool,
    #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let d = i % width as usize;
        let kind = i / width as usize % 3;
        let head = i / (width as usize * 3) % per as usize;
        let token = i / (width as usize * 3 * per as usize) % tokens as usize;
        let destination = i / (width as usize * 3 * per as usize * tokens as usize);
        let source =
            (token * heads as usize + destination * per as usize + head) * width as usize + d;
        let mut value = f32::cast_from(q[source]);
        if kind == 1 {
            value = f32::cast_from(k[source]);
        }
        if kind == 2 {
            value = f32::cast_from(v[source]);
        }
        out[i] = u8::cast_from(rublas::tensor_float_quant::fp8_encode(
            value / scale[kind],
            e5m2,
        ));
    }
}
