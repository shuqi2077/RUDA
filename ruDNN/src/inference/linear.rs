//! Native linear, segmented GEMM and low-rank/quantized projection composition.
use super::{InferenceError, alloc, check, float};
pub use rublas::tensor_float_quant::{
    BlockFloatTensor, FloatFormat, QuantOptions, ScaleFormat, ScaleLayout,
};
use ruda_core::tensor::DType;
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};

pub enum Matrix<'a, R: Runtime> {
    Dense(&'a RudaTensor<R>),
    BlockFloat(&'a BlockFloatTensor<R>),
    Fp8 {
        data: &'a RudaTensor<R>,
        decoder_scale: &'a RudaTensor<R>,
        format: FloatFormat,
    },
    Fp8Blocks {
        data: &'a RudaTensor<R>,
        scales: &'a RudaTensor<R>,
        format: FloatFormat,
        block_rows: usize,
        block_cols: usize,
    },
}
impl<R: Runtime> Matrix<'_, R> {
    pub fn decode(&self) -> Result<RudaTensor<R>, InferenceError> {
        match self {
            Self::Dense(t) => super::norm::cast(t, DType::F32),
            Self::BlockFloat(t) => t.dequantize(DType::F32).map_err(|e| InferenceError(e.0)),
            Self::Fp8 {
                data,
                decoder_scale,
                format,
            } => decode_fp8(data, decoder_scale, *format),
            Self::Fp8Blocks {
                data,
                scales,
                format,
                block_rows,
                block_cols,
            } => decode_fp8_blocks(data, scales, *format, *block_rows, *block_cols),
        }
    }
}
/// Fixed-address format decoder for prepared multi-operator graphs.
pub struct PreparedMatrix<'a, R: Runtime> {
    source: Matrix<'a, R>,
    pub output: RudaTensor<R>,
    broadcast: Option<super::norm::PreparedBroadcast<R>>,
    pack: Option<super::PreparedPack<R>>,
}
impl<'a, R: Runtime> PreparedMatrix<'a, R> {
    pub fn prepare(source: Matrix<'a, R>) -> Result<Self, InferenceError> {
        let (like, shape) = match &source {
            Matrix::Dense(t) => (*t, t.meta.shape().clone()),
            Matrix::BlockFloat(t) => (&t.data, t.logical_shape.clone()),
            Matrix::Fp8 { data, .. } | Matrix::Fp8Blocks { data, .. } => {
                (*data, data.meta.shape().clone())
            }
        };
        super::check_view(like, like)?;
        let pack = match &source {
            Matrix::Dense(t) => Some(super::PreparedPack::prepare(t)?),
            _ => {
                check(like, like)?;
                None
            }
        };
        let output = alloc(like, shape, DType::F32);
        let mut broadcast = None;
        match &source {
            Matrix::Dense(t) => float(t)?,
            Matrix::BlockFloat(_) => {}
            Matrix::Fp8 {
                data,
                decoder_scale,
                format,
            } => {
                if data.dtype != DType::U8 || *format == FloatFormat::E2m1 {
                    return Err(InferenceError("prepared FP8 storage invalid"));
                }
                broadcast = Some(super::norm::PreparedBroadcast::prepare(
                    decoder_scale,
                    &output,
                )?);
            }
            Matrix::Fp8Blocks {
                data,
                scales,
                format,
                block_rows,
                block_cols,
            } => {
                check(scales, data)?;
                let shape = data.meta.shape();
                if shape.len() < 2
                    || data.dtype != DType::U8
                    || scales.dtype != DType::F32
                    || *format == FloatFormat::E2m1
                    || *block_rows == 0
                    || *block_cols == 0
                {
                    return Err(InferenceError("prepared block FP8 recipe invalid"));
                }
                let m = shape[shape.len() - 2];
                let k = shape[shape.len() - 1];
                if m == 0
                    || k == 0
                    || scales.meta.num_elements()
                        != data.meta.num_elements() / m / k
                            * m.div_ceil(*block_rows)
                            * k.div_ceil(*block_cols)
                {
                    return Err(InferenceError("prepared block FP8 scales mismatch"));
                }
            }
        }
        Ok(Self {
            source,
            output,
            broadcast,
            pack,
        })
    }
    pub unsafe fn run(&self) -> Result<(), InferenceError> {
        let out = &self.output;
        let n = out.meta.num_elements();
        match &self.source {
            Matrix::Dense(_) => unsafe {
                let pack = self.pack.as_ref().unwrap();
                pack.run();
                super::norm::cast_into(&pack.output, out)
            },
            Matrix::BlockFloat(t) => unsafe {
                t.dequantize_into(out).map_err(|e| InferenceError(e.0))
            },
            Matrix::Fp8 { data, format, .. } => {
                let broadcast = self.broadcast.as_ref().unwrap();
                unsafe {
                    broadcast.run();
                }
                if n > 0 {
                    let dim = RudaDim::new(out.client.properties(), n);
                    let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
                    fp8_values::launch::<R>(
                        &out.client,
                        count,
                        dim,
                        (*data).clone().into_array_arg(),
                        broadcast.output.clone().into_array_arg(),
                        out.clone().into_array_arg(),
                        n as u32,
                        *format == FloatFormat::E5m2,
                    );
                }
                Ok(())
            }
            Matrix::Fp8Blocks {
                data,
                scales,
                format,
                block_rows,
                block_cols,
            } => {
                let shape = data.meta.shape();
                let rows = shape[shape.len() - 2];
                let cols = shape[shape.len() - 1];
                if n > 0 {
                    let dim = RudaDim::new(out.client.properties(), n);
                    let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
                    fp8_blocks::launch::<R>(
                        &out.client,
                        count,
                        dim,
                        (*data).clone().into_array_arg(),
                        (*scales).clone().into_array_arg(),
                        out.clone().into_array_arg(),
                        n as u32,
                        rows as u32,
                        cols as u32,
                        rows.div_ceil(*block_rows) as u32,
                        cols.div_ceil(*block_cols) as u32,
                        *block_rows as u32,
                        *block_cols as u32,
                        *format == FloatFormat::E5m2,
                    );
                }
                Ok(())
            }
        }
    }
}
/// FP32 block scales [groups,ceil(M/Br),ceil(K/Bc)] cover per-token (Br=1),
/// per-tensor, and two-dimensional weight-block scaling without approximation.
pub fn decode_fp8_blocks<R: Runtime>(
    data: &RudaTensor<R>,
    scales: &RudaTensor<R>,
    format: FloatFormat,
    block_rows: usize,
    block_cols: usize,
) -> Result<RudaTensor<R>, InferenceError> {
    check(data, data)?;
    check(scales, data)?;
    let rank = data.meta.num_dims();
    if rank < 2
        || data.dtype != DType::U8
        || scales.dtype != DType::F32
        || block_rows == 0
        || block_cols == 0
        || format == FloatFormat::E2m1
    {
        return Err(InferenceError("invalid block-scaled FP8 matrix"));
    }
    let rows = data.meta.shape()[rank - 2];
    let cols = data.meta.shape()[rank - 1];
    if rows == 0 || cols == 0 {
        return Err(InferenceError(
            "block-scaled FP8 matrix axes must be nonempty",
        ));
    }
    let groups = data.meta.num_elements() / rows / cols;
    let sr = rows.div_ceil(block_rows);
    let sc = cols.div_ceil(block_cols);
    if scales.meta.num_elements() != groups * sr * sc {
        return Err(InferenceError("FP8 block scale count mismatch"));
    }
    let out = alloc(data, data.meta.shape().clone(), DType::F32);
    let n = data.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(data.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&data.client, n, dim);
        fp8_blocks::launch::<R>(
            &data.client,
            count,
            dim,
            data.clone().into_array_arg(),
            scales.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            rows as u32,
            cols as u32,
            sr as u32,
            sc as u32,
            block_rows as u32,
            block_cols as u32,
            format == FloatFormat::E5m2,
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn fp8_blocks(
    data: &Array<u8>,
    scales: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    rows: u32,
    cols: u32,
    sr: u32,
    sc: u32,
    br: u32,
    bc: u32,
    #[comptime] e5m2: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % cols as usize;
        let row = i / cols as usize % rows as usize;
        let group = i / (cols as usize * rows as usize);
        let sf =
            scales[(group * sr as usize + row / br as usize) * sc as usize + col / bc as usize];
        out[i] = rublas::tensor_float_quant::fp8_decode(u32::cast_from(data[i]), e5m2) * sf;
    }
}
pub fn decode_fp8<R: Runtime>(
    data: &RudaTensor<R>,
    decoder_scale: &RudaTensor<R>,
    format: FloatFormat,
) -> Result<RudaTensor<R>, InferenceError> {
    check(data, data)?;
    if data.dtype != DType::U8 || format == FloatFormat::E2m1 {
        return Err(InferenceError("FP8 storage must be U8 E4M3/E5M2"));
    }
    let out = alloc(data, data.meta.shape().clone(), DType::F32);
    let scales = super::norm::broadcast_f32(decoder_scale, &out)?;
    let n = data.meta.num_elements();
    if n != 0 {
        let dim = RudaDim::new(data.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&data.client, n, dim);
        fp8_values::launch::<R>(
            &data.client,
            count,
            dim,
            data.clone().into_array_arg(),
            scales.into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            format == FloatFormat::E5m2,
        );
    }
    Ok(out)
}
pub fn encode_fp8<R: Runtime>(
    input: &RudaTensor<R>,
    decoder_scale: &RudaTensor<R>,
    format: FloatFormat,
) -> Result<RudaTensor<R>, InferenceError> {
    let x = super::norm::cast(input, DType::F32)?;
    if format == FloatFormat::E2m1 {
        return Err(InferenceError("encode_fp8 expects E4M3 or E5M2"));
    }
    let s = super::norm::broadcast_f32(decoder_scale, &x)?;
    let out = alloc(&x, x.meta.shape().clone(), DType::U8);
    let n = x.meta.num_elements();
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        fp8_pack::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            s.into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            format == FloatFormat::E5m2,
        );
    }
    Ok(out)
}
pub struct ScaledFp8<R: Runtime> {
    pub data: RudaTensor<R>,
    pub scales: RudaTensor<R>,
    pub block_rows: usize,
    pub block_cols: usize,
    pub format: FloatFormat,
}
impl<R: Runtime> ScaledFp8<R> {
    pub fn matrix(&self) -> Matrix<'_, R> {
        Matrix::Fp8Blocks {
            data: &self.data,
            scales: &self.scales,
            format: self.format,
            block_rows: self.block_rows,
            block_cols: self.block_cols,
        }
    }
    pub fn allocate(
        like: &RudaTensor<R>,
        block_rows: usize,
        block_cols: usize,
        format: FloatFormat,
    ) -> Result<Self, InferenceError> {
        check(like, like)?;
        let s = like.meta.shape();
        if s.len() < 2
            || block_rows == 0
            || block_cols == 0
            || format == FloatFormat::E2m1
            || s[s.len() - 2] == 0
            || s[s.len() - 1] == 0
        {
            return Err(InferenceError(
                "FP8 block quantizer needs nonempty matrices and positive blocks",
            ));
        }
        let m = s[s.len() - 2];
        let k = s[s.len() - 1];
        let groups = like.meta.num_elements() / m / k;
        let sr = m.div_ceil(block_rows);
        let sc = k.div_ceil(block_cols);
        groups
            .checked_mul(sr)
            .and_then(|n| n.checked_mul(sc))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("FP8 scales exceed U32"))?;
        Ok(Self {
            data: alloc(like, s.clone(), DType::U8),
            scales: alloc(like, [groups, sr, sc], DType::F32),
            block_rows,
            block_cols,
            format,
        })
    }
    pub unsafe fn quantize_into(&self, input: &RudaTensor<R>) -> Result<(), InferenceError> {
        check(input, &self.data)?;
        check(&self.scales, &self.data)?;
        let s = self.data.meta.shape();
        if s.len() < 2
            || input.dtype != DType::F32
            || input.meta.shape() != s
            || self.data.dtype != DType::U8
            || self.scales.dtype != DType::F32
            || self.block_rows == 0
            || self.block_cols == 0
            || self.format == FloatFormat::E2m1
        {
            return Err(InferenceError("retained FP8 encoder operands invalid"));
        }
        let rows = s[s.len() - 2];
        let cols = s[s.len() - 1];
        if rows == 0 || cols == 0 {
            return Err(InferenceError("FP8 matrix axes must be nonempty"));
        }
        let groups = input.meta.num_elements() / rows / cols;
        let sr = rows.div_ceil(self.block_rows);
        let sc = cols.div_ceil(self.block_cols);
        if self.scales.meta.shape()[..] != [groups, sr, sc] {
            return Err(InferenceError("retained FP8 scale shape mismatch"));
        }
        let n = groups * sr * sc;
        if n > 0 {
            let dim = RudaDim::new(input.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
            encode_blocks::launch::<R>(
                &input.client,
                count,
                dim,
                input.clone().into_array_arg(),
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                n as u32,
                rows as u32,
                cols as u32,
                sr as u32,
                sc as u32,
                self.block_rows as u32,
                self.block_cols as u32,
                self.format == FloatFormat::E5m2,
            );
        }
        Ok(())
    }
    pub unsafe fn decode_into(&self, out: &RudaTensor<R>) -> Result<(), InferenceError> {
        check(out, &self.data)?;
        check(&self.scales, &self.data)?;
        let s = self.data.meta.shape();
        if s.len() < 2
            || self.block_rows == 0
            || self.block_cols == 0
            || self.data.dtype != DType::U8
            || self.scales.dtype != DType::F32
            || out.dtype != DType::F32
            || out.meta.shape() != s
            || self.format == FloatFormat::E2m1
        {
            return Err(InferenceError("retained FP8 decoder operands invalid"));
        }
        let rows = s[s.len() - 2];
        let cols = s[s.len() - 1];
        if rows == 0 || cols == 0 {
            return Err(InferenceError("FP8 matrix axes must be nonempty"));
        }
        let sr = rows.div_ceil(self.block_rows);
        let sc = cols.div_ceil(self.block_cols);
        if self.scales.meta.num_elements() != out.meta.num_elements() / rows / cols * sr * sc {
            return Err(InferenceError("retained FP8 decoder scales mismatch"));
        }
        let n = out.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(out.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
            fp8_blocks::launch::<R>(
                &out.client,
                count,
                dim,
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                rows as u32,
                cols as u32,
                sr as u32,
                sc as u32,
                self.block_rows as u32,
                self.block_cols as u32,
                self.format == FloatFormat::E5m2,
            );
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Fp8Recipe {
    pub block_rows: usize,
    pub block_cols: usize,
    pub format: FloatFormat,
}

/// FP32 decoder scales (not rounded E8M0), supporting per-token and rectangular
/// weight blocks. Groups are the leading batch axes of [...,M,K].
pub fn quantize_fp8_blocks<R: Runtime>(
    input: &RudaTensor<R>,
    block_rows: usize,
    block_cols: usize,
    format: FloatFormat,
) -> Result<ScaledFp8<R>, InferenceError> {
    let x = super::norm::cast(input, DType::F32)?;
    let out = ScaledFp8::allocate(&x, block_rows, block_cols, format)?;
    unsafe {
        out.quantize_into(&x)?;
    }
    Ok(out)
}

#[ruda(launch)]
pub fn encode_blocks(
    x: &Array<f32>,
    data: &mut Array<u8>,
    scales: &mut Array<f32>,
    n: u32,
    rows: u32,
    cols: u32,
    sr: u32,
    sc: u32,
    br: u32,
    bc: u32,
    #[comptime] e5m2: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let block_col = i % sc as usize;
        let block_row = i / sc as usize % sr as usize;
        let group = i / (sc as usize * sr as usize);
        let r0 = block_row * br as usize;
        let c0 = block_col * bc as usize;
        let r1 = usize::min(rows as usize, r0 + br as usize);
        let c1 = usize::min(cols as usize, c0 + bc as usize);
        let mut maximum: f32 = 0.0f32;
        for row in r0..r1 {
            for col in c0..c1 {
                maximum = f32::max(
                    maximum,
                    x[(group * rows as usize + row) * cols as usize + col].abs(),
                );
            }
        }
        let mut limit: f32 = 448.0f32;
        if comptime!(e5m2) {
            limit = 57344.0;
        }
        let mut scale: f32 = 1.0f32;
        if maximum > 0.0 {
            scale = maximum / limit;
        }
        scales[i] = scale;
        for row in r0..r1 {
            for col in c0..c1 {
                let index = (group * rows as usize + row) * cols as usize + col;
                data[index] = u8::cast_from(rublas::tensor_float_quant::fp8_encode(
                    x[index] / scale,
                    e5m2,
                ));
            }
        }
    }
}
#[ruda(launch)]
pub fn fp8_values(
    x: &Array<u8>,
    scales: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    #[comptime] e5m2: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = rublas::tensor_float_quant::fp8_decode(u32::cast_from(x[i]), e5m2) * scales[i];
    }
}
#[ruda(launch)]
pub fn fp8_pack(
    x: &Array<f32>,
    scales: &Array<f32>,
    out: &mut Array<u8>,
    n: u32,
    #[comptime] e5m2: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = u8::cast_from(rublas::tensor_float_quant::fp8_encode(
            x[i] / scales[i],
            e5m2,
        ));
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LinearOptions {
    pub transpose_weight: bool,
    pub alpha: f32,
    pub beta: f32,
    pub output_dtype: DType,
}
impl Default for LinearOptions {
    fn default() -> Self {
        Self {
            transpose_weight: true,
            alpha: 1.0,
            beta: 0.0,
            output_dtype: DType::BF16,
        }
    }
}
/// Batch axes are broadcast by the existing native RUDA GEMM engine.
pub fn matmul<R: Runtime>(
    input: Matrix<'_, R>,
    weight: Matrix<'_, R>,
    addend: Option<&RudaTensor<R>>,
    options: LinearOptions,
) -> Result<RudaTensor<R>, InferenceError> {
    if !options.alpha.is_finite() || !options.beta.is_finite() {
        return Err(InferenceError("linear coefficients must be finite"));
    }
    let x = input.decode()?;
    let w = weight.decode()?;
    let c = addend
        .map(|t| super::norm::cast(t, DType::F32))
        .transpose()?;
    let out = rublas::tensor_matmul::gemm(
        x,
        w,
        c,
        rublas::tensor_matmul::GemmOptions {
            alpha: options.alpha as f64,
            beta: options.beta as f64,
            transpose_rhs: options.transpose_weight,
            ..Default::default()
        },
        rublas::tensor_matmul::MatmulStrategy::default(),
    )
    .map_err(|_| InferenceError("native linear GEMM setup failed"))?;
    super::norm::cast(&out, options.output_dtype)
}
/// Fixed-address native matrix multiplication. Decoder parameters are reread
/// each replay. Leading batch axes use the same right-aligned broadcasting as
/// ordinary matmul. Addends may broadcast to the result.
pub struct PreparedLinear<'a, R: Runtime> {
    input: PreparedMatrix<'a, R>,
    weight: PreparedMatrix<'a, R>,
    addend: Option<super::norm::PreparedBroadcast<R>>,
    product: RudaTensor<R>,
    pub output: RudaTensor<R>,
    options: LinearOptions,
}
impl<'a, R: Runtime> PreparedLinear<'a, R> {
    pub fn prepare(
        input: Matrix<'a, R>,
        weight: Matrix<'a, R>,
        addend: Option<&RudaTensor<R>>,
        options: LinearOptions,
    ) -> Result<Self, InferenceError> {
        let input = PreparedMatrix::prepare(input)?;
        let weight = PreparedMatrix::prepare(weight)?;
        let x = &input.output;
        let w = &weight.output;
        check(w, x)?;
        let a = x.meta.shape();
        let b = w.meta.shape();
        if a.len() < 2 || b.len() < 2 || !options.alpha.is_finite() || !options.beta.is_finite() {
            return Err(InferenceError(
                "prepared linear requires matrices and finite coefficients",
            ));
        }
        let (k, n) = if options.transpose_weight {
            (b[b.len() - 1], b[b.len() - 2])
        } else {
            (b[b.len() - 2], b[b.len() - 1])
        };
        if a[a.len() - 1] != k {
            return Err(InferenceError(
                "prepared linear reduction dimensions mismatch",
            ));
        }
        let rank = a.len().max(b.len());
        let mut shape = vec![1usize; rank];
        for i in 0..rank - 2 {
            let ai = i.checked_sub(rank - a.len()).map_or(1, |j| a[j]);
            let bi = i.checked_sub(rank - b.len()).map_or(1, |j| b[j]);
            if ai != bi && ai != 1 && bi != 1 {
                return Err(InferenceError(
                    "prepared linear batch dimensions do not broadcast",
                ));
            }
            shape[i] = ai.max(bi);
        }
        shape[rank - 2] = a[a.len() - 2];
        shape[rank - 1] = n;
        if shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .filter(|n| *n <= u32::MAX as usize)
            .is_none()
        {
            return Err(InferenceError("prepared linear output exceeds U32"));
        }
        let shape = ruda_core::tensor::Shape::from(shape);
        let product = alloc(x, shape.clone(), DType::F32);
        let output = alloc(x, shape, options.output_dtype);
        float(&output)?;
        let addend = addend
            .map(|c| super::norm::PreparedBroadcast::prepare(c, &product))
            .transpose()?;
        Ok(Self {
            input,
            weight,
            addend,
            product,
            output,
            options,
        })
    }
    /// Retained outputs and scratch are exclusive on the bound execution queue.
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        unsafe {
            self.input.run()?;
            self.weight.run()?;
            if let Some(c) = &self.addend {
                c.run();
            }
        }
        let w = if self.options.transpose_weight {
            let rank = self.weight.output.meta.num_dims();
            ruda_kernel::tensor::permutation::swap_dims(
                self.weight.output.clone(),
                rank - 2,
                rank - 1,
            )
        } else {
            self.weight.output.clone()
        };
        rublas::tensor_matmul::matmul(
            self.input.output.clone(),
            w,
            Some(self.product.clone()),
            rublas::tensor_matmul::MatmulStrategy::Naive,
            DType::F32,
        )
        .map_err(|_| InferenceError("prepared native GEMM failed"))?;
        let n = self.output.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(self.output.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.output.client, n, dim);
            linear_epilogue::launch::<R>(
                &self.output.client,
                count,
                dim,
                self.product.clone().into_array_arg(),
                self.addend
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(&self.product)
                    .clone()
                    .into_array_arg(),
                self.output.clone().into_array_arg(),
                n as u32,
                self.options.alpha,
                self.options.beta,
                self.addend.is_some(),
                self.output.dtype.into(),
            );
        }
        Ok(&self.output)
    }
}
#[ruda(launch)]
pub fn linear_epilogue<F: Float>(
    product: &Array<f32>,
    addend: &Array<f32>,
    output: &mut Array<F>,
    n: u32,
    alpha: f32,
    beta: f32,
    #[comptime] has_addend: bool,
    #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut value = alpha * product[i];
        if comptime!(has_addend) {
            value = fma(beta, addend[i], value);
        }
        output[i] = F::cast_from(value);
    }
}
/// Native checkpoint-byte preparation for shuffled [K/128,N,128] FP8 weights.
/// Unlike a backend handle, the result is ordinary portable device storage.
pub fn prepare_low_latency_gemm_weights<R: Runtime>(
    weights: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    check(weights, weights)?;
    if weights.dtype != DType::U8
        || weights.meta.num_dims() != 2
        || weights.meta.shape()[0] % 32 != 0
        || weights.meta.shape()[1] % 128 != 0
    {
        return Err(InferenceError(
            "low-latency weight preparation needs FP8 bytes [N,K], N%32=0 and K%128=0",
        ));
    }
    let rows = weights.meta.shape()[0];
    let cols = weights.meta.shape()[1];
    let out = alloc(weights, [cols / 128, rows, 128], DType::U8);
    let n = weights.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(weights.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&weights.client, n, dim);
        blocked_weight::launch::<R>(
            &weights.client,
            count,
            dim,
            weights.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            rows as u32,
            cols as u32,
            false,
        );
    }
    Ok(out)
}
pub fn restore_low_latency_gemm_weights<R: Runtime>(
    blocked: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    check(blocked, blocked)?;
    if blocked.dtype != DType::U8
        || blocked.meta.num_dims() != 3
        || blocked.meta.shape()[1] % 32 != 0
        || blocked.meta.shape()[2] != 128
    {
        return Err(InferenceError("blocked FP8 weights must be [K/128,N,128]"));
    }
    let rows = blocked.meta.shape()[1];
    let cols = blocked.meta.shape()[0] * 128;
    let out = alloc(blocked, [rows, cols], DType::U8);
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(blocked.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&blocked.client, n, dim);
        blocked_weight::launch::<R>(
            &blocked.client,
            count,
            dim,
            blocked.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            rows as u32,
            cols as u32,
            true,
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn blocked_weight(
    input: &Array<u8>,
    output: &mut Array<u8>,
    n: u32,
    rows: u32,
    cols: u32,
    #[comptime] inverse: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / cols as usize;
        let col = i % cols as usize;
        let local = row % 32;
        let shuffled = row / 32 * 32 + (local % 4) * 8 + local / 4;
        let blocked = (col / 128 * rows as usize + shuffled) * 128 + col % 128;
        if comptime!(inverse) {
            output[i] = input[blocked];
        } else {
            output[blocked] = input[i];
        }
    }
}
pub struct LowRankAdapter<R: Runtime> {
    pub down: RudaTensor<R>,
    pub up: RudaTensor<R>,
    pub scale: f32,
    pub generation: u64,
}
impl<R: Runtime> LowRankAdapter<R> {
    /// Preparation owns decoded device matrices; later checkpoint mutations cannot
    /// silently change a prepared adapter. Generation IDs identify replacements.
    pub fn prepare(
        down: Matrix<'_, R>,
        up: Matrix<'_, R>,
        scale: f32,
        generation: u64,
    ) -> Result<Self, InferenceError> {
        let down = down.decode()?;
        let up = up.decode()?;
        check(&up, &down)?;
        if down.meta.num_dims() != 2
            || up.meta.num_dims() != 2
            || down.meta.shape()[0] != up.meta.shape()[1]
            || !scale.is_finite()
        {
            return Err(InferenceError("LoRA expects down [rank,K], up [N,rank]"));
        }
        Ok(Self {
            down,
            up,
            scale,
            generation,
        })
    }
}
pub fn matmul_lora<R: Runtime>(
    input: Matrix<'_, R>,
    weight: Matrix<'_, R>,
    adapter: &LowRankAdapter<R>,
    expected_generation: u64,
    bias: Option<&RudaTensor<R>>,
    output_dtype: DType,
) -> Result<RudaTensor<R>, InferenceError> {
    if adapter.generation != expected_generation {
        return Err(InferenceError(
            "prepared low-rank adapter generation is stale",
        ));
    }
    let x = input.decode()?;
    let base = matmul(
        Matrix::Dense(&x),
        weight,
        None,
        LinearOptions {
            output_dtype: DType::F32,
            ..Default::default()
        },
    )?;
    let low = matmul(
        Matrix::Dense(&x),
        Matrix::Dense(&adapter.down),
        None,
        LinearOptions {
            output_dtype: DType::F32,
            ..Default::default()
        },
    )?;
    let delta = matmul(
        Matrix::Dense(&low),
        Matrix::Dense(&adapter.up),
        Some(&base),
        LinearOptions {
            alpha: adapter.scale,
            beta: 1.0,
            output_dtype: DType::F32,
            ..Default::default()
        },
    )?;
    affine(&delta, bias, None, 1.0, output_dtype)
}
pub fn nvfp4_quantize_smooth<R: Runtime>(
    input: &RudaTensor<R>,
    pre_quant_scale: &RudaTensor<R>,
    encoder_global: &RudaTensor<R>,
) -> Result<BlockFloatTensor<R>, InferenceError> {
    let smoothed = affine(input, None, Some(pre_quant_scale), 1.0, DType::F32)?;
    let global = rublas::tensor_float_quant::invert_global_scale(encoder_global)
        .map_err(|e| InferenceError(e.0))?;
    rublas::tensor_float_quant::quantize(&smoothed, Some(&global), None, QuantOptions::nvfp4())
        .map_err(|e| InferenceError(e.0))
}
/// alpha * (A_fp4 B_fp4^T + down up^T) + bias. Decoder scales in A/B
/// describe the residual product before the separate GPU alpha multiplier.
pub fn nvfp4_svdquant<R: Runtime>(
    a: &BlockFloatTensor<R>,
    b: &BlockFloatTensor<R>,
    down: &RudaTensor<R>,
    up: &RudaTensor<R>,
    alpha: &RudaTensor<R>,
    bias: Option<&RudaTensor<R>>,
    dtype: DType,
) -> Result<RudaTensor<R>, InferenceError> {
    let base = matmul(
        Matrix::BlockFloat(a),
        Matrix::BlockFloat(b),
        None,
        LinearOptions {
            output_dtype: DType::F32,
            ..Default::default()
        },
    )?;
    let summed = matmul(
        Matrix::Dense(down),
        Matrix::Dense(up),
        Some(&base),
        LinearOptions {
            beta: 1.0,
            output_dtype: DType::F32,
            ..Default::default()
        },
    )?;
    let scaled = affine(&summed, None, Some(alpha), 1.0, DType::F32)?;
    affine(&scaled, bias, None, 1.0, dtype)
}
pub fn svdquant_linear<R: Runtime>(
    input: &RudaTensor<R>,
    weight: &BlockFloatTensor<R>,
    alpha: &RudaTensor<R>,
    smooth: &RudaTensor<R>,
    down_smoothed_transposed: &RudaTensor<R>,
    up_scaled: &RudaTensor<R>,
    encoder_global: &RudaTensor<R>,
    bias: Option<&RudaTensor<R>>,
) -> Result<RudaTensor<R>, InferenceError> {
    let quantized = nvfp4_quantize_smooth(input, smooth, encoder_global)?;
    let down = matmul(
        Matrix::Dense(input),
        Matrix::Dense(down_smoothed_transposed),
        None,
        LinearOptions {
            transpose_weight: false,
            output_dtype: DType::BF16,
            ..Default::default()
        },
    )?;
    nvfp4_svdquant(
        &quantized,
        weight,
        &down,
        up_scaled,
        alpha,
        bias,
        DType::BF16,
    )
}
/// y = (x + broadcast(bias)) * broadcast(scale) * alpha.
pub fn affine<R: Runtime>(
    input: &RudaTensor<R>,
    bias: Option<&RudaTensor<R>>,
    scale: Option<&RudaTensor<R>>,
    alpha: f32,
    dtype: DType,
) -> Result<RudaTensor<R>, InferenceError> {
    let x = super::norm::cast(input, DType::F32)?;
    let b = bias
        .map(|t| super::norm::broadcast_f32(t, &x))
        .transpose()?;
    let s = scale
        .map(|t| super::norm::broadcast_f32(t, &x))
        .transpose()?;
    let out = alloc(&x, x.meta.shape().clone(), dtype);
    float(&out)?;
    let n = x.meta.num_elements();
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        affine_values::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            b.as_ref().unwrap_or(&x).clone().into_array_arg(),
            s.as_ref().unwrap_or(&x).clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            alpha,
            b.is_some(),
            s.is_some(),
            dtype.into(),
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn affine_values<O: Float>(
    x: &Array<f32>,
    bias: &Array<f32>,
    scale: &Array<f32>,
    out: &mut Array<O>,
    n: u32,
    alpha: f32,
    #[comptime] biased: bool,
    #[comptime] scaled: bool,
    #[define(O)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut v = x[i];
        if comptime!(biased) {
            v += bias[i];
        }
        if comptime!(scaled) {
            v *= scale[i];
        }
        out[i] = O::cast_from(v * alpha);
    }
}

/// Packed rows [T,K] with per-segment weights [E,N,K]. Scheduling indices and
/// boundaries remain on the device. A row's segment selects its weight matrix.
pub fn segmented_matmul<R: Runtime>(
    input: Matrix<'_, R>,
    weights: Matrix<'_, R>,
    indptr: &RudaTensor<R>,
    weight_indices: Option<&RudaTensor<R>>,
    bias: Option<&RudaTensor<R>>,
    output_dtype: DType,
) -> Result<RudaTensor<R>, InferenceError> {
    let x = input.decode()?;
    let w = weights.decode()?;
    for t in [&w, indptr] {
        check(t, &x)?;
    }
    if x.meta.num_dims() != 2
        || w.meta.num_dims() != 3
        || x.meta.shape()[1] != w.meta.shape()[2]
        || indptr.dtype != DType::I32
        || indptr.meta.num_dims() != 1
        || indptr.meta.shape()[0] < 2
    {
        return Err(InferenceError(
            "segmented GEMM needs X[T,K], W[E,N,K], I32 indptr[S+1]",
        ));
    }
    let segments = indptr.meta.shape()[0] - 1;
    let experts = w.meta.shape()[0];
    let tokens = x.meta.shape()[0];
    let width = x.meta.shape()[1];
    let columns = w.meta.shape()[1];
    if let Some(ids) = weight_indices {
        check(ids, &x)?;
        if ids.dtype != DType::I32 || ids.meta.shape()[..] != [segments] {
            return Err(InferenceError("segment weight indices need I32 [S]"));
        }
    } else if segments > experts {
        return Err(InferenceError("segmented weights insufficient"));
    }
    let b = bias.map(|t| super::norm::cast(t, DType::F32)).transpose()?;
    if let Some(b) = &b {
        check(b, &x)?;
        if b.meta.shape()[..] != [experts, columns] {
            return Err(InferenceError("segment bias needs [E,N]"));
        }
    }
    let n = tokens
        .checked_mul(columns)
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(InferenceError("segmented output exceeds U32"))?;
    let out = alloc(&x, [tokens, columns], output_dtype);
    float(&out)?;
    if n > 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        segmented::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            w.clone().into_array_arg(),
            indptr.clone().into_array_arg(),
            weight_indices.unwrap_or(indptr).clone().into_array_arg(),
            b.as_ref().unwrap_or(&x).clone().into_array_arg(),
            out.clone().into_array_arg(),
            tokens as u32,
            width as u32,
            columns as u32,
            segments as u32,
            experts as u32,
            weight_indices.is_some(),
            b.is_some(),
            output_dtype.into(),
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn segmented<O: Float>(
    x: &Array<f32>,
    w: &Array<f32>,
    cu: &Array<i32>,
    ids: &Array<i32>,
    bias: &Array<f32>,
    out: &mut Array<O>,
    tokens: u32,
    width: u32,
    columns: u32,
    segments: u32,
    experts: u32,
    #[comptime] indexed: bool,
    #[comptime] biased: bool,
    #[define(O)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * columns as usize {
        let row = i / columns as usize;
        let col = i % columns as usize;
        let mut lo: usize = 0usize;
        let mut hi = segments as usize;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if cu[mid] <= row as i32 {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut sum: f32 = 0.0f32;
        if lo < segments as usize && row < cu[lo + 1] as usize {
            let mut e = lo as i32;
            if comptime!(indexed) {
                e = ids[lo];
            }
            if e >= 0 && e < experts as i32 {
                if comptime!(biased) {
                    sum = bias[e as usize * columns as usize + col];
                }
                for k in 0..width as usize {
                    sum = fma(
                        x[row * width as usize + k],
                        w[(e as usize * columns as usize + col) * width as usize + k],
                        sum,
                    );
                }
            }
        }
        out[i] = O::cast_from(sum);
    }
}

/// Prepared grouped/segmented GEMM with device-resident boundaries and optional
/// segment-to-weight table. Formats are decoded into retained native buffers.
pub struct PreparedSegmented<'a, R: Runtime> {
    input: PreparedMatrix<'a, R>,
    weights: PreparedMatrix<'a, R>,
    indptr: super::metadata::PreparedIndices<R>,
    indices: Option<super::metadata::PreparedIndices<R>>,
    bias: Option<super::PreparedPack<R>>,
    bias_f32: Option<RudaTensor<R>>,
    pub output: RudaTensor<R>,
}
impl<'a, R: Runtime> PreparedSegmented<'a, R> {
    pub fn prepare(
        input: Matrix<'a, R>,
        weights: Matrix<'a, R>,
        indptr: &RudaTensor<R>,
        indices: Option<&RudaTensor<R>>,
        bias: Option<&RudaTensor<R>>,
        dtype: DType,
    ) -> Result<Self, InferenceError> {
        let input = PreparedMatrix::prepare(input)?;
        let weights = PreparedMatrix::prepare(weights)?;
        let x = &input.output;
        let w = &weights.output;
        let indptr = super::metadata::PreparedIndices::prepare(indptr, DType::I32)?;
        let indices = indices
            .map(|t| super::metadata::PreparedIndices::prepare(t, DType::I32))
            .transpose()?;
        for t in [w, &indptr.output] {
            check(t, x)?;
        }
        if x.meta.num_dims() != 2
            || w.meta.num_dims() != 3
            || x.meta.shape()[1] != w.meta.shape()[2]
            || indptr.output.meta.num_dims() != 1
            || indptr.output.meta.num_elements() < 2
        {
            return Err(InferenceError("prepared segmented GEMM geometry invalid"));
        }
        let segments = indptr.output.meta.num_elements() - 1;
        let experts = w.meta.shape()[0];
        let n = w.meta.shape()[1];
        if let Some(i) = &indices {
            check(&i.output, x)?;
            if i.output.meta.shape()[..] != [segments] {
                return Err(InferenceError("segment weight IDs require [segments]"));
            }
        } else if segments > experts {
            return Err(InferenceError("insufficient segmented weights"));
        }
        let bias = bias.map(super::PreparedPack::prepare).transpose()?;
        let bias_f32 = if let Some(b) = &bias {
            check(&b.output, x)?;
            float(&b.output)?;
            if b.output.meta.shape()[..] != [experts, n] {
                return Err(InferenceError("segment bias must be [E,N]"));
            }
            Some(alloc(x, [experts, n], DType::F32))
        } else {
            None
        };
        x.meta.shape()[0]
            .checked_mul(n)
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("segmented output exceeds U32"))?;
        let output = alloc(x, [x.meta.shape()[0], n], dtype);
        float(&output)?;
        Ok(Self {
            input,
            weights,
            indptr,
            indices,
            bias,
            bias_f32,
            output,
        })
    }
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        unsafe {
            self.input.run()?;
            self.weights.run()?;
            self.indptr.run();
            if let Some(i) = &self.indices {
                i.run();
            }
            if let (Some(b), Some(f32)) = (&self.bias, &self.bias_f32) {
                b.run();
                super::norm::cast_into(&b.output, f32)?;
            }
        }
        let x = &self.input.output;
        let w = &self.weights.output;
        let n = self.output.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            segmented::launch::<R>(
                &x.client,
                count,
                dim,
                x.clone().into_array_arg(),
                w.clone().into_array_arg(),
                self.indptr.output.clone().into_array_arg(),
                self.indices
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(&self.indptr.output)
                    .clone()
                    .into_array_arg(),
                self.bias_f32.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                x.meta.shape()[0] as u32,
                x.meta.shape()[1] as u32,
                w.meta.shape()[1] as u32,
                (self.indptr.output.meta.num_elements() - 1) as u32,
                w.meta.shape()[0] as u32,
                self.indices.is_some(),
                self.bias.is_some(),
                self.output.dtype.into(),
            );
        }
        Ok(&self.output)
    }
}
/// Prepared LoRA keeps independent optional full-precision adapter inputs,
/// avoiding an unintended quantization boundary on the low-rank branch.
pub struct PreparedLora<'a, R: Runtime> {
    input: PreparedMatrix<'a, R>,
    weight: PreparedMatrix<'a, R>,
    adapter_input: Option<PreparedMatrix<'a, R>>,
    adapter: &'a LowRankAdapter<R>,
    generation: u64,
    base: RudaTensor<R>,
    low: RudaTensor<R>,
    delta: RudaTensor<R>,
    bias: Option<super::norm::PreparedBroadcast<R>>,
    pub output: RudaTensor<R>,
}
impl<'a, R: Runtime> PreparedLora<'a, R> {
    pub fn prepare(
        input: Matrix<'a, R>,
        weight: Matrix<'a, R>,
        adapter_input: Option<Matrix<'a, R>>,
        adapter: &'a LowRankAdapter<R>,
        generation: u64,
        bias: Option<&RudaTensor<R>>,
        dtype: DType,
    ) -> Result<Self, InferenceError> {
        if adapter.generation != generation {
            return Err(InferenceError(
                "prepared low-rank adapter generation is stale",
            ));
        }
        let input = PreparedMatrix::prepare(input)?;
        let weight = PreparedMatrix::prepare(weight)?;
        let adapter_input = adapter_input.map(PreparedMatrix::prepare).transpose()?;
        let x = &input.output;
        let w = &weight.output;
        for t in [w, &adapter.down, &adapter.up] {
            check(t, x)?;
        }
        if x.meta.num_dims() != 2
            || w.meta.num_dims() != 2
            || adapter.down.dtype != DType::F32
            || adapter.up.dtype != DType::F32
            || adapter.down.meta.num_dims() != 2
            || adapter.up.meta.num_dims() != 2
            || w.meta.shape()[1] != x.meta.shape()[1]
            || adapter.down.meta.shape()[1] != x.meta.shape()[1]
            || adapter.up.meta.shape()[0] != w.meta.shape()[0]
            || adapter.up.meta.shape()[1] != adapter.down.meta.shape()[0]
            || adapter_input
                .as_ref()
                .is_some_and(|p| p.output.meta.shape() != x.meta.shape())
        {
            return Err(InferenceError("prepared LoRA geometry mismatch"));
        }
        let m = x.meta.shape()[0];
        let n = w.meta.shape()[0];
        let rank = adapter.down.meta.shape()[0];
        let base = alloc(x, [m, n], DType::F32);
        let low = alloc(x, [m, rank], DType::F32);
        let delta = alloc(x, [m, n], DType::F32);
        let output = alloc(x, [m, n], dtype);
        float(&output)?;
        let bias = bias
            .map(|p| super::norm::PreparedBroadcast::prepare(p, &base))
            .transpose()?;
        Ok(Self {
            input,
            weight,
            adapter_input,
            adapter,
            generation,
            base,
            low,
            delta,
            bias,
            output,
        })
    }
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        if self.adapter.generation != self.generation {
            return Err(InferenceError(
                "prepared low-rank adapter generation is stale",
            ));
        }
        unsafe {
            self.input.run()?;
            self.weight.run()?;
            if let Some(p) = &self.adapter_input {
                p.run()?;
            }
            if let Some(p) = &self.bias {
                p.run();
            }
        }
        let x = &self.input.output;
        let adapter_x = self.adapter_input.as_ref().map_or(x, |p| &p.output);
        let transpose =
            |t: &RudaTensor<R>| ruda_kernel::tensor::permutation::swap_dims(t.clone(), 0, 1);
        for (a, b, out) in [
            (x, transpose(&self.weight.output), &self.base),
            (adapter_x, transpose(&self.adapter.down), &self.low),
            (&self.low, transpose(&self.adapter.up), &self.delta),
        ] {
            rublas::tensor_matmul::matmul(
                a.clone(),
                b,
                Some(out.clone()),
                rublas::tensor_matmul::MatmulStrategy::Naive,
                DType::F32,
            )
            .map_err(|_| InferenceError("prepared low-rank native GEMM failed"))?;
        }
        let n = self.output.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            lora_epilogue::launch::<R>(
                &x.client,
                count,
                dim,
                self.base.clone().into_array_arg(),
                self.delta.clone().into_array_arg(),
                self.bias
                    .as_ref()
                    .map(|p| &p.output)
                    .unwrap_or(&self.base)
                    .clone()
                    .into_array_arg(),
                self.output.clone().into_array_arg(),
                n as u32,
                self.adapter.scale,
                self.bias.is_some(),
                self.output.dtype.into(),
            );
        }
        Ok(&self.output)
    }
}
#[ruda(launch)]
pub fn lora_epilogue<F: Float>(
    base: &Array<f32>,
    delta: &Array<f32>,
    bias: &Array<f32>,
    out: &mut Array<F>,
    n: u32,
    scale: f32,
    #[comptime] biased: bool,
    #[define(F)] _f: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut value = fma(delta[i], scale, base[i]);
        if comptime!(biased) {
            value += bias[i];
        }
        out[i] = F::cast_from(value);
    }
}
