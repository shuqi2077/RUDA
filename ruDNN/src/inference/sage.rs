//! Native SageAttention token-block Q/K and channel-wise FP8 V quantization.
//! Ragged block numbering restarts at each sequence, entirely on the device.
use super::{
    InferenceError, alloc, check,
    linear::{Matrix, PreparedMatrix},
};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SageFormat {
    Int8,
    E4m3,
}
pub struct TokenBlockTensor<R: Runtime> {
    pub data: RudaTensor<R>,
    pub scales: RudaTensor<R>,
    pub block_indptr: RudaTensor<R>,
    pub cu_seqlens: RudaTensor<R>,
    pub block_size: usize,
    pub format: SageFormat,
    capacity: usize,
    tokens: usize,
    heads: usize,
    width: usize,
    sequences: usize,
}
impl<R: Runtime> TokenBlockTensor<R> {
    pub fn allocate(
        input: &RudaTensor<R>,
        cu_seqlens: &RudaTensor<R>,
        block_size: usize,
        format: SageFormat,
    ) -> Result<Self, InferenceError> {
        check(input, input)?;
        check(cu_seqlens, input)?;
        if input.meta.num_dims() != 3
            || input.meta.shape()[1] == 0
            || input.meta.shape()[2] == 0
            || cu_seqlens.dtype != DType::I32
            || cu_seqlens.meta.num_dims() != 1
            || cu_seqlens.meta.num_elements() < 2
            || !matches!(block_size, 1 | 4 | 16)
        {
            return Err(InferenceError(
                "Sage Q/K needs [T,H,D], I32 indptr and block 1/4/16",
            ));
        }
        let s = input.meta.shape();
        let tokens = s[0];
        let heads = s[1];
        let width = s[2];
        let sequences = cu_seqlens.meta.num_elements() - 1;
        let capacity = tokens.div_ceil(block_size) + sequences - 1;
        let data = alloc(
            input,
            input.meta.shape().clone(),
            if format == SageFormat::Int8 {
                DType::I8
            } else {
                DType::U8
            },
        );
        let scales = alloc(input, [heads, capacity], DType::F32);
        let block_indptr = alloc(input, [sequences + 1], DType::U32);
        if heads
            .checked_mul(capacity)
            .filter(|n| *n <= u32::MAX as usize)
            .is_none()
        {
            return Err(InferenceError("Sage block scales exceed U32"));
        }
        Ok(Self {
            data,
            scales,
            block_indptr,
            cu_seqlens: cu_seqlens.clone(),
            block_size,
            format,
            capacity,
            tokens,
            heads,
            width,
            sequences,
        })
    }
    /// FP32 input and optional [H,D] mean are prepared device operands.
    pub unsafe fn quantize_into(
        &self,
        input: &RudaTensor<R>,
        mean: Option<&RudaTensor<R>>,
    ) -> Result<(), InferenceError> {
        check(input, &self.data)?;
        if input.dtype != DType::F32 || input.meta.shape() != self.data.meta.shape() {
            return Err(InferenceError("Sage prepared input mismatch"));
        }
        if let Some(m) = mean {
            check(m, input)?;
            if m.dtype != DType::F32 || m.meta.shape()[..] != [self.heads, self.width] {
                return Err(InferenceError("Sage key mean must be [H,D] F32"));
            }
        }
        let n = self.sequences + 1;
        let dim = RudaDim::new(input.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
        block_offsets::launch::<R>(
            &input.client,
            count,
            dim,
            self.cu_seqlens.clone().into_array_arg(),
            self.block_indptr.clone().into_array_arg(),
            self.sequences as u32,
            self.block_size as u32,
        );
        let n = self.heads * self.capacity;
        if n > 0 {
            let dim = RudaDim::new(input.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
            quantize_blocks::launch::<R>(
                &input.client,
                count,
                dim,
                input.clone().into_array_arg(),
                self.cu_seqlens.clone().into_array_arg(),
                self.block_indptr.clone().into_array_arg(),
                mean.unwrap_or(input).clone().into_array_arg(),
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                self.tokens as u32,
                self.heads as u32,
                self.width as u32,
                self.sequences as u32,
                self.capacity as u32,
                self.block_size as u32,
                self.format == SageFormat::E4m3,
                mean.is_some(),
                self.data.dtype.into(),
            );
        }
        Ok(())
    }
    pub fn decode(&self) -> Result<RudaTensor<R>, InferenceError> {
        let out = alloc(&self.data, self.data.meta.shape().clone(), DType::F32);
        unsafe {
            self.decode_into(&out)?;
        }
        Ok(out)
    }
    pub unsafe fn decode_into(&self, out: &RudaTensor<R>) -> Result<(), InferenceError> {
        check(out, &self.data)?;
        if out.dtype != DType::F32 || out.meta.shape() != self.data.meta.shape() {
            return Err(InferenceError("Sage decoder destination mismatch"));
        }
        let n = out.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(out.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
            decode_blocks::launch::<R>(
                &out.client,
                count,
                dim,
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                self.cu_seqlens.clone().into_array_arg(),
                self.block_indptr.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                self.heads as u32,
                self.width as u32,
                self.sequences as u32,
                self.capacity as u32,
                self.block_size as u32,
                self.format == SageFormat::E4m3,
                self.data.dtype.into(),
            );
        }
        Ok(())
    }
}
pub struct PreparedSage<'a, R: Runtime> {
    query_input: PreparedMatrix<'a, R>,
    key_input: PreparedMatrix<'a, R>,
    value_input: PreparedMatrix<'a, R>,
    pub query: TokenBlockTensor<R>,
    pub key: TokenBlockTensor<R>,
    pub value: RudaTensor<R>,
    pub value_scales: RudaTensor<R>,
    pub key_mean: RudaTensor<R>,
    smooth_key: bool,
}
impl<'a, R: Runtime> PreparedSage<'a, R> {
    pub fn prepare(
        query: &'a RudaTensor<R>,
        key: &'a RudaTensor<R>,
        value: &'a RudaTensor<R>,
        q_indptr: &RudaTensor<R>,
        kv_indptr: &RudaTensor<R>,
        q_block: usize,
        k_block: usize,
        format: SageFormat,
        smooth_key: bool,
    ) -> Result<Self, InferenceError> {
        let query_input = PreparedMatrix::prepare(Matrix::Dense(query))?;
        let key_input = PreparedMatrix::prepare(Matrix::Dense(key))?;
        let value_input = PreparedMatrix::prepare(Matrix::Dense(value))?;
        let q = &query_input.output;
        let k = &key_input.output;
        let v = &value_input.output;
        for t in [k, v] {
            check(t, q)?;
        }
        if q.meta.num_dims() != 3
            || k.meta.num_dims() != 3
            || k.meta.shape() != v.meta.shape()
            || q.meta.shape()[2] != k.meta.shape()[2]
            || q_indptr.meta.shape() != kv_indptr.meta.shape()
        {
            return Err(InferenceError("Sage Q/K/V dimensions mismatch"));
        }
        let query = TokenBlockTensor::allocate(q, q_indptr, q_block, format)?;
        let key = TokenBlockTensor::allocate(k, kv_indptr, k_block, format)?;
        let value = alloc(v, v.meta.shape().clone(), DType::U8);
        let value_scales = alloc(v, [v.meta.shape()[1], v.meta.shape()[2]], DType::F32);
        let key_mean = alloc(k, [k.meta.shape()[1], k.meta.shape()[2]], DType::F32);
        Ok(Self {
            query_input,
            key_input,
            value_input,
            query,
            key,
            value,
            value_scales,
            key_mean,
            smooth_key,
        })
    }
    /// Runs entirely into prepared storage; key_mean is the global per-head
    /// channel mean, and remains zero when smoothing is disabled.
    pub unsafe fn run(&self) -> Result<(), InferenceError> {
        unsafe {
            self.query_input.run()?;
            self.key_input.run()?;
            self.value_input.run()?;
        }
        let k = &self.key_input.output;
        let v = &self.value_input.output;
        let n = self.value_scales.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(k.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&k.client, n, dim);
            channel_statistics::launch::<R>(
                &k.client,
                count,
                dim,
                k.clone().into_array_arg(),
                v.clone().into_array_arg(),
                self.key_mean.clone().into_array_arg(),
                self.value_scales.clone().into_array_arg(),
                k.meta.shape()[0] as u32,
                k.meta.shape()[1] as u32,
                k.meta.shape()[2] as u32,
                self.smooth_key,
            );
        }
        unsafe {
            self.query.quantize_into(&self.query_input.output, None)?;
            self.key.quantize_into(
                k,
                if self.smooth_key {
                    Some(&self.key_mean)
                } else {
                    None
                },
            )?;
        }
        let n = v.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(v.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&v.client, n, dim);
            quantize_value::launch::<R>(
                &v.client,
                count,
                dim,
                v.clone().into_array_arg(),
                self.value_scales.clone().into_array_arg(),
                self.value.clone().into_array_arg(),
                n as u32,
                (v.meta.shape()[1] * v.meta.shape()[2]) as u32,
            );
        }
        Ok(())
    }
}
#[ruda(launch)]
pub fn block_offsets(cu: &Array<i32>, out: &mut Array<u32>, sequences: u32, block: u32) {
    let seq = ABSOLUTE_POS as usize;
    if seq <= sequences as usize {
        let mut prefix: u32 = 0u32;
        for i in 0..seq {
            let length = i32::max(0, cu[i + 1] - cu[i]) as u32;
            prefix += (length + block - 1) / block;
        }
        out[seq] = prefix;
    }
}
#[ruda]
fn find_sequence(cu: &Array<i32>, token: usize, sequences: usize) -> usize {
    let mut lo: usize = 0usize;
    let mut hi = sequences;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if cu[mid] <= token as i32 {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}
#[ruda(launch)]
pub fn quantize_blocks<I: Int>(
    x: &Array<f32>,
    cu: &Array<i32>,
    offsets: &Array<u32>,
    mean: &Array<f32>,
    out: &mut Array<I>,
    scales: &mut Array<f32>,
    tokens: u32,
    heads: u32,
    width: u32,
    sequences: u32,
    capacity: u32,
    block: u32,
    #[comptime] fp8: bool,
    #[comptime] smooth: bool,
    #[define(I)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < heads as usize * capacity as usize {
        let head = i / capacity as usize;
        let physical_block = i % capacity as usize;
        let mut seq: usize = 0usize;
        let mut hi = sequences as usize;
        while seq < hi {
            let mid = (seq + hi + 1) / 2;
            if offsets[mid] <= physical_block as u32 {
                seq = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut sf: f32 = 1.0f32;
        if seq < sequences as usize && physical_block < offsets[seq + 1] as usize {
            let begin =
                cu[seq] as usize + (physical_block - offsets[seq] as usize) * block as usize;
            let end = usize::min(
                usize::min(cu[seq + 1] as usize, begin + block as usize),
                tokens as usize,
            );
            let mut maximum: f32 = 0.0f32;
            for t in begin..end {
                for d in 0..width as usize {
                    let mut value = x[(t * heads as usize + head) * width as usize + d];
                    if comptime!(smooth) {
                        value -= mean[head * width as usize + d];
                    }
                    maximum = f32::max(maximum, value.abs());
                }
            }
            let mut limit: f32 = 127.0f32;
            if comptime!(fp8) {
                limit = 448.0;
            }
            if maximum > 0.0 {
                sf = maximum / limit;
            }
            for t in begin..end {
                for d in 0..width as usize {
                    let index = (t * heads as usize + head) * width as usize + d;
                    let mut value = x[index];
                    if comptime!(smooth) {
                        value -= mean[head * width as usize + d];
                    }
                    value /= sf;
                    if comptime!(fp8) {
                        out[index] =
                            I::cast_from(rublas::tensor_float_quant::fp8_encode(value, false));
                    } else {
                        let floor = value.floor();
                        let fraction = value - floor;
                        let mut rounded = floor;
                        if fraction > 0.5 || fraction == 0.5 && (floor as i32 & 1i32) != 0 {
                            rounded += 1.0;
                        }
                        out[index] = I::cast_from(f32::min(f32::max(rounded, -127.0), 127.0));
                    }
                }
            }
        }
        scales[i] = sf;
    }
}
#[ruda(launch)]
pub fn decode_blocks<I: Int>(
    data: &Array<I>,
    scales: &Array<f32>,
    cu: &Array<i32>,
    offsets: &Array<u32>,
    out: &mut Array<f32>,
    n: u32,
    heads: u32,
    width: u32,
    sequences: u32,
    capacity: u32,
    block: u32,
    #[comptime] fp8: bool,
    #[define(I)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let token = i / (heads as usize * width as usize);
        let head = i / width as usize % heads as usize;
        let seq = find_sequence(cu, token, sequences as usize);
        let mut value: f32 = 0.0f32;
        if seq < sequences as usize {
            let scale = scales[head * capacity as usize
                + offsets[seq] as usize
                + (token - cu[seq] as usize) / block as usize];
            value = f32::cast_from(data[i]);
            if comptime!(fp8) {
                value = rublas::tensor_float_quant::fp8_decode(u32::cast_from(data[i]), false);
            }
            value *= scale;
        }
        out[i] = value;
    }
}
#[ruda(launch)]
pub fn channel_statistics(
    k: &Array<f32>,
    v: &Array<f32>,
    mean: &mut Array<f32>,
    scale: &mut Array<f32>,
    tokens: u32,
    heads: u32,
    width: u32,
    #[comptime] smooth: bool,
) {
    let i = ABSOLUTE_POS as usize;
    let span = heads as usize * width as usize;
    if i < span {
        let mut sum: f32 = 0.0f32;
        let mut maximum: f32 = 0.0f32;
        for t in 0..tokens as usize {
            if comptime!(smooth) {
                sum += k[t * span + i];
            }
            maximum = f32::max(maximum, v[t * span + i].abs());
        }
        let mut m: f32 = 0.0f32;
        if tokens > 0 {
            m = sum / tokens as f32;
        }
        mean[i] = m;
        let mut sf: f32 = 1.0f32;
        if maximum > 0.0 {
            sf = maximum / 448.0;
        }
        scale[i] = sf;
    }
}
#[ruda(launch)]
pub fn quantize_value(x: &Array<f32>, scales: &Array<f32>, out: &mut Array<u8>, n: u32, span: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = u8::cast_from(rublas::tensor_float_quant::fp8_encode(
            x[i] / scales[i % span as usize],
            false,
        ));
    }
}
