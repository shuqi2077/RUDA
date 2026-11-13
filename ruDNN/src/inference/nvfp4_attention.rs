//! Native NVFP4 attention checkpoint layout, centering and compact QK correction.
//! The 64-row scale swizzle and K-token permutation follow FlashInfer v0.7.4 /
//! SageAttention's Apache-2.0 format. No vendor numerical kernels are invoked.
use super::{InferenceError, alloc, check};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};

/// Raw E2M1 nibbles and E4M3 decoder scales. All seven fields have the same
/// physical shapes as the stable SM120 interface, including transposed V.
pub struct PackedQkv<R: Runtime> {
    pub query: RudaTensor<R>,
    pub key: RudaTensor<R>,
    pub value_transposed: RudaTensor<R>,
    pub query_scale: RudaTensor<R>,
    pub key_scale: RudaTensor<R>,
    pub value_scale_transposed: RudaTensor<R>,
    pub correction: RudaTensor<R>,
    pub per_block_mean: bool,
}
impl<R: Runtime> PackedQkv<R> {
    fn geometry(&self) -> Result<(usize, usize, usize, usize, usize, usize), InferenceError> {
        let q = &self.query;
        for t in [
            &self.key,
            &self.value_transposed,
            &self.query_scale,
            &self.key_scale,
            &self.value_scale_transposed,
            &self.correction,
        ] {
            check(t, q)?;
        }
        check(q, q)?;
        if q.meta.num_dims() != 4 || self.key.meta.num_dims() != 4 {
            return Err(InferenceError(
                "NVFP4 attention needs rank-four Q/K payloads",
            ));
        }
        let s = q.meta.shape();
        let k = self.key.meta.shape();
        let (b, h, hk, m, n, d) = (s[0], s[1], k[1], s[2], k[2], s[3] * 2);
        if b == 0
            || h == 0
            || hk == 0
            || h % hk != 0
            || m == 0
            || n == 0
            || m % 128 != 0
            || n % 128 != 0
            || !matches!(d, 64 | 128)
            || k[0] != b
            || k[3] * 2 != d
        {
            return Err(InferenceError(
                "invalid NVFP4 attention batch/head/padded geometry",
            ));
        }
        let groups = if self.per_block_mean { m / 128 } else { 1 };
        for (t, shape) in [
            (&self.value_transposed, vec![b, hk, d, n / 2]),
            (&self.query_scale, vec![b, h, m, d / 16]),
            (&self.key_scale, vec![b, hk, n, d / 16]),
            (&self.value_scale_transposed, vec![b, hk, d, n / 16]),
        ] {
            if t.dtype != DType::U8 || t.meta.shape()[..] != shape[..] {
                return Err(InferenceError("NVFP4 packed payload/scale shape mismatch"));
            }
        }
        if q.dtype != DType::U8
            || self.key.dtype != DType::U8
            || self.correction.dtype != DType::F32
            || self.correction.meta.shape()[..] != [b, h, groups, n]
        {
            return Err(InferenceError(
                "NVFP4 correction must be compact F32 [B,H,G,Npad]",
            ));
        }
        Ok((b, h, hk, m, n, d))
    }
}
/// Allocation-only preparation. Dense inputs are [B,H,T,D], F16/BF16/F32.
/// `run` refreshes centering, padding, scales, packed data and QK correction.
pub struct PreparedQuantize<R: Runtime> {
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    q_mean: RudaTensor<R>,
    k_mean: RudaTensor<R>,
    q_center: RudaTensor<R>,
    k_center: RudaTensor<R>,
    v_pad: RudaTensor<R>,
    pub packed: PackedQkv<R>,
    b: usize,
    h: usize,
    hk: usize,
    m: usize,
    n: usize,
    mp: usize,
    np: usize,
    d: usize,
    g: usize,
}
impl<R: Runtime> PreparedQuantize<R> {
    pub fn prepare(
        q: &RudaTensor<R>,
        k: &RudaTensor<R>,
        v: &RudaTensor<R>,
        per_block_mean: bool,
    ) -> Result<Self, InferenceError> {
        for t in [q, k, v] {
            check(t, q)?;
            super::float(t)?;
        }
        if q.meta.num_dims() != 4
            || k.meta.num_dims() != 4
            || k.meta.shape() != v.meta.shape()
            || q.dtype != k.dtype
            || q.dtype != v.dtype
        {
            return Err(InferenceError(
                "NVFP4 dense QKV need matching floating [B,H,T,D]",
            ));
        }
        let s = q.meta.shape();
        let ks = k.meta.shape();
        let (b, h, hk, m, n, d) = (s[0], s[1], ks[1], s[2], ks[2], s[3]);
        if b == 0
            || h == 0
            || hk == 0
            || h % hk != 0
            || m == 0
            || n == 0
            || ks[0] != b
            || ks[3] != d
            || !matches!(d, 64 | 128)
        {
            return Err(InferenceError(
                "NVFP4 dense geometry needs nonempty GQA and D=64/128",
            ));
        }
        let mp = m.div_ceil(128) * 128;
        let np = n.div_ceil(128) * 128;
        let g = if per_block_mean { mp / 128 } else { 1 };
        for size in [
            b.checked_mul(h)
                .and_then(|x| x.checked_mul(mp))
                .and_then(|x| x.checked_mul(d)),
            b.checked_mul(hk)
                .and_then(|x| x.checked_mul(np))
                .and_then(|x| x.checked_mul(d)),
            b.checked_mul(h)
                .and_then(|x| x.checked_mul(g))
                .and_then(|x| x.checked_mul(np)),
        ] {
            if size.is_none_or(|x| x > u32::MAX as usize) {
                return Err(InferenceError("NVFP4 padded storage exceeds U32"));
            }
        }
        Ok(Self {
            query: q.clone(),
            key: k.clone(),
            value: v.clone(),
            q_mean: alloc(q, [b, h, g, d], DType::F32),
            k_mean: alloc(q, [b, hk, 1, d], DType::F32),
            q_center: alloc(q, [b, h, mp, d], q.dtype),
            k_center: alloc(q, [b, hk, np, d], q.dtype),
            v_pad: alloc(q, [b, hk, np, d], q.dtype),
            packed: PackedQkv {
                query: alloc(q, [b, h, mp, d / 2], DType::U8),
                key: alloc(q, [b, hk, np, d / 2], DType::U8),
                value_transposed: alloc(q, [b, hk, d, np / 2], DType::U8),
                query_scale: alloc(q, [b, h, mp, d / 16], DType::U8),
                key_scale: alloc(q, [b, hk, np, d / 16], DType::U8),
                value_scale_transposed: alloc(q, [b, hk, d, np / 16], DType::U8),
                correction: alloc(q, [b, h, g, np], DType::F32),
                per_block_mean,
            },
            b,
            h,
            hk,
            m,
            n,
            mp,
            np,
            d,
            g,
        })
    }
    /// Destinations and retained scratch must not be concurrently used.
    pub unsafe fn run(&self) -> Result<&PackedQkv<R>, InferenceError> {
        for (x, mean, tokens, pad, groups, query) in [
            (&self.query, &self.q_mean, self.m, self.mp, self.g, true),
            (&self.key, &self.k_mean, self.n, self.np, 1, false),
        ] {
            let n = mean.meta.num_elements();
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            means::launch::<R>(
                &x.client,
                count,
                dim,
                x.clone().into_array_arg(),
                mean.clone().into_array_arg(),
                n as u32,
                tokens as u32,
                pad as u32,
                groups as u32,
                self.d as u32,
                query,
                self.packed.per_block_mean,
                x.dtype.into(),
            );
        }
        for (x, mean, out, tokens, pad, groups, kind) in [
            (
                &self.query,
                &self.q_mean,
                &self.q_center,
                self.m,
                self.mp,
                self.g,
                0u32,
            ),
            (
                &self.key,
                &self.k_mean,
                &self.k_center,
                self.n,
                self.np,
                1,
                1,
            ),
            (
                &self.value,
                &self.k_mean,
                &self.v_pad,
                self.n,
                self.np,
                1,
                2,
            ),
        ] {
            let n = out.meta.num_elements();
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            center::launch::<R>(
                &x.client,
                count,
                dim,
                x.clone().into_array_arg(),
                mean.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                tokens as u32,
                pad as u32,
                groups as u32,
                self.d as u32,
                kind,
                self.packed.per_block_mean,
                x.dtype.into(),
            );
        }
        let p = &self.packed;
        for (x, out, sf, rows, cols, mode) in [
            (
                &self.q_center,
                &p.query,
                &p.query_scale,
                self.mp,
                self.d,
                0u32,
            ),
            (&self.k_center, &p.key, &p.key_scale, self.np, self.d, 1),
            (
                &self.v_pad,
                &p.value_transposed,
                &p.value_scale_transposed,
                self.d,
                self.np,
                2,
            ),
        ] {
            let n = sf.meta.num_elements();
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            quantize_vectors::launch::<R>(
                &x.client,
                count,
                dim,
                x.clone().into_array_arg(),
                out.clone().into_array_arg(),
                sf.clone().into_array_arg(),
                n as u32,
                rows as u32,
                cols as u32,
                mode,
                x.dtype.into(),
            );
        }
        let n = p.correction.meta.num_elements();
        let dim = RudaDim::new(self.query.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&self.query.client, n, dim);
        correction::launch::<R>(
            &self.query.client,
            count,
            dim,
            self.q_mean.clone().into_array_arg(),
            self.k_center.clone().into_array_arg(),
            p.correction.clone().into_array_arg(),
            n as u32,
            self.h as u32,
            self.hk as u32,
            self.g as u32,
            self.np as u32,
            self.d as u32,
            self.query.dtype.into(),
        );
        Ok(p)
    }
}
pub fn quantize_qkv<R: Runtime>(
    q: &RudaTensor<R>,
    k: &RudaTensor<R>,
    v: &RudaTensor<R>,
    per_block_mean: bool,
) -> Result<PackedQkv<R>, InferenceError> {
    let prepared = PreparedQuantize::prepare(q, k, v, per_block_mean)?;
    unsafe {
        prepared.run()?;
    }
    Ok(prepared.packed)
}
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub scale: Option<f32>,
    pub causal: bool,
    pub unpadded_k_len: Option<usize>,
    pub output_dtype: DType,
    pub return_lse: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            scale: None,
            causal: false,
            unpadded_k_len: None,
            output_dtype: DType::BF16,
            return_lse: true,
        }
    }
}
pub struct Output<R: Runtime> {
    pub values: RudaTensor<R>,
    pub lse: Option<RudaTensor<R>>,
}
pub struct PreparedAttention<'a, R: Runtime> {
    packed: &'a PackedQkv<R>,
    pub output: Output<R>,
    options: Options,
}
impl<'a, R: Runtime> PreparedAttention<'a, R> {
    pub fn prepare(packed: &'a PackedQkv<R>, options: Options) -> Result<Self, InferenceError> {
        let (b, h, _, m, n, d) = packed.geometry()?;
        if options.scale.is_some_and(|v| !v.is_finite())
            || options.unpadded_k_len.is_some_and(|v| v == 0 || v > n)
            || !matches!(options.output_dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(InferenceError("invalid NVFP4 attention options"));
        }
        let output = Output {
            values: alloc(&packed.query, [b, h, m, d], options.output_dtype),
            lse: if options.return_lse {
                Some(alloc(&packed.query, [b, h, m], DType::F32))
            } else {
                None
            },
        };
        Ok(Self {
            packed,
            output,
            options,
        })
    }
    pub unsafe fn run(&self) -> Result<&Output<R>, InferenceError> {
        unsafe {
            self.run_into(&self.output)?;
        }
        Ok(&self.output)
    }
    pub unsafe fn run_into(&self, destination: &Output<R>) -> Result<(), InferenceError> {
        let (b, h, hk, m, n, d) = self.packed.geometry()?;
        let out = &destination.values;
        check(out, &self.packed.query)?;
        if out.dtype != self.options.output_dtype
            || out.meta.shape()[..] != [b, h, m, d]
            || destination.lse.is_some() != self.options.return_lse
        {
            return Err(InferenceError("NVFP4 output destination mismatch"));
        }
        if let Some(lse) = &destination.lse {
            check(lse, out)?;
            if lse.dtype != DType::F32 || lse.meta.shape()[..] != [b, h, m] {
                return Err(InferenceError("NVFP4 LSE destination mismatch"));
            }
        }
        if super::plane(out, m, b * h)? != 32 {
            return Err(InferenceError(
                "NVFP4 packed attention requires 32-lane device planes",
            ));
        }
        let p = self.packed;
        attend::launch::<R>(
            &out.client,
            RudaCount::Static(m as u32, (b * h) as u32, 1),
            RudaDim::new_1d(32),
            p.query.clone().into_array_arg(),
            p.key.clone().into_array_arg(),
            p.value_transposed.clone().into_array_arg(),
            p.query_scale.clone().into_array_arg(),
            p.key_scale.clone().into_array_arg(),
            p.value_scale_transposed.clone().into_array_arg(),
            p.correction.clone().into_array_arg(),
            out.clone().into_array_arg(),
            destination
                .lse
                .as_ref()
                .unwrap_or(&p.correction)
                .clone()
                .into_array_arg(),
            h as u32,
            hk as u32,
            m as u32,
            n as u32,
            self.options.unpadded_k_len.unwrap_or(n) as u32,
            self.options.scale.unwrap_or(1.0 / (d as f32).sqrt()),
            d,
            self.options.causal,
            p.per_block_mean,
            self.options.return_lse,
            out.dtype.into(),
        );
        Ok(())
    }
}
pub fn forward<R: Runtime>(
    packed: &PackedQkv<R>,
    options: Options,
) -> Result<Output<R>, InferenceError> {
    let p = PreparedAttention::prepare(packed, options)?;
    unsafe {
        p.run()?;
    }
    Ok(p.output)
}
#[ruda(launch)]
pub fn means<F: Float>(
    input: &Array<F>,
    mean: &mut Array<f32>,
    n: u32,
    tokens: u32,
    pad: u32,
    groups: u32,
    d: u32,
    #[comptime] query: bool,
    #[comptime] block: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % d as usize;
        let g = i / d as usize % groups as usize;
        let head = i / (d * groups) as usize;
        let mut begin: usize = 0usize;
        let mut end = tokens as usize;
        let mut denominator = tokens as f32;
        if comptime!(query) {
            end = pad as usize;
            denominator = pad as f32;
            if comptime!(block) {
                begin = g * 128;
                end = begin + 128;
                denominator = 128.0;
            }
        }
        let mut sum: f32 = 0.0f32;
        for token in begin..end {
            if token < tokens as usize {
                sum += f32::cast_from(input[(head * tokens as usize + token) * d as usize + col]);
            }
        }
        mean[i] = f32::cast_from(F::cast_from(sum / denominator));
    }
}
#[ruda(launch)]
pub fn center<F: Float>(
    input: &Array<F>,
    mean: &Array<f32>,
    out: &mut Array<F>,
    n: u32,
    tokens: u32,
    pad: u32,
    groups: u32,
    d: u32,
    #[comptime] kind: u32,
    #[comptime] block: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % d as usize;
        let token = i / d as usize % pad as usize;
        let head = i / (d * pad) as usize;
        let mut value: f32 = 0.0f32;
        if token < tokens as usize {
            value = f32::cast_from(input[(head * tokens as usize + token) * d as usize + col]);
        }
        if comptime!(kind == 0) {
            let mut g: usize = 0usize;
            if comptime!(block) {
                g = token / 128;
            }
            value -= mean[(head * groups as usize + g) * d as usize + col];
        }
        if comptime!(kind == 1) {
            if token < tokens as usize {
                value -= mean[head * d as usize + col];
            }
        }
        out[i] = F::cast_from(value);
    }
}
#[ruda]
pub fn scale_index(row: usize, col: usize, cols: usize) -> usize {
    row / 64 * 64 * cols + col / 4 * 256 + (row % 16) * 16 + (row % 64 / 16) * 4 + col % 4
}
#[ruda]
pub fn permuted_token(token: usize) -> usize {
    let local = token % 32;
    token / 32 * 32 + local / 8 * 2 + (local % 8 / 2) * 8 + local % 2
}
#[ruda(launch)]
pub fn quantize_vectors<F: Float>(
    input: &Array<F>,
    out: &mut Array<u8>,
    scales: &mut Array<u8>,
    n: u32,
    rows: u32,
    cols: u32,
    #[comptime] mode: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let blocks = cols as usize / 16;
        let block = i % blocks;
        let row = i / blocks % rows as usize;
        let head = i / (blocks * rows as usize);
        let mut source = row;
        if comptime!(mode == 1) {
            source = permuted_token(row);
        }
        let mut maximum: f32 = 0.0f32;
        let mut values = Array::<f32>::new(16usize);
        #[unroll]
        for j in 0usize..16 {
            let col = block * 16 + j;
            let mut at = (head * rows as usize + source) * cols as usize + col;
            if comptime!(mode == 2) {
                at = (head * cols as usize + col) * rows as usize + row;
            }
            let value = f32::cast_from(input[at]);
            values[j] = value;
            maximum = f32::max(maximum, value.abs());
        }
        let bits = rublas::tensor_float_quant::fp8_encode(maximum / 6.0, false);
        let scale = rublas::tensor_float_quant::fp8_decode(bits, false);
        let mut inv: f32 = 0.0f32;
        if scale != 0.0 {
            inv = 1.0 / scale;
        }
        scales[head * rows as usize * blocks + scale_index(row, block, blocks)] =
            u8::cast_from(bits);
        #[unroll]
        for j in 0usize..8 {
            let a = rublas::tensor_float_quant::e2m1_encode(values[j * 2] * inv);
            let b = rublas::tensor_float_quant::e2m1_encode(values[j * 2 + 1] * inv);
            out[(head * rows as usize + row) * (cols as usize / 2) + block * 8 + j] =
                u8::cast_from(a | (b << 4));
        }
    }
}
#[ruda(launch)]
pub fn correction<F: Float>(
    mean: &Array<f32>,
    key: &Array<F>,
    out: &mut Array<f32>,
    n: u32,
    h: u32,
    hk: u32,
    groups: u32,
    np: u32,
    d: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let token = i % np as usize;
        let g = i / np as usize % groups as usize;
        let head = i / (np * groups) as usize;
        let kv = head / h as usize * hk as usize + (head % h as usize) / (h / hk) as usize;
        let mut dot: f32 = 0.0f32;
        for col in 0..d as usize {
            dot = fma(
                mean[(head * groups as usize + g) * d as usize + col],
                f32::cast_from(key[(kv * np as usize + token) * d as usize + col]),
                dot,
            );
        }
        out[i] = dot;
    }
}
#[ruda]
pub fn unpack(
    data: &Array<u8>,
    scales: &Array<u8>,
    head: usize,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
) -> f32 {
    let byte = u32::cast_from(data[(head * rows + row) * (cols / 2) + col / 2]);
    let value = rublas::tensor_float_quant::e2m1_decode((byte >> ((col % 2) * 4) as u32) & 15u32);
    let sf = rublas::tensor_float_quant::fp8_decode(
        u32::cast_from(scales[head * rows * (cols / 16) + scale_index(row, col / 16, cols / 16)]),
        false,
    );
    value * sf
}
#[ruda(launch)]
pub fn attend<F: Float>(
    q: &Array<u8>,
    k: &Array<u8>,
    v: &Array<u8>,
    qs: &Array<u8>,
    ks: &Array<u8>,
    vs: &Array<u8>,
    correction: &Array<f32>,
    out: &mut Array<F>,
    lse: &mut Array<f32>,
    heads: u32,
    kv_heads: u32,
    m: u32,
    n: u32,
    valid: u32,
    scale: f32,
    #[comptime] d: usize,
    #[comptime] causal: bool,
    #[comptime] block_mean: bool,
    #[comptime] return_lse: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let kv = head / heads as usize * kv_heads as usize
        + (head % heads as usize) / (heads / kv_heads) as usize;
    let mut query = Array::<f32>::new((d + 31) / 32);
    let mut acc = Array::<f32>::new((d + 31) / 32);
    #[unroll]
    for j in 0usize..((d + 31) / 32) {
        let col = lane + j * 32;
        query[j] = 0.0;
        acc[j] = 0.0;
        if col < d {
            query[j] = unpack(q, qs, head, row, col, m as usize, d);
        }
    }
    let mut groups: usize = 1usize;
    let mut group: usize = 0usize;
    if comptime!(block_mean) {
        groups = m as usize / 128;
        group = row / 128;
    }
    let mut maximum: f32 = 0.0f32;
    let mut total: f32 = 0.0f32;
    for token in 0..valid as usize {
        let mut visible = true;
        if comptime!(causal) {
            visible = (token as i64) <= row as i64 + n as i64 - m as i64;
        }
        if visible {
            let mut dot: f32 = 0.0f32;
            #[unroll]
            for j in 0usize..((d + 31) / 32) {
                let col = lane + j * 32;
                if col < d {
                    dot = fma(
                        query[j],
                        unpack(k, ks, kv, permuted_token(token), col, n as usize, d),
                        dot,
                    );
                }
            }
            let score =
                (plane_sum(dot) + correction[(head * groups + group) * n as usize + token]) * scale;
            let mut next = score;
            let mut old: f32 = 0.0f32;
            if total != 0.0 {
                next = f32::max(maximum, score);
                old = (maximum - next).exp();
            }
            let weight = (score - next).exp();
            total = fma(total, old, weight);
            #[unroll]
            for j in 0usize..((d + 31) / 32) {
                let col = lane + j * 32;
                if col < d {
                    acc[j] = fma(
                        weight,
                        unpack(v, vs, kv, col, token, d, n as usize),
                        acc[j] * old,
                    );
                }
            }
            maximum = next;
        }
    }
    let index = head * m as usize + row;
    #[unroll]
    for j in 0usize..((d + 31) / 32) {
        let col = lane + j * 32;
        if col < d {
            let mut value: f32 = 0.0f32;
            if total != 0.0 {
                value = acc[j] / total;
            }
            out[index * d + col] = F::cast_from(value);
        }
    }
    if comptime!(return_lse) {
        if lane == 0 {
            let mut value = f32::NEG_INFINITY;
            if total != 0.0 {
                value = maximum + total.ln();
            }
            lse[index] = value;
        }
    }
}
