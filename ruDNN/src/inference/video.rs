//! Across-head Q/K RMSNorm, spatial/frame RoPE and independent V copy/FP8 pack.
use super::{InferenceError, alloc, check, float, plane};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
#[derive(Clone, Copy, Debug)]
pub struct VideoOptions {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub frame_channels: usize,
    pub height_channels: usize,
    pub width_channels: usize,
    pub query_heads: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub epsilon: f32,
    pub base: f32,
    pub interleaved: bool,
    pub factor: f32,
    pub low: f32,
    pub high: f32,
    pub attention_factor: f32,
    pub normalize: bool,
    pub output_fp8: bool,
    pub output_multiplier: f32,
    pub value_multiplier: f32,
}
pub fn qk_rmsnorm_rope<R: Runtime>(
    qkv: &RudaTensor<R>,
    q_weight: &RudaTensor<R>,
    k_weight: &RudaTensor<R>,
    options: VideoOptions,
) -> Result<super::fused::Qkv<R>, InferenceError> {
    let prepared = PreparedVideo::prepare(qkv, q_weight, k_weight, options)?;
    unsafe {
        prepared.run()?;
    }
    Ok(super::fused::Qkv {
        query: prepared.output.query.clone(),
        key: prepared.output.key.clone(),
        value: prepared.output.value.clone(),
    })
}
/// Fixed-address video Q/K normalization, frame/height/width RoPE and V pack.
pub struct PreparedVideo<R: Runtime> {
    input: super::PreparedPack<R>,
    qw: super::PreparedPack<R>,
    kw: super::PreparedPack<R>,
    qw_f32: RudaTensor<R>,
    kw_f32: RudaTensor<R>,
    pub output: super::fused::Qkv<R>,
    options: VideoOptions,
    row_width: usize,
    sequence: usize,
    tokens: usize,
    lanes: u32,
}
impl<R: Runtime> PreparedVideo<R> {
    pub fn prepare(
        qkv: &RudaTensor<R>,
        q_weight: &RudaTensor<R>,
        k_weight: &RudaTensor<R>,
        o: VideoOptions,
    ) -> Result<Self, InferenceError> {
        let input = super::PreparedPack::prepare(qkv)?;
        let x = &input.output;
        float(x)?;
        let d = o.head_dim;
        let heads = o
            .query_heads
            .checked_add(o.key_heads)
            .and_then(|n| n.checked_add(o.value_heads))
            .ok_or(InferenceError("video head count overflow"))?;
        let row_width = heads
            .checked_mul(d)
            .ok_or(InferenceError("video QKV width overflow"))?;
        let sequence = o
            .frames
            .checked_mul(o.height)
            .and_then(|n| n.checked_mul(o.width))
            .ok_or(InferenceError("video grid overflow"))?;
        if !(x.meta.num_dims() == 2 || x.meta.num_dims() == 3)
            || row_width == 0
            || *x.meta.shape().last().unwrap() != row_width
            || sequence == 0
            || d == 0
            || d % 2 != 0
            || o.query_heads == 0
            || o.key_heads == 0
            || o.value_heads == 0
            || o.frame_channels
                .checked_add(o.height_channels)
                .and_then(|n| n.checked_add(o.width_channels))
                != Some(d)
            || [o.frame_channels, o.height_channels, o.width_channels]
                .iter()
                .any(|c| c % 2 != 0)
            || !o.epsilon.is_finite()
            || o.epsilon <= 0.0
            || !o.base.is_finite()
            || o.base <= 0.0
            || !o.factor.is_finite()
            || o.factor <= 0.0
            || !o.low.is_finite()
            || !o.high.is_finite()
            || !o.attention_factor.is_finite()
            || !o.output_multiplier.is_finite()
            || !o.value_multiplier.is_finite()
        {
            return Err(InferenceError("video QKV/spatial RoPE recipe invalid"));
        }
        let tokens = x.meta.num_elements() / row_width;
        if tokens % sequence != 0 {
            return Err(InferenceError(
                "video tokens must contain whole spatial grids",
            ));
        }
        let qw = super::PreparedPack::prepare(q_weight)?;
        let kw = super::PreparedPack::prepare(k_weight)?;
        for (w, h) in [(&qw.output, o.query_heads), (&kw.output, o.key_heads)] {
            check(w, x)?;
            float(w)?;
            if w.meta.shape()[..] != [h * d] {
                return Err(InferenceError("video RMS weights must span all Q/K heads"));
            }
        }
        let qw_f32 = alloc(x, [o.query_heads * d], DType::F32);
        let kw_f32 = alloc(x, [o.key_heads * d], DType::F32);
        let dtype = if o.output_fp8 { DType::U8 } else { x.dtype };
        let output = super::fused::Qkv {
            query: alloc(x, [tokens, o.query_heads, d], dtype),
            key: alloc(x, [tokens, o.key_heads, d], dtype),
            value: alloc(x, [tokens, o.value_heads, d], dtype),
        };
        let lanes = plane(x, tokens, 1)?;
        Ok(Self {
            input,
            qw,
            kw,
            qw_f32,
            kw_f32,
            output,
            options: o,
            row_width,
            sequence,
            tokens,
            lanes,
        })
    }
    pub unsafe fn run(&self) -> Result<&super::fused::Qkv<R>, InferenceError> {
        unsafe {
            self.input.run();
            self.qw.run();
            self.kw.run();
            super::norm::cast_into(&self.qw.output, &self.qw_f32)?;
            super::norm::cast_into(&self.kw.output, &self.kw_f32)?;
        }
        let x = &self.input.output;
        let o = self.options;
        let d = o.head_dim;
        for (kind, out, weight, count, start) in [
            (
                0u32,
                &self.output.query,
                &self.qw_f32,
                o.query_heads,
                0usize,
            ),
            (
                1,
                &self.output.key,
                &self.kw_f32,
                o.key_heads,
                o.query_heads * d,
            ),
            (
                2,
                &self.output.value,
                &self.qw_f32,
                o.value_heads,
                (o.query_heads + o.key_heads) * d,
            ),
        ] {
            if self.tokens > 0 {
                video_component::launch::<R>(
                    &x.client,
                    RudaCount::Static(self.tokens as u32, 1, 1),
                    RudaDim::new_1d(self.lanes),
                    x.clone().into_array_arg(),
                    weight.clone().into_array_arg(),
                    out.clone().into_array_arg(),
                    self.row_width as u32,
                    start as u32,
                    count as u32,
                    d as u32,
                    self.sequence as u32,
                    o.height as u32,
                    o.width as u32,
                    o.frame_channels as u32,
                    o.height_channels as u32,
                    o.width_channels as u32,
                    o.epsilon,
                    o.base,
                    o.factor,
                    o.low,
                    o.high,
                    o.attention_factor,
                    if kind == 2 {
                        o.value_multiplier
                    } else {
                        o.output_multiplier
                    },
                    kind,
                    o.normalize,
                    o.interleaved,
                    o.output_fp8,
                    x.dtype.into(),
                    out.dtype.into(),
                );
            }
        }
        Ok(&self.output)
    }
}
#[ruda(launch)]
pub fn video_component<F: Float, O: Numeric>(
    input: &Array<F>,
    weight: &Array<f32>,
    out: &mut Array<O>,
    row_width: u32,
    start: u32,
    heads: u32,
    d: u32,
    seq: u32,
    height: u32,
    width: u32,
    fc: u32,
    hc: u32,
    wc: u32,
    epsilon: f32,
    base: f32,
    factor: f32,
    low: f32,
    high: f32,
    attention: f32,
    multiplier: f32,
    #[comptime] kind: u32,
    #[comptime] normalized: bool,
    #[comptime] interleaved: bool,
    #[comptime] fp8: bool,
    #[define(F)] _f: StorageType,
    #[define(O)] _o: StorageType,
) {
    let token = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let span = heads as usize * d as usize;
    let source = token * row_width as usize + start as usize;
    let mut inv: f32 = 1.0f32;
    if comptime!(kind != 2 && normalized) {
        let mut j = lane;
        let mut square: f32 = 0.0f32;
        while j < span {
            let x = f32::cast_from(input[source + j]);
            square += x * x;
            j += PLANE_DIM as usize;
        }
        inv = 1.0 / (plane_sum(square) / span as f32 + epsilon).sqrt();
    }
    if comptime!(kind == 2) {
        let mut j = lane;
        while j < span {
            let value = f32::cast_from(input[source + j]);
            let mut result = O::cast_from(value);
            if comptime!(fp8) {
                result = O::cast_from(rublas::tensor_float_quant::fp8_encode(
                    value * multiplier,
                    false,
                ));
            }
            out[token * span + j] = result;
            j += PLANE_DIM as usize;
        }
    } else {
        let mut pair = lane;
        while pair < span / 2 {
            let head = pair / (d as usize / 2);
            let local = pair % (d as usize / 2);
            let mut first = local;
            let mut second = local + d as usize / 2;
            if comptime!(interleaved) {
                first = local * 2;
                second = first + 1;
            }
            let a_index = head * d as usize + first;
            let b_index = head * d as usize + second;
            let mut a = f32::cast_from(input[source + a_index]);
            let mut b = f32::cast_from(input[source + b_index]);
            if comptime!(normalized) {
                a = a * inv * weight[a_index];
                b = b * inv * weight[b_index];
                if comptime!(!fp8) {
                    a = f32::cast_from(F::cast_from(a));
                    b = f32::cast_from(F::cast_from(b));
                }
            }
            let grid_token = token % seq as usize;
            let mut position = grid_token / (height as usize * width as usize);
            let mut channels = fc as usize;
            let mut freq_index = local;
            if local >= fc as usize / 2 {
                channels = hc as usize;
                freq_index = local - fc as usize / 2;
                position = grid_token / width as usize % height as usize;
            }
            if local >= (fc + hc) as usize / 2 {
                channels = wc as usize;
                freq_index = local - (fc + hc) as usize / 2;
                position = grid_token % width as usize;
            }
            let mut frequency = f32::powf(base, -2.0 * freq_index as f32 / channels as f32);
            if factor != 1.0 {
                let mut upper = high;
                if (low - upper).abs() <= 0.000001 {
                    upper += 0.001;
                }
                let ramp = f32::min(
                    f32::max((freq_index as f32 - low) / (upper - low), 0.0),
                    1.0,
                );
                frequency = frequency * (1.0 - ramp) + frequency / factor * ramp;
            }
            let angle = position as f32 * frequency;
            let cosine = angle.cos() * attention;
            let sine = angle.sin() * attention;
            let first_value = fma(a, cosine, -b * sine);
            let second_value = fma(b, cosine, a * sine);
            let mut oa = O::cast_from(first_value);
            let mut ob = O::cast_from(second_value);
            if comptime!(fp8) {
                oa = O::cast_from(rublas::tensor_float_quant::fp8_encode(
                    first_value * multiplier,
                    false,
                ));
                ob = O::cast_from(rublas::tensor_float_quant::fp8_encode(
                    second_value * multiplier,
                    false,
                ));
            }
            out[token * span + a_index] = oa;
            out[token * span + b_index] = ob;
            pair += PLANE_DIM as usize;
        }
    }
}
