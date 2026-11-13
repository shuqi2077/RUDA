//! Per-vector block-scaled recurrent-state storage with Philox rounding.
use super::{InferenceError, alloc, check, plane};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateFormat {
    I16,
    I8,
    E4m3,
    E5m2,
}
pub struct QuantizedState<R: Runtime> {
    pub data: RudaTensor<R>,
    pub scale: RudaTensor<R>,
    pub format: StateFormat,
}
#[derive(Clone, Copy, Debug)]
pub struct Rounding {
    pub seed: u64,
    pub offset: u64,
    pub philox_rounds: usize,
}
impl Default for Rounding {
    fn default() -> Self {
        Self {
            seed: 0,
            offset: 0,
            philox_rounds: 0,
        }
    }
}
impl<R: Runtime> QuantizedState<R> {
    pub fn copy(&self) -> Self {
        let copy = |t: &RudaTensor<R>| {
            let out = alloc(t, t.meta.shape().clone(), t.dtype);
            ruda_kernel::library::tensor::copy_into(
                &t.client,
                t.clone().binding(),
                out.clone().binding(),
                t.dtype.into(),
            );
            out
        };
        Self {
            data: copy(&self.data),
            scale: copy(&self.scale),
            format: self.format,
        }
    }
    /// Requantize into retained storage, optionally reading an I64/U64 seed on
    /// the GPU so graph replay can change RNG state without a host readback.
    pub unsafe fn encode_into(
        &self,
        input: &RudaTensor<R>,
        rounding: Rounding,
        seed: Option<&RudaTensor<R>>,
    ) -> Result<(), InferenceError> {
        self.validate()?;
        check(input, &self.data)?;
        if input.dtype != DType::F32
            || input.meta.shape() != self.data.meta.shape()
            || rounding.philox_rounds > 64
        {
            return Err(InferenceError(
                "prepared state encoder input/rounding mismatch",
            ));
        }
        if let Some(seed) = seed {
            check(seed, input)?;
            if !matches!(seed.dtype, DType::I64 | DType::U64) || seed.meta.num_elements() != 1 {
                return Err(InferenceError("state rounding seed must be I64/U64 scalar"));
            }
        }
        let width = *input
            .meta
            .shape()
            .last()
            .ok_or(InferenceError("state encoding needs an axis"))?;
        if width == 0 || self.scale.meta.num_elements() != input.meta.num_elements() / width {
            return Err(InferenceError("prepared state scale shape mismatch"));
        }
        let rows = input.meta.num_elements() / width;
        let lanes = plane(input, rows, 1)?;
        if rows > 0 {
            encode_rows::launch::<R>(
                &input.client,
                RudaCount::Static(rows as u32, 1, 1),
                RudaDim::new_1d(lanes),
                input.clone().into_array_arg(),
                seed.unwrap_or(input).clone().into_array_arg(),
                self.data.clone().into_array_arg(),
                self.scale.clone().into_array_arg(),
                width as u32,
                rounding.seed,
                rounding.offset,
                match self.format {
                    StateFormat::I16 => 32767.0,
                    StateFormat::I8 => 127.0,
                    StateFormat::E4m3 => 448.0,
                    StateFormat::E5m2 => 57344.0,
                },
                matches!(self.format, StateFormat::E4m3 | StateFormat::E5m2),
                self.format == StateFormat::E5m2,
                rounding.philox_rounds,
                seed.is_some(),
                self.data.dtype.into(),
            );
        }
        Ok(())
    }
    pub(crate) fn validate(&self) -> Result<usize, InferenceError> {
        check(&self.data, &self.data)?;
        check(&self.scale, &self.data)?;
        let shape = self.data.meta.shape();
        let width = *shape
            .last()
            .ok_or(InferenceError("quantized state needs a last axis"))?;
        let expected = match self.format {
            StateFormat::I16 => DType::I16,
            StateFormat::I8 => DType::I8,
            _ => DType::U8,
        };
        if width == 0
            || width > u32::MAX as usize
            || self.data.dtype != expected
            || self.scale.dtype != DType::F32
            || self.scale.meta.shape()[..] != shape[..shape.len() - 1]
        {
            return Err(InferenceError("quantized-state shape/storage mismatch"));
        }
        Ok(width)
    }
    /// Allocate persistent encoded storage without reading or encoding input.
    pub fn allocate(
        like: &RudaTensor<R>,
        shape: Shape,
        format: StateFormat,
    ) -> Result<Self, InferenceError> {
        check(like, like)?;
        let width = *shape
            .last()
            .ok_or(InferenceError("quantized state needs a last axis"))?;
        if width == 0
            || shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("quantized state allocation exceeds U32"));
        }
        let dtype = match format {
            StateFormat::I16 => DType::I16,
            StateFormat::I8 => DType::I8,
            _ => DType::U8,
        };
        let scale = alloc(
            like,
            Shape::from(shape[..shape.len() - 1].to_vec()),
            DType::F32,
        );
        Ok(Self {
            data: alloc(like, shape, dtype),
            scale,
            format,
        })
    }
    pub fn decode(&self) -> Result<RudaTensor<R>, InferenceError> {
        self.validate()?;
        let out = alloc(&self.data, self.data.meta.shape().clone(), DType::F32);
        unsafe {
            self.decode_into(&out)?;
        }
        Ok(out)
    }
    /// Output is retained FP32 storage, disjoint from payload and scale arrays.
    pub unsafe fn decode_into(&self, out: &RudaTensor<R>) -> Result<(), InferenceError> {
        let width = self.validate()?;
        check(out, &self.data)?;
        if out.dtype != DType::F32 || out.meta.shape() != self.data.meta.shape() {
            return Err(InferenceError("state decoder output shape/dtype mismatch"));
        }
        let n = out.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(out.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
            decode::launch::<R>(
                &out.client,
                count,
                dim,
                self.data.clone().into_array_arg(),
                self.scale.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                width as u32,
                matches!(self.format, StateFormat::E4m3 | StateFormat::E5m2),
                self.format == StateFormat::E5m2,
                self.data.dtype.into(),
            );
        }
        Ok(())
    }
}
/// Raw payload/scale row copy, preserving even zero-prefix checkpoint bits.
/// `flags_zero` selects flags == 0 instead of flags != 0. Destination IDs must
/// be unique among enabled requests. All owners must remain live/exclusive.
pub unsafe fn copy_selected_rows<R: Runtime>(
    source: &QuantizedState<R>,
    source_rows: &RudaTensor<R>,
    destination: &QuantizedState<R>,
    destination_rows: &RudaTensor<R>,
    flags: Option<&RudaTensor<R>>,
    flags_zero: bool,
) -> Result<(), InferenceError> {
    source.validate()?;
    destination.validate()?;
    for t in [
        &source.data,
        &source.scale,
        source_rows,
        &destination.data,
        &destination.scale,
        destination_rows,
    ] {
        check(t, &source.data)?;
    }
    if source.format != destination.format
        || source.data.dtype != destination.data.dtype
        || source.data.meta.num_dims() < 2
        || destination.data.meta.num_dims() != source.data.meta.num_dims()
        || source.data.meta.shape()[1..] != destination.data.meta.shape()[1..]
        || source_rows.dtype != DType::I32
        || destination_rows.dtype != DType::I32
        || source_rows.meta.num_dims() != 1
        || source_rows.meta.shape() != destination_rows.meta.shape()
    {
        return Err(InferenceError(
            "quantized state row copy dimensions mismatch",
        ));
    }
    if let Some(f) = flags {
        check(f, &source.data)?;
        if f.dtype != DType::I32 || f.meta.shape() != source_rows.meta.shape() {
            return Err(InferenceError("quantized state copy flags mismatch"));
        }
    }
    let width: usize = source.data.meta.shape()[1..].iter().product();
    let scale_width: usize = source.scale.meta.shape()[1..].iter().product();
    let n = source_rows
        .meta
        .num_elements()
        .checked_mul(width)
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(InferenceError("quantized row copy exceeds U32"))?;
    if n > 0 {
        let dim = RudaDim::new(source.data.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&source.data.client, n, dim);
        copy_rows::launch::<R>(
            &source.data.client,
            count,
            dim,
            source.data.clone().into_array_arg(),
            source.scale.clone().into_array_arg(),
            source_rows.clone().into_array_arg(),
            destination_rows.clone().into_array_arg(),
            flags.unwrap_or(source_rows).clone().into_array_arg(),
            destination.data.clone().into_array_arg(),
            destination.scale.clone().into_array_arg(),
            n as u32,
            width as u32,
            scale_width as u32,
            source.data.meta.shape()[0] as u32,
            destination.data.meta.shape()[0] as u32,
            flags.is_some(),
            flags_zero,
            source.data.dtype.into(),
        );
    }
    Ok(())
}
#[ruda(launch)]
pub fn copy_rows<I: Int>(
    source: &Array<I>,
    source_scale: &Array<f32>,
    source_rows: &Array<i32>,
    destination_rows: &Array<i32>,
    flags: &Array<i32>,
    out: &mut Array<I>,
    scale: &mut Array<f32>,
    n: u32,
    width: u32,
    scale_width: u32,
    source_slots: u32,
    destination_slots: u32,
    #[comptime] conditional: bool,
    #[comptime] flags_zero: bool,
    #[define(I)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let request = i / width as usize;
        let col = i % width as usize;
        let src = source_rows[request];
        let dst = destination_rows[request];
        let mut enabled = true;
        if comptime!(conditional) {
            enabled = flags[request] != 0;
            if comptime!(flags_zero) {
                enabled = flags[request] == 0;
            }
        }
        if enabled
            && src >= 0
            && src < source_slots as i32
            && dst >= 0
            && dst < destination_slots as i32
        {
            out[dst as usize * width as usize + col] = source[src as usize * width as usize + col];
            if col < scale_width as usize {
                scale[dst as usize * scale_width as usize + col] =
                    source_scale[src as usize * scale_width as usize + col];
            }
        }
    }
}
pub fn encode<R: Runtime>(
    input: &RudaTensor<R>,
    format: StateFormat,
    rounding: Rounding,
) -> Result<QuantizedState<R>, InferenceError> {
    let x = super::norm::cast(input, DType::F32)?;
    let out = QuantizedState::allocate(&x, x.meta.shape().clone(), format)?;
    unsafe {
        out.encode_into(&x, rounding, None)?;
    }
    Ok(out)
}

#[ruda]
pub(crate) fn philox_word(seed: u64, counter: u64, #[comptime] rounds: usize) -> u32 {
    let mut a = counter as u32;
    let mut b = (counter >> 32) as u32;
    let mut c: u32 = 0u32;
    let mut d: u32 = 0u32;
    let mut k0 = seed as u32;
    let mut k1 = (seed >> 32) as u32;
    for r in 0usize..rounds {
        let p0 = a as u64 * 0xD2511F53u64;
        let p1 = c as u64 * 0xCD9E8D57u64;
        let na = (p1 >> 32) as u32 ^ b ^ k0;
        let nb = p1 as u32;
        let nc = (p0 >> 32) as u32 ^ d ^ k1;
        let nd = p0 as u32;
        a = na;
        b = nb;
        c = nc;
        d = nd;
        k0 = k0 + 0x9E3779B9u32;
        k1 = k1 + 0xBB67AE85u32;
    }
    a
}
#[ruda(launch)]
pub fn decode<I: Int>(
    data: &Array<I>,
    scale: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    width: u32,
    #[comptime] fp8: bool,
    #[comptime] e5m2: bool,
    #[define(I)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut value = f32::cast_from(data[i]);
        if comptime!(fp8) {
            value = rublas::tensor_float_quant::fp8_decode(u32::cast_from(data[i]), e5m2);
        }
        out[i] = value * scale[i / width as usize];
    }
}
#[ruda(launch)]
pub fn encode_rows<I: Int>(
    x: &Array<f32>,
    seed_values: &Array<u64>,
    data: &mut Array<I>,
    scale: &mut Array<f32>,
    width: u32,
    seed: u64,
    offset: u64,
    limit: f32,
    #[comptime] fp8: bool,
    #[comptime] e5m2: bool,
    #[comptime] rounds: usize,
    #[comptime] device_seed: bool,
    #[define(I)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut rng_seed = seed;
    if comptime!(device_seed) {
        rng_seed = seed_values[0];
    }
    let mut j = lane;
    let mut maximum: f32 = 0.0f32;
    while j < width as usize {
        maximum = f32::max(maximum, x[row * width as usize + j].abs());
        j += PLANE_DIM as usize;
    }
    maximum = plane_max(maximum);
    let mut sf = maximum / limit;
    if sf == 0.0 {
        sf = 1.0;
    }
    if lane == 0 {
        scale[row] = sf;
    }
    j = lane;
    while j < width as usize {
        let i = row * width as usize + j;
        let value = x[i] / sf;
        let mut encoded: u32 = 0u32;
        if comptime!(fp8) {
            encoded = rublas::tensor_float_quant::fp8_encode(value, e5m2);
            if comptime!(rounds > 0) {
                let mut lo = encoded & 127u32;
                let magnitude = value.abs();
                let rounded = rublas::tensor_float_quant::fp8_decode(lo, e5m2);
                let mut hi = lo;
                if rounded > magnitude && lo > 0 {
                    lo -= 1;
                } else {
                    hi = u32::min(lo + 1, 126u32);
                    if comptime!(e5m2) {
                        hi = u32::min(lo + 1, 123u32);
                    }
                }
                let low = rublas::tensor_float_quant::fp8_decode(lo, e5m2);
                let high = rublas::tensor_float_quant::fp8_decode(hi, e5m2);
                let u = (philox_word(rng_seed, offset + i as u64, rounds) >> 8) as f32
                    * 0.000000059604644775390625;
                let mut chosen = lo;
                if high > low && u < (magnitude - low) / (high - low) {
                    chosen = hi;
                }
                encoded = chosen | ((u32::reinterpret(value) >> 24) & 128u32);
            }
            data[i] = I::cast_from(encoded);
        } else {
            let low = value.floor();
            let fraction = value - low;
            let mut rounded = low;
            if fraction > 0.5 || fraction == 0.5 && (low as i32 & 1i32) != 0 {
                rounded = low + 1.0;
            }
            if comptime!(rounds > 0) {
                let low = value.floor();
                let u = (philox_word(rng_seed, offset + i as u64, rounds) >> 8) as f32
                    * 0.000000059604644775390625;
                rounded = low;
                if u < value - low {
                    rounded += 1.0;
                }
            }
            data[i] = I::cast_from(f32::min(f32::max(rounded, -limit), limit));
        }
        j += PLANE_DIM as usize;
    }
}
