use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{Shape, element::Element};
use crate::collective::radix::RudaRadixKey;
use super::{RudaPrimitiveError, check_type};

#[ruda]
pub trait RudaHistogramSample: Numeric {
    fn even_bin(sample: Self, lower: Self, upper: Self, bins: usize) -> usize;
}

// floor(numerator * multiplier / denominator), with numerator < denominator.
// Quotient/remainder recurrence avoids a 128-bit multiply for 64-bit samples.
#[ruda]
fn mul_div(numerator: u64, multiplier: u64, denominator: u64) -> u64 {
    let mut quotient = 0u64;
    let mut remainder = 0u64;
    for step in 0..64u32 {
        quotient *= 2u64;
        if remainder >= denominator - remainder {
            remainder -= denominator - remainder;
            quotient += 1u64;
        } else {
            remainder *= 2u64;
        }
        if ((multiplier >> (63u64 - step as u64)) & 1u64) != 0 {
            if remainder >= denominator - numerator {
                remainder -= denominator - numerator;
                quotient += 1u64;
            } else {
                remainder += numerator;
            }
        }
    }
    quotient
}

macro_rules! integer_sample {
    ($($ty:ty),* $(,)?) => {$(
        #[ruda]
        impl RudaHistogramSample for $ty {
            fn even_bin(sample: Self, lower: Self, upper: Self, bins: usize) -> usize {
                let start = Self::ordered_bits(lower);
                let numerator = Self::ordered_bits(sample) - start;
                let denominator = Self::ordered_bits(upper) - start;
                mul_div(numerator, bins as u64, denominator) as usize
            }
        }
    )*};
}
integer_sample!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! float_sample {
    ($($ty:ty),* $(,)?) => {$(
        #[ruda]
        impl RudaHistogramSample for $ty {
            fn even_bin(sample: Self, lower: Self, upper: Self, bins: usize) -> usize {
                usize::cast_from((sample - lower) * (Self::cast_from(bins) / (upper - lower)))
            }
        }
    )*};
}
float_sample!(f32, f64, half::f16, half::bf16);

pub trait RudaHistogramParameters<L> {
    fn valid_parameters(bins: usize, lower: L, upper: L) -> bool;
}

#[ruda]
pub trait RudaHistogramPair<L: Numeric>: Numeric {
    fn mixed_bin(sample: Self, lower: L, upper: L, bins: usize) -> usize;
}

macro_rules! histogram_pair {
    ($sample:ty, $level:ty, $common:ty, $kind:ident) => {
        #[ruda]
        impl RudaHistogramPair<$level> for $sample {
            fn mixed_bin(sample: Self, lower: $level, upper: $level, bins: usize) -> usize {
                let value = <$common as Cast>::cast_from(sample);
                let low = <$common as Cast>::cast_from(lower);
                let high = <$common as Cast>::cast_from(upper);
                let mut bin = bins;
                if value >= low && value < high {
                    bin = <$common as RudaHistogramSample>::even_bin(value, low, high, bins);
                }
                bin
            }
        }
        impl RudaHistogramParameters<$level> for $sample {
            fn valid_parameters(bins: usize, lower: $level, upper: $level) -> bool {
                let low = lower as $common;
                let high = upper as $common;
                bins > 0 && low < high && histogram_pair!(@valid $kind, bins, low, high)
            }
        }
    };
    (@valid integer, $bins:ident, $low:ident, $high:ident) => {
        (($high as i128 - $low as i128) as u128)
            .checked_mul($bins as u128).is_some_and(|product| product <= u64::MAX as u128)
    };
    (@valid float, $bins:ident, $low:ident, $high:ident) => { true };
}

macro_rules! histogram_pairs {
    ($sample:ty; $( $level:ty => $common:ty : $kind:ident ),* $(,)?) => {
        $(histogram_pair!($sample, $level, $common, $kind);)*
    };
}

histogram_pairs!(u8; u8=>u8:integer, i8=>i32:integer, u16=>i32:integer, i16=>i32:integer,
    u32=>u32:integer, i32=>i32:integer, u64=>u64:integer, i64=>i64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(i8; u8=>i32:integer, i8=>i8:integer, u16=>i32:integer, i16=>i32:integer,
    u32=>u32:integer, i32=>i32:integer, u64=>u64:integer, i64=>i64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(u16; u8=>i32:integer, i8=>i32:integer, u16=>u16:integer, i16=>i32:integer,
    u32=>u32:integer, i32=>i32:integer, u64=>u64:integer, i64=>i64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(i16; u8=>i32:integer, i8=>i32:integer, u16=>i32:integer, i16=>i16:integer,
    u32=>u32:integer, i32=>i32:integer, u64=>u64:integer, i64=>i64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(u32; u8=>u32:integer, i8=>u32:integer, u16=>u32:integer, i16=>u32:integer,
    u32=>u32:integer, i32=>u32:integer, u64=>u64:integer, i64=>i64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(i32; u8=>i32:integer, i8=>i32:integer, u16=>i32:integer, i16=>i32:integer,
    u32=>u32:integer, i32=>i32:integer, u64=>u64:integer, i64=>i64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(u64; u8=>u64:integer, i8=>u64:integer, u16=>u64:integer, i16=>u64:integer,
    u32=>u64:integer, i32=>u64:integer, u64=>u64:integer, i64=>u64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(i64; u8=>i64:integer, i8=>i64:integer, u16=>i64:integer, i16=>i64:integer,
    u32=>i64:integer, i32=>i64:integer, u64=>u64:integer, i64=>i64:integer, f32=>f32:float, f64=>f64:float);
histogram_pairs!(f32; u8=>f32:float, i8=>f32:float, u16=>f32:float, i16=>f32:float,
    u32=>f32:float, i32=>f32:float, u64=>f32:float, i64=>f32:float, f32=>f32:float, f64=>f64:float);
histogram_pairs!(f64; u8=>f64:float, i8=>f64:float, u16=>f64:float, i16=>f64:float,
    u32=>f64:float, i32=>f64:float, u64=>f64:float, i64=>f64:float, f32=>f64:float, f64=>f64:float);
histogram_pairs!(half::f16; f32=>f32:float, f64=>f64:float);
histogram_pairs!(half::bf16; f32=>f32:float, f64=>f64:float);

/// An interleaved image region. Strides and width are measured in elements
/// and pixels, respectively; active channels are the first channels of a pixel.
#[derive(Clone, Copy, Debug)]
pub struct RudaHistogramRegion {
    pub width: usize,
    pub rows: usize,
    pub channels: usize,
    pub row_stride: usize,
}

impl RudaHistogramRegion {
    fn validate<R: Runtime>(&self, input: &RudaTensor<R>, active: usize) -> Result<usize, RudaPrimitiveError> {
        if self.channels == 0 || active == 0 || active > self.channels {
            return Err(RudaPrimitiveError::Configuration("invalid histogram channel count"));
        }
        let row = self.width.checked_mul(self.channels).ok_or(RudaPrimitiveError::Configuration("histogram row size overflow"))?;
        if self.row_stride < row { return Err(RudaPrimitiveError::Configuration("histogram row stride is too small")); }
        let needed = if self.rows == 0 || self.width == 0 { 0 } else {
            (self.rows - 1).checked_mul(self.row_stride).and_then(|n| n.checked_add(row))
                .ok_or(RudaPrimitiveError::Configuration("histogram extent overflow"))?
        };
        if needed > input.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
        self.width.checked_mul(self.rows).ok_or(RudaPrimitiveError::Configuration("histogram pixel count overflow"))
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn clear<C: Numeric>(output: &mut Tensor<C>, bins: usize) {
    if ABSOLUTE_POS < bins { output[ABSOLUTE_POS] = C::from_int(0); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn even_kernel<T: RudaHistogramSample, C: Numeric>(
    input: &LinearView<T>, output: &mut Tensor<Atomic<C>>, lower: InputScalar, upper: InputScalar,
    bins: usize, pixels: usize, width: usize, row_stride: usize, channels: usize, channel: usize,
) {
    let pixel = ABSOLUTE_POS;
    if pixel < pixels {
        let sample = input[pixel / width * row_stride + pixel % width * channels + channel];
        let low = lower.get::<T>();
        let high = upper.get::<T>();
        if sample >= low && sample < high {
            let bin = T::even_bin(sample, low, high, bins);
            if bin < bins { output[bin].fetch_add(C::from_int(1)); }
        }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn range_kernel<T: Numeric, L: Numeric, C: Numeric>(
    input: &LinearView<T>, levels: &LinearView<L>, output: &mut Tensor<Atomic<C>>,
    pixels: usize, width: usize, row_stride: usize, channels: usize, channel: usize,
) {
    let pixel = ABSOLUTE_POS;
    if pixel < pixels {
        let sample = L::cast_from(input[pixel / width * row_stride + pixel % width * channels + channel]);
        if sample >= levels[0] && sample < levels[levels.shape() - 1] {
            let mut low = 1usize;
            let mut high = levels.shape();
            while low < high {
                let middle = low + (high - low) / 2;
                if sample >= levels[middle] { low = middle + 1; } else { high = middle; }
            }
            output[low - 1].fetch_add(C::from_int(1));
        }
    }
}

fn allocate<R: Runtime, C: TensorElement>(input: &RudaTensor<R>, bins: usize) -> RudaTensor<R> {
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([bins]), <C as Element>::dtype());
    let dim = RudaDim::new(input.client.properties(), bins);
    let grid = calculate_ruda_count_elemwise(&input.client, bins, dim);
    unsafe {
        clear::launch_unchecked::<C, R>(&input.client, grid, dim, address_type!(output), output.clone().into_tensor_arg(), bins);
    }
    output
}

/// Equal-width histograms, one `(bins, lower, upper)` descriptor per active
/// channel. Upper bounds are excluded. Counter atomic addition must be
/// supported by the selected backend and counter type.
pub fn even<R, T, C>(
    input: &RudaTensor<R>, region: RudaHistogramRegion, channels: &[(usize, T, T)],
) -> Result<Vec<RudaTensor<R>>, RudaPrimitiveError>
where R: Runtime, T: TensorElement + RudaHistogramSample, C: TensorElement + Int,
{
    check_type::<R, T>(input)?;
    let pixels = region.validate(input, channels.len())?;
    for &(bins, lower, upper) in channels {
        if bins == 0 || !(lower < upper) { return Err(RudaPrimitiveError::Configuration("histogram requires positive bins and lower < upper")); }
    }
    let mut outputs = Vec::with_capacity(channels.len());
    for (channel, &(bins, lower, upper)) in channels.iter().enumerate() {
        let output = allocate::<R, C>(input, bins);
        if pixels > 0 {
            let dim = RudaDim::new(input.client.properties(), pixels);
            let grid = calculate_ruda_count_elemwise(&input.client, pixels, dim);
            unsafe {
                even_kernel::launch_unchecked::<T, C, R>(
                    &input.client, grid, dim, address_type!(input, output), input.clone().into_linear_view(),
                    output.clone().into_tensor_arg(), InputScalar::new(lower, input.dtype), InputScalar::new(upper, input.dtype),
                    bins, pixels, region.width, region.row_stride, region.channels, channel,
                );
            }
        }
        outputs.push(output);
    }
    Ok(outputs)
}

/// Histograms with nondecreasing, device-resident boundary arrays, one per
/// active channel. Each boundary array has at least two elements.
pub fn range<R, T, C>(
    input: &RudaTensor<R>, region: RudaHistogramRegion, levels: &[RudaTensor<R>],
) -> Result<Vec<RudaTensor<R>>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, C: TensorElement + Int,
{
    range_mixed::<R, T, T, C>(input, region, levels)
}

/// Range histograms convert samples to the boundary type before searching.
pub fn range_mixed<R, T, L, C>(
    input: &RudaTensor<R>, region: RudaHistogramRegion, levels: &[RudaTensor<R>],
) -> Result<Vec<RudaTensor<R>>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, L: TensorElement, C: TensorElement + Int,
{
    check_type::<R, T>(input)?;
    let pixels = region.validate(input, levels.len())?;
    for boundaries in levels {
        check_type::<R, L>(boundaries)?;
        if boundaries.meta.num_elements() < 2 { return Err(RudaPrimitiveError::Configuration("histogram requires at least two levels")); }
        if boundaries.device.to_id() != input.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    let mut outputs = Vec::with_capacity(levels.len());
    for (channel, boundaries) in levels.iter().enumerate() {
        let output = allocate::<R, C>(input, boundaries.meta.num_elements() - 1);
        if pixels > 0 {
            let dim = RudaDim::new(input.client.properties(), pixels);
            let grid = calculate_ruda_count_elemwise(&input.client, pixels, dim);
            unsafe {
                range_kernel::launch_unchecked::<T, L, C, R>(
                    &input.client, grid, dim, address_type!(input, boundaries, output),
                    input.clone().into_linear_view(), boundaries.clone().into_linear_view(), output.clone().into_tensor_arg(),
                    pixels, region.width, region.row_stride, region.channels, channel,
                );
            }
        }
        outputs.push(output);
    }
    Ok(outputs)
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn even_mixed_kernel<T: RudaHistogramPair<L>, L: Numeric, C: Numeric>(
    input: &LinearView<T>, output: &mut Tensor<Atomic<C>>, lower: InputScalar, upper: InputScalar,
    bins: usize, pixels: usize, width: usize, row_stride: usize, channels: usize, channel: usize,
) {
    let pixel = ABSOLUTE_POS;
    if pixel < pixels {
        let sample = input[pixel / width * row_stride + pixel % width * channels + channel];
        let bin = T::mixed_bin(sample, lower.get::<L>(), upper.get::<L>(), bins);
        if bin < bins { output[bin].fetch_add(C::from_int(1)); }
    }
}

/// Equal-width histograms using the arithmetic common type of samples and levels.
pub fn even_mixed<R, T, L, C>(
    input: &RudaTensor<R>, region: RudaHistogramRegion, channels: &[(usize, L, L)],
) -> Result<Vec<RudaTensor<R>>, RudaPrimitiveError>
where R: Runtime, T: TensorElement + RudaHistogramPair<L> + RudaHistogramParameters<L>,
    L: TensorElement, C: TensorElement + Int,
{
    check_type::<R, T>(input)?;
    let pixels = region.validate(input, channels.len())?;
    for &(bins, lower, upper) in channels {
        if !T::valid_parameters(bins, lower, upper) {
            return Err(RudaPrimitiveError::Configuration("invalid histogram bounds or integer bin scaling overflow"));
        }
    }
    let mut outputs = Vec::with_capacity(channels.len());
    for (channel, &(bins, lower, upper)) in channels.iter().enumerate() {
        let output = allocate::<R, C>(input, bins);
        if pixels > 0 {
            let dim = RudaDim::new(input.client.properties(), pixels);
            let grid = calculate_ruda_count_elemwise(&input.client, pixels, dim);
            unsafe {
                even_mixed_kernel::launch_unchecked::<T, L, C, R>(
                    &input.client, grid, dim, address_type!(input, output), input.clone().into_linear_view(),
                    output.clone().into_tensor_arg(), InputScalar::new(lower, <L as Element>::dtype()),
                    InputScalar::new(upper, <L as Element>::dtype()), bins, pixels,
                    region.width, region.row_stride, region.channels, channel,
                );
            }
        }
        outputs.push(output);
    }
    Ok(outputs)
}
