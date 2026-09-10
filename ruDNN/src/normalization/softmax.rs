use super::{NormalizationError, RudaTensor, Runtime};
use ruda_core::tensor::DType;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{allocation::empty_device_contiguous_dtype, contiguous::into_contiguous};

/// Stable last-axis softmax with FP32 arithmetic and a plane-local reduction.
/// Input and output are F32. NaN and infinite rows follow exp(x - max(x)) / sum(exp(x - max(x))).
pub fn softmax_last_axis<R: Runtime>(input: RudaTensor<R>) -> Result<RudaTensor<R>, NormalizationError> {
    let shape = input.meta.shape();
    let width = *shape.last().ok_or(NormalizationError("softmax requires an axis"))?;
    let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if width == 0 || elements.is_none_or(|n| n > u32::MAX as usize)
        || width > u32::MAX as usize || input.dtype != DType::F32 || input.qparams.is_some()
    {
        return Err(NormalizationError("softmax requires an unquantized F32 tensor with a nonempty last axis"));
    }
    let plane = input.client.properties().hardware.plane_size_max;
    if !plane.is_power_of_two() || plane > input.client.properties().hardware.max_ruda_dim.0 {
        return Err(NormalizationError("softmax requires a power-of-two plane"));
    }
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape.clone().into(), DType::F32);
    let rows = elements.unwrap() / width;
    if rows == 0 { return Ok(output); }
    let client = input.client.clone();
    row_softmax::launch::<R>(
        &client, RudaCount::Static(rows as u32, 1, 1), RudaDim::new_1d(plane),
        into_contiguous(input).into_array_arg(), output.clone().into_array_arg(),
        width as u32, include_str!("softmax.rs").to_owned(),
    );
    Ok(output)
}

#[ruda(launch)]
fn row_softmax(input: &Array<f32>, output: &mut Array<f32>, width: u32, #[comptime] _source: String) {
    let width = width as usize;
    let base = RUDA_POS_X as usize * width;
    let lane = UNIT_POS as usize;
    let step = PLANE_DIM as usize;
    let mut maximum = input[base];
    let mut column = lane;
    while column < width {
        maximum = f32::max(maximum, input[base + column]);
        if width - column <= step { break; }
        column += step;
    }
    let mut offset = PLANE_DIM / 2;
    while offset > 0 {
        maximum = f32::max(maximum, plane_shuffle_xor(maximum, offset));
        offset /= 2;
    }
    let mut sum = 0f32;
    column = lane;
    while column < width {
        sum += f32::exp(input[base + column] - maximum);
        if width - column <= step { break; }
        column += step;
    }
    offset = PLANE_DIM / 2;
    while offset > 0 {
        sum += plane_shuffle_xor(sum, offset);
        offset /= 2;
    }
    column = lane;
    while column < width {
        output[base + column] = f32::exp(input[base + column] - maximum) / sum;
        if width - column <= step { break; }
        column += step;
    }
}
