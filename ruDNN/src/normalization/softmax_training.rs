use super::{NormalizationError, RudaTensor, Runtime};
use ruda_core::{device::Device, ir::features::Plane, tensor::DType};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{allocation::empty_device_contiguous_dtype, contiguous::into_contiguous};

fn layout<R: Runtime>(input: &RudaTensor<R>) -> Result<(usize, usize, RudaDim), NormalizationError> {
    let width = *input.meta.shape().last().ok_or(NormalizationError("softmax requires an axis"))?;
    let elements = input.meta.shape().iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(NormalizationError("softmax shape overflow"))?;
    if width == 0 || width > u32::MAX as usize || elements > u32::MAX as usize
        || input.qparams.is_some() || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(NormalizationError("softmax training requires unquantized F32/F16/BF16 and a nonempty last axis"));
    }
    let properties = input.client.properties();
    let hardware = &properties.hardware;
    let plane = hardware.plane_size_max;
    let rows = elements / width;
    if !plane.is_power_of_two() || plane != hardware.plane_size_min
        || !properties.features.plane.contains(Plane::Ops)
        || plane > hardware.max_ruda_dim.0.min(hardware.max_units_per_ruda)
        || rows > hardware.max_ruda_count.0 as usize {
        return Err(NormalizationError("softmax training requires a supported fixed plane and row grid"));
    }
    Ok((rows, width, RudaDim::new_1d(plane)))
}

/// Last-axis softmax or log-softmax in FP32, retaining the unrounded output for backward.
/// F16/BF16 inputs are read directly; no full FP32 input copy is required.
pub fn softmax_last_axis_working<R: Runtime>(input: RudaTensor<R>, logarithmic: bool)
    -> Result<RudaTensor<R>, NormalizationError> {
    let (rows, width, dim) = layout(&input)?;
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(),
        input.meta.shape().clone(), DType::F32);
    if rows == 0 { return Ok(output); }
    let dtype = input.dtype;
    forward::launch(&output.client, RudaCount::Static(rows as u32, 1, 1), dim,
        into_contiguous(input).into_array_arg(), output.clone().into_array_arg(), width as u32,
        logarithmic, include_str!("softmax_training.rs").to_owned(), dtype.into());
    Ok(output)
}

/// First-order VJP from the saved FP32 softmax/log-softmax output, returned in FP32.
/// No probability matrix is reconstructed for ordinary softmax, and no host reduction is used.
pub fn softmax_last_axis_backward<R: Runtime>(working: RudaTensor<R>, grad: RudaTensor<R>, logarithmic: bool)
    -> Result<RudaTensor<R>, NormalizationError> {
    let (rows, width, dim) = layout(&working)?;
    if working.dtype != DType::F32 || grad.meta.shape() != working.meta.shape()
        || grad.qparams.is_some() || !matches!(grad.dtype, DType::F32 | DType::F16 | DType::BF16)
        || grad.device.to_id() != working.device.to_id()
        || !grad.client.same_execution_queue(&working.client) {
        return Err(NormalizationError("softmax backward requires matching same-queue gradients and saved FP32 output"));
    }
    let output = empty_device_contiguous_dtype(working.client.clone(), working.device.clone(),
        working.meta.shape().clone(), DType::F32);
    if rows == 0 { return Ok(output); }
    let dtype = grad.dtype;
    backward::launch(&output.client, RudaCount::Static(rows as u32, 1, 1), dim,
        into_contiguous(working).into_array_arg(), into_contiguous(grad).into_array_arg(),
        output.clone().into_array_arg(), width as u32, logarithmic,
        include_str!("softmax_training.rs").to_owned(), dtype.into());
    Ok(output)
}

#[ruda(launch)]
fn forward<F: Float>(input: &Array<F>, output: &mut Array<f32>, width: u32,
    #[comptime] logarithmic: bool, #[comptime] _source: String, #[define(F)] _dtype: StorageType) {
    let width = width as usize;
    let base = RUDA_POS_X as usize * width;
    let lane = UNIT_POS as usize;
    let step = PLANE_DIM as usize;
    let mut maximum = f32::cast_from(input[base]);
    let mut column = lane;
    while column < width {
        maximum = f32::max(maximum, f32::cast_from(input[base + column]));
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
        sum += f32::exp(f32::cast_from(input[base + column]) - maximum);
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
        let shifted = f32::cast_from(input[base + column]) - maximum;
        output[base + column] = if logarithmic { shifted - f32::ln(sum) } else { f32::exp(shifted) / sum };
        if width - column <= step { break; }
        column += step;
    }
}

#[ruda(launch)]
fn backward<G: Float>(working: &Array<f32>, grad: &Array<G>, output: &mut Array<f32>, width: u32,
    #[comptime] logarithmic: bool, #[comptime] _source: String, #[define(G)] _dtype: StorageType) {
    let width = width as usize;
    let base = RUDA_POS_X as usize * width;
    let lane = UNIT_POS as usize;
    let step = PLANE_DIM as usize;
    let mut sum = 0f32;
    let mut column = lane;
    while column < width {
        let value = f32::cast_from(grad[base + column]);
        sum += if logarithmic { value } else { value * working[base + column] };
        if width - column <= step { break; }
        column += step;
    }
    let mut offset = PLANE_DIM / 2;
    while offset > 0 {
        sum += plane_shuffle_xor(sum, offset);
        offset /= 2;
    }
    column = lane;
    while column < width {
        let value = f32::cast_from(grad[base + column]);
        let saved = working[base + column];
        output[base + column] = if logarithmic { value - f32::exp(saved) * sum } else { saved * (value - sum) };
        if width - column <= step { break; }
        column += step;
    }
}
