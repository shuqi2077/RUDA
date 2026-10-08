use super::{NormalizationError, rms};
use ruda_core::{device::Device, ir::features::Plane, tensor::{DType, Shape}};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous},
};

fn layout<R: Runtime>(input: &RudaTensor<R>) -> Result<(usize, usize, u32, bool), NormalizationError> {
    let width = *input.meta.shape().last().ok_or(NormalizationError("RMSNorm requires an axis"))?;
    let elements = input.meta.shape().iter().try_fold(1usize, |n, &extent| n.checked_mul(extent));
    if width == 0 || width > u32::MAX as usize || elements.is_none_or(|size| size > u32::MAX as usize)
        || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16) || input.qparams.is_some() {
        return Err(NormalizationError("invalid RMSNorm training shape or storage"));
    }
    let rows = elements.unwrap() / width;
    let properties = input.client.properties();
    let hardware = &properties.hardware;
    let maximum = hardware.max_ruda_dim.0.min(hardware.max_units_per_ruda);
    let plane = hardware.plane_size_max;
    if !properties.features.plane.contains(Plane::Ops) || !plane.is_power_of_two()
        || plane > maximum || rows > hardware.max_ruda_count.0 as usize {
        return Err(NormalizationError("RMSNorm runtime cannot launch row reductions"));
    }
    let (threads, vectorized) = rms::launch_parameters(width, rows, plane, maximum);
    Ok((rows, width, threads, vectorized))
}

fn binding<R: Runtime>(input: &RudaTensor<R>, value: &RudaTensor<R>) -> Result<(), NormalizationError> {
    if value.qparams.is_some() || value.device.to_id() != input.device.to_id()
        || !input.client.same_execution_queue(&value.client) {
        return Err(NormalizationError("RMSNorm training bindings must share their execution queue and device"));
    }
    Ok(())
}

fn weight<R: Runtime>(input: &RudaTensor<R>, gamma: &RudaTensor<R>, width: usize) -> Result<(), NormalizationError> {
    binding(input, gamma)?;
    if gamma.meta.shape()[..] != [width] || !matches!(gamma.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(NormalizationError("RMSNorm weight must be a floating feature vector"));
    }
    Ok(())
}

/// Native RMSNorm output and saved FP32 reciprocal row norms.
/// Reuses the original FP32 statistics and affine arithmetic with one output cast.
pub fn rms_norm_with_stats<R: Runtime>(input: RudaTensor<R>, gamma: RudaTensor<R>, epsilon: f32)
    -> Result<[RudaTensor<R>; 2], NormalizationError> {
    let (rows, width, threads, vectorized) = layout(&input)?;
    weight(&input, &gamma, width)?;
    if !epsilon.is_finite() || epsilon <= 0.0 { return Err(NormalizationError("invalid RMSNorm epsilon")); }
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let rstd = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), Shape::new([rows]), DType::F32);
    if rows > 0 {
        let types = [input.dtype.into(), gamma.dtype.into()];
        rms::training_forward::launch(&output.client, RudaCount::Static(rows as u32, 1, 1), RudaDim::new_1d(threads),
            into_contiguous(input).into_array_arg(), into_contiguous(gamma).into_array_arg(),
            output.clone().into_array_arg(), rstd.clone().into_array_arg(), width as u32, epsilon,
            threads, vectorized, types);
    }
    Ok([output, rstd])
}

/// Native RMSNorm input and weight gradients from the exact saved row statistics.
/// Uses FP32 partial reductions without floating atomics or host reductions.
pub fn rms_norm_backward<R: Runtime>(input: RudaTensor<R>, gamma: RudaTensor<R>, grad: RudaTensor<R>, rstd: RudaTensor<R>)
    -> Result<[RudaTensor<R>; 2], NormalizationError> {
    let (rows, width, threads) = backward_layout(&input, &gamma, &grad, &rstd)?;
    let input_grad = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let weight_grad = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), Shape::new([width]), gamma.dtype);
    let input = into_contiguous(input);
    let gamma = into_contiguous(gamma);
    let grad = into_contiguous(grad);
    let rstd = into_contiguous(rstd);
    let client = input.client.clone();
    if rows > 0 {
        rms::input_backward::launch(&client, RudaCount::Static(rows as u32, 1, 1), RudaDim::new_1d(threads),
            input.clone().into_array_arg(), gamma.clone().into_array_arg(), grad.clone().into_array_arg(),
            rstd.clone().into_array_arg(), input_grad.clone().into_array_arg(), width as u32, threads,
            [input.dtype.into(), gamma.dtype.into(), grad.dtype.into()]);
    }
    let parts = rows.div_ceil(32).clamp(1, 128);
    let work = parts * width;
    let partial = empty_device_contiguous_dtype(client.clone(), input.device.clone(), Shape::new([parts, width]), DType::F32);
    let part_dim = RudaDim::new(client.properties(), work);
    let part_count = calculate_ruda_count_elemwise(&client, work, part_dim);
    rms::weight_partial::launch(&client, part_count, part_dim, input.clone().into_array_arg(), grad.clone().into_array_arg(),
        rstd.into_array_arg(), partial.clone().into_array_arg(), width as u32, parts as u32,
        [input.dtype.into(), grad.dtype.into()]);
    let merge_dim = RudaDim::new(client.properties(), width);
    let merge_count = calculate_ruda_count_elemwise(&client, width, merge_dim);
    rms::weight_merge::launch(&client, merge_count, merge_dim, partial.into_array_arg(), weight_grad.clone().into_array_arg(),
        parts as u32, gamma.dtype.into());
    Ok([input_grad, weight_grad])
}

fn backward_layout<R: Runtime>(input: &RudaTensor<R>, gamma: &RudaTensor<R>, grad: &RudaTensor<R>, rstd: &RudaTensor<R>)
    -> Result<(usize, usize, u32), NormalizationError> {
    let (rows, width, threads, _) = layout(input)?;
    weight(input, gamma, width)?;
    for value in [grad, rstd] { binding(input, value)?; }
    if grad.meta.shape() != input.meta.shape() || !matches!(grad.dtype, DType::F32 | DType::F16 | DType::BF16)
        || rstd.dtype != DType::F32 || rstd.meta.shape()[..] != [rows] {
        return Err(NormalizationError("invalid RMSNorm gradient or saved statistics"));
    }
    Ok((rows, width, threads))
}

/// Compute only requested input and weight gradients, without unused output or scratch buffers.
pub fn rms_norm_backward_select<R: Runtime>(input: RudaTensor<R>, gamma: RudaTensor<R>, grad: RudaTensor<R>,
    rstd: RudaTensor<R>, mask: [bool; 2]) -> Result<[Option<RudaTensor<R>>; 2], NormalizationError> {
    if mask == [false; 2] { return Ok([None, None]); }
    if mask == [true; 2] { return Ok(rms_norm_backward(input, gamma, grad, rstd)?.map(Some)); }
    let (rows, width, threads) = backward_layout(&input, &gamma, &grad, &rstd)?;
    let allocate = |shape, dtype| empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, dtype);
    let input_grad = mask[0].then(|| allocate(input.meta.shape().clone(), input.dtype));
    let weight_grad = mask[1].then(|| allocate(Shape::new([width]), gamma.dtype));
    let input = into_contiguous(input);
    let gamma = if mask[0] { into_contiguous(gamma) } else { gamma };
    let grad = into_contiguous(grad);
    let rstd = into_contiguous(rstd);
    let client = input.client.clone();
    if let Some(output) = &input_grad {
        if rows > 0 {
            rms::input_backward::launch(&client, RudaCount::Static(rows as u32, 1, 1), RudaDim::new_1d(threads),
                input.clone().into_array_arg(), gamma.clone().into_array_arg(), grad.clone().into_array_arg(),
                rstd.clone().into_array_arg(), output.clone().into_array_arg(), width as u32, threads,
                [input.dtype.into(), gamma.dtype.into(), grad.dtype.into()]);
        }
    }
    if let Some(output) = &weight_grad {
        let parts = rows.div_ceil(32).clamp(1, 128);
        let work = parts * width;
        let partial = empty_device_contiguous_dtype(client.clone(), input.device.clone(), Shape::new([parts, width]), DType::F32);
        let dim = RudaDim::new(client.properties(), work);
        let count = calculate_ruda_count_elemwise(&client, work, dim);
        rms::weight_partial::launch(&client, count, dim, input.clone().into_array_arg(), grad.clone().into_array_arg(),
            rstd.into_array_arg(), partial.clone().into_array_arg(), width as u32, parts as u32,
            [input.dtype.into(), grad.dtype.into()]);
        let dim = RudaDim::new(client.properties(), width);
        let count = calculate_ruda_count_elemwise(&client, width, dim);
        rms::weight_merge::launch(&client, count, dim, partial.into_array_arg(), output.clone().into_array_arg(),
            parts as u32, gamma.dtype.into());
    }
    Ok([input_grad, weight_grad])
}
