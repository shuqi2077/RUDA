use super::{NormalizationError, kernel};
use ruda_core::{device::Device, ir::features::Plane, tensor::{DType, Shape}};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous},
};

fn layout<R: Runtime>(input: &RudaTensor<R>) -> Result<(usize, usize, RudaDim), NormalizationError> {
    let shape = input.meta.shape();
    let width = *shape.last().ok_or(NormalizationError("LayerNorm input must have an axis"))?;
    let elements = shape.iter().try_fold(1usize, |n, &extent| n.checked_mul(extent));
    if width == 0 || width > u32::MAX as usize || elements.is_none_or(|size| size > u32::MAX as usize)
        || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16) || input.qparams.is_some()
    { return Err(NormalizationError("invalid LayerNorm training shape or storage")); }
    let properties = input.client.properties();
    let hardware = &properties.hardware;
    let plane = hardware.plane_size_max;
    if !plane.is_power_of_two() || plane != hardware.plane_size_min
        || !properties.features.plane.contains(Plane::Ops)
        || plane > hardware.max_ruda_dim.0 || hardware.max_ruda_dim.1 < 4
        || plane > hardware.max_units_per_ruda / 4
        || elements.unwrap() / width > hardware.max_ruda_count.0 as usize {
        return Err(NormalizationError("LayerNorm runtime cannot launch four planes"));
    }
    Ok((elements.unwrap() / width, width, RudaDim::new_2d(plane, 4)))
}

fn binding<R: Runtime>(input: &RudaTensor<R>, other: &RudaTensor<R>) -> Result<(), NormalizationError> {
    if other.qparams.is_some() || other.device.to_id() != input.device.to_id()
        || !input.client.same_execution_queue(&other.client)
    { return Err(NormalizationError("LayerNorm training bindings must share the execution queue and device")); }
    Ok(())
}

fn affine<R: Runtime>(input: &RudaTensor<R>, value: &RudaTensor<R>, width: usize) -> Result<(), NormalizationError> {
    binding(input, value)?;
    if value.meta.shape()[..] != [width] || !matches!(value.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(NormalizationError("LayerNorm affine must be a floating feature vector"));
    }
    Ok(())
}

/// Native last-axis LayerNorm with original output storage and saved FP32 row statistics.
/// Affine leaves retain their actual F32/F16/BF16 storage independently of activations.
pub fn layer_norm_with_stats<R: Runtime>(
    input: RudaTensor<R>, gamma: RudaTensor<R>, beta: Option<RudaTensor<R>>, epsilon: f32,
) -> Result<[RudaTensor<R>; 3], NormalizationError> {
    let (rows, width, dim) = layout(&input)?;
    affine(&input, &gamma, width)?;
    if let Some(beta) = &beta { affine(&input, beta, width)?; }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(NormalizationError("LayerNorm epsilon must be finite and positive"));
    }
    let allocate = |shape, dtype| empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, dtype);
    let output = allocate(input.meta.shape().clone(), input.dtype);
    let mean = allocate(Shape::new([rows]), DType::F32);
    let rstd = allocate(Shape::new([rows]), DType::F32);
    if rows == 0 { return Ok([output, mean, rstd]); }
    let has_beta = beta.is_some();
    let beta = beta.unwrap_or_else(|| gamma.clone());
    let types = [input.dtype.into(), gamma.dtype.into(), beta.dtype.into()];
    kernel::layer_norm_training::launch(&output.client, RudaCount::Static(rows as u32, 1, 1), dim,
        into_contiguous(input).into_array_arg(), into_contiguous(gamma).into_array_arg(),
        into_contiguous(beta).into_array_arg(), output.clone().into_array_arg(),
        mean.clone().into_array_arg(), rstd.clone().into_array_arg(), width as u32, epsilon, has_beta, types);
    Ok([output, mean, rstd])
}

/// Native first-order LayerNorm input, weight and bias derivatives.
/// Uses saved FP32 statistics without recomputing forward; affine reductions use
/// deterministic FP32 partials, with no floating atomic updates or host reductions.
pub fn layer_norm_backward<R: Runtime>(
    input: RudaTensor<R>, gamma: RudaTensor<R>, grad: RudaTensor<R>,
    mean: RudaTensor<R>, rstd: RudaTensor<R>,
) -> Result<[RudaTensor<R>; 3], NormalizationError> {
    let (rows, width, dim) = backward_layout(&input, &gamma, &grad, &mean, &rstd)?;
    let allocate = |shape, dtype| empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, dtype);
    let input_grad = allocate(input.meta.shape().clone(), input.dtype);
    let weight_grad = allocate(Shape::new([width]), gamma.dtype);
    let bias_grad = allocate(Shape::new([width]), DType::F32);
    let input = into_contiguous(input);
    let gamma = into_contiguous(gamma);
    let grad = into_contiguous(grad);
    let mean = into_contiguous(mean);
    let rstd = into_contiguous(rstd);
    let client = input.client.clone();
    if rows > 0 {
        kernel::layer_norm_input_backward::launch(&client, RudaCount::Static(rows as u32, 1, 1), dim,
            input.clone().into_array_arg(), gamma.clone().into_array_arg(), grad.clone().into_array_arg(),
            mean.clone().into_array_arg(), rstd.clone().into_array_arg(), input_grad.clone().into_array_arg(),
            width as u32, [input.dtype.into(), gamma.dtype.into(), grad.dtype.into()]);
    }
    let parts = rows.div_ceil(32).clamp(1, 128);
    let work = parts * width;
    let weight_parts = empty_device_contiguous_dtype(client.clone(), input.device.clone(), Shape::new([parts, width]), DType::F32);
    let bias_parts = empty_device_contiguous_dtype(client.clone(), input.device.clone(), Shape::new([parts, width]), DType::F32);
    let part_dim = RudaDim::new(client.properties(), work);
    let part_count = calculate_ruda_count_elemwise(&client, work, part_dim);
    kernel::layer_norm_affine_partial::launch(&client, part_count, part_dim,
        input.clone().into_array_arg(), grad.clone().into_array_arg(), mean.into_array_arg(), rstd.into_array_arg(),
        weight_parts.clone().into_array_arg(), bias_parts.clone().into_array_arg(), width as u32, parts as u32,
        [input.dtype.into(), grad.dtype.into()]);
    let merge_dim = RudaDim::new(client.properties(), width);
    let merge_count = calculate_ruda_count_elemwise(&client, width, merge_dim);
    kernel::layer_norm_affine_merge::launch(&client, merge_count, merge_dim,
        weight_parts.into_array_arg(), bias_parts.into_array_arg(), weight_grad.clone().into_array_arg(),
        bias_grad.clone().into_array_arg(), parts as u32, gamma.dtype.into());
    Ok([input_grad, weight_grad, bias_grad])
}

fn backward_layout<R: Runtime>(input: &RudaTensor<R>, gamma: &RudaTensor<R>, grad: &RudaTensor<R>,
    mean: &RudaTensor<R>, rstd: &RudaTensor<R>) -> Result<(usize, usize, RudaDim), NormalizationError> {
    let (rows, width, dim) = layout(input)?;
    affine(input, gamma, width)?;
    for value in [grad, mean, rstd] { binding(input, value)?; }
    if grad.meta.shape() != input.meta.shape() || !matches!(grad.dtype, DType::F32 | DType::F16 | DType::BF16)
        || mean.meta.shape()[..] != [rows] || rstd.meta.shape() != mean.meta.shape()
        || mean.dtype != DType::F32 || rstd.dtype != DType::F32 {
        return Err(NormalizationError("invalid LayerNorm gradient or saved statistics"));
    }
    Ok((rows, width, dim))
}

/// Compute only requested input, weight and bias gradients, in that order.
/// Unrequested outputs are absent, without allocating substitute gradient buffers.
pub fn layer_norm_backward_select<R: Runtime>(input: RudaTensor<R>, gamma: RudaTensor<R>, grad: RudaTensor<R>,
    mean: RudaTensor<R>, rstd: RudaTensor<R>, mask: [bool; 3])
    -> Result<[Option<RudaTensor<R>>; 3], NormalizationError> {
    if mask == [false; 3] { return Ok([None, None, None]); }
    if mask == [true; 3] { return Ok(layer_norm_backward(input, gamma, grad, mean, rstd)?.map(Some)); }
    let (rows, width, dim) = backward_layout(&input, &gamma, &grad, &mean, &rstd)?;
    let allocate = |shape, dtype| empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, dtype);
    let input_grad = mask[0].then(|| allocate(input.meta.shape().clone(), input.dtype));
    let weight_grad = mask[1].then(|| allocate(Shape::new([width]), gamma.dtype));
    let bias_grad = mask[2].then(|| allocate(Shape::new([width]), DType::F32));
    let input = if mask[0] || mask[1] { into_contiguous(input) } else { input };
    let gamma = if mask[0] { into_contiguous(gamma) } else { gamma };
    let grad = into_contiguous(grad);
    let mean = if mask[0] || mask[1] { into_contiguous(mean) } else { mean };
    let rstd = if mask[0] || mask[1] { into_contiguous(rstd) } else { rstd };
    let client = input.client.clone();
    if let Some(output) = &input_grad {
        if rows > 0 {
            kernel::layer_norm_input_backward::launch(&client, RudaCount::Static(rows as u32, 1, 1), dim,
                input.clone().into_array_arg(), gamma.clone().into_array_arg(), grad.clone().into_array_arg(),
                mean.clone().into_array_arg(), rstd.clone().into_array_arg(), output.clone().into_array_arg(),
                width as u32, [input.dtype.into(), gamma.dtype.into(), grad.dtype.into()]);
        }
    }
    if mask[1] || mask[2] {
        let parts = rows.div_ceil(32).clamp(1, 128);
        let work = parts * width;
        let allocate_partial = || empty_device_contiguous_dtype(client.clone(), input.device.clone(),
            Shape::new([parts, width]), DType::F32);
        let dim = RudaDim::new(client.properties(), work);
        let count = calculate_ruda_count_elemwise(&client, work, dim);
        let merge_dim = RudaDim::new(client.properties(), width);
        let merge_count = calculate_ruda_count_elemwise(&client, width, merge_dim);
        match (&weight_grad, &bias_grad) {
            (Some(weight), Some(bias)) => {
                let wp = allocate_partial();
                let bp = allocate_partial();
                kernel::layer_norm_affine_partial::launch(&client, count, dim, input.clone().into_array_arg(),
                    grad.clone().into_array_arg(), mean.into_array_arg(), rstd.into_array_arg(),
                    wp.clone().into_array_arg(), bp.clone().into_array_arg(), width as u32, parts as u32,
                    [input.dtype.into(), grad.dtype.into()]);
                kernel::layer_norm_affine_merge::launch(&client, merge_count, merge_dim, wp.into_array_arg(),
                    bp.into_array_arg(), weight.clone().into_array_arg(), bias.clone().into_array_arg(),
                    parts as u32, gamma.dtype.into());
            }
            (Some(weight), None) => {
                let partial = allocate_partial();
                kernel::layer_norm_weight_partial::launch(&client, count, dim, input.clone().into_array_arg(),
                    grad.clone().into_array_arg(), mean.into_array_arg(), rstd.into_array_arg(),
                    partial.clone().into_array_arg(), width as u32, parts as u32, [input.dtype.into(), grad.dtype.into()]);
                super::rms::weight_merge::launch(&client, merge_count, merge_dim, partial.into_array_arg(),
                    weight.clone().into_array_arg(), parts as u32, gamma.dtype.into());
            }
            (None, Some(bias)) => {
                let partial = allocate_partial();
                kernel::layer_norm_bias_partial::launch(&client, count, dim, grad.clone().into_array_arg(),
                    partial.clone().into_array_arg(), width as u32, rows as u32, parts as u32, grad.dtype.into());
                super::rms::weight_merge::launch(&client, merge_count, merge_dim, partial.into_array_arg(),
                    bias.clone().into_array_arg(), parts as u32, DType::F32.into());
            }
            (None, None) => unreachable!(),
        }
    }
    Ok([input_grad, weight_grad, bias_grad])
}
