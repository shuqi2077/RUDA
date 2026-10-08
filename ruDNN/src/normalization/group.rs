use super::{NormalizationError, group_kernel as kernel};
use ruda_core::{device::Device, ir::features::Plane, tensor::{DType, Shape}};
use ruda_kernel::{dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous}};

struct Layout { batch: usize, channels: usize, spatial: usize, width: usize, rows: usize, dim: RudaDim }

fn layout<R: Runtime>(input: &RudaTensor<R>, groups: usize) -> Result<Layout, NormalizationError> {
    let shape = input.meta.shape();
    if shape.num_dims() < 2 || groups == 0 || shape[1] == 0 || shape[1] % groups != 0 {
        return Err(NormalizationError("GroupNorm requires [batch, channels, ...] and evenly divided positive channel groups"));
    }
    let channels = shape[1];
    let spatial = shape[2..].iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(NormalizationError("GroupNorm spatial extent overflow"))?;
    let width = (channels / groups).checked_mul(spatial).ok_or(NormalizationError("GroupNorm width overflow"))?;
    let rows = shape[0].checked_mul(groups).ok_or(NormalizationError("GroupNorm row count overflow"))?;
    let elements = rows.checked_mul(width).ok_or(NormalizationError("GroupNorm element count overflow"))?;
    if width == 0 || channels > u32::MAX as usize || width > u32::MAX as usize || elements > u32::MAX as usize
        || input.qparams.is_some() || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(NormalizationError("GroupNorm requires nonempty groups and unquantized F32/F16/BF16 storage"));
    }
    let properties = input.client.properties();
    let hardware = &properties.hardware;
    let plane = hardware.plane_size_max;
    if !plane.is_power_of_two() || plane != hardware.plane_size_min || !properties.features.plane.contains(Plane::Ops)
        || plane > hardware.max_ruda_dim.0 || hardware.max_ruda_dim.1 < 4 || plane > hardware.max_units_per_ruda / 4
        || rows > hardware.max_ruda_count.0 as usize {
        return Err(NormalizationError("GroupNorm runtime cannot launch four fixed planes per group"));
    }
    Ok(Layout { batch: shape[0], channels, spatial, width, rows, dim: RudaDim::new_2d(plane, 4) })
}

fn binding<R: Runtime>(input: &RudaTensor<R>, value: &RudaTensor<R>) -> Result<(), NormalizationError> {
    if value.qparams.is_some() || value.device.to_id() != input.device.to_id()
        || !value.client.same_execution_queue(&input.client) {
        return Err(NormalizationError("GroupNorm bindings must share the original device and execution queue"));
    }
    Ok(())
}
fn affine<R: Runtime>(input: &RudaTensor<R>, value: &RudaTensor<R>, channels: usize) -> Result<(), NormalizationError> {
    binding(input, value)?;
    if value.meta.shape()[..] != [channels] || !matches!(value.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(NormalizationError("GroupNorm affine must be an actual F32/F16/BF16 channel vector"));
    }
    Ok(())
}

/// Biased-variance GroupNorm on actual `[batch, channels, ...]` input, with saved FP32 `[batch, groups]` statistics.
/// Gamma and beta are independently optional; no artificial ones/zeros are allocated for absent affine leaves.
pub fn group_norm_with_stats<R: Runtime>(input: RudaTensor<R>, gamma: Option<RudaTensor<R>>, beta: Option<RudaTensor<R>>,
    groups: usize, epsilon: f32) -> Result<[RudaTensor<R>; 3], NormalizationError> {
    let info = layout(&input, groups)?;
    for value in gamma.iter().chain(beta.iter()) { affine(&input, value, info.channels)?; }
    if info.rows > 0 {
        let has_gamma = gamma.is_some();
        let has_beta = beta.is_some();
        let mut operands = vec![input.clone()];
        operands.extend(gamma.iter().cloned());
        operands.extend(beta.iter().cloned());
        let candidates = ruda_kernel::tensor::tuning::elementwise_candidates(&input);
        let output = ruda_kernel::tensor::tuning::execute_variants(operands, "group_norm_forward_v1",
            format!("groups={groups};epsilon={:08x};weight={has_gamma};bias={has_beta}", epsilon.to_bits()), candidates,
            move |values, units| {
                let gamma = has_gamma.then(|| values[1].clone());
                let beta = has_beta.then(|| values[1 + usize::from(has_gamma)].clone());
                group_norm_forward_inner(values[0].clone(), gamma, beta, groups, epsilon, units)
                    .map(|output| output.into_iter().collect()).map_err(|error| error.to_string())
            }).map_err(|_| NormalizationError("GroupNorm forward autotune failed without replay"))?;
        if let Some(output) = output { return Ok(output.try_into().expect("GroupNorm output and both saved statistics")); }
    }
    group_norm_forward_inner(input, gamma, beta, groups, epsilon, 0)
}

fn group_norm_forward_inner<R: Runtime>(input: RudaTensor<R>, gamma: Option<RudaTensor<R>>, beta: Option<RudaTensor<R>>,
    groups: usize, epsilon: f32, units: u32) -> Result<[RudaTensor<R>; 3], NormalizationError> {
    let info = layout(&input, groups)?;
    for value in gamma.iter().chain(beta.iter()) { affine(&input, value, info.channels)?; }
    let allocate = |shape, dtype| empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, dtype);
    let output = allocate(input.meta.shape().clone(), input.dtype);
    let mean = allocate(Shape::new([info.batch, groups]), DType::F32);
    let rstd = allocate(Shape::new([info.batch, groups]), DType::F32);
    if info.rows == 0 { return Ok([output, mean, rstd]); }
    let input = into_contiguous(input);
    kernel::statistics::launch(&input.client, RudaCount::Static(info.rows as u32, 1, 1), info.dim,
        input.clone().into_array_arg(), mean.clone().into_array_arg(), rstd.clone().into_array_arg(), info.width as u32,
        epsilon, [include_str!("group_kernel.rs"), include_str!("kernel.rs")].concat(), input.dtype.into());
    let dim = if units == 0 { RudaDim::new(input.client.properties(), input.meta.num_elements()) } else { RudaDim::new_1d(units) };
    let count = calculate_ruda_count_elemwise(&input.client, input.meta.num_elements(), dim);
    match (gamma.map(into_contiguous), beta.map(into_contiguous)) {
        (Some(gamma), Some(beta)) => kernel::forward_affine::launch(&input.client, count, dim,
            input.clone().into_array_arg(), gamma.clone().into_array_arg(), beta.clone().into_array_arg(),
            mean.clone().into_array_arg(), rstd.clone().into_array_arg(), output.clone().into_array_arg(),
            info.width as u32, info.channels as u32, info.spatial as u32, [input.dtype.into(), gamma.dtype.into(), beta.dtype.into()]),
        (Some(gamma), None) => kernel::forward_weight::launch(&input.client, count, dim,
            input.clone().into_array_arg(), gamma.clone().into_array_arg(), mean.clone().into_array_arg(), rstd.clone().into_array_arg(),
            output.clone().into_array_arg(), info.width as u32, info.channels as u32, info.spatial as u32, [input.dtype.into(), gamma.dtype.into()]),
        (None, Some(beta)) => kernel::forward_bias::launch(&input.client, count, dim,
            input.clone().into_array_arg(), beta.clone().into_array_arg(), mean.clone().into_array_arg(), rstd.clone().into_array_arg(),
            output.clone().into_array_arg(), info.width as u32, info.channels as u32, info.spatial as u32, [input.dtype.into(), beta.dtype.into()]),
        (None, None) => kernel::forward_plain::launch(&input.client, count, dim, input.clone().into_array_arg(),
            mean.clone().into_array_arg(), rstd.clone().into_array_arg(), output.clone().into_array_arg(), info.width as u32, input.dtype.into()),
    }
    Ok([output, mean, rstd])
}

/// Independently requested original input/weight/bias VJPs, using saved statistics and ordered FP32 affine partials.
/// Bias-only work does not read input/weight/statistic values. Unrequested outputs and scratch are absent.
pub fn group_norm_backward_select<R: Runtime>(input: RudaTensor<R>, gamma: Option<RudaTensor<R>>, grad: RudaTensor<R>,
    mean: RudaTensor<R>, rstd: RudaTensor<R>, groups: usize, mask: [bool; 3])
    -> Result<[Option<RudaTensor<R>>; 3], NormalizationError> {
    if mask == [false; 3] { return Ok([None, None, None]); }
    let info = layout(&input, groups)?;
    if info.rows > 0 && (mask[1] || mask[2]) {
        let has_gamma = gamma.is_some();
        let mut operands = vec![input.clone(), grad.clone(), mean.clone(), rstd.clone()];
        operands.extend(gamma.iter().cloned());
        let rows = info.batch * info.spatial;
        let default = rows.div_ceil(32).clamp(1, 128);
        let mut candidates = vec![("original_partitions", 0usize)];
        for (name, parts) in [("parts_1", 1usize), ("parts_8", 8), ("parts_32", 32), ("parts_128", 128)] {
            if parts <= rows && parts != default && parts.checked_mul(info.channels).is_some_and(|work| work <= u32::MAX as usize) {
                candidates.push((name, parts));
            }
        }
        let output = ruda_kernel::tensor::tuning::execute_variants(operands, "group_norm_backward_v1",
            format!("groups={groups};weight={has_gamma};leaves={mask:?}"), candidates, move |values, parts| {
                group_norm_backward_inner(values[0].clone(), has_gamma.then(|| values[4].clone()), values[1].clone(),
                    values[2].clone(), values[3].clone(), groups, mask, parts)
                    .map(|output| output.into_iter().flatten().collect()).map_err(|error| error.to_string())
            }).map_err(|_| NormalizationError("GroupNorm backward autotune failed without replay"))?;
        if let Some(output) = output {
            let mut output = output.into_iter();
            return Ok(core::array::from_fn(|index| mask[index].then(|| output.next().expect("requested GroupNorm derivative"))));
        }
    }
    group_norm_backward_inner(input, gamma, grad, mean, rstd, groups, mask, 0)
}

fn group_norm_backward_inner<R: Runtime>(input: RudaTensor<R>, gamma: Option<RudaTensor<R>>, grad: RudaTensor<R>,
    mean: RudaTensor<R>, rstd: RudaTensor<R>, groups: usize, mask: [bool; 3], partitions: usize)
    -> Result<[Option<RudaTensor<R>>; 3], NormalizationError> {
    if mask == [false; 3] { return Ok([None, None, None]); }
    let info = layout(&input, groups)?;
    if let Some(gamma) = &gamma { affine(&input, gamma, info.channels)?; }
    for value in [&grad, &mean, &rstd] { binding(&input, value)?; }
    if grad.meta.shape() != input.meta.shape() || !matches!(grad.dtype, DType::F32 | DType::F16 | DType::BF16)
        || mean.meta.shape()[..] != [info.batch, groups] || rstd.meta.shape() != mean.meta.shape()
        || mean.dtype != DType::F32 || rstd.dtype != DType::F32 || (mask[1] && gamma.is_none()) {
        return Err(NormalizationError("invalid GroupNorm gradient, actual weight or saved statistics"));
    }
    let allocate = |shape, dtype| empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, dtype);
    let dx = mask[0].then(|| allocate(input.meta.shape().clone(), input.dtype));
    let dw = mask[1].then(|| allocate(Shape::new([info.channels]), gamma.as_ref().expect("requested weight").dtype));
    let db = mask[2].then(|| allocate(Shape::new([info.channels]), DType::F32));
    let client = input.client.clone();
    let device = input.device.clone();
    let input = if mask[0] || mask[1] { into_contiguous(input) } else { input };
    let mean = if mask[0] || mask[1] { into_contiguous(mean) } else { mean };
    let rstd = if mask[0] || mask[1] { into_contiguous(rstd) } else { rstd };
    let grad = into_contiguous(grad);
    if let Some(dx) = &dx {
        if info.rows > 0 {
            let count = RudaCount::Static(info.rows as u32, 1, 1);
            match gamma.as_ref() {
                Some(gamma) => {
                    let gamma = into_contiguous(gamma.clone());
                    kernel::input_weighted::launch(&client, count, info.dim, input.clone().into_array_arg(), gamma.clone().into_array_arg(),
                        grad.clone().into_array_arg(), mean.clone().into_array_arg(), rstd.clone().into_array_arg(), dx.clone().into_array_arg(),
                        info.width as u32, info.channels as u32, info.spatial as u32, [input.dtype.into(), gamma.dtype.into(), grad.dtype.into()]);
                }
                None => kernel::input_plain::launch(&client, count, info.dim, input.clone().into_array_arg(), grad.clone().into_array_arg(),
                    mean.clone().into_array_arg(), rstd.clone().into_array_arg(), dx.clone().into_array_arg(), info.width as u32, [input.dtype.into(), grad.dtype.into()]),
            }
        }
    }
    if dw.is_some() || db.is_some() {
        let parts = if partitions == 0 { (info.batch * info.spatial).div_ceil(32).clamp(1, 128) } else { partitions };
        let work = parts * info.channels;
        let scratch = || empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([parts, info.channels]), DType::F32);
        let wp = dw.as_ref().map(|_| scratch());
        let bp = db.as_ref().map(|_| scratch());
        let dim = RudaDim::new(client.properties(), work);
        let count = calculate_ruda_count_elemwise(&client, work, dim);
        match (&wp, &bp) {
            (Some(wp), Some(bp)) => kernel::affine_partial::launch(&client, count, dim, input.clone().into_array_arg(), grad.clone().into_array_arg(),
                mean.into_array_arg(), rstd.into_array_arg(), wp.clone().into_array_arg(), bp.clone().into_array_arg(),
                info.width as u32, info.channels as u32, info.spatial as u32, parts as u32, [input.dtype.into(), grad.dtype.into()]),
            (Some(wp), None) => kernel::weight_partial::launch(&client, count, dim, input.clone().into_array_arg(), grad.clone().into_array_arg(),
                mean.into_array_arg(), rstd.into_array_arg(), wp.clone().into_array_arg(), info.width as u32, info.channels as u32, info.spatial as u32,
                parts as u32, [input.dtype.into(), grad.dtype.into()]),
            (None, Some(bp)) => kernel::bias_partial::launch(&client, count, dim, grad.clone().into_array_arg(), bp.clone().into_array_arg(),
                info.channels as u32, info.spatial as u32, parts as u32, grad.dtype.into()),
            (None, None) => unreachable!(),
        }
        let dim = RudaDim::new(client.properties(), info.channels);
        let count = calculate_ruda_count_elemwise(&client, info.channels, dim);
        match (&dw, &db, wp, bp) {
            (Some(dw), Some(db), Some(wp), Some(bp)) => super::kernel::layer_norm_affine_merge::launch(&client, count, dim,
                wp.into_array_arg(), bp.into_array_arg(), dw.clone().into_array_arg(), db.clone().into_array_arg(), parts as u32, dw.dtype.into()),
            (Some(dw), None, Some(wp), None) => super::rms::weight_merge::launch(&client, count, dim,
                wp.into_array_arg(), dw.clone().into_array_arg(), parts as u32, dw.dtype.into()),
            (None, Some(db), None, Some(bp)) => super::rms::weight_merge::launch(&client, count, dim,
                bp.into_array_arg(), db.clone().into_array_arg(), parts as u32, DType::F32.into()),
            _ => unreachable!(),
        }
    }
    Ok([dx, dw, db])
}
