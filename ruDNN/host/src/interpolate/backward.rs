use super::*;

// ============================================================================
// Backward implementations
// ============================================================================

/// Nearest neighbor backward: accumulates gradients at source positions.
pub(super) fn interpolate_nearest_backward_impl<T>(
    x: HostTensor,
    grad: HostTensor,
    output_size: [usize; 2],
    _align_corners: bool,
) -> HostTensor
where
    T: Float + ruda_core::tensor::element::Element + bytemuck::Pod + Send + Sync,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_map = nearest_index_map(in_height, out_height);
    let x_map = nearest_index_map(in_width, out_width);

    let in_numel = batch * channels * in_height * in_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    // Scatter-add gradients from one output plane into one input gradient plane.
    #[inline]
    fn scatter_plane<T: Float + Copy>(
        grad_data: &[T],
        grad_base: usize,
        input_grad: &mut [T],
        in_width: usize,
        out_width: usize,
        y_map: &[usize],
        x_map: &[usize],
    ) {
        for (oh, &ih) in y_map.iter().enumerate() {
            let grad_row = grad_base + oh * out_width;
            for (ow, &iw) in x_map.iter().enumerate() {
                input_grad[ih * in_width + iw] =
                    input_grad[ih * in_width + iw] + grad_data[grad_row + ow];
            }
        }
    }

    let mut input_grad = vec![T::zero(); in_numel];
    let bc = batch * channels;

    // Each (batch, channel) plane is independent, so parallelize across planes.
    // Gate on output size since the work is proportional to iterating output pixels.
    #[cfg(feature = "rayon")]
    if bc * out_hw >= ruda_core::tensor::host::parallel::PARALLEL_THRESHOLD {
        use rayon::prelude::*;

        input_grad
            .par_chunks_mut(in_hw)
            .enumerate()
            .for_each(|(bc_idx, grad_plane)| {
                let grad_base = bc_idx * out_hw;
                scatter_plane(
                    grad_data, grad_base, grad_plane, in_width, out_width, &y_map, &x_map,
                );
            });
    } else {
        for bc_idx in 0..bc {
            let grad_base = bc_idx * out_hw;
            let in_start = bc_idx * in_hw;
            scatter_plane(
                grad_data,
                grad_base,
                &mut input_grad[in_start..in_start + in_hw],
                in_width,
                out_width,
                &y_map,
                &x_map,
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    for bc_idx in 0..bc {
        let grad_base = bc_idx * out_hw;
        let in_start = bc_idx * in_hw;
        scatter_plane(
            grad_data,
            grad_base,
            &mut input_grad[in_start..in_start + in_hw],
            in_width,
            out_width,
            &y_map,
            &x_map,
        );
    }

    HostTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

/// Bilinear backward: distributes gradients to 4 source positions weighted by bilinear coefficients.
pub(super) fn interpolate_bilinear_backward_impl<T>(
    x: HostTensor,
    grad: HostTensor,
    output_size: [usize; 2],
    align_corners: bool,
) -> HostTensor
where
    T: Float + ruda_core::tensor::element::Element + bytemuck::Pod,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = coord_ratio(in_height, out_height, align_corners);
    let x_ratio = coord_ratio(in_width, out_width, align_corners);

    let in_numel = batch * channels * in_height * in_width;
    let mut input_grad = vec![T::zero(); in_numel];

    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    for b in 0..batch {
        for c in 0..channels {
            let in_base = b * channels * in_hw + c * in_hw;
            let out_base = b * channels * out_hw + c * out_hw;

            for oh in 0..out_height {
                let y_in = map_coord(oh, y_ratio, align_corners);
                let y_low = (y_in.floor().max(0.0)) as usize;
                let y_high = (y_low + 1).min(in_height - 1);
                let y_weight = T::from((y_in - y_low as f64).max(0.0)).unwrap();

                for ow in 0..out_width {
                    let x_in = map_coord(ow, x_ratio, align_corners);
                    let x_low = (x_in.floor().max(0.0)) as usize;
                    let x_high = (x_low + 1).min(in_width - 1);
                    let x_weight = T::from((x_in - x_low as f64).max(0.0)).unwrap();

                    let grad_val = grad_data[out_base + oh * out_width + ow];
                    let one = T::one();

                    input_grad[in_base + y_low * in_width + x_low] = input_grad
                        [in_base + y_low * in_width + x_low]
                        + grad_val * (one - x_weight) * (one - y_weight);
                    input_grad[in_base + y_low * in_width + x_high] = input_grad
                        [in_base + y_low * in_width + x_high]
                        + grad_val * x_weight * (one - y_weight);
                    input_grad[in_base + y_high * in_width + x_low] = input_grad
                        [in_base + y_high * in_width + x_low]
                        + grad_val * (one - x_weight) * y_weight;
                    input_grad[in_base + y_high * in_width + x_high] = input_grad
                        [in_base + y_high * in_width + x_high]
                        + grad_val * x_weight * y_weight;
                }
            }
        }
    }

    HostTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

/// Bicubic backward: distributes gradients to 16 source positions weighted by cubic coefficients.
pub(super) fn interpolate_bicubic_backward_impl<T>(
    x: HostTensor,
    grad: HostTensor,
    output_size: [usize; 2],
    align_corners: bool,
) -> HostTensor
where
    T: Float + ruda_core::tensor::element::Element + bytemuck::Pod,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = coord_ratio(in_height, out_height, align_corners);
    let x_ratio = coord_ratio(in_width, out_width, align_corners);

    let in_numel = batch * channels * in_height * in_width;
    let mut input_grad = vec![T::zero(); in_numel];

    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;
    let a = -0.75_f64;

    for b in 0..batch {
        for c in 0..channels {
            let in_base = b * channels * in_hw + c * in_hw;
            let out_base = b * channels * out_hw + c * out_hw;

            for oh in 0..out_height {
                let y_in = map_coord(oh, y_ratio, align_corners);
                let y0 = y_in.floor() as isize;

                for ow in 0..out_width {
                    let x_in = map_coord(ow, x_ratio, align_corners);
                    let x0 = x_in.floor() as isize;

                    let grad_val = <T as num_traits::ToPrimitive>::to_f64(
                        &grad_data[out_base + oh * out_width + ow],
                    )
                    .unwrap_or(0.0);

                    for dy in -1..=2_isize {
                        let y = y0 + dy;
                        let y_idx = y.clamp(0, in_height as isize - 1) as usize;
                        let ty = (y_in - y0 as f64) - dy as f64;
                        let wy = cubic_weight(ty, a);

                        for dx in -1..=2_isize {
                            let x = x0 + dx;
                            let x_idx = x.clamp(0, in_width as isize - 1) as usize;
                            let tx = (x_in - x0 as f64) - dx as f64;
                            let wx = cubic_weight(tx, a);

                            let weight = wx * wy * grad_val;
                            input_grad[in_base + y_idx * in_width + x_idx] = input_grad
                                [in_base + y_idx * in_width + x_idx]
                                + T::from(weight).unwrap();
                        }
                    }
                }
            }
        }
    }

    HostTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

