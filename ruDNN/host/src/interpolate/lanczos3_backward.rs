use super::*;

fn axis_taps(
    input_size: usize,
    output_size: usize,
    align_corners: bool,
) -> Vec<[Option<(usize, f64)>; 6]> {
    let ratio = coord_ratio(input_size, output_size, align_corners);
    let last = input_size as isize - 1;
    (0..output_size)
        .map(|output_index| {
            let coordinate = map_coord(output_index, ratio, align_corners);
            let origin = coordinate.floor();
            core::array::from_fn(|tap| {
                let offset = tap as isize - 2;
                let index = origin as isize + offset;
                if index < 0 || index > last {
                    None
                } else {
                    Some((index as usize, lanczos3_weight(coordinate - (origin + offset as f64))))
                }
            })
        })
        .collect()
}

pub(super) fn interpolate_lanczos3_backward_impl<T>(
    x: HostTensor,
    grad: HostTensor,
    output_size: [usize; 2],
    align_corners: bool,
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
    let y_taps = axis_taps(in_height, out_height, align_corners);
    let x_taps = axis_taps(in_width, out_width, align_corners);
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;
    let planes = batch * channels;
    let mut input_grad = vec![T::zero(); planes * in_hw];

    let scatter_plane = |plane: usize, input_grad: &mut [T]| {
        for (oh, y_taps) in y_taps.iter().enumerate() {
            for (ow, x_taps) in x_taps.iter().enumerate() {
                let mut weight_sum = 0.0_f64;
                for &(_, wy) in y_taps.iter().flatten() {
                    for &(_, wx) in x_taps.iter().flatten() {
                        weight_sum += wy * wx;
                    }
                }
                let value = &grad_data[plane * out_hw + oh * out_width + ow];
                let mut value = <T as num_traits::ToPrimitive>::to_f64(value).unwrap();
                if weight_sum != 0.0 {
                    value /= weight_sum;
                }
                for &(iy, wy) in y_taps.iter().flatten() {
                    for &(ix, wx) in x_taps.iter().flatten() {
                        let weight = wy * wx;
                        let index = iy * in_width + ix;
                        input_grad[index] = input_grad[index] + T::from(value * weight).unwrap();
                    }
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        if planes * out_hw >= ruda_core::tensor::host::parallel::PARALLEL_THRESHOLD {
            input_grad.par_chunks_mut(in_hw).enumerate().for_each(|(plane, grad)| {
                scatter_plane(plane, grad);
            });
        } else {
            for (plane, grad) in input_grad.chunks_mut(in_hw).enumerate() {
                scatter_plane(plane, grad);
            }
        }
    }
    #[cfg(not(feature = "rayon"))]
    for (plane, grad) in input_grad.chunks_mut(in_hw).enumerate() {
        scatter_plane(plane, grad);
    }

    HostTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}
