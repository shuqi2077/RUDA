use super::lanczos3::{lanczos3_coordinate, lanczos3_weight};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::num_traits::Zero;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearLayout;
use ruda_kernel::library::tensor::layout::*;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::tensor::layout::{address_type, linear_layout, max_vector_size_many, shape_divmod};

#[ruda]
fn contributing_range(
    input_index: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] align_corners: bool,
) -> (usize, usize) {
    let mut left = 0usize;
    let mut right = output_size;
    while left < right {
        let middle = left + (right - left) / 2;
        let coordinate = lanczos3_coordinate(middle, input_size, output_size, align_corners);
        let upper = clamp_min(coordinate.floor() + 3.0, 0.0) as usize;
        if upper < input_index {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    let start = left;
    right = output_size;
    while left < right {
        let middle = left + (right - left) / 2;
        let coordinate = lanczos3_coordinate(middle, input_size, output_size, align_corners);
        let lower = clamp_min(coordinate.floor() - 2.0, 0.0) as usize;
        if lower <= input_index {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    (start, left)
}

#[ruda]
fn axis_weights(coordinate: f32, last: f32) -> Sequence<f32> {
    let origin = coordinate.floor();
    let mut weights = Sequence::new();
    #[unroll]
    for tap in 0..6usize {
        let offset = comptime![tap as f32 - 2.0];
        let position = origin + offset;
        let weight = if position >= 0.0 && position <= last {
            lanczos3_weight(coordinate - position)
        } else {
            0.0f32.runtime()
        };
        weights.push(weight);
    }
    weights
}

#[ruda(launch, address_type = "dynamic")]
fn interpolate_lanczos3_backward_kernel<F: Float, N: Size>(
    grad: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[comptime] align_corners: bool,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let vector_size = grad.vector_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);
    let (rem, channel) = shape_out[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, input_x) = shape_out[2].div_mod(rem);
    let (batch, input_y) = shape_out[1].div_mod(rem);
    let height = output.shape(1);
    let width = output.shape(2);
    let last_y = (height - 1) as f32;
    let last_x = (width - 1) as f32;
    let grad_height = grad.shape(1);
    let grad_width = grad.shape(2);
    let (y_start, y_end) = contributing_range(input_y, height, grad_height, align_corners);
    let (x_start, x_end) = contributing_range(input_x, width, grad_width, align_corners);
    let index_base = batch * grad.stride(0) + channel * grad.stride(3);
    let mut sum = Vector::zero();

    for y in y_start..y_end {
        let coordinate = lanczos3_coordinate(y, height, grad_height, align_corners);
        let y_floor = coordinate.floor();
        let weights_y = axis_weights(coordinate, last_y);

        for x in x_start..x_end {
            let coordinate = lanczos3_coordinate(x, width, grad_width, align_corners);
            let x_floor = coordinate.floor();
            let weights_x = axis_weights(coordinate, last_x);
            let mut weight_sum = 0.0f32;

            #[unroll]
            for ky in 0..6usize {
                let offset_y = comptime![ky as f32 - 2.0];
                let y_position = y_floor + offset_y;
                if y_position >= 0.0 && y_position <= last_y {
                    #[unroll]
                    for kx in 0..6usize {
                        let offset_x = comptime![kx as f32 - 2.0];
                        let x_position = x_floor + offset_x;
                        if x_position >= 0.0 && x_position <= last_x {
                            weight_sum += weights_y[ky] * weights_x[kx];
                        }
                    }
                }
            }

            let index = index_base + y * grad.stride(1) + x * grad.stride(2);
            let mut value = grad[index / vector_size];
            if weight_sum != 0.0 {
                value *= Vector::new(F::cast_from(1.0 / weight_sum));
            }

            #[unroll]
            for ky in 0..6usize {
                let offset_y = comptime![ky as f32 - 2.0];
                let y_position = y_floor + offset_y;
                if y_position >= 0.0 && y_position <= last_y {
                    if y_position as usize == input_y {
                        #[unroll]
                        for kx in 0..6usize {
                            let offset_x = comptime![kx as f32 - 2.0];
                            let x_position = x_floor + offset_x;
                            if x_position >= 0.0 && x_position <= last_x {
                                if x_position as usize == input_x {
                                    let weight = weights_y[ky] * weights_x[kx];
                                    sum += value * Vector::new(F::cast_from(weight));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    output[out_idx] = sum;
}

pub fn interpolate_lanczos3_backward_launch<R: Runtime>(
    out_grad: RudaTensor<R>,
    output: RudaTensor<R>,
    align_corners: bool,
) -> RudaTensor<R> {
    if output.meta.num_elements() == 0 {
        return output;
    }
    let vector_size = max_vector_size_many(&[&out_grad, &output], 3);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);
    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(out_grad.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&out_grad.client, working_units, ruda_dim);

    interpolate_lanczos3_backward_kernel::launch(
        &output.client,
        ruda_count,
        ruda_dim,
        address_type!(out_grad, output),
        vector_size,
        out_grad.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        out_shape,
        out_layout,
        align_corners,
        output.dtype.into(),
    );

    output
}
