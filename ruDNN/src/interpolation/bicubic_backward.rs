use super::bicubic::{bicubic_coordinate, cubic_coefficient};
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
    let last = (input_size - 1) as f32;
    let mut left = 0usize;
    let mut right = output_size;
    while left < right {
        let middle = left + (right - left) / 2;
        let coordinate = bicubic_coordinate(middle, input_size, output_size, align_corners);
        let upper = clamp(coordinate.floor() + 2.0, 0.0, last) as usize;
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
        let coordinate = bicubic_coordinate(middle, input_size, output_size, align_corners);
        let lower = clamp(coordinate.floor() - 1.0, 0.0, last) as usize;
        if lower <= input_index {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    (start, left)
}

#[ruda(launch, address_type = "dynamic")]
fn interpolate_bicubic_backward_kernel<F: Float, N: Size>(
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
        let coordinate = bicubic_coordinate(y, height, grad_height, align_corners);
        let y_floor = coordinate.floor();
        let yw = Vector::<F, N>::new(F::cast_from(coordinate - y_floor));

        for x in x_start..x_end {
            let coordinate = bicubic_coordinate(x, width, grad_width, align_corners);
            let x_floor = coordinate.floor();
            let xw = Vector::<F, N>::new(F::cast_from(coordinate - x_floor));
            let index = index_base + y * grad.stride(1) + x * grad.stride(2);
            let value = grad[index / vector_size];

            #[unroll]
            for ky in 0..4usize {
                let offset_y = comptime![ky as f32 - 1.0];
                let y_index = clamp(y_floor + offset_y, 0.0, last_y) as usize;
                if y_index == input_y {
                    let row_grad = value * cubic_coefficient(yw, ky);
                    #[unroll]
                    for kx in 0..4usize {
                        let offset_x = comptime![kx as f32 - 1.0];
                        let x_index = clamp(x_floor + offset_x, 0.0, last_x) as usize;
                        if x_index == input_x {
                            sum += row_grad * cubic_coefficient(xw, kx);
                        }
                    }
                }
            }
        }
    }

    output[out_idx] = sum;
}

pub fn interpolate_bicubic_backward_launch<R: Runtime>(
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

    interpolate_bicubic_backward_kernel::launch(
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
