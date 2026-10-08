use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl::num_traits::Zero;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearLayout;
use ruda_kernel::library::tensor::layout::*;

use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::linear_layout;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::max_vector_size_many;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::ir::AddressType;
use crate::indexing::scaled_index_division;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn interpolate_nearest_backward_kernel<F: Float, G: Float, A: Float, N: Size>(
    grad: &Tensor<Vector<G, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[comptime] max_value: usize,
    #[define(F, G, A)] _types: [StorageType; 3],
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let vector_size = grad.vector_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let out_h = output.shape(1);
    let out_w = output.shape(2);
    let grad_h = grad.shape(1);
    let grad_w = grad.shape(2);

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, out_x) = shape_out[2].div_mod(rem);
    let (b, out_y) = shape_out[1].div_mod(rem);

    let grad_y_start = boundary_index(out_y, grad_h, out_h, max_value);
    let grad_y_end = boundary_index(out_y + 1, grad_h, out_h, max_value);
    let grad_x_start = boundary_index(out_x, grad_w, out_w, max_value);
    let grad_x_end = boundary_index(out_x + 1, grad_w, out_w, max_value);

    let index_grad_base = b * grad.stride(0) + c * grad.stride(3);

    let mut sum = Vector::<A, N>::zero();

    for grad_y in grad_y_start..grad_y_end {
        for grad_x in grad_x_start..grad_x_end {
            let index_grad = index_grad_base + grad_y * grad.stride(1) + grad_x * grad.stride(2);

            sum += Vector::<A, N>::cast_from(grad[index_grad / vector_size]);
        }
    }

    output[out_idx] = Vector::<F, N>::cast_from(sum);
}

#[ruda]
fn boundary_index(input_index: usize, output_size: usize, input_size: usize,
    #[comptime] max_value: usize) -> usize {
    let (quotient, remainder) = scaled_index_division(input_index, output_size, input_size, max_value);
    if remainder == 0 {
        quotient
    } else {
        quotient + 1
    }
}

pub fn interpolate_nearest_backward_launch<R: Runtime>(
    out_grad: RudaTensor<R>,
    output: RudaTensor<R>,
) -> RudaTensor<R> {
    if output.meta.num_elements() == 0 { return output; }
    let types = super::base::backward_types(output.dtype, out_grad.dtype);
    let vector_size = max_vector_size_many(&[&out_grad, &output], 3);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);
    let address = address_type!(out_grad, output);
    let max_value = if address == AddressType::U64 { usize::MAX } else { u32::MAX as usize };

    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(out_grad.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&out_grad.client, working_units, ruda_dim);

    unsafe {
        interpolate_nearest_backward_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address,
            vector_size,
            out_grad.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            out_shape,
            out_layout,
            max_value,
            types,
        )
    };

    output
}
