use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearLayout;
use ruda_kernel::library::tensor::layout::*;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;

use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::linear_layout;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::RudaTensor;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn interpolate_nearest_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let vector_size = input.vector_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let out_pos = ABSOLUTE_POS * vector_size;

    let (h_in, w_in) = (input.shape(1), input.shape(2));
    let (h_out, w_out) = (output.shape(1), output.shape(2));

    let (rem, c) = shape_out[3].div_mod(out_pos);
    let (rem, x) = shape_out[2].div_mod(rem);
    let (b, y) = shape_out[1].div_mod(rem);

    let y = y * h_in / h_out;
    let x = x * w_in / w_out;

    let in_idx =
        b * input.stride(0) + y * input.stride(1) + x * input.stride(2) + c * input.stride(3);

    output[out_idx] = input[in_idx / vector_size];
}

pub fn interpolate_nearest_launch<R: Runtime>(
    input: RudaTensor<R>,
    output: RudaTensor<R>,
) -> RudaTensor<R> {
    let client = input.client.clone();

    let vector_size = max_vector_size(&input);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(input.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, working_units, ruda_dim);

    let shape_out = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);

    unsafe {
        interpolate_nearest_kernel::launch_unchecked(
            &client,
            ruda_count,
            ruda_dim,
            address_type!(input, output),
            vector_size,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            shape_out,
            out_layout,
            output.dtype.into(),
        )
    };

    output
}
