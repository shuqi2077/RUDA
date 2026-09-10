use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn repeat_dim_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    out_shape: Sequence<FastDivmod<usize>>,
    in_shape: FastDivmod<usize>,
    #[comptime] dim: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let rank = out_shape.len().comptime();

    let mut pos = ABSOLUTE_POS;
    let mut offset_input = 0;
    let mut offset_output = 0;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;

        let (rem, mut local_pos) = out_shape[i].div_mod(pos);
        pos = rem;

        offset_output += local_pos * output.stride(i);

        if i == dim {
            local_pos = in_shape.modulo(local_pos);
        }

        offset_input += local_pos * input.stride(i);
    }

    output[offset_output] = input[offset_input];
}

pub fn repeat_dim<R: Runtime>(
    mut input: RudaTensor<R>,
    dim: usize,
    times: usize,
) -> RudaTensor<R> {
    if input.meta.shape()[dim] == 1 {
        input.meta.strides[dim] = 0;
        input.meta.shape = input.meta.shape.clone().repeat(dim, times).unwrap();
        return input;
    }

    let shape = input.meta.shape.clone().repeat(dim, times).unwrap();

    // Create output handle
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape,
        input.dtype,
    );

    let working_units = output.meta.num_elements();
    if working_units == 0 {
        return output;
    }

    let ruda_dim = RudaDim::new(input.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, working_units, ruda_dim);

    let shape_arg = input.meta.shape()[dim];

    unsafe {
        repeat_dim_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(input, output),
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            shape_divmod(&output),
            shape_arg,
            dim,
            output.dtype.into(),
        )
    };

    output
}
