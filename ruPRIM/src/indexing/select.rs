use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_core::tensor::TensorMetadata;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn select_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<T>,
    indices: &LinearView<I>,
    output: &mut LinearView<T, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= output.shape() {
        terminate!();
    }

    let rank = out_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        let (rem, offset_local) = out_shape[i].div_mod(offset);
        offset = rem;

        let offset_local = ruda_kernel::dsl::prelude::select(
            i == dim,
            usize::cast_from(indices.read_checked(offset_local)),
            offset_local,
        );

        offset_input += offset_local * input.stride(i);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub fn select<R: Runtime>(
    tensor: RudaTensor<R>,
    dim: usize,
    indices: RudaTensor<R>,
) -> RudaTensor<R> {
    let mut shape_output = tensor.shape();
    shape_output[dim] = indices.meta.shape()[0];
    let total_elem = shape_output.num_elements();

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output,
        tensor.dtype,
    );

    if total_elem == 0 {
        return output;
    }

    let working_units = total_elem;
    let ruda_dim = RudaDim::new(indices.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&indices.client, working_units, ruda_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    unsafe {
        select_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, indices, output),
            tensor.into_tensor_arg(),
            indices.into_linear_view(),
            output.clone().into_linear_view(),
            shape_divmod(&output),
            dim,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    };
    output
}
