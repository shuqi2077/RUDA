use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::broadcast_strides;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::TensorMetadata;
use ruda_kernel::dsl::frontend::ABSOLUTE_POS;
use ruda_kernel::dsl::frontend::Numeric;
use ruda_kernel::dsl::frontend::Tensor;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::index_offset_contiguous_fastdivmod;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn gather_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<T>,
    indices: &LinearView<I>,
    output: &mut LinearView<T, ReadWrite>,
    in_strides: Sequence<usize>, // zeroed out for broadcast dims and `dim`
    out_shape: Sequence<FastDivmod<usize>>,
    dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if !indices.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let mut offset = index_offset_contiguous_fastdivmod(
        ABSOLUTE_POS,
        &out_shape,
        &in_strides,
        input.vector_size(),
    );

    offset += usize::cast_from(indices[ABSOLUTE_POS]) * input.stride(dim);

    output[ABSOLUTE_POS] = input[offset];
}

pub fn gather<R: Runtime>(
    dim: usize,
    tensor: RudaTensor<R>,
    indices: RudaTensor<R>,
) -> RudaTensor<R> {
    let shape_output = indices.shape();
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

    let ruda_dim = RudaDim::new(tensor.client.properties(), total_elem);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, total_elem, ruda_dim);
    let mut in_strides = broadcast_strides(&output, &tensor);
    in_strides.values[dim] = 0; // Zero `dim` to exclude it from the indexing

    let (dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    unsafe {
        gather_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, indices, output),
            tensor.into_tensor_arg(),
            indices.into_linear_view(),
            output.clone().into_linear_view(),
            in_strides,
            shape_divmod(&output),
            dim,
            [dtype.into(), indices_dtype.into()],
        )
    }

    output
}
