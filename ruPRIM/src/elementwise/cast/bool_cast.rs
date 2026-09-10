use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::TensorMetadata;
use ruda_core::tensor::DType;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::num_traits::One;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::linear::LinearView;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn bool_cast_kernel<B: Int, T: Numeric, N: Size>(
    input: &LinearView<Vector<B, N>>,
    output: &mut LinearView<Vector<T, N>, ReadWrite>,
    #[define(B, T)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS] & Vector::one());
}

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32 or u8
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: Runtime>(tensor: RudaTensor<R>, out_dtype: DType) -> RudaTensor<R> {
    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        out_dtype,
    );

    let vector_size = max_vector_size(&tensor);
    let num_elems = tensor.meta.num_elements();
    let working_units = num_elems / vector_size as usize;
    let ruda_dim = RudaDim::new(tensor.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, working_units, ruda_dim);

    let dtype = tensor.dtype;

    unsafe {
        bool_cast_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, output),
            vector_size,
            tensor.into_linear_view(),
            output.clone().into_linear_view(),
            [dtype.into(), out_dtype.into()],
        )
    };

    output
}
