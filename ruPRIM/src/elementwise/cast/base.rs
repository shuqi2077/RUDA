use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::DType;
use ruda_core::tensor::TensorMetadata;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
pub fn cast_element<I: Numeric, O: Numeric, N: Size>(
    input: &LinearView<Vector<I, N>>,
    output: &mut LinearView<Vector<O, N>, ReadWrite>,
    #[define(I, O)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS]);
}

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: Runtime>(input: RudaTensor<R>, dtype: DType) -> RudaTensor<R> {
    let dtype_output = match dtype {
        DType::Flex32 => DType::F32,
        _ => dtype,
    };
    let dtype_input = match input.dtype {
        DType::Flex32 => DType::F32,
        _ => input.dtype,
    };

    if dtype_input == dtype_output {
        return input;
    }

    let client = input.client.clone();

    let vector_size = max_vector_size(&input);

    let num_elems: usize = input.meta.num_elements();

    let working_units = num_elems / vector_size as usize;
    let ruda_dim = RudaDim::new(client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&client, working_units, ruda_dim);

    let output = empty_device_dtype(
        client.clone(),
        input.device.clone(),
        input.shape(),
        dtype, // We take the same dtype as passed as input (Flex32 not F32)
    );

    cast_element::launch(
        &client,
        ruda_count,
        ruda_dim,
        address_type!(input, output),
        vector_size,
        input.into_linear_view(),
        output.clone().into_linear_view(),
        [dtype_input.into(), dtype_output.into()],
    );

    output
}
