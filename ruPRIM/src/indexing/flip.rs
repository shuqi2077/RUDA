use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::DType;
use ruda_core::tensor::TensorMetadata;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearView;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn flip_kernel<E: Numeric, Bool: Int>(
    input: &Tensor<E>,
    output: &mut LinearView<E, ReadWrite>,
    in_shape: Sequence<FastDivmod<usize>>,
    indices: Sequence<InputScalar>,
    #[define(E, Bool)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = in_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let dim = rank - i - 1;
        let shape = input.shape(dim);

        let (rem, offset_local) = in_shape[dim].div_mod(offset);
        offset = rem;

        let flip = indices.index(dim).get::<Bool>() == Bool::from_int(1);
        let offset_local = select(flip, shape - offset_local - 1, offset_local);

        offset_input += offset_local * input.stride(dim);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub fn flip<R: Runtime>(
    tensor: RudaTensor<R>,
    indices: &[usize],
    dtype_bool: DType,
) -> RudaTensor<R> {
    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        tensor.dtype,
    );
    flip_on_output(tensor, output, indices, dtype_bool)
}

pub fn flip_on_output<R: Runtime>(
    tensor: RudaTensor<R>,
    output: RudaTensor<R>,
    indices: &[usize],
    dtype_bool: DType,
) -> RudaTensor<R> {
    if output.meta.num_elements() == 0 {
        return output;
    }
    let dtype_input = tensor.dtype;
    let ndims = tensor.meta.num_dims();
    let mut indices_sequence = SequenceArg::<R, InputScalar>::new();

    for i in 0..ndims {
        indices_sequence.push({
            let val = indices.contains(&i) as u8;
            InputScalar::new(val, dtype_bool)
        });
    }

    let num_elements = output.meta.num_elements();
    let ruda_dim = RudaDim::new(tensor.client.properties(), num_elements);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, num_elements, ruda_dim);

    let shape = shape_divmod(&tensor);
    unsafe {
        flip_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, output),
            tensor.into_tensor_arg(),
            output.clone().into_linear_view(),
            shape,
            indices_sequence,
            [dtype_input.into(), dtype_bool.into()],
        )
    }

    output
}
