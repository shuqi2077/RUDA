use crate::library::tensor::{TensorHandle, copy_gpu_ref, into_contiguous};
use crate::dsl::prelude::*;
use ruda_kernel::dsl::{
    calculate_ruda_count_elemwise, tensor_vector_size_parallel,
};
use std::cmp::min;

/// Kernel for converting a non-contiguous tensor into a contiguous one when
/// the vectorization axis is perpendicular to the last dimension.
///
/// This kernel handles the case where memory is laid out such that the unit-stride
/// is not on the last dimension, requiring a "gather-and-transpose" pattern
/// to write out contiguous vectors.
#[ruda(launch_unchecked, address_type = "dynamic")]
fn copy_perpendicular<T: Numeric, N: Size>(
    input: &Tensor<Vector<T, N>>,
    output: &mut Tensor<Vector<T, N>>,
    axis_vectorized: usize,
    working_units: usize,
    #[define(T)] _elem: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }
    let vector_size = input.vector_size();
    let last_axis = input.rank() - 1;
    let mut remaining = ABSOLUTE_POS;
    let mut input_base = 0usize;
    let mut output_base = 0usize;
    for reverse_axis in 0..input.rank() {
        let axis = last_axis - reverse_axis;
        let tiled = axis == axis_vectorized || axis == last_axis;
        let mut size = output.shape(axis);
        if tiled {
            size /= vector_size;
        }
        let mut coordinate = remaining % size;
        remaining /= size;
        if tiled {
            coordinate *= vector_size;
        }
        input_base += coordinate * input.stride(axis);
        output_base += coordinate * output.stride(axis);
    }

    let mut accumulators = Sequence::<Vector<T, N>>::new();

    #[unroll]
    for _ in 0..vector_size {
        accumulators.push(Vector::empty());
    }

    for i in 0..vector_size {
        let index = (input_base + i * input.stride(last_axis)) / vector_size;
        let batched = input[index];
        #[unroll]
        for o in 0..vector_size {
            let vector = accumulators.index_mut(o);
            vector[i] = batched[o];
        }
    }

    #[unroll]
    for o in 0..vector_size {
        let index = (output_base + o * output.stride(axis_vectorized)) / vector_size;
        output[index] = accumulators[o];
    }
}

/// Launches the perpendicular contiguous kernel.
///
/// This is used when the input tensor's memory layout is such that the last dimension
/// is not the one with a stride of 1 (the vectorized dimension). It optimizes
/// the copy by using hardware vectorization (Vectors) and an in-register transpose.
pub fn launch_into_contiguous_perpendicular<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    dtype: StorageType,
) -> TensorHandle<R> {
    // Fallback for 1D tensors where perpendicularity doesn't apply.
    if input.shape.len() <= 1 {
        return into_contiguous(client, input, dtype);
    }

    let output = TensorHandle::empty(client, input.shape.to_vec(), dtype);
    launch_copy_perpendicular_ref(client, input, output.clone().binding(), dtype);

    output
}

/// Launches the perpendicular contiguous kernel.
///
/// This is used when the input tensor's memory layout is such that the last dimension
/// is not the one with a stride of 1 (the vectorized dimension). It optimizes
/// the copy by using hardware vectorization (Vectors) and an in-register transpose.
pub fn launch_copy_perpendicular_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
) {
    let num_elems = output.shape.iter().product::<usize>();
    if num_elems == 0 {
        return;
    }
    let rank = input.shape.len();
    let axis = input.strides.iter().enumerate().find_map(|(axis, &stride)| {
        (axis + 1 < rank && stride == 1).then_some(axis)
    });
    let Some(axis) = axis else {
        copy_gpu_ref(client, input, output, dtype);
        return;
    };
    if input.shape != output.shape {
        copy_gpu_ref(client, input, output, dtype);
        return;
    }

    let vector_size_input = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        axis,
    );
    let vector_size_parallel = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        rank - 1,
    );
    let vector_size = min(vector_size_input, vector_size_parallel);
    if vector_size == 1 {
        copy_gpu_ref(client, input, output, dtype);
        return;
    }

    let working_units = num_elems / (vector_size * vector_size);
    let ruda_dim = RudaDim::new(client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(client, working_units, ruda_dim);
    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        copy_perpendicular::launch_unchecked::<R>(
            client,
            ruda_count,
            ruda_dim,
            address_type,
            vector_size,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            working_units,
            dtype,
        );
    }
}
