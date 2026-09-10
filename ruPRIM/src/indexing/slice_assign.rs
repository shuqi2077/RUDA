use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::tensor_vector_size_parallel;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::intrinsic;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearView;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn slice_assign_kernel<E: Numeric, N: Size>(
    input: &mut Tensor<Vector<E, N>>,
    value: &LinearView<Vector<E, N>>,
    slice_shape: Sequence<FastDivmod<usize>>,
    slice_offsets: Sequence<usize>,
    #[define(E)] _dtype: StorageType,
) {
    if !value.is_in_bounds(ABSOLUTE_POS) {
        terminate!()
    }

    let rank = comptime!(slice_shape.len());

    let line_size = input.vector_size();
    let mut offset_remainder = ABSOLUTE_POS * line_size;
    let mut offset_input = 0;

    #[allow(clippy::explicit_counter_loop)]
    #[unroll]
    for i in 0..rank {
        let dim = rank - i - 1;
        let (rem, offset_local) = slice_shape[dim].div_mod(offset_remainder);

        let range_start = slice_offsets[dim];
        let offset_local_input = offset_local + range_start;

        offset_input += offset_local_input * input.stride(dim);
        offset_remainder = rem;
    }

    // Value tensor is accessed linearly since it's a LinearView
    input[offset_input / line_size] = value[ABSOLUTE_POS];
}

/// Kernel for slice assign with steps
#[ruda(launch_unchecked, address_type = "dynamic")]
fn slice_assign_with_steps_kernel<E: Numeric>(
    input: &mut Tensor<E>,
    value: &LinearView<E>,
    value_shape: Sequence<FastDivmod<usize>>,
    starts: Sequence<usize>,
    ends: Sequence<usize>,
    steps: Sequence<usize>,
    reversed: Sequence<usize>,
    #[define(E)] _dtype: StorageType,
) {
    if !value.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![value_shape.len()];
    let mut value_offset = ABSOLUTE_POS;
    let mut input_offset = 0;

    // Calculate the input offset based on value position and slice info
    #[unroll]
    for i in 0..rank {
        // Iterate in reverse to use divmod
        let dim = rank - i - 1;
        let start = starts[dim];
        let end = ends[dim];
        let step = steps[dim];

        let (rem, value_idx) = value_shape[dim].div_mod(value_offset);
        value_offset = rem;

        let input_idx = if reversed[dim] == 0 {
            // Forward stepping
            start + value_idx * step
        } else {
            // Backward stepping - start from end-1
            // For negative steps, we iterate backwards through the selected indices
            let end_minus_1 = end - 1;
            end_minus_1 - value_idx * step
        };

        input_offset += input_idx * input.stride(dim);
    }

    input[input_offset] = value[ABSOLUTE_POS];
}

pub fn slice_assign<R: Runtime>(
    tensor: RudaTensor<R>,
    indices: &[ruda_core::tensor::Slice],
    value: RudaTensor<R>,
) -> RudaTensor<R> {
    assert!(indices.iter().all(|slice| slice.step != 0), "Step cannot be zero");
    if value.meta.num_elements() == 0 {
        return tensor;
    }
    // Check if any slice has non-unit step
    let has_non_unit_step = indices.iter().any(|s| s.step != 1);

    if has_non_unit_step {
        // Use slice_assign_with_steps
        return slice_assign_with_steps(tensor, indices, value);
    }

    let client = tensor.client.clone();
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };
    let ndims = tensor.meta.num_dims();
    let ranges = (0..ndims)
        .map(|axis| indices.get(axis).copied().unwrap_or_default().to_range(tensor.meta.shape()[axis]))
        .collect::<Vec<_>>();
    let mut slice_shape = tensor.meta.shape().clone();
    for (dim, range) in slice_shape.iter_mut().zip(&ranges) {
        *dim = range.end - range.start;
    }
    let base_offset = ranges.iter().zip(tensor.meta.strides().iter())
        .map(|(range, stride)| range.start * stride).sum::<usize>();
    let vector_size = if ndims > 0 {
        let candidates = client.io_optimized_vector_sizes(tensor.dtype.size()).filter(|&size| {
            base_offset.is_multiple_of(size)
                && tensor_vector_size_parallel(
                    core::iter::once(size), &slice_shape, tensor.meta.strides(), ndims - 1,
                ) == size
        });
        tensor_vector_size_parallel(candidates, value.meta.shape(), value.meta.strides(), ndims - 1)
    } else {
        1
    };

    let mut shape = SequenceArg::<R, FastDivmod<usize>>::new();
    let mut offsets = SequenceArg::<R, usize>::new();

    for range in ranges {
        shape.push(range.end - range.start);
        offsets.push(range.start);
    }

    let working_units = value.meta.num_elements() / vector_size;
    let ruda_dim = RudaDim::new(tensor.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, working_units, ruda_dim);

    unsafe {
        slice_assign_kernel::launch_unchecked(
            &tensor.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, value),
            vector_size,
            tensor.clone().into_tensor_arg(),
            value.into_linear_view(),
            shape,
            offsets,
            tensor.dtype.into(),
        )
    };

    tensor
}

/// Slice assign with steps support
///
/// This function handles slice assignment with arbitrary step values, including negative steps.
/// It follows NumPy/PyTorch semantics where values[i] is assigned to selected_indices[i].
///
/// For example, with s![0..6;-1] which selects indices [5,4,3,2,1,0]:
/// - values[0] goes to index 5
/// - values[1] goes to index 4
/// - etc.
pub fn slice_assign_with_steps<R: Runtime>(
    tensor: RudaTensor<R>,
    slices: &[ruda_core::tensor::Slice],
    value: RudaTensor<R>,
) -> RudaTensor<R> {
    assert!(slices.iter().all(|slice| slice.step != 0), "Step cannot be zero");
    if value.meta.num_elements() == 0 {
        return tensor;
    }
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    // Prepare sequences for kernel
    let mut starts = SequenceArg::<R, usize>::new();
    let mut ends = SequenceArg::<R, usize>::new();
    let mut steps = SequenceArg::<R, usize>::new();
    let mut reversed = SequenceArg::<R, usize>::new();

    for (dim, slice) in slices.iter().enumerate() {
        let range = slice.to_range(tensor.meta.shape()[dim]);
        starts.push(range.start);
        ends.push(range.end);
        steps.push(slice.step.unsigned_abs());
        reversed.push(usize::from(slice.is_reversed()));
    }

    // Pad with default values if needed to match tensor dimensions
    for dim in slices.len()..tensor.meta.num_dims() {
        starts.push(0);
        ends.push(tensor.meta.shape[dim]);
        steps.push(1);
        reversed.push(0);
    }

    // Launch kernel
    let working_units = value.meta.num_elements();
    let ruda_dim = RudaDim::new(tensor.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, working_units, ruda_dim);

    let shape = shape_divmod(&value);
    let step_address = AddressType::from_len(
        slices.iter().map(|slice| slice.step.unsigned_abs()).max().unwrap_or(1),
    );
    unsafe {
        slice_assign_with_steps_kernel::launch_unchecked(
            &tensor.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, value).max(step_address),
            tensor.clone().into_tensor_arg(),
            value.into_linear_view(),
            shape,
            starts,
            ends,
            steps,
            reversed,
            tensor.dtype.into(),
        );
    }

    tensor
}

/// Helper function for unwrap
#[allow(unused)]
#[ruda]
fn unwrap(value: u32) -> comptime_type!(u32) {
    intrinsic!(|_| value.constant().unwrap().as_u32())
}
