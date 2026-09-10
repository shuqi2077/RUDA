use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::Slice;
use ruda_core::tensor::TensorMetadata;
use ruda_core::tensor::Metadata;
use ruda_core::tensor::SliceOps;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::intrinsic;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use std::ops::Range;

/// Slice a jit tensor with a set of ranges
pub fn slice<R: Runtime>(tensor: RudaTensor<R>, indices: &[Range<usize>]) -> RudaTensor<R> {
    let mut dims = tensor.shape();
    for (dim, range) in dims.iter_mut().zip(indices) {
        *dim = range.end.saturating_sub(range.start);
    }
    if dims.contains(&0) {
        return empty_device_dtype(tensor.client, tensor.device, dims, tensor.dtype);
    }
    let mut offset_start = 0u64;
    let mut offset_end = 0u64;

    for i in 0..indices.len() {
        offset_start += (tensor.meta.strides()[i] * indices[i].start) as u64;
        offset_end += (tensor.meta.strides()[i] * (tensor.meta.shape()[i] - indices[i].end)) as u64;
    }

    let offset_start = offset_start * tensor.dtype.size() as u64;
    let offset_end = offset_end * tensor.dtype.size() as u64;

    let memory_offset_alignment = tensor.client.properties().memory.alignment;

    if offset_start.is_multiple_of(memory_offset_alignment)
        && offset_end.is_multiple_of(memory_offset_alignment)
    {
        RudaTensor::new(
            tensor.client.clone(),
            tensor
                .handle
                .clone()
                .offset_start(offset_start)
                .offset_end(offset_end),
            Metadata::new(dims, tensor.meta.strides.clone()),
            tensor.device.clone(),
            tensor.dtype,
        )
    } else {
        let output = empty_device_dtype(
            tensor.client.clone(),
            tensor.device.clone(),
            dims,
            tensor.dtype,
        );
        slice_on_output(tensor, output, indices)
    }
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn slice_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut LinearView<E, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    indices: Sequence<usize>,
    #[define(E)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![out_shape.len()];
    let mut offset_output = ABSOLUTE_POS;
    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        // Iterate in reverse to use divmod
        let dim = rank - i - 1;

        let range_start = indices[dim];
        let (rem, offset_local) = out_shape[dim].div_mod(offset_output);
        offset_output = rem;

        let offset_local = offset_local + range_start;

        offset_input += offset_local * input.stride(dim);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub fn slice_on_output<R: Runtime>(
    tensor: RudaTensor<R>,
    output: RudaTensor<R>,
    indices: &[Range<usize>],
) -> RudaTensor<R> {
    if output.meta.num_elements() == 0 {
        return output;
    }
    let ndims = tensor.meta.num_dims();
    let mut indices_sequence = SequenceArg::<R, usize>::new();

    for i in 0..ndims {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        indices_sequence.push(start);
    }

    let working_units = output.meta.num_elements();
    let ruda_dim = RudaDim::new(tensor.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, working_units, ruda_dim);
    let dtype = tensor.dtype;

    unsafe {
        slice_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, output),
            tensor.into_tensor_arg(),
            output.clone().into_linear_view(),
            shape_divmod(&output),
            indices_sequence,
            dtype.into(),
        )
    };

    output
}

/// Kernel for slicing with steps
#[ruda(launch_unchecked, address_type = "dynamic")]
fn slice_with_steps_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut LinearView<E, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    starts: Sequence<usize>,
    ends: Sequence<usize>,
    steps: Sequence<usize>,
    reversed: Sequence<usize>,
    #[define(E)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![out_shape.len()];
    let mut output_offset = ABSOLUTE_POS;
    let mut input_offset = 0;

    // Calculate the input offset based on output position and slice info
    #[unroll]
    for i in 0..rank {
        // Iterate in reverse to use divmod
        let dim = rank - i - 1;
        let start = starts[dim];
        let end = ends[dim];
        let step = steps[dim];

        let (rem, output_idx) = out_shape[dim].div_mod(output_offset);
        output_offset = rem;

        let input_idx = if reversed[dim] == 0 {
            // Forward stepping
            start + output_idx * step
        } else {
            // Backward stepping - start from end-1
            let end_minus_1 = end - 1;
            end_minus_1 - output_idx * step
        };

        input_offset += input_idx * input.stride(dim);
    }

    output[ABSOLUTE_POS] = input[input_offset];
}

/// Slice a tensor with steps
pub fn slice_with_steps<R: Runtime>(tensor: RudaTensor<R>, slices: &[Slice]) -> RudaTensor<R> {
    // Check if all steps are 1 - if so, use the optimized regular slice
    let all_steps_one = slices.iter().all(|info| info.step == 1);

    if all_steps_one {
        // Convert Slice to Range for step=1
        let simple_ranges: Vec<Range<usize>> = slices
            .iter()
            .enumerate()
            .map(|(i, slice)| slice.to_range(tensor.meta.shape()[i]))
            .collect();
        return slice(tensor, &simple_ranges);
    }

    // Calculate output shape
    let shape_output = tensor.shape().slice(slices).unwrap();

    // Create output tensor
    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output.clone(),
        tensor.dtype,
    );

    if shape_output.num_elements() == 0 {
        return output;
    }

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
    let working_units = shape_output.num_elements();
    let ruda_dim = RudaDim::new(tensor.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, working_units, ruda_dim);
    let dtype = tensor.dtype;

    let step_address = AddressType::from_len(
        slices.iter().map(|slice| slice.step.unsigned_abs()).max().unwrap_or(1),
    );
    unsafe {
        slice_with_steps_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(tensor, output).max(step_address),
            tensor.into_tensor_arg(),
            output.clone().into_linear_view(),
            shape_divmod(&output),
            starts,
            ends,
            steps,
            reversed,
            dtype.into(),
        );
    }

    output
}

/// This is annoying and we need to find a way to do this automatically at some point
#[allow(unused)]
#[ruda]
fn unwrap(value: u32) -> comptime_type!(u32) {
    intrinsic!(|_| value.constant().unwrap().as_u32())
}
