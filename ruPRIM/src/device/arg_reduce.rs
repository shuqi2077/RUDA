use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use super::{RudaPrimitiveError, check_type, scan_threads, segments};

pub struct RudaArgReduction<R: Runtime> {
    pub values: RudaTensor<R>,
    pub indices: RudaTensor<R>,
}

pub fn segmented_fixed<R: Runtime, T: TensorElement>(input: &RudaTensor<R>, num_segments: usize,
    segment_size: usize, largest: bool, threads: u32) -> Result<RudaArgReduction<R>, RudaPrimitiveError>
{
    check_type::<R, T>(input)?;
    configuration::<R, T>(input, threads)?;
    let (begins, ends) = segments::fixed_offsets(input, num_segments, segment_size)?;
    segmented::<R, T>(input, &begins, &ends, largest, threads)
}

#[ruda]
fn better<T: Numeric>(left: T, left_index: u64, right: T, right_index: u64, #[comptime] largest: bool) -> bool {
    let ordered = if largest { right > left } else { right < left };
    ordered || (right == left && right_index < left_index)
}

#[ruda]
fn block_arg<T: Numeric>(
    value: &mut T, index: &mut u64, scratch: &mut SharedMemory<T>, indices: &mut SharedMemory<u64>,
    valid: usize, #[comptime] threads: usize, #[comptime] largest: bool,
) {
    let lane = UNIT_POS as usize;
    if lane < valid { scratch[lane] = *value; indices[lane] = *index; }
    sync_ruda();
    let mut distance = 1usize;
    while distance < threads {
        if lane % (distance * 2) == 0 && lane + distance < valid {
            let next = lane + distance;
            if better::<T>(scratch[lane], indices[lane], scratch[next], indices[next], largest) {
                scratch[lane] = scratch[next];
                indices[lane] = indices[next];
            }
        }
        sync_ruda();
        distance *= 2;
    }
    *value = scratch[0];
    *index = indices[0];
    sync_ruda();
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn tile<T: Numeric>(
    input: &LinearView<T>, input_indices: &LinearView<u64>,
    output: &mut LinearView<T, ReadWrite>, output_indices: &mut LinearView<u64, ReadWrite>,
    #[comptime] first: bool, #[comptime] largest: bool, #[comptime] threads: usize,
) {
    let start = RUDA_POS as usize * threads;
    if start >= input.shape() { terminate!(); }
    let valid = min(threads, input.shape() - start);
    let position = start + UNIT_POS as usize;
    let mut value = T::from_int(0);
    let mut index = 0u64;
    if position < input.shape() {
        value = input[position];
        index = if first { position as u64 } else { input_indices[position] };
    }
    let mut scratch = SharedMemory::<T>::new(threads);
    let mut indices = SharedMemory::<u64>::new(threads);
    block_arg::<T>(&mut value, &mut index, &mut scratch, &mut indices, valid, threads, largest);
    if UNIT_POS == 0 { output[RUDA_POS as usize] = value; output_indices[RUDA_POS as usize] = index; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn empty_result<T: Numeric>(output: &mut LinearView<T, ReadWrite>, indices: &mut LinearView<u64, ReadWrite>, #[comptime] largest: bool) {
    if ABSOLUTE_POS == 0 {
        output[0] = if largest { T::min_value() } else { T::max_value() };
        indices[0] = 1;
    }
}

fn configuration<R: Runtime, T: TensorElement>(input: &RudaTensor<R>, threads: u32) -> Result<(), RudaPrimitiveError> {
    scan_threads::<R, T>(input, threads)?;
    if threads as usize * (core::mem::size_of::<T>() + 8) > input.client.properties().hardware.max_shared_memory_size {
        return Err(RudaPrimitiveError::Configuration("argument reduction scratch exceeds shared memory"));
    }
    Ok(())
}

fn allocate<R: Runtime>(input: &RudaTensor<R>, count: usize) -> RudaArgReduction<R> {
    let buffer = |dtype| empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([count]), dtype);
    RudaArgReduction { values: buffer(input.dtype), indices: buffer(DType::U64) }
}

/// Argument min/max, breaking equal-value ties by lowest input index.
/// Empty input yields index 1 and the corresponding finite type-limit sentinel.
pub fn reduce<R: Runtime, T: TensorElement>(
    input: &RudaTensor<R>, largest: bool, threads: u32,
) -> Result<RudaArgReduction<R>, RudaPrimitiveError> {
    check_type::<R, T>(input)?;
    configuration::<R, T>(input, threads)?;
    let count = input.meta.num_elements();
    if count == 0 {
        let output = allocate(input, 1);
        unsafe {
            empty_result::launch_unchecked::<T, R>(&input.client, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1),
                address_type!((output.values), (output.indices)), output.values.clone().into_linear_view(), output.indices.clone().into_linear_view(), largest);
        }
        return Ok(output);
    }
    let mut source = input.clone();
    let mut source_indices = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([0]), DType::U64);
    let mut first = true;
    let dim = RudaDim::new_1d(threads);
    loop {
        let count = source.meta.num_elements();
        let partials = count.div_ceil(threads as usize);
        let output = allocate(input, partials);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            tile::launch_unchecked::<T, R>(
                &input.client, grid, dim, address_type!(source, source_indices, (output.values), (output.indices)),
                source.into_linear_view(), source_indices.into_linear_view(), output.values.clone().into_linear_view(),
                output.indices.clone().into_linear_view(), first, largest, threads as usize,
            );
        }
        if partials == 1 { return Ok(output); }
        source = output.values;
        source_indices = output.indices;
        first = false;
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn segmented_kernel<T: Numeric>(
    input: &LinearView<T>, begins: &LinearView<u64>, ends: &LinearView<u64>,
    output: &mut LinearView<T, ReadWrite>, output_indices: &mut LinearView<u64, ReadWrite>,
    #[comptime] largest: bool, #[comptime] threads: usize,
) {
    let segment = RUDA_POS as usize;
    if segment >= begins.shape() { terminate!(); }
    let begin = begins[segment] as usize;
    let end = ends[segment] as usize;
    let mut best = if largest { T::min_value() } else { T::max_value() };
    let mut best_index = 1u64;
    let mut any = false;
    let mut start = begin;
    let mut scratch = SharedMemory::<T>::new(threads);
    let mut indices = SharedMemory::<u64>::new(threads);
    while start < end {
        let valid = min(threads, end - start);
        let mut value = T::from_int(0);
        let mut index = 0u64;
        if (UNIT_POS as usize) < valid {
            value = input[start + UNIT_POS as usize];
            index = (start - begin + UNIT_POS as usize) as u64;
        }
        block_arg::<T>(&mut value, &mut index, &mut scratch, &mut indices, valid, threads, largest);
        if !any || better::<T>(best, best_index, value, index, largest) { best = value; best_index = index; }
        any = true;
        start += valid;
    }
    if UNIT_POS == 0 { output[segment] = best; output_indices[segment] = best_index; }
}

/// Independent argument reductions. Indices are relative to each segment.
pub fn segmented<R: Runtime, T: TensorElement>(
    input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>, largest: bool, threads: u32,
) -> Result<RudaArgReduction<R>, RudaPrimitiveError> {
    check_type::<R, T>(input)?;
    configuration::<R, T>(input, threads)?;
    let count = segments::check_offsets(input, begins, ends)?;
    let output = allocate(input, count);
    if count > 0 {
        let work = count.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("segment launch size overflow"))?;
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
        unsafe {
            segmented_kernel::launch_unchecked::<T, R>(
                &input.client, grid, dim, address_type!(input, begins, ends, (output.values), (output.indices)),
                input.clone().into_linear_view(), begins.clone().into_linear_view(), ends.clone().into_linear_view(),
                output.values.clone().into_linear_view(), output.indices.clone().into_linear_view(), largest, threads as usize,
            );
        }
    }
    Ok(output)
}
