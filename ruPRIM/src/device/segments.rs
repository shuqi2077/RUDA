use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::Shape;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};
use super::{RudaPrimitiveError, check_type, scan_threads};

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn fixed_offsets_kernel(begins: &mut LinearView<u64, ReadWrite>, ends: &mut LinearView<u64, ReadWrite>, size: usize) {
    let index = ABSOLUTE_POS;
    if index < begins.shape() {
        begins[index] = (index * size) as u64;
        ends[index] = ((index + 1) * size) as u64;
    }
}

pub(crate) fn fixed_offsets<R: Runtime>(input: &RudaTensor<R>, count: usize, size: usize)
    -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError>
{
    let end = count.checked_mul(size).ok_or(RudaPrimitiveError::Configuration("fixed segment extent overflow"))?;
    if end > input.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    let allocate = || empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([count]), ruda_core::tensor::DType::U64);
    let begins = allocate();
    let ends = allocate();
    if count > 0 {
        let dim = RudaDim::new(input.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            fixed_offsets_kernel::launch_unchecked::<R>(&input.client, grid, dim,
                address_type!(input, begins, ends).max(AddressType::from_len(count)),
                begins.clone().into_linear_view(), ends.clone().into_linear_view(), size);
        }
    }
    Ok((begins, ends))
}

pub fn reduce_fixed<R, T, O>(input: &RudaTensor<R>, num_segments: usize, segment_size: usize,
    initial: T, op: O::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    scan_threads::<R, T>(input, threads)?;
    let (begins, ends) = fixed_offsets(input, num_segments, segment_size)?;
    reduce::<R, T, O>(input, &begins, &ends, initial, op, threads)
}

/// Offset arrays reside on the device. Each nonempty range must be within the
/// input; segment order and adjacency are unrestricted. End <= begin is empty.
pub(crate) fn check_offsets<R: Runtime>(
    input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
) -> Result<usize, RudaPrimitiveError> {
    check_type::<R, u64>(begins)?;
    check_type::<R, u64>(ends)?;
    if begins.meta.num_elements() != ends.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != begins.device.to_id() || input.device.to_id() != ends.device.to_id() { return Err(RudaPrimitiveError::Device); }
    Ok(begins.meta.num_elements())
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn reduce_kernel<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    input: &LinearView<T>, begins: &LinearView<u64>, ends: &LinearView<u64>,
    output: &mut LinearView<T, ReadWrite>, initial: InputScalar, op: &O,
    #[comptime] threads: usize,
) {
    let segment = RUDA_POS as usize;
    if segment >= begins.shape() { terminate!(); }
    let begin = begins[segment] as usize;
    let end = ends[segment] as usize;
    let mut accumulated = initial.get::<T>();
    let mut start = begin;
    let mut local = Array::<T>::new(1usize);
    let mut scratch = SharedMemory::<T>::new(threads);
    while start < end {
        let valid = min(threads, end - start);
        if (UNIT_POS as usize) < valid { local[0] = input[start + UNIT_POS as usize]; }
        let tile = crate::block::reduce::<T, O>(&local, &mut scratch, op, valid, threads, 1usize);
        accumulated = op.combine(accumulated, tile);
        start += valid;
    }
    if UNIT_POS == 0 { output[segment] = accumulated; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn scan_kernel<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    input: &LinearView<T>, begins: &LinearView<u64>, ends: &LinearView<u64>,
    output_begins: &LinearView<u64>, output: &mut LinearView<T, ReadWrite>,
    initial: InputScalar, op: &O, #[comptime] seeded: bool, #[comptime] exclusive: bool,
    #[comptime] threads: usize,
) {
    let segment = RUDA_POS as usize;
    if segment >= begins.shape() { terminate!(); }
    let begin = begins[segment] as usize;
    let end = ends[segment] as usize;
    let out_begin = output_begins[segment] as usize;
    let lane = UNIT_POS as usize;
    let mut start = begin;
    let mut carry = initial.get::<T>();
    let mut has_carry = false;
    if seeded { has_carry = true; }
    let mut local = Array::<T>::new(1usize);
    let mut scanned = Array::<T>::new(1usize);
    let mut scratch = SharedMemory::<T>::new(threads);
    while start < end {
        let valid = min(threads, end - start);
        if lane < valid { local[0] = input[start + lane]; }
        crate::block::inclusive_scan::<T, O>(&local, &mut scanned, &mut scratch, op, valid, threads, 1usize);
        if lane < valid {
            let mut value = scanned[0];
            if exclusive {
                value = carry;
                if lane > 0 { value = op.combine(carry, scratch[lane - 1]); }
            } else if has_carry {
                value = op.combine(carry, value);
            }
            output[out_begin + start - begin + lane] = value;
        }
        let aggregate = scratch[valid - 1];
        carry = if has_carry { op.combine(carry, aggregate) } else { aggregate };
        has_carry = true;
        sync_ruda();
        start += valid;
    }
}

/// Reduce arbitrary independent U64-offset ranges. Empty segments yield
/// `initial`; overlapping read-only input segments are permitted.
pub fn reduce<R, T, O>(
    input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    initial: T, op: O::RuntimeArg<R>, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    scan_threads::<R, T>(input, threads)?;
    let segments = check_offsets(input, begins, ends)?;
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([segments]), input.dtype);
    if segments > 0 {
        let work = segments.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("segment launch size overflow"))?;
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
        unsafe {
            reduce_kernel::launch_unchecked::<T, O, R>(
                &input.client, grid, dim, address_type!(input, begins, ends, output),
                input.clone().into_linear_view(), begins.clone().into_linear_view(), ends.clone().into_linear_view(),
                output.clone().into_linear_view(), InputScalar::new(initial, input.dtype), op, threads as usize,
            );
        }
    }
    Ok(output)
}

/// Scan independent U64-offset ranges into caller-provided output ranges.
/// Unwritten output positions are preserved. Output ranges must not overlap.
/// Exact in-place input/output with identical offsets is supported; other
/// input/output overlap is not. `initial` also supports seeded inclusive scans.
pub fn scan_into<R, T, O>(
    input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    output_begins: &RudaTensor<R>, output: &RudaTensor<R>, initial: Option<T>,
    exclusive: bool, op: O::RuntimeArg<R>, threads: u32,
) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    check_type::<R, T>(output)?;
    check_type::<R, u64>(output_begins)?;
    scan_threads::<R, T>(input, threads)?;
    let segments = check_offsets(input, begins, ends)?;
    if output_begins.meta.num_elements() != segments { return Err(RudaPrimitiveError::Length); }
    if output.device.to_id() != input.device.to_id() || output_begins.device.to_id() != input.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if exclusive && initial.is_none() { return Err(RudaPrimitiveError::Configuration("exclusive scan requires an initial value")); }
    if segments > 0 {
        let work = segments.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("segment launch size overflow"))?;
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
        let seeded = initial.is_some();
        let value = initial.unwrap_or_else(|| T::from_int(0));
        unsafe {
            scan_kernel::launch_unchecked::<T, O, R>(
                &input.client, grid, dim, address_type!(input, begins, ends, output_begins, output),
                input.clone().into_linear_view(), begins.clone().into_linear_view(), ends.clone().into_linear_view(),
                output_begins.clone().into_linear_view(), output.clone().into_linear_view(),
                InputScalar::new(value, input.dtype), op, seeded, exclusive, threads as usize,
            );
        }
    }
    Ok(())
}
