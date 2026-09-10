use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::RudaTensor;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand, RudaRecordArray, RudaRecordShared};
use crate::device::{RudaPrimitiveError, check_type};
use super::{RudaRecordBuffer, RudaRecordBytes, configuration};

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn reduce_kernel<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(
    input: &RudaRecordBytes, begins: &LinearView<u64>, ends: &LinearView<u64>, initial: &RudaRecordBytes,
    output: &mut RudaRecordBytes, op: &O, #[comptime] threads: usize,
) {
    let segment = RUDA_POS as usize;
    if segment >= begins.shape() { terminate!(); }
    let mut start = begins[segment] as usize;
    let end = ends[segment] as usize;
    let mut carry = <RudaRecordBytes as RudaRead<T>>::read(initial, 0);
    let mut local = RudaRecordArray::<T>::new(1usize);
    let mut scratch = RudaRecordShared::<T>::new(threads);
    while start < end {
        let valid = min(threads, end - start);
        if (UNIT_POS as usize) < valid { local.write(0, <RudaRecordBytes as RudaRead<T>>::read(input, start + UNIT_POS as usize)); }
        let aggregate = crate::block::record::reduce::<T, O>(&local, &mut scratch, op, valid, threads, 1usize);
        carry = op.combine(carry, aggregate);
        start += valid;
    }
    if UNIT_POS == 0 { <RudaRecordBytes as RudaWrite<T>>::write(output, segment, carry); }
}

fn check<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    initial: &RudaRecordBuffer<R, T>, threads: u32) -> Result<usize, RudaPrimitiveError>
{
    configuration(input, threads)?;
    check_type::<R, u64>(begins)?;
    check_type::<R, u64>(ends)?;
    if begins.meta.num_elements() != ends.meta.num_elements() || initial.len() != 1 { return Err(RudaPrimitiveError::Length); }
    if input.bytes.device.to_id() != begins.device.to_id() || input.bytes.device.to_id() != ends.device.to_id() || input.bytes.device.to_id() != initial.bytes.device.to_id() {
        return Err(RudaPrimitiveError::Device);
    }
    Ok(begins.meta.num_elements())
}

/// Independent record reductions, preserving operand order. Each nonempty
/// range must lie within input; overlapping input ranges are allowed.
pub fn reduce<R, T, O>(input: &RudaRecordBuffer<R, T>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    initial: &RudaRecordBuffer<R, T>, op: O::RuntimeArg<R>, threads: u32) -> Result<RudaRecordBuffer<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg,
{
    let count = check(input, begins, ends, initial, threads)?;
    let output = input.empty(count)?;
    if count > 0 {
        let work = count.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("segment launch size overflow"))?;
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(input.client(), work, dim);
        unsafe {
            reduce_kernel::launch_unchecked::<T, O, R>(input.client(), grid, dim, input.view(), begins.clone().into_linear_view(),
                ends.clone().into_linear_view(), initial.view(), output.view(), op, threads as usize);
        }
    }
    Ok(output)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn scan_kernel<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(
    input: &RudaRecordBytes, begins: &LinearView<u64>, ends: &LinearView<u64>, output_begins: &LinearView<u64>,
    initial: &RudaRecordBytes, output: &mut RudaRecordBytes, op: &O,
    #[comptime] seeded: bool, #[comptime] exclusive: bool, #[comptime] threads: usize,
) {
    let segment = RUDA_POS as usize;
    if segment >= begins.shape() { terminate!(); }
    let begin = begins[segment] as usize;
    let end = ends[segment] as usize;
    let target = output_begins[segment] as usize;
    let mut carry = RudaRecordArray::<T>::new(1usize);
    if seeded { carry.write(0, <RudaRecordBytes as RudaRead<T>>::read(initial, 0)); }
    let mut has_carry = false;
    if seeded { has_carry = true; }
    let mut start = begin;
    let lane = UNIT_POS as usize;
    let mut local = RudaRecordArray::<T>::new(1usize);
    let mut result = RudaRecordArray::<T>::new(1usize);
    let mut scratch = RudaRecordShared::<T>::new(threads);
    while start < end {
        let valid = min(threads, end - start);
        if lane < valid { local.write(0, <RudaRecordBytes as RudaRead<T>>::read(input, start + lane)); }
        crate::block::record::inclusive_scan::<T, O>(&local, &mut result, &mut scratch, op, valid, threads, 1usize);
        if lane < valid {
            let mut value = result.read(0);
            if exclusive {
                value = carry.read(0);
                if lane > 0 { value = op.combine(carry.read(0), scratch.read(lane - 1)); }
            } else if has_carry { value = op.combine(carry.read(0), value); }
            <RudaRecordBytes as RudaWrite<T>>::write(output, target + start - begin + lane, value);
        }
        let aggregate = scratch.read(valid - 1);
        if has_carry { carry.write(0, op.combine(carry.read(0), aggregate)); }
        else { carry.write(0, aggregate); }
        has_carry = true;
        sync_ruda();
        start += valid;
    }
}

/// Scan disjoint output ranges. Exact in-place ranges are supported; other
/// input/output overlap is not. Initial is absent for unseeded inclusive scans.
pub fn scan_into<R, T, O>(input: &RudaRecordBuffer<R, T>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    output_begins: &RudaTensor<R>, output: &RudaRecordBuffer<R, T>, initial: Option<&RudaRecordBuffer<R, T>>,
    op: O::RuntimeArg<R>, exclusive: bool, threads: u32) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg,
{
    configuration(input, threads)?;
    check_type::<R, u64>(begins)?;
    check_type::<R, u64>(ends)?;
    let count = begins.meta.num_elements();
    if count != ends.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if begins.device.to_id() != input.bytes.device.to_id() || ends.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if let Some(initial) = initial {
        if initial.len() != 1 { return Err(RudaPrimitiveError::Length); }
        if initial.bytes.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    let seeded = initial.is_some();
    check_type::<R, u64>(output_begins)?;
    if output_begins.meta.num_elements() != count { return Err(RudaPrimitiveError::Length); }
    if output.bytes.device.to_id() != input.bytes.device.to_id() || output_begins.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if exclusive && !seeded { return Err(RudaPrimitiveError::Configuration("exclusive record scan requires an initial value")); }
    if count > 0 {
        let work = count.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("segment launch size overflow"))?;
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(input.client(), work, dim);
        unsafe {
            scan_kernel::launch_unchecked::<T, O, R>(input.client(), grid, dim, input.view(), begins.clone().into_linear_view(),
                ends.clone().into_linear_view(), output_begins.clone().into_linear_view(), initial.unwrap_or(input).view(), output.view(),
                op, seeded, exclusive, threads as usize);
        }
    }
    Ok(())
}
