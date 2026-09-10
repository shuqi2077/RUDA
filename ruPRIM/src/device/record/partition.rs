use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaSum, RudaSumLaunch};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::device::{RudaPrimitiveError, scan, select::{RudaPredicate, RudaPredicateExpand}};
use super::{RudaRecordBuffer, RudaRecordBytes};

pub struct RudaRecordTwoWayPartition<R: Runtime, T: RudaRecord> {
    pub selected: RudaRecordBuffer<R, T>,
    pub rejected: RudaRecordBuffer<R, T>,
    pub counts: RudaTensor<R>,
}

pub struct RudaRecordPartition<R: Runtime, T: RudaRecord> {
    pub first: RudaRecordBuffer<R, T>,
    pub second: RudaRecordBuffer<R, T>,
    pub remaining: RudaRecordBuffer<R, T>,
    pub counts: RudaTensor<R>,
}

fn counters<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, count: usize) -> RudaTensor<R> {
    empty_device_dtype(input.bytes.client.clone(), input.bytes.device.clone(), Shape::new([count]), DType::U64)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn split<T: RudaRecord>(input: &RudaRecordBytes, selected_count: &LinearView<u64>,
    selected: &mut RudaRecordBytes, rejected: &mut RudaRecordBytes, counts: &mut LinearView<u64, ReadWrite>, count: usize,
) {
    let index = ABSOLUTE_POS;
    let accepted = selected_count[0] as usize;
    if index == 0 { counts[0] = accepted as u64; counts[1] = (count - accepted) as u64; }
    if index < accepted { <RudaRecordBytes as RudaWrite<T>>::write(selected, index, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
    if index < count - accepted { <RudaRecordBytes as RudaWrite<T>>::write(rejected, index, <RudaRecordBytes as RudaRead<T>>::read(input, count - 1 - index)); }
}

/// Two separate stable output sequences, with [selected, rejected] U64 counts.
pub fn two_way<R, T, P>(input: &RudaRecordBuffer<R, T>, predicate: P::RuntimeArg<R>, threads: u32)
    -> Result<RudaRecordTwoWayPartition<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, P: RudaPredicate<T> + LaunchArg,
{
    let partition = super::select::select_if::<R, T, P>(input, predicate, true, threads)?;
    let output = RudaRecordTwoWayPartition { selected: input.empty(input.len())?, rejected: input.empty(input.len())?,
        counts: counters(input, 2) };
    let dim = RudaDim::new(input.client().properties(), input.len().max(1));
    let grid = calculate_ruda_count_elemwise(input.client(), input.len().max(1), dim);
    unsafe { split::launch_unchecked::<T, R>(input.client(), grid, dim, partition.values.view(), partition.count.into_linear_view(),
        output.selected.view(), output.rejected.view(), output.counts.clone().into_linear_view(), input.len()); }
    Ok(output)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn classify<T: RudaRecord, P: RudaPredicate<T> + LaunchArg, Q: RudaPredicate<T> + LaunchArg>(
    input: &RudaRecordBytes, first: &mut LinearView<u64, ReadWrite>, second: &mut LinearView<u64, ReadWrite>,
    first_predicate: &P, second_predicate: &Q, count: usize,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let value = <RudaRecordBytes as RudaRead<T>>::read(input, index);
        let accepted = first_predicate.test(value);
        let mut accepted_second = false;
        if !accepted { accepted_second = second_predicate.test(value); }
        first[index] = u64::cast_from(accepted);
        second[index] = u64::cast_from(accepted_second);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn scatter<T: RudaRecord>(
    input: &RudaRecordBytes, first: &LinearView<u64>, second: &LinearView<u64>,
    first_prefix: &LinearView<u64>, second_prefix: &LinearView<u64>,
    first_output: &mut RudaRecordBytes, second_output: &mut RudaRecordBytes, remaining: &mut RudaRecordBytes,
    counts: &mut LinearView<u64, ReadWrite>, count: usize,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut a = 0u64;
        let mut b = 0u64;
        if count > 0 { a = first_prefix[count - 1]; b = second_prefix[count - 1]; }
        counts[0] = a;
        counts[1] = b;
        counts[2] = count as u64 - a - b;
    }
    if index < count {
        let a = first_prefix[index] as usize;
        let b = second_prefix[index] as usize;
        let value = <RudaRecordBytes as RudaRead<T>>::read(input, index);
        if first[index] != 0 { <RudaRecordBytes as RudaWrite<T>>::write(first_output, a - 1, value); }
        else if second[index] != 0 { <RudaRecordBytes as RudaWrite<T>>::write(second_output, b - 1, value); }
        else { <RudaRecordBytes as RudaWrite<T>>::write(remaining, index - a - b, value); }
    }
}

/// First predicate takes priority over the second. Each output is stable;
/// counts are [first, second, remaining] U64 device values.
pub fn three_way<R, T, P, Q>(input: &RudaRecordBuffer<R, T>, first_predicate: P::RuntimeArg<R>,
    second_predicate: Q::RuntimeArg<R>, threads: u32) -> Result<RudaRecordPartition<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, P: RudaPredicate<T> + LaunchArg, Q: RudaPredicate<T> + LaunchArg,
{
    let first = counters(input, input.len());
    let second = counters(input, input.len());
    let dim = RudaDim::new(input.client().properties(), input.len().max(1));
    if !input.is_empty() {
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe { classify::launch_unchecked::<T, P, Q, R>(input.client(), grid, dim, input.view(),
            first.clone().into_linear_view(), second.clone().into_linear_view(), first_predicate, second_predicate, input.len()); }
    }
    let first_prefix = scan::inclusive_scan::<R, u64, RudaSum>(&first, RudaSumLaunch::new(), threads)?;
    let second_prefix = scan::inclusive_scan::<R, u64, RudaSum>(&second, RudaSumLaunch::new(), threads)?;
    let output = RudaRecordPartition { first: input.empty(input.len())?, second: input.empty(input.len())?,
        remaining: input.empty(input.len())?, counts: counters(input, 3) };
    let grid = calculate_ruda_count_elemwise(input.client(), input.len().max(1), dim);
    unsafe { scatter::launch_unchecked::<T, R>(input.client(), grid, dim, input.view(), first.into_linear_view(),
        second.into_linear_view(), first_prefix.into_linear_view(), second_prefix.into_linear_view(),
        output.first.view(), output.second.view(), output.remaining.view(), output.counts.clone().into_linear_view(), input.len()); }
    Ok(output)
}
