use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaSum, RudaSumLaunch};
use super::{RudaPrimitiveError, check_type, empty_like, scan, select::RudaPredicate};
use super::select::RudaPredicateExpand;

pub struct RudaPartition<R: Runtime> {
    pub first: RudaTensor<R>,
    pub second: RudaTensor<R>,
    pub remaining: RudaTensor<R>,
    /// Three U64 counts, in the same order as the output buffers.
    pub counts: RudaTensor<R>,
}

pub struct RudaTwoWayPartition<R: Runtime> {
    pub selected: RudaTensor<R>,
    pub rejected: RudaTensor<R>,
    /// Selected and rejected lengths, as two U64 device values.
    pub counts: RudaTensor<R>,
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn split_partition<T: Numeric>(
    input: &LinearView<T>, selected_count: &LinearView<u64>, selected: &mut LinearView<T, ReadWrite>,
    rejected: &mut LinearView<T, ReadWrite>, counts: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    let count = selected_count[0] as usize;
    if index == 0 { counts[0] = count as u64; counts[1] = (input.shape() - count) as u64; }
    if index < count { selected[index] = input[index]; }
    if index < input.shape() - count { rejected[index] = input[input.shape() - 1 - index]; }
}

/// Separate-output stable partition; the predicate is evaluated once per item.
pub fn two_way<R, T, P>(input: &RudaTensor<R>, predicate: P::RuntimeArg<R>, threads: u32)
    -> Result<RudaTwoWayPartition<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, P: RudaPredicate<T> + LaunchArg,
{
    let partition = super::select::select_if::<R, T, P>(input, predicate, threads, true)?;
    let output = RudaTwoWayPartition {
        selected: empty_like(input), rejected: empty_like(input),
        counts: empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([2]), DType::U64),
    };
    let work = input.meta.num_elements().max(1);
    let dim = RudaDim::new(input.client.properties(), work);
    let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
    unsafe {
        split_partition::launch_unchecked::<T, R>(&input.client, grid, dim,
            address_type!((partition.values), (partition.count), (output.selected), (output.rejected), (output.counts)),
            partition.values.into_linear_view(), partition.count.into_linear_view(), output.selected.clone().into_linear_view(),
            output.rejected.clone().into_linear_view(), output.counts.clone().into_linear_view());
    }
    Ok(output)
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn classify<T: Numeric, P: RudaPredicate<T> + LaunchArg, Q: RudaPredicate<T> + LaunchArg>(
    input: &LinearView<T>, first: &mut LinearView<u64, ReadWrite>, second: &mut LinearView<u64, ReadWrite>,
    first_predicate: &P, second_predicate: &Q,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let value = input[index];
        let accepted = first_predicate.test(value);
        let mut accepted_second = false;
        if !accepted { accepted_second = second_predicate.test(value); }
        first[index] = u64::cast_from(accepted);
        second[index] = u64::cast_from(accepted_second);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn scatter<T: Numeric>(
    input: &LinearView<T>, first: &LinearView<u64>, second: &LinearView<u64>,
    first_prefix: &LinearView<u64>, second_prefix: &LinearView<u64>,
    first_output: &mut LinearView<T, ReadWrite>, second_output: &mut LinearView<T, ReadWrite>,
    remaining: &mut LinearView<T, ReadWrite>, counts: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut first_count = 0u64;
        let mut second_count = 0u64;
        if input.shape() > 0 { first_count = first_prefix[input.shape() - 1]; second_count = second_prefix[input.shape() - 1]; }
        counts[0] = first_count;
        counts[1] = second_count;
        counts[2] = input.shape() as u64 - first_count - second_count;
    }
    if index < input.shape() {
        let a = first_prefix[index] as usize;
        let b = second_prefix[index] as usize;
        if first[index] != 0 { first_output[a - 1] = input[index]; }
        else if second[index] != 0 { second_output[b - 1] = input[index]; }
        else { remaining[index - a - b] = input[index]; }
    }
}

/// Stable three-way partition. First predicate takes priority, then the second;
/// items accepted by neither go to `remaining`. All counts remain on the device.
pub fn three_way<R, T, P, Q>(
    input: &RudaTensor<R>, first_predicate: P::RuntimeArg<R>, second_predicate: Q::RuntimeArg<R>, threads: u32,
) -> Result<RudaPartition<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, P: RudaPredicate<T> + LaunchArg, Q: RudaPredicate<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    let size = input.meta.num_elements();
    let flags = || empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([size]), DType::U64);
    let first = flags();
    let second = flags();
    let dim = RudaDim::new(input.client.properties(), size.max(1));
    if size > 0 {
        let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
        unsafe {
            classify::launch_unchecked::<T, P, Q, R>(&input.client, grid, dim, address_type!(input, first, second),
                input.clone().into_linear_view(), first.clone().into_linear_view(), second.clone().into_linear_view(), first_predicate, second_predicate);
        }
    }
    let first_prefix = scan::inclusive_scan::<R, u64, RudaSum>(&first, RudaSumLaunch::new(), threads)?;
    let second_prefix = scan::inclusive_scan::<R, u64, RudaSum>(&second, RudaSumLaunch::new(), threads)?;
    let output = RudaPartition {
        first: empty_like(input), second: empty_like(input), remaining: empty_like(input),
        counts: empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([3]), DType::U64),
    };
    let grid = calculate_ruda_count_elemwise(&input.client, size.max(1), dim);
    unsafe {
        scatter::launch_unchecked::<T, R>(
            &input.client, grid, dim, address_type!(input, first, second, first_prefix, second_prefix, (output.first), (output.second), (output.remaining), (output.counts)),
            input.clone().into_linear_view(), first.into_linear_view(), second.into_linear_view(), first_prefix.into_linear_view(), second_prefix.into_linear_view(),
            output.first.clone().into_linear_view(), output.second.clone().into_linear_view(), output.remaining.clone().into_linear_view(), output.counts.clone().into_linear_view(),
        );
    }
    Ok(output)
}
