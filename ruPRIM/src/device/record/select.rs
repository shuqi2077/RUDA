use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaKeyEqual, RudaKeyEqualExpand, RudaSum, RudaSumLaunch, RudaMinimum, RudaMinimumLaunch};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::device::{RudaPrimitiveError, scan, reduce, select::{RudaPredicate, RudaPredicateExpand}};
use super::{RudaRecordBuffer, RudaRecordBytes};

pub struct RudaRecordSelection<R: Runtime, T: RudaRecord> {
    pub values: RudaRecordBuffer<R, T>,
    pub count: RudaTensor<R>,
}

fn allocate<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, count: usize) -> RudaTensor<R> {
    empty_device_dtype(input.bytes.client.clone(), input.bytes.device.clone(), Shape::new([count]), DType::U64)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn normalize(input: &LinearView<u64>, output: &mut LinearView<u64, ReadWrite>) {
    if ABSOLUTE_POS < input.shape() { output[ABSOLUTE_POS] = u64::cast_from(input[ABSOLUTE_POS] != 0); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn predicate_flags<T: RudaRecord, P: RudaPredicate<T> + LaunchArg>(
    input: &RudaRecordBytes, flags: &mut LinearView<u64, ReadWrite>, predicate: &P, count: usize,
    #[comptime] indices: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let accepted = predicate.test(<RudaRecordBytes as RudaRead<T>>::read(input, index));
        if indices { flags[index] = if accepted { index as u64 } else { count as u64 }; }
        else { flags[index] = u64::cast_from(accepted); }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn head_flags<T: RudaRecord, E: RudaKeyEqual<T> + LaunchArg>(
    input: &RudaRecordBytes, flags: &mut LinearView<u64, ReadWrite>, equal: &E, count: usize,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut head = true;
        if index > 0 {
            head = !equal.equal(<RudaRecordBytes as RudaRead<T>>::read(input, index - 1),
                <RudaRecordBytes as RudaRead<T>>::read(input, index));
        }
        flags[index] = u64::cast_from(head);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn scatter<T: RudaRecord>(input: &RudaRecordBytes, flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output: &mut RudaRecordBytes, output_count: &mut LinearView<u64, ReadWrite>, count: usize, #[comptime] partition: bool,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut total = 0u64;
        if count > 0 { total = prefixes[count - 1]; }
        output_count[0] = total;
    }
    if index < count {
        let prefix = prefixes[index] as usize;
        if flags[index] != 0 { <RudaRecordBytes as RudaWrite<T>>::write(output, prefix - 1, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
        else if partition { <RudaRecordBytes as RudaWrite<T>>::write(output, count - 1 - (index - prefix), <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
    }
}

/// Flagged selection and single-output partition of records. Flags are
/// U64 device values, with nonzero meaning selected. Selected output is stable; partition
/// places the rejected items after it in reverse input order.
pub fn flagged<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, flags: &RudaTensor<R>,
    partition: bool, threads: u32) -> Result<RudaRecordSelection<R, T>, RudaPrimitiveError>
{
    crate::device::check_type::<R, u64>(flags)?;
    if flags.meta.num_elements() != input.len() { return Err(RudaPrimitiveError::Length); }
    if flags.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let normalized = allocate(input, input.len());
    if !input.is_empty() {
        let dim = RudaDim::new(input.client().properties(), input.len());
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe {
            normalize::launch_unchecked::<R>(input.client(), grid, dim, flags.clone().into_linear_view(), normalized.clone().into_linear_view());
        }
    }
    compact(input, &normalized, partition, threads)
}

fn compact<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, normalized: &RudaTensor<R>,
    partition: bool, threads: u32) -> Result<RudaRecordSelection<R, T>, RudaPrimitiveError>
{
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(normalized, RudaSumLaunch::new(), threads)?;
    let values = input.empty(input.len())?;
    let count = allocate(input, 1);
    let work = input.len().max(1);
    let dim = RudaDim::new(input.client().properties(), work);
    let grid = calculate_ruda_count_elemwise(input.client(), work, dim);
    unsafe {
        scatter::launch_unchecked::<T, R>(input.client(), grid, dim, input.view(), normalized.clone().into_linear_view(), prefixes.into_linear_view(),
            values.view(), count.clone().into_linear_view(), input.len(), partition);
    }
    Ok(RudaRecordSelection { values, count })
}

pub fn select_if<R, T, P>(input: &RudaRecordBuffer<R, T>, predicate: P::RuntimeArg<R>, partition: bool, threads: u32)
    -> Result<RudaRecordSelection<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, P: RudaPredicate<T> + LaunchArg,
{
    let flags = allocate(input, input.len());
    if !input.is_empty() {
        let dim = RudaDim::new(input.client().properties(), input.len());
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe {
            predicate_flags::launch_unchecked::<T, P, R>(input.client(), grid, dim, input.view(), flags.clone().into_linear_view(), predicate, input.len(), false);
        }
    }
    compact(input, &flags, partition, threads)
}

pub fn unique<R, T, E>(input: &RudaRecordBuffer<R, T>, equal: E::RuntimeArg<R>, threads: u32)
    -> Result<RudaRecordSelection<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, E: RudaKeyEqual<T> + LaunchArg,
{
    let flags = allocate(input, input.len());
    if !input.is_empty() {
        let dim = RudaDim::new(input.client().properties(), input.len());
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe {
            head_flags::launch_unchecked::<T, E, R>(input.client(), grid, dim, input.view(), flags.clone().into_linear_view(), equal, input.len());
        }
    }
    compact(input, &flags, false, threads)
}

pub fn find_if<R, T, P>(input: &RudaRecordBuffer<R, T>, predicate: P::RuntimeArg<R>, threads: u32)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, P: RudaPredicate<T> + LaunchArg,
{
    let indices = allocate(input, input.len());
    if !input.is_empty() {
        let dim = RudaDim::new(input.client().properties(), input.len());
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe {
            predicate_flags::launch_unchecked::<T, P, R>(input.client(), grid, dim, input.view(), indices.clone().into_linear_view(), predicate, input.len(), true);
        }
    }
    reduce::reduce::<R, u64, RudaMinimum>(&indices, input.len() as u64, RudaMinimumLaunch::new(), threads)
}

pub struct RudaRecordPairSelection<R: Runtime, K: RudaRecord, V: RudaRecord> {
    pub keys: RudaRecordBuffer<R, K>,
    pub values: RudaRecordBuffer<R, V>,
    pub count: RudaTensor<R>,
}

/// Retain the first key and its associated value in each adjacent equal run.
pub fn unique_by_key<R, K, V, E>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    equal: E::RuntimeArg<R>, threads: u32) -> Result<RudaRecordPairSelection<R, K, V>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, E: RudaKeyEqual<K> + LaunchArg,
{
    if keys.len() != values.len() { return Err(RudaPrimitiveError::Length); }
    if keys.bytes.device.to_id() != values.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let flags = allocate(keys, keys.len());
    let dim = RudaDim::new(keys.client().properties(), keys.len().max(1));
    if !keys.is_empty() {
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe { head_flags::launch_unchecked::<K, E, R>(keys.client(), grid, dim,
            keys.view(), flags.clone().into_linear_view(), equal, keys.len()); }
    }
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
    let output_keys = keys.empty(keys.len())?;
    let output_values = values.empty(values.len())?;
    let count = allocate(keys, 1);
    let grid = calculate_ruda_count_elemwise(keys.client(), keys.len().max(1), dim);
    unsafe {
        scatter::launch_unchecked::<K, R>(keys.client(), grid.clone(), dim, keys.view(), flags.clone().into_linear_view(),
            prefixes.clone().into_linear_view(), output_keys.view(), count.clone().into_linear_view(), keys.len(), false);
        scatter::launch_unchecked::<V, R>(keys.client(), grid, dim, values.view(), flags.into_linear_view(),
            prefixes.into_linear_view(), output_values.view(), count.clone().into_linear_view(), values.len(), false);
    }
    Ok(RudaRecordPairSelection { keys: output_keys, values: output_values, count })
}

/// Select records using a predicate on the separate record-valued flag sequence.
pub fn flagged_if<R, T, F, P>(input: &RudaRecordBuffer<R, T>, flags: &RudaRecordBuffer<R, F>,
    predicate: P::RuntimeArg<R>, partition: bool, threads: u32) -> Result<RudaRecordSelection<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, F: RudaRecord, P: RudaPredicate<F> + LaunchArg,
{
    if input.len() != flags.len() { return Err(RudaPrimitiveError::Length); }
    if input.bytes.device.to_id() != flags.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let normalized = allocate(input, input.len());
    if !input.is_empty() {
        let dim = RudaDim::new(input.client().properties(), input.len());
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe { predicate_flags::launch_unchecked::<F, P, R>(input.client(), grid, dim, flags.view(),
            normalized.clone().into_linear_view(), predicate, input.len(), false); }
    }
    compact(input, &normalized, partition, threads)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn write_selection<T: RudaRecord>(input: &RudaRecordBytes, count: &LinearView<u64>, output: &mut RudaRecordBytes, capacity: usize) {
    let index = ABSOLUTE_POS;
    if index < capacity {
        if (index as u64) < count[0] { <RudaRecordBytes as RudaWrite<T>>::write(output, index, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
    }
}

/// Copy the valid prefix, preserving the output tail. Output capacity must be
/// at least the selection's allocated capacity; count is not read back.
pub fn write_into<R: Runtime, T: RudaRecord>(selection: RudaRecordSelection<R, T>, output: &RudaRecordBuffer<R, T>)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
{
    crate::device::check_type::<R, u64>(&selection.count)?;
    if output.len() < selection.values.len() || selection.count.meta.num_elements() != 1 { return Err(RudaPrimitiveError::Length); }
    if output.bytes.device.to_id() != selection.values.bytes.device.to_id() || output.bytes.device.to_id() != selection.count.device.to_id() {
        return Err(RudaPrimitiveError::Device);
    }
    if !selection.values.is_empty() {
        let dim = RudaDim::new(output.client().properties(), selection.values.len());
        let grid = calculate_ruda_count_elemwise(output.client(), selection.values.len(), dim);
        unsafe { write_selection::launch_unchecked::<T, R>(output.client(), grid, dim, selection.values.view(),
            selection.count.clone().into_linear_view(), output.view(), selection.values.len()); }
    }
    Ok(selection.count)
}

pub fn select_if_in_place<R, T, P>(input: &RudaRecordBuffer<R, T>, predicate: P::RuntimeArg<R>, threads: u32)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, P: RudaPredicate<T> + LaunchArg,
{
    let result = select_if::<R, T, P>(input, predicate, false, threads)?;
    write_into(result, input)
}

pub fn flagged_in_place<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, flags: &RudaTensor<R>, threads: u32)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
{
    let result = flagged(input, flags, false, threads)?;
    write_into(result, input)
}

pub fn flagged_if_in_place<R, T, F, P>(input: &RudaRecordBuffer<R, T>, flags: &RudaRecordBuffer<R, F>,
    predicate: P::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, F: RudaRecord, P: RudaPredicate<F> + LaunchArg,
{
    let result = flagged_if::<R, T, F, P>(input, flags, predicate, false, threads)?;
    write_into(result, input)
}

pub fn unique_in_place<R, T, E>(input: &RudaRecordBuffer<R, T>, equal: E::RuntimeArg<R>, threads: u32)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, E: RudaKeyEqual<T> + LaunchArg,
{
    let result = unique::<R, T, E>(input, equal, threads)?;
    write_into(result, input)
}

pub fn unique_by_key_in_place<R, K, V, E>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    equal: E::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, E: RudaKeyEqual<K> + LaunchArg,
{
    let result = unique_by_key::<R, K, V, E>(keys, values, equal, threads)?;
    write_into(RudaRecordSelection { values: result.keys, count: result.count.clone() }, keys)?;
    write_into(RudaRecordSelection { values: result.values, count: result.count }, values)
}
