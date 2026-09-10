use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaCompare, RudaCompareExpand, RudaSum, RudaSumLaunch};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::collective::decompose::RudaDecomposer;
use crate::device::RudaPrimitiveError;
use super::{RudaRecordBuffer, RudaRecordBytes};

#[ruda]
fn rank<K: RudaRecord, C: RudaCompare<K>>(keys: &RudaRecordBytes, compare: &C,
    index: usize, begin: usize, end: usize, run: usize,
) -> usize {
    let group = begin + (index - begin) / run / 2 * run * 2;
    let middle = group + min(run, end - group);
    let stop = middle + min(run, end - middle);
    let left = index < middle;
    let other_begin = if left { middle } else { group };
    let mut low = other_begin;
    let mut high = if left { stop } else { middle };
    let key = <RudaRecordBytes as RudaRead<K>>::read(keys, index);
    while low < high {
        let probe = low + (high - low) / 2;
        let other = <RudaRecordBytes as RudaRead<K>>::read(keys, probe);
        let before = if left { compare.before(other, key) } else { !compare.before(key, other) };
        if before { low = probe + 1; } else { high = probe; }
    }
    group + (index - if left { group } else { middle }) + low - other_begin
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn merge_pass<K: RudaRecord, V: RudaRecord, C: RudaCompare<K> + LaunchArg>(
    keys: &RudaRecordBytes, values: &RudaRecordBytes, begins: &LinearView<u64>, ends: &LinearView<u64>,
    output_keys: &mut RudaRecordBytes, output_values: &mut RudaRecordBytes,
    compare: &C, count: usize, run: usize, #[comptime] pairs: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut position = index;
        let end = ends[index] as usize;
        if end > 0 { position = rank::<K, C>(keys, compare, index, begins[index] as usize, end, run); }
        <RudaRecordBytes as RudaWrite<K>>::write(output_keys, position, <RudaRecordBytes as RudaRead<K>>::read(keys, index));
        if pairs { <RudaRecordBytes as RudaWrite<V>>::write(output_values, position, <RudaRecordBytes as RudaRead<V>>::read(values, index)); }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
pub(super) fn radix_pass<K: RudaRecord, V: RudaRecord>(keys: &RudaRecordBytes, values: &RudaRecordBytes,
    ends: &LinearView<u64>, flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output_keys: &mut RudaRecordBytes, output_values: &mut RudaRecordBytes, count: usize, #[comptime] pairs: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut position = index;
        let end = ends[index] as usize;
        if end > 0 {
            let ones = prefixes[end - 1] as usize;
            let before = (prefixes[index] - flags[index]) as usize;
            position = if flags[index] == 0 { index - before } else { end - ones + before };
        }
        <RudaRecordBytes as RudaWrite<K>>::write(output_keys, position, <RudaRecordBytes as RudaRead<K>>::read(keys, index));
        if pairs { <RudaRecordBytes as RudaWrite<V>>::write(output_values, position, <RudaRecordBytes as RudaRead<V>>::read(values, index)); }
    }
}

fn check<R: Runtime, K: RudaRecord, V: RudaRecord>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>, pairs: bool)
    -> Result<(), RudaPrimitiveError>
{
    if pairs && keys.len() != values.len() { return Err(RudaPrimitiveError::Length); }
    if keys.bytes.device.to_id() != values.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    Ok(())
}

fn merge_impl<R, K, V, C>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    begins: &RudaTensor<R>, ends: &RudaTensor<R>, compare: C::RuntimeArg<R>, pairs: bool, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check(keys, values, pairs)?;
    let map = crate::device::segmented_sort::make_map_len(&keys.bytes, keys.len(), begins, ends, threads)?;
    let mut source_keys = keys.clone();
    let mut source_values = values.clone();
    if keys.len() < 2 { return Ok((source_keys, source_values)); }
    let key_buffers = [keys.empty(keys.len())?, keys.empty(keys.len())?];
    let value_buffers = [values.empty(if pairs { values.len() } else { 0 })?, values.empty(if pairs { values.len() } else { 0 })?];
    let dim = RudaDim::new(keys.client().properties(), keys.len());
    let mut run = 1usize;
    let mut selector = 0usize;
    while run < keys.len() {
        let output_keys = &key_buffers[selector];
        let output_values = &value_buffers[selector];
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe {
            merge_pass::launch_unchecked::<K, V, C, R>(keys.client(), grid, dim, source_keys.view(), source_values.view(),
                map.begins.clone().into_linear_view(), map.ends.clone().into_linear_view(), output_keys.view(), output_values.view(),
                compare.clone(), keys.len(), run, pairs);
        }
        source_keys = output_keys.clone();
        source_values = output_values.clone();
        run = run.saturating_mul(2);
        selector ^= 1;
    }
    Ok((source_keys, source_values))
}

/// Stable sorting of arbitrary nonoverlapping record ranges. Descriptors may
/// be unordered and have gaps; positions outside every range are preserved.
pub fn sort_pairs<R, K, V, C>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    begins: &RudaTensor<R>, ends: &RudaTensor<R>, compare: C::RuntimeArg<R>, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    merge_impl::<R, K, V, C>(keys, values, begins, ends, compare, true, threads)
}

pub fn sort_keys<R, K, C>(keys: &RudaRecordBuffer<R, K>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    compare: C::RuntimeArg<R>, threads: u32) -> Result<RudaRecordBuffer<R, K>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    Ok(merge_impl::<R, K, K, C>(keys, keys, begins, ends, compare, false, threads)?.0)
}

fn radix_impl<R, K, V, D>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    begins: &RudaTensor<R>, ends: &RudaTensor<R>, decomposer: D::RuntimeArg<R>,
    begin_bit: usize, end_bit: usize, descending: bool, pairs: bool, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    check(keys, values, pairs)?;
    if begin_bit > end_bit || end_bit > D::BITS { return Err(RudaPrimitiveError::Configuration("invalid decomposed radix bit interval")); }
    let map = crate::device::segmented_sort::make_map_len(&keys.bytes, keys.len(), begins, ends, threads)?;
    let mut source_keys = keys.clone();
    let mut source_values = values.clone();
    if keys.len() < 2 || begin_bit == end_bit { return Ok((source_keys, source_values)); }
    let key_buffers = [keys.empty(keys.len())?, keys.empty(keys.len())?];
    let value_buffers = [values.empty(if pairs { values.len() } else { 0 })?, values.empty(if pairs { values.len() } else { 0 })?];
    let flags = empty_device_dtype(keys.bytes.client.clone(), keys.bytes.device.clone(), Shape::new([keys.len()]), DType::U64);
    let dim = RudaDim::new(keys.client().properties(), keys.len());
    for (pass, bit) in (begin_bit..end_bit).enumerate() {
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe {
            super::digit_flags::launch_unchecked::<K, D, R>(keys.client(), grid, dim, source_keys.view(), flags.clone().into_linear_view(),
                decomposer.clone(), keys.len(), bit, descending);
        }
        let prefixes = crate::device::segmented::scan_by_heads::<R, u64, RudaSum>(&flags, &map.heads, RudaSumLaunch::new(), None)?;
        let output_keys = &key_buffers[pass % 2];
        let output_values = &value_buffers[pass % 2];
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe {
            radix_pass::launch_unchecked::<K, V, R>(keys.client(), grid, dim, source_keys.view(), source_values.view(), map.ends.clone().into_linear_view(),
                flags.clone().into_linear_view(), prefixes.into_linear_view(), output_keys.view(), output_values.view(), keys.len(), pairs);
        }
        source_keys = output_keys.clone();
        source_values = output_values.clone();
    }
    Ok((source_keys, source_values))
}

pub fn radix_sort_pairs<R, K, V, D>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    begins: &RudaTensor<R>, ends: &RudaTensor<R>, decomposer: D::RuntimeArg<R>,
    begin_bit: usize, end_bit: usize, descending: bool, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    radix_impl::<R, K, V, D>(keys, values, begins, ends, decomposer, begin_bit, end_bit, descending, true, threads)
}

pub fn radix_sort_keys<R, K, D>(keys: &RudaRecordBuffer<R, K>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    decomposer: D::RuntimeArg<R>, begin_bit: usize, end_bit: usize, descending: bool, threads: u32)
    -> Result<RudaRecordBuffer<R, K>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    Ok(radix_impl::<R, K, K, D>(keys, keys, begins, ends, decomposer, begin_bit, end_bit, descending, false, threads)?.0)
}
