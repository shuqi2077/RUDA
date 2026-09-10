use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaCompare, RudaCompareExpand, RudaSum, RudaSumLaunch, radix::RudaRadixKey};
use super::{RudaPrimitiveError, check_type, empty_like, scan_threads, segments, segmented};

pub(super) struct SegmentMap<R: Runtime> {
    pub(super) begins: RudaTensor<R>,
    pub(super) ends: RudaTensor<R>,
    pub(super) heads: RudaTensor<R>,
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn clear_map(begins: &mut LinearView<u64, ReadWrite>, ends: &mut LinearView<u64, ReadWrite>, heads: &mut LinearView<u32, ReadWrite>) {
    if ABSOLUTE_POS < begins.shape() {
        begins[ABSOLUTE_POS] = 0;
        ends[ABSOLUTE_POS] = 0;
        heads[ABSOLUTE_POS] = 0;
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn fill_map(
    begins: &LinearView<u64>, ends: &LinearView<u64>,
    map_begins: &mut LinearView<u64, ReadWrite>, map_ends: &mut LinearView<u64, ReadWrite>,
    heads: &mut LinearView<u32, ReadWrite>, #[comptime] threads: usize,
) {
    let segment = RUDA_POS as usize;
    if segment < begins.shape() {
        let begin = begins[segment] as usize;
        let end = ends[segment] as usize;
        let mut index = begin + UNIT_POS as usize;
        while index < end {
            map_begins[index] = begin as u64;
            map_ends[index] = end as u64;
            heads[index] = u32::cast_from(index == begin);
            index += threads;
        }
    }
}

fn make_map<R: Runtime>(
    input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>, threads: u32,
) -> Result<SegmentMap<R>, RudaPrimitiveError> {
    make_map_len(input, input.meta.num_elements(), begins, ends, threads)
}

pub(super) fn make_map_len<R: Runtime>(
    input: &RudaTensor<R>, count: usize, begins: &RudaTensor<R>, ends: &RudaTensor<R>, threads: u32,
) -> Result<SegmentMap<R>, RudaPrimitiveError> {
    let segments = segments::check_offsets(input, begins, ends)?;
    scan_threads::<R, u64>(input, threads)?;
    let allocate = |dtype| empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([count]), dtype);
    let map = SegmentMap { begins: allocate(DType::U64), ends: allocate(DType::U64), heads: allocate(DType::U32) };
    if count > 0 {
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            clear_map::launch_unchecked::<R>(
                &input.client, grid, dim, address_type!((map.begins), (map.ends), (map.heads)),
                map.begins.clone().into_linear_view(), map.ends.clone().into_linear_view(), map.heads.clone().into_linear_view(),
            );
        }
    }
    if segments > 0 {
        let work = segments.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("segment launch size overflow"))?;
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
        unsafe {
            fill_map::launch_unchecked::<R>(
                &input.client, grid, dim, address_type!(begins, ends, (map.begins), (map.ends), (map.heads)),
                begins.clone().into_linear_view(), ends.clone().into_linear_view(), map.begins.clone().into_linear_view(),
                map.ends.clone().into_linear_view(), map.heads.clone().into_linear_view(), threads as usize,
            );
        }
    }
    Ok(map)
}

#[ruda]
fn merge_rank<K: Numeric, C: RudaCompare<K>>(
    input: &LinearView<K>, compare: &C, index: usize, begin: usize, end: usize, run: usize,
) -> usize {
    let group = begin + ((index - begin) / run / 2) * run * 2;
    let middle = group + min(run, end - group);
    let stop = middle + min(run, end - middle);
    let left = index < middle;
    let own_begin = select(left, group, middle);
    let other_begin = select(left, middle, group);
    let mut low = other_begin;
    let mut high = select(left, stop, middle);
    let key = input[index];
    while low < high {
        let mid = low + (high - low) / 2;
        let advance = if left { compare.before(input[mid], key) } else { !compare.before(key, input[mid]) };
        if advance { low = mid + 1; } else { high = mid; }
    }
    group + (index - own_begin) + (low - other_begin)
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn merge_keys<K: Numeric, C: RudaCompare<K> + LaunchArg>(
    input: &LinearView<K>, begins: &LinearView<u64>, ends: &LinearView<u64>,
    output: &mut LinearView<K, ReadWrite>, compare: &C, run: usize,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let mut rank = index;
        let end = ends[index] as usize;
        if end > 0 { rank = merge_rank::<K, C>(input, compare, index, begins[index] as usize, end, run); }
        output[rank] = input[index];
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn merge_pairs<K: Numeric, V: Numeric, C: RudaCompare<K> + LaunchArg>(
    input: &LinearView<K>, values: &LinearView<V>, begins: &LinearView<u64>, ends: &LinearView<u64>,
    output: &mut LinearView<K, ReadWrite>, output_values: &mut LinearView<V, ReadWrite>, compare: &C, run: usize,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let mut rank = index;
        let end = ends[index] as usize;
        if end > 0 { rank = merge_rank::<K, C>(input, compare, index, begins[index] as usize, end, run); }
        output[rank] = input[index];
        output_values[rank] = values[index];
    }
}

/// Stable sort of disjoint U64-offset segments. Descriptors need not be ordered
/// or adjacent; positions outside the segments are preserved.
pub fn sort_keys<R, K, C>(
    input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>, compare: C::RuntimeArg<R>, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(input)?;
    let map = make_map(input, begins, ends, threads)?;
    let count = input.meta.num_elements();
    let mut source = input.clone();
    if count < 2 { return Ok(source); }
    let buffers = [empty_like(input), empty_like(input)];
    let dim = RudaDim::new(input.client.properties(), count);
    let mut run = 1usize;
    let mut pass = 0usize;
    while run < count {
        let output = buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            merge_keys::launch_unchecked::<K, C, R>(
                &input.client, grid, dim, address_type!(source, (map.begins), (map.ends), output),
                source.into_linear_view(), map.begins.clone().into_linear_view(), map.ends.clone().into_linear_view(),
                output.clone().into_linear_view(), compare.clone(), run,
            );
        }
        source = output;
        run = run.saturating_mul(2);
        pass += 1;
    }
    Ok(source)
}

pub fn sort_pairs<R, K, V, C>(
    input: &RudaTensor<R>, values: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    compare: C::RuntimeArg<R>, threads: u32,
) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError>
where R: Runtime, K: TensorElement, V: TensorElement, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(input)?;
    check_type::<R, V>(values)?;
    if input.meta.num_elements() != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let map = make_map(input, begins, ends, threads)?;
    let count = input.meta.num_elements();
    let mut source = input.clone();
    let mut source_values = values.clone();
    if count < 2 { return Ok((source, source_values)); }
    let buffers = [empty_like(input), empty_like(input)];
    let value_buffers = [empty_like(values), empty_like(values)];
    let dim = RudaDim::new(input.client.properties(), count);
    let mut run = 1usize;
    let mut pass = 0usize;
    while run < count {
        let output = buffers[pass % 2].clone();
        let output_values = value_buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            merge_pairs::launch_unchecked::<K, V, C, R>(
                &input.client, grid, dim, address_type!(source, source_values, (map.begins), (map.ends), output, output_values),
                source.into_linear_view(), source_values.into_linear_view(),
                map.begins.clone().into_linear_view(), map.ends.clone().into_linear_view(),
                output.clone().into_linear_view(), output_values.clone().into_linear_view(), compare.clone(), run,
            );
        }
        source = output;
        source_values = output_values;
        run = run.saturating_mul(2);
        pass += 1;
    }
    Ok((source, source_values))
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn radix_flags<K: RudaRadixKey>(
    input: &LinearView<K>, ends: &LinearView<u64>, flags: &mut LinearView<u64, ReadWrite>,
    #[comptime] bit: u32, #[comptime] descending: bool,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let mut flag = 0u64;
        if ends[index] > 0 {
            flag = (K::ordered_bits(input[index]) >> bit) & 1u64;
            if descending { flag ^= 1u64; }
        }
        flags[index] = flag;
    }
}

#[ruda]
fn radix_rank(
    begins: &LinearView<u64>, ends: &LinearView<u64>, flags: &LinearView<u64>, prefixes: &LinearView<u64>, index: usize,
) -> usize {
    let end = ends[index] as usize;
    let mut rank = index;
    if end > 0 {
        let ones = prefixes[end - 1] as usize;
        let before = (prefixes[index] - flags[index]) as usize;
        rank = if flags[index] == 0 { index - before } else { end - ones + before };
    }
    rank
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn radix_scatter_keys<K: Numeric>(
    input: &LinearView<K>, begins: &LinearView<u64>, ends: &LinearView<u64>, flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output: &mut LinearView<K, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() { output[radix_rank(begins, ends, flags, prefixes, index)] = input[index]; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn radix_scatter_pairs<K: Numeric, V: Numeric>(
    input: &LinearView<K>, values: &LinearView<V>, begins: &LinearView<u64>, ends: &LinearView<u64>,
    flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output: &mut LinearView<K, ReadWrite>, output_values: &mut LinearView<V, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let rank = radix_rank(begins, ends, flags, prefixes, index);
        output[rank] = input[index];
        output_values[rank] = values[index];
    }
}

fn prefixes<R: Runtime, K: TensorElement + RudaRadixKey>(
    input: &RudaTensor<R>, map: &SegmentMap<R>, flags: &RudaTensor<R>, bit: u32, descending: bool,
) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    let count = input.meta.num_elements();
    let dim = RudaDim::new(input.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
    unsafe {
        radix_flags::launch_unchecked::<K, R>(&input.client, grid, dim, address_type!(input, (map.ends), flags),
            input.clone().into_linear_view(), map.ends.clone().into_linear_view(), flags.clone().into_linear_view(), bit, descending);
    }
    segmented::scan_by_heads::<R, u64, RudaSum>(flags, &map.heads, RudaSumLaunch::new(), None)
}

/// Stable LSD radix sorting within disjoint segments. The bit interval applies
/// after signed/floating radix encoding; gaps retain their original values.
pub fn radix_sort_keys<R: Runtime, K: TensorElement + RudaRadixKey>(
    input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    check_type::<R, K>(input)?;
    super::radix::check_bits::<K>(begin_bit, end_bit)?;
    let map = make_map(input, begins, ends, threads)?;
    let count = input.meta.num_elements();
    let mut source = input.clone();
    if count < 2 || begin_bit == end_bit { return Ok(source); }
    let buffers = [empty_like(input), empty_like(input)];
    let flags = empty_like(&map.begins);
    let dim = RudaDim::new(input.client.properties(), count);
    for (pass, bit) in (begin_bit..end_bit).enumerate() {
        let scanned = prefixes::<R, K>(&source, &map, &flags, bit, descending)?;
        let output = buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            radix_scatter_keys::launch_unchecked::<K, R>(
                &input.client, grid, dim, address_type!(source, (map.begins), (map.ends), flags, scanned, output),
                source.into_linear_view(), map.begins.clone().into_linear_view(), map.ends.clone().into_linear_view(),
                flags.clone().into_linear_view(), scanned.into_linear_view(), output.clone().into_linear_view(),
            );
        }
        source = output;
    }
    Ok(source)
}

pub fn radix_sort_pairs<R: Runtime, K: TensorElement + RudaRadixKey, V: TensorElement>(
    input: &RudaTensor<R>, values: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError> {
    check_type::<R, K>(input)?;
    check_type::<R, V>(values)?;
    if input.meta.num_elements() != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    super::radix::check_bits::<K>(begin_bit, end_bit)?;
    let map = make_map(input, begins, ends, threads)?;
    let count = input.meta.num_elements();
    let mut source = input.clone();
    let mut source_values = values.clone();
    if count < 2 || begin_bit == end_bit { return Ok((source, source_values)); }
    let buffers = [empty_like(input), empty_like(input)];
    let value_buffers = [empty_like(values), empty_like(values)];
    let flags = empty_like(&map.begins);
    let dim = RudaDim::new(input.client.properties(), count);
    for (pass, bit) in (begin_bit..end_bit).enumerate() {
        let scanned = prefixes::<R, K>(&source, &map, &flags, bit, descending)?;
        let output = buffers[pass % 2].clone();
        let output_values = value_buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            radix_scatter_pairs::launch_unchecked::<K, V, R>(
                &input.client, grid, dim, address_type!(source, source_values, (map.begins), (map.ends), flags, scanned, output, output_values),
                source.into_linear_view(), source_values.into_linear_view(), map.begins.clone().into_linear_view(),
                map.ends.clone().into_linear_view(), flags.clone().into_linear_view(), scanned.into_linear_view(),
                output.clone().into_linear_view(), output_values.clone().into_linear_view(),
            );
        }
        source = output;
        source_values = output_values;
    }
    Ok((source, source_values))
}
