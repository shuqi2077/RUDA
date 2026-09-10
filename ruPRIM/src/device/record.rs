use ruda_core::device::Device;
use core::marker::PhantomData;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand, RudaCompare, RudaCompareExpand};
use crate::collective::record::{RudaRecord, RudaRecordArray, RudaRecordShared,
    RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand, RudaAddress, RudaAddressExpand, RudaReference};
use super::RudaPrimitiveError;
use crate::collective::decompose::{RudaDecomposer, RudaDecomposerExpand};
use crate::collective::{RudaSum, RudaSumLaunch};

pub mod select;
pub mod segments;
pub mod segmented_sort;
pub mod grouped;
pub mod run_length;
pub mod merge;
pub mod adjacent;
pub mod partition;
pub mod double_buffer;

#[derive(RudaType, RudaLaunch)]
pub struct RudaRecordBytes {
    pub data: LinearView<u8, ReadWrite>,
}

#[ruda]
impl<T: RudaRecord> RudaAddress<T> for RudaRecordBytes {
    fn reference(&self, index: usize) -> RudaReference<T> {
        let base = native_address(&self.data.to_linear_slice(), 0);
        let align = comptime![T::ALIGN as u64];
        RudaReference::<T>::new((base + align - 1) / align * align + index as u64 * comptime![T::SIZE as u64])
    }
}
#[ruda]
impl<T: RudaRecord> RudaRead<T> for RudaRecordBytes {
    fn read(&self, index: usize) -> T {
        let reference = <RudaRecordBytes as RudaAddress<T>>::reference(self, index);
        reference.read()
    }
}
#[ruda]
impl<T: RudaRecord> RudaWrite<T> for RudaRecordBytes {
    fn write(&mut self, index: usize, value: T) {
        let reference = <RudaRecordBytes as RudaAddress<T>>::reference(self, index);
        reference.write(value);
    }
}

/// Owned, aligned device storage for native-layout records. Use view() with
/// access::transform/copy to populate records without host synchronization.
pub struct RudaRecordBuffer<R: Runtime, T: RudaRecord> {
    bytes: RudaTensor<R>,
    count: usize,
    marker: PhantomData<T>,
}

impl<R: Runtime, T: RudaRecord> Clone for RudaRecordBuffer<R, T> {
    fn clone(&self) -> Self { Self { bytes: self.bytes.clone(), count: self.count, marker: PhantomData } }
}

impl<R: Runtime, T: RudaRecord> RudaRecordBuffer<R, T> {
    pub fn allocate(anchor: &RudaTensor<R>, count: usize) -> Result<Self, RudaPrimitiveError> {
        if T::SIZE == 0 || !T::ALIGN.is_power_of_two() {
            return Err(RudaPrimitiveError::Configuration("invalid record layout"));
        }
        let size = count.checked_mul(T::SIZE).and_then(|n| n.checked_add(T::ALIGN))
            .ok_or(RudaPrimitiveError::Configuration("record allocation size overflow"))?;
        let bytes = empty_device_dtype(anchor.client.clone(), anchor.device.clone(), Shape::new([size]), DType::U8);
        Ok(Self { bytes, count, marker: PhantomData })
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn client(&self) -> &ComputeClient<R> { &self.bytes.client }
    pub fn view(&self) -> RudaRecordBytesLaunch<R> { RudaRecordBytesLaunch::new(self.bytes.clone().into_linear_view()) }
    fn empty(&self, count: usize) -> Result<Self, RudaPrimitiveError> { Self::allocate(&self.bytes, count) }
}

fn configuration<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, threads: u32) -> Result<(), RudaPrimitiveError> {
    let hardware = &input.client().properties().hardware;
    if threads < 2 || threads > hardware.max_units_per_ruda || threads > hardware.max_ruda_dim.0 {
        return Err(RudaPrimitiveError::Configuration("record collective block size is invalid"));
    }
    if (threads as usize).checked_mul(T::SIZE).is_none_or(|bytes| bytes > hardware.max_shared_memory_size) {
        return Err(RudaPrimitiveError::Configuration("record collective exceeds shared memory"));
    }
    Ok(())
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn scan_tile<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(
    input: &RudaRecordBytes, output: &mut RudaRecordBytes, totals: &mut RudaRecordBytes,
    op: &O, count: usize, #[comptime] threads: usize,
) {
    let start = RUDA_POS as usize * threads;
    if start >= count { terminate!(); }
    let valid = min(threads, count - start);
    let index = start + UNIT_POS as usize;
    let mut local = RudaRecordArray::<T>::new(1usize);
    let mut result = RudaRecordArray::<T>::new(1usize);
    let mut scratch = RudaRecordShared::<T>::new(threads);
    if index < count { local.write(0, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
    crate::block::record::inclusive_scan::<T, O>(&local, &mut result, &mut scratch, op, valid, threads, 1usize);
    if index < count { <RudaRecordBytes as RudaWrite<T>>::write(output, index, result.read(0)); }
    if UNIT_POS == 0 { <RudaRecordBytes as RudaWrite<T>>::write(totals, RUDA_POS as usize, scratch.read(valid - 1)); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn prefix<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(
    output: &mut RudaRecordBytes, totals: &RudaRecordBytes, op: &O, count: usize, #[comptime] threads: usize,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let tile = index / threads;
        if tile > 0 {
            <RudaRecordBytes as RudaWrite<T>>::write(output, index, op.combine(<RudaRecordBytes as RudaRead<T>>::read(totals, tile - 1),
                <RudaRecordBytes as RudaRead<T>>::read(output, index)));
        }
    }
}

pub fn inclusive_scan<R, T, O>(input: &RudaRecordBuffer<R, T>, op: O::RuntimeArg<R>, threads: u32)
    -> Result<RudaRecordBuffer<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    configuration(input, threads)?;
    let output = input.empty(input.count)?;
    if input.is_empty() { return Ok(output); }
    let tiles = input.count.div_ceil(threads as usize);
    let totals = input.empty(tiles)?;
    let dim = RudaDim::new_1d(threads);
    let grid = calculate_ruda_count_elemwise(input.client(), input.count, dim);
    unsafe {
        scan_tile::launch_unchecked::<T, O, R>(input.client(), grid, dim,
            input.view(), output.view(), totals.view(), op.clone(), input.count, threads as usize);
    }
    if tiles > 1 {
        let scanned = inclusive_scan::<R, T, O>(&totals, op.clone(), threads)?;
        let grid = calculate_ruda_count_elemwise(input.client(), input.count, dim);
        unsafe {
            prefix::launch_unchecked::<T, O, R>(input.client(), grid, dim, output.view(), scanned.view(), op, input.count, threads as usize);
        }
    }
    Ok(output)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn seed<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(
    input: &RudaRecordBytes, output: &mut RudaRecordBytes, initial: &RudaRecordBytes,
    op: &O, count: usize, #[comptime] exclusive: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut value = <RudaRecordBytes as RudaRead<T>>::read(initial, 0);
        if exclusive {
            if index > 0 { value = op.combine(value, <RudaRecordBytes as RudaRead<T>>::read(input, index - 1)); }
        } else { value = op.combine(value, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
        <RudaRecordBytes as RudaWrite<T>>::write(output, index, value);
    }
}

pub fn scan_init<R, T, O>(input: &RudaRecordBuffer<R, T>, initial: &RudaRecordBuffer<R, T>,
    op: O::RuntimeArg<R>, exclusive: bool, threads: u32) -> Result<RudaRecordBuffer<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    if initial.len() != 1 { return Err(RudaPrimitiveError::Length); }
    if initial.bytes.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let scanned = inclusive_scan::<R, T, O>(input, op.clone(), threads)?;
    let output = input.empty(input.count)?;
    if input.count > 0 {
        let dim = RudaDim::new_1d(threads);
        let grid = calculate_ruda_count_elemwise(input.client(), input.count, dim);
        unsafe { seed::launch_unchecked::<T, O, R>(input.client(), grid, dim, scanned.view(), output.view(), initial.view(), op, input.count, exclusive); }
    }
    Ok(output)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn reduce_tile<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(
    input: &RudaRecordBytes, output: &mut RudaRecordBytes, op: &O, count: usize, #[comptime] threads: usize,
) {
    let start = RUDA_POS as usize * threads;
    if start >= count { terminate!(); }
    let valid = min(threads, count - start);
    let index = start + UNIT_POS as usize;
    let mut local = RudaRecordArray::<T>::new(1usize);
    let mut scratch = RudaRecordShared::<T>::new(threads);
    if index < count { local.write(0, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
    let value = crate::block::record::reduce::<T, O>(&local, &mut scratch, op, valid, threads, 1usize);
    if UNIT_POS == 0 { <RudaRecordBytes as RudaWrite<T>>::write(output, RUDA_POS as usize, value); }
}

pub fn reduce<R, T, O>(input: &RudaRecordBuffer<R, T>, initial: &RudaRecordBuffer<R, T>,
    op: O::RuntimeArg<R>, threads: u32) -> Result<RudaRecordBuffer<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    configuration(input, threads)?;
    if initial.len() != 1 { return Err(RudaPrimitiveError::Length); }
    if initial.bytes.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if input.is_empty() { return Ok(initial.clone()); }
    let dim = RudaDim::new_1d(threads);
    let mut source = input.clone();
    while source.count > 1 {
        let output = input.empty(source.count.div_ceil(threads as usize))?;
        let grid = calculate_ruda_count_elemwise(input.client(), source.count, dim);
        unsafe {
            reduce_tile::launch_unchecked::<T, O, R>(input.client(), grid, dim, source.view(), output.view(), op.clone(), source.count, threads as usize);
        }
        source = output;
    }
    let output = input.empty(1)?;
    unsafe {
        seed::launch_unchecked::<T, O, R>(input.client(), RudaCount::Static(1, 1, 1), RudaDim::new_1d(1),
            source.view(), output.view(), initial.view(), op, 1, false);
    }
    Ok(output)
}

#[ruda]
fn merge_rank<K: RudaRecord, C: RudaCompare<K>>(
    keys: &RudaRecordBytes, compare: &C, index: usize, count: usize, run: usize,
) -> usize {
    let group = index / (run * 2) * (run * 2);
    let middle = min(group + run, count);
    let end = min(group + run * 2, count);
    let key = <RudaRecordBytes as RudaRead<K>>::read(keys, index);
    let left = index < middle;
    let begin = if left { middle } else { group };
    let mut low = begin;
    let mut high = if left { end } else { middle };
    while low < high {
        let probe = low + (high - low) / 2;
        let other = <RudaRecordBytes as RudaRead<K>>::read(keys, probe);
        let before = if left { compare.before(other, key) } else { !compare.before(key, other) };
        if before { low = probe + 1; } else { high = probe; }
    }
    group + (if left { index - group } else { index - middle }) + low - begin
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn merge_pass<K: RudaRecord, V: RudaRecord, C: RudaCompare<K> + LaunchArg>(
    keys: &RudaRecordBytes, values: &RudaRecordBytes, output_keys: &mut RudaRecordBytes,
    output_values: &mut RudaRecordBytes, compare: &C, count: usize, run: usize, #[comptime] pairs: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let rank = merge_rank::<K, C>(keys, compare, index, count, run);
        <RudaRecordBytes as RudaWrite<K>>::write(output_keys, rank, <RudaRecordBytes as RudaRead<K>>::read(keys, index));
        if pairs { <RudaRecordBytes as RudaWrite<V>>::write(output_values, rank, <RudaRecordBytes as RudaRead<V>>::read(values, index)); }
    }
}

pub fn merge_sort_pairs<R, K, V, C>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>, compare: C::RuntimeArg<R>)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    if keys.len() != values.len() { return Err(RudaPrimitiveError::Length); }
    if keys.bytes.device.to_id() != values.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let mut source_keys = keys.clone();
    let mut source_values = values.clone();
    if keys.len() < 2 { return Ok((source_keys, source_values)); }
    let key_buffers = [keys.empty(keys.len())?, keys.empty(keys.len())?];
    let value_buffers = [values.empty(values.len())?, values.empty(values.len())?];
    let dim = RudaDim::new(keys.client().properties(), keys.len());
    let mut run = 1usize;
    let mut selector = 0usize;
    while run < keys.len() {
        let out_keys = &key_buffers[selector];
        let out_values = &value_buffers[selector];
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe {
            merge_pass::launch_unchecked::<K, V, C, R>(keys.client(), grid, dim, source_keys.view(), source_values.view(),
                out_keys.view(), out_values.view(), compare.clone(), keys.len(), run, true);
        }
        source_keys = out_keys.clone();
        source_values = out_values.clone();
        selector ^= 1;
        run = run.saturating_mul(2);
    }
    Ok((source_keys, source_values))
}

pub fn merge_sort_keys<R, K, C>(keys: &RudaRecordBuffer<R, K>, compare: C::RuntimeArg<R>)
    -> Result<RudaRecordBuffer<R, K>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    let mut source = keys.clone();
    if keys.len() < 2 { return Ok(source); }
    let buffers = [keys.empty(keys.len())?, keys.empty(keys.len())?];
    let dim = RudaDim::new(keys.client().properties(), keys.len());
    let mut run = 1usize;
    let mut selector = 0usize;
    while run < keys.len() {
        let output = &buffers[selector];
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe {
            merge_pass::launch_unchecked::<K, K, C, R>(keys.client(), grid, dim, source.view(), source.view(),
                output.view(), output.view(), compare.clone(), keys.len(), run, false);
        }
        source = output.clone();
        selector ^= 1;
        run = run.saturating_mul(2);
    }
    Ok(source)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn digit_flags<K: RudaRecord, D: RudaDecomposer<K> + LaunchArg>(
    keys: &RudaRecordBytes, flags: &mut LinearView<u64, ReadWrite>, decomposer: &D,
    count: usize, #[comptime] bit: usize, #[comptime] descending: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        flags[index] = u64::cast_from(decomposer.bit(<RudaRecordBytes as RudaRead<K>>::read(keys, index), bit) != descending);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn radix_scatter<K: RudaRecord, V: RudaRecord>(keys: &RudaRecordBytes, values: &RudaRecordBytes,
    flags: &LinearView<u64>, prefixes: &LinearView<u64>, output_keys: &mut RudaRecordBytes,
    output_values: &mut RudaRecordBytes, count: usize, #[comptime] pairs: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let ones = prefixes[count - 1] as usize;
        let before = (prefixes[index] - flags[index]) as usize;
        let rank = if flags[index] == 0 { index - before } else { count - ones + before };
        <RudaRecordBytes as RudaWrite<K>>::write(output_keys, rank, <RudaRecordBytes as RudaRead<K>>::read(keys, index));
        if pairs { <RudaRecordBytes as RudaWrite<V>>::write(output_values, rank, <RudaRecordBytes as RudaRead<V>>::read(values, index)); }
    }
}

fn radix_impl<R, K, V, D>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    decomposer: D::RuntimeArg<R>, begin_bit: usize, end_bit: usize, descending: bool, pairs: bool, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    if begin_bit > end_bit || end_bit > D::BITS { return Err(RudaPrimitiveError::Configuration("invalid decomposed radix bit interval")); }
    if pairs && keys.len() != values.len() { return Err(RudaPrimitiveError::Length); }
    if keys.bytes.device.to_id() != values.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    super::scan_threads::<R, u64>(&keys.bytes, threads)?;
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
            digit_flags::launch_unchecked::<K, D, R>(keys.client(), grid, dim, source_keys.view(), flags.clone().into_linear_view(),
                decomposer.clone(), keys.len(), bit, descending);
        }
        let prefixes = super::scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
        let output_keys = &key_buffers[pass % 2];
        let output_values = &value_buffers[pass % 2];
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe {
            radix_scatter::launch_unchecked::<K, V, R>(keys.client(), grid, dim, source_keys.view(), source_values.view(),
                flags.clone().into_linear_view(), prefixes.into_linear_view(), output_keys.view(), output_values.view(), keys.len(), pairs);
        }
        source_keys = output_keys.clone();
        source_values = output_values.clone();
    }
    Ok((source_keys, source_values))
}

pub fn radix_sort_pairs<R, K, V, D>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    decomposer: D::RuntimeArg<R>, begin_bit: usize, end_bit: usize, descending: bool, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    radix_impl::<R, K, V, D>(keys, values, decomposer, begin_bit, end_bit, descending, true, threads)
}

pub fn radix_sort_keys<R, K, D>(keys: &RudaRecordBuffer<R, K>, decomposer: D::RuntimeArg<R>,
    begin_bit: usize, end_bit: usize, descending: bool, threads: u32) -> Result<RudaRecordBuffer<R, K>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    Ok(radix_impl::<R, K, K, D>(keys, keys, decomposer, begin_bit, end_bit, descending, false, threads)?.0)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn topk_flags<K: RudaRecord, D: RudaDecomposer<K> + LaunchArg>(
    keys: &RudaRecordBytes, candidates: &LinearView<u64>, flags: &mut LinearView<u64, ReadWrite>,
    decomposer: &D, count: usize, #[comptime] bit: usize, #[comptime] largest: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut flag = 0u64;
        if candidates[index] != 0 {
            flag = u64::cast_from(decomposer.bit(<RudaRecordBytes as RudaRead<K>>::read(keys, index), bit) == largest);
        }
        flags[index] = flag;
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn pack<K: RudaRecord, V: RudaRecord>(keys: &RudaRecordBytes, values: &RudaRecordBytes,
    selected: &LinearView<u64>, prefixes: &LinearView<u64>, output_keys: &mut RudaRecordBytes,
    output_values: &mut RudaRecordBytes, count: usize, #[comptime] pairs: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        if selected[index] != 0 {
            let rank = prefixes[index] as usize - 1;
            <RudaRecordBytes as RudaWrite<K>>::write(output_keys, rank, <RudaRecordBytes as RudaRead<K>>::read(keys, index));
            if pairs { <RudaRecordBytes as RudaWrite<V>>::write(output_values, rank, <RudaRecordBytes as RudaRead<V>>::read(values, index)); }
        }
    }
}

fn topk_impl<R, K, V, D>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    decomposer: D::RuntimeArg<R>, k: usize, largest: bool, pairs: bool, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    if pairs && keys.len() != values.len() { return Err(RudaPrimitiveError::Length); }
    if keys.bytes.device.to_id() != values.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    super::scan_threads::<R, u64>(&keys.bytes, threads)?;
    let k = k.min(keys.len());
    let output_keys = keys.empty(k)?;
    let output_values = values.empty(if pairs { k } else { 0 })?;
    if k == 0 { return Ok((output_keys, output_values)); }
    let allocate = |len| empty_device_dtype(keys.bytes.client.clone(), keys.bytes.device.clone(), Shape::new([len]), DType::U64);
    let candidates = allocate(keys.len());
    let selected = allocate(keys.len());
    let flags = allocate(keys.len());
    let state = allocate(2);
    let dim = RudaDim::new(keys.client().properties(), keys.len());
    let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
    unsafe {
        super::topk::initialise::launch_unchecked::<R>(keys.client(), grid, dim, AddressType::U64,
            candidates.clone().into_linear_view(), selected.clone().into_linear_view(), state.clone().into_linear_view(), k);
    }
    for bit in (0..D::BITS).rev() {
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe {
            topk_flags::launch_unchecked::<K, D, R>(keys.client(), grid, dim, keys.view(), candidates.clone().into_linear_view(),
                flags.clone().into_linear_view(), decomposer.clone(), keys.len(), bit, largest);
        }
        let total = super::reduce::reduce::<R, u64, RudaSum>(&flags, 0, RudaSumLaunch::new(), threads)?;
        unsafe {
            super::topk::decision::launch_unchecked::<R>(keys.client(), RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), AddressType::U64,
                total.into_linear_view(), state.clone().into_linear_view());
            let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
            super::topk::refine::launch_unchecked::<R>(keys.client(), grid, dim, AddressType::U64,
                flags.clone().into_linear_view(), state.clone().into_linear_view(), candidates.clone().into_linear_view(), selected.clone().into_linear_view());
        }
    }
    let prefixes = super::scan::inclusive_scan::<R, u64, RudaSum>(&candidates, RudaSumLaunch::new(), threads)?;
    let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
    unsafe {
        super::topk::resolve_ties::launch_unchecked::<R>(keys.client(), grid, dim, AddressType::U64, candidates.into_linear_view(),
            prefixes.into_linear_view(), state.into_linear_view(), selected.clone().into_linear_view());
    }
    let prefixes = super::scan::inclusive_scan::<R, u64, RudaSum>(&selected, RudaSumLaunch::new(), threads)?;
    let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
    unsafe {
        pack::launch_unchecked::<K, V, R>(keys.client(), grid, dim, keys.view(), values.view(), selected.into_linear_view(), prefixes.into_linear_view(),
            output_keys.view(), output_values.view(), keys.len(), pairs);
    }
    Ok((output_keys, output_values))
}

/// Radix-refinement TopK over decomposed records, without full sorting.
/// Output and cutoff ties retain input order; k is capped to the input size.
pub fn topk_pairs<R, K, V, D>(keys: &RudaRecordBuffer<R, K>, values: &RudaRecordBuffer<R, V>,
    decomposer: D::RuntimeArg<R>, k: usize, largest: bool, threads: u32)
    -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    topk_impl::<R, K, V, D>(keys, values, decomposer, k, largest, true, threads)
}

pub fn topk_keys<R, K, D>(keys: &RudaRecordBuffer<R, K>, decomposer: D::RuntimeArg<R>, k: usize, largest: bool, threads: u32)
    -> Result<RudaRecordBuffer<R, K>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    Ok(topk_impl::<R, K, K, D>(keys, keys, decomposer, k, largest, false, threads)?.0)
}
