use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand, RudaKeyEqual, RudaKeyEqualExpand, RudaSum, RudaSumLaunch};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::device::{RudaPrimitiveError, check_type, scan, transform::{self, RudaCast, RudaCastLaunch}};
use super::{RudaRecordBuffer, RudaRecordBytes};

fn flags<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>, dtype: DType, count: usize) -> RudaTensor<R> {
    empty_device_dtype(input.bytes.client.clone(), input.bytes.device.clone(), Shape::new([count]), dtype)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn heads_kernel<K: RudaRecord, E: RudaKeyEqual<K> + LaunchArg>(
    keys: &RudaRecordBytes, heads: &mut LinearView<u32, ReadWrite>, equal: &E, count: usize,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut head = true;
        if index > 0 {
            head = !equal.equal(<RudaRecordBytes as RudaRead<K>>::read(keys, index - 1), <RudaRecordBytes as RudaRead<K>>::read(keys, index));
        }
        heads[index] = u32::cast_from(head);
    }
}

pub fn key_heads<R, K, E>(keys: &RudaRecordBuffer<R, K>, equal: E::RuntimeArg<R>) -> RudaTensor<R>
where R: Runtime, K: RudaRecord, E: RudaKeyEqual<K> + LaunchArg,
{
    let heads = flags(keys, DType::U32, keys.len());
    if !keys.is_empty() {
        let dim = RudaDim::new(keys.client().properties(), keys.len());
        let grid = calculate_ruda_count_elemwise(keys.client(), keys.len(), dim);
        unsafe { heads_kernel::launch_unchecked::<K, E, R>(keys.client(), grid, dim, keys.view(), heads.clone().into_linear_view(), equal, keys.len()); }
    }
    heads
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn step<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(input: &RudaRecordBytes, heads: &LinearView<u32>,
    output: &mut RudaRecordBytes, output_heads: &mut LinearView<u32, ReadWrite>, op: &O, count: usize, distance: usize,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut value = <RudaRecordBytes as RudaRead<T>>::read(input, index);
        let mut boundary = heads[index] != 0 || index == 0;
        if index >= distance {
            if !boundary { value = op.combine(<RudaRecordBytes as RudaRead<T>>::read(input, index - distance), value); }
            boundary = boundary || heads[index - distance] != 0 || index == distance;
        }
        <RudaRecordBytes as RudaWrite<T>>::write(output, index, value);
        output_heads[index] = u32::cast_from(boundary);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn seed<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(input: &RudaRecordBytes, heads: &LinearView<u32>,
    output: &mut RudaRecordBytes, initial: &RudaRecordBytes, op: &O, count: usize, #[comptime] exclusive: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut value = <RudaRecordBytes as RudaRead<T>>::read(initial, 0);
        if exclusive {
            if index > 0 && heads[index] == 0 { value = op.combine(value, <RudaRecordBytes as RudaRead<T>>::read(input, index - 1)); }
        } else { value = op.combine(value, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
        <RudaRecordBytes as RudaWrite<T>>::write(output, index, value);
    }
}

pub fn scan_by_heads<R, T, O>(input: &RudaRecordBuffer<R, T>, heads: &RudaTensor<R>,
    initial: Option<&RudaRecordBuffer<R, T>>, exclusive: bool, op: O::RuntimeArg<R>) -> Result<RudaRecordBuffer<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    check_type::<R, u32>(heads)?;
    if heads.meta.num_elements() != input.len() { return Err(RudaPrimitiveError::Length); }
    if heads.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if exclusive && initial.is_none() { return Err(RudaPrimitiveError::Configuration("exclusive scan requires an initial value")); }
    if let Some(initial) = initial {
        if initial.len() != 1 { return Err(RudaPrimitiveError::Length); }
        if initial.bytes.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    if input.is_empty() { return input.empty(0); }
    let buffers = [input.empty(input.len())?, input.empty(input.len())?];
    let head_buffers = [flags(input, DType::U32, input.len()), flags(input, DType::U32, input.len())];
    let mut source = input.clone();
    let mut source_heads = heads.clone();
    let dim = RudaDim::new(input.client().properties(), input.len());
    let mut distance = 1usize;
    let mut selector = 0usize;
    while distance < input.len() {
        let output = &buffers[selector];
        let output_heads = &head_buffers[selector];
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe {
            step::launch_unchecked::<T, O, R>(input.client(), grid, dim, source.view(), source_heads.into_linear_view(), output.view(),
                output_heads.clone().into_linear_view(), op.clone(), input.len(), distance);
        }
        source = output.clone();
        source_heads = output_heads.clone();
        distance = distance.saturating_mul(2);
        selector ^= 1;
    }
    if let Some(initial) = initial {
        let output = input.empty(input.len())?;
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe {
            seed::launch_unchecked::<T, O, R>(input.client(), grid, dim, source.view(), heads.clone().into_linear_view(), output.view(),
                initial.view(), op, input.len(), exclusive);
        }
        source = output;
    }
    Ok(source)
}

pub fn scan_by_key<R, K, T, O, E>(keys: &RudaRecordBuffer<R, K>, input: &RudaRecordBuffer<R, T>,
    initial: Option<&RudaRecordBuffer<R, T>>, exclusive: bool, op: O::RuntimeArg<R>, equal: E::RuntimeArg<R>)
    -> Result<RudaRecordBuffer<R, T>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone, E: RudaKeyEqual<K> + LaunchArg,
{
    if keys.len() != input.len() { return Err(RudaPrimitiveError::Length); }
    if keys.bytes.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let heads = key_heads::<R, K, E>(keys, equal);
    scan_by_heads::<R, T, O>(input, &heads, initial, exclusive, op)
}

pub struct RudaRecordKeyReduction<R: Runtime, K: RudaRecord, T: RudaRecord> {
    pub keys: RudaRecordBuffer<R, K>,
    pub aggregates: RudaRecordBuffer<R, T>,
    pub count: RudaTensor<R>,
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn compact<K: RudaRecord, T: RudaRecord>(keys: &RudaRecordBytes, scanned: &RudaRecordBytes,
    heads: &LinearView<u32>, prefixes: &LinearView<u64>, output_keys: &mut RudaRecordBytes,
    aggregates: &mut RudaRecordBytes, output_count: &mut LinearView<u64, ReadWrite>, count: usize,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut total = 0u64;
        if count > 0 { total = prefixes[count - 1]; }
        output_count[0] = total;
    }
    if index < count {
        let rank = prefixes[index] as usize - 1;
        if heads[index] != 0 { <RudaRecordBytes as RudaWrite<K>>::write(output_keys, rank, <RudaRecordBytes as RudaRead<K>>::read(keys, index)); }
        let mut tail = index + 1 == count;
        if index + 1 < count { tail = heads[index + 1] != 0; }
        if tail { <RudaRecordBytes as RudaWrite<T>>::write(aggregates, rank, <RudaRecordBytes as RudaRead<T>>::read(scanned, index)); }
    }
}

pub fn reduce_by_key<R, K, T, O, E>(keys: &RudaRecordBuffer<R, K>, input: &RudaRecordBuffer<R, T>,
    op: O::RuntimeArg<R>, equal: E::RuntimeArg<R>, threads: u32) -> Result<RudaRecordKeyReduction<R, K, T>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone, E: RudaKeyEqual<K> + LaunchArg,
{
    if keys.len() != input.len() { return Err(RudaPrimitiveError::Length); }
    if keys.bytes.device.to_id() != input.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let heads = key_heads::<R, K, E>(keys, equal);
    let lengths = transform::unary::<R, u32, u64, RudaCast>(&heads, RudaCastLaunch::new())?;
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&lengths, RudaSumLaunch::new(), threads)?;
    let scanned = scan_by_heads::<R, T, O>(input, &heads, None, false, op)?;
    let output = RudaRecordKeyReduction { keys: keys.empty(keys.len())?, aggregates: input.empty(input.len())?, count: flags(keys, DType::U64, 1) };
    let work = keys.len().max(1);
    let dim = RudaDim::new(keys.client().properties(), work);
    let grid = calculate_ruda_count_elemwise(keys.client(), work, dim);
    unsafe {
        compact::launch_unchecked::<K, T, R>(keys.client(), grid, dim, keys.view(), scanned.view(), heads.into_linear_view(), prefixes.into_linear_view(),
            output.keys.view(), output.aggregates.view(), output.count.clone().into_linear_view(), keys.len());
    }
    Ok(output)
}
