use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaCompare, RudaCompareExpand};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::device::RudaPrimitiveError;
use super::{RudaRecordBuffer, RudaRecordBytes};

#[ruda]
fn insertion<K: RudaRecord, C: RudaCompare<K>>(
    input: &RudaRecordBytes, key: K, count: usize, compare: &C, #[comptime] upper: bool,
) -> usize {
    let mut low = 0usize;
    let mut high = count;
    while low < high {
        let middle = low + (high - low) / 2;
        let value = <RudaRecordBytes as RudaRead<K>>::read(input, middle);
        let before = if upper { !compare.before(key, value) } else { compare.before(value, key) };
        if before { low = middle + 1; } else { high = middle; }
    }
    low
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn keys_kernel<K: RudaRecord, C: RudaCompare<K> + LaunchArg>(
    left: &RudaRecordBytes, right: &RudaRecordBytes, output: &mut RudaRecordBytes,
    compare: &C, left_count: usize, right_count: usize,
) {
    let index = ABSOLUTE_POS;
    if index < left_count {
        let key = <RudaRecordBytes as RudaRead<K>>::read(left, index);
        <RudaRecordBytes as RudaWrite<K>>::write(output, index + insertion::<K, C>(right, key, right_count, compare, false), key);
    } else if index - left_count < right_count {
        let item = index - left_count;
        let key = <RudaRecordBytes as RudaRead<K>>::read(right, item);
        <RudaRecordBytes as RudaWrite<K>>::write(output, item + insertion::<K, C>(left, key, left_count, compare, true), key);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn pairs_kernel<K: RudaRecord, V: RudaRecord, C: RudaCompare<K> + LaunchArg>(
    left: &RudaRecordBytes, left_values: &RudaRecordBytes,
    right: &RudaRecordBytes, right_values: &RudaRecordBytes,
    keys: &mut RudaRecordBytes, values: &mut RudaRecordBytes, compare: &C,
    left_count: usize, right_count: usize,
) {
    let index = ABSOLUTE_POS;
    if index < left_count {
        let key = <RudaRecordBytes as RudaRead<K>>::read(left, index);
        let rank = index + insertion::<K, C>(right, key, right_count, compare, false);
        <RudaRecordBytes as RudaWrite<K>>::write(keys, rank, key);
        <RudaRecordBytes as RudaWrite<V>>::write(values, rank, <RudaRecordBytes as RudaRead<V>>::read(left_values, index));
    } else if index - left_count < right_count {
        let item = index - left_count;
        let key = <RudaRecordBytes as RudaRead<K>>::read(right, item);
        let rank = item + insertion::<K, C>(left, key, left_count, compare, true);
        <RudaRecordBytes as RudaWrite<K>>::write(keys, rank, key);
        <RudaRecordBytes as RudaWrite<V>>::write(values, rank, <RudaRecordBytes as RudaRead<V>>::read(right_values, item));
    }
}

/// Stable merge; equivalent keys from the left sequence precede right keys.
pub fn keys<R, K, C>(left: &RudaRecordBuffer<R, K>, right: &RudaRecordBuffer<R, K>, compare: C::RuntimeArg<R>)
    -> Result<RudaRecordBuffer<R, K>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, C: RudaCompare<K> + LaunchArg,
{
    if left.bytes.device.to_id() != right.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let count = left.len().checked_add(right.len()).ok_or(RudaPrimitiveError::Configuration("merged length overflow"))?;
    let output = left.empty(count)?;
    if count > 0 {
        let dim = RudaDim::new(left.client().properties(), count);
        let grid = calculate_ruda_count_elemwise(left.client(), count, dim);
        unsafe { keys_kernel::launch_unchecked::<K, C, R>(left.client(), grid, dim,
            left.view(), right.view(), output.view(), compare, left.len(), right.len()); }
    }
    Ok(output)
}

pub fn pairs<R, K, V, C>(
    left: &RudaRecordBuffer<R, K>, left_values: &RudaRecordBuffer<R, V>,
    right: &RudaRecordBuffer<R, K>, right_values: &RudaRecordBuffer<R, V>, compare: C::RuntimeArg<R>,
) -> Result<(RudaRecordBuffer<R, K>, RudaRecordBuffer<R, V>), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, C: RudaCompare<K> + LaunchArg,
{
    if left.len() != left_values.len() || right.len() != right_values.len() { return Err(RudaPrimitiveError::Length); }
    if left.bytes.device.to_id() != right.bytes.device.to_id() || left.bytes.device.to_id() != left_values.bytes.device.to_id()
        || left.bytes.device.to_id() != right_values.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let count = left.len().checked_add(right.len()).ok_or(RudaPrimitiveError::Configuration("merged length overflow"))?;
    let output_keys = left.empty(count)?;
    let output_values = left_values.empty(count)?;
    if count > 0 {
        let dim = RudaDim::new(left.client().properties(), count);
        let grid = calculate_ruda_count_elemwise(left.client(), count, dim);
        unsafe { pairs_kernel::launch_unchecked::<K, V, C, R>(left.client(), grid, dim,
            left.view(), left_values.view(), right.view(), right_values.view(), output_keys.view(), output_values.view(),
            compare, left.len(), right.len()); }
    }
    Ok((output_keys, output_values))
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn bounds_kernel<K: RudaRecord, C: RudaCompare<K> + LaunchArg>(
    input: &RudaRecordBytes, queries: &RudaRecordBytes, output: &mut LinearView<u64, ReadWrite>,
    compare: &C, count: usize, query_count: usize, #[comptime] upper: bool,
) {
    let index = ABSOLUTE_POS;
    if index < query_count {
        let key = <RudaRecordBytes as RudaRead<K>>::read(queries, index);
        output[index] = insertion::<K, C>(input, key, count, compare, upper) as u64;
    }
}

/// One lower/upper insertion bound per device-resident query record.
pub fn bounds<R, K, C>(input: &RudaRecordBuffer<R, K>, queries: &RudaRecordBuffer<R, K>,
    compare: C::RuntimeArg<R>, upper: bool) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: RudaRecord, C: RudaCompare<K> + LaunchArg,
{
    if input.bytes.device.to_id() != queries.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let output = empty_device_dtype(input.bytes.client.clone(), input.bytes.device.clone(), Shape::new([queries.len()]), DType::U64);
    if !queries.is_empty() {
        let dim = RudaDim::new(input.client().properties(), queries.len());
        let grid = calculate_ruda_count_elemwise(input.client(), queries.len(), dim);
        unsafe { bounds_kernel::launch_unchecked::<K, C, R>(input.client(), grid, dim, input.view(), queries.view(),
            output.clone().into_linear_view(), compare, input.len(), queries.len(), upper); }
    }
    Ok(output)
}
