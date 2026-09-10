use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaSum, RudaSumLaunch, radix::RudaRadixKey};
use super::{RudaPrimitiveError, check_type, empty_like, scan};

pub fn sort_keys_into<R: Runtime, K: TensorElement + RudaRadixKey>(input: &RudaTensor<R>, output: &RudaTensor<R>,
    begin_bit: u32, end_bit: u32, descending: bool, threads: u32) -> Result<(), RudaPrimitiveError>
{
    check_type::<R, K>(output)?;
    if input.meta.num_elements() != output.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != output.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let sorted = sort_keys::<R, K>(input, begin_bit, end_bit, descending, threads)?;
    super::iteration::copy_into::<R, K>(&sorted, output)
}

pub fn sort_pairs_into<R: Runtime, K: TensorElement + RudaRadixKey, V: TensorElement>(keys: &RudaTensor<R>, values: &RudaTensor<R>,
    output_keys: &RudaTensor<R>, output_values: &RudaTensor<R>, begin_bit: u32, end_bit: u32, descending: bool, threads: u32)
    -> Result<(), RudaPrimitiveError>
{
    check_type::<R, K>(output_keys)?;
    check_type::<R, V>(output_values)?;
    if keys.meta.num_elements() != output_keys.meta.num_elements() || values.meta.num_elements() != output_values.meta.num_elements() {
        return Err(RudaPrimitiveError::Length);
    }
    if keys.device.to_id() != output_keys.device.to_id() || keys.device.to_id() != output_values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let (sorted_keys, sorted_values) = sort_pairs::<R, K, V>(keys, values, begin_bit, end_bit, descending, threads)?;
    super::iteration::copy_into::<R, K>(&sorted_keys, output_keys)?;
    super::iteration::copy_into::<R, V>(&sorted_values, output_values)
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn flags_kernel<K: RudaRadixKey>(
    keys: &LinearView<K>, flags: &mut LinearView<u64, ReadWrite>,
    #[comptime] bit: u32, #[comptime] descending: bool,
) {
    let index = ABSOLUTE_POS;
    if index < keys.shape() {
        let mut bit_value = (K::ordered_bits(keys[index]) >> bit) & 1u64;
        if descending { bit_value ^= 1u64; }
        flags[index] = bit_value;
    }
}

#[ruda]
fn position(flags: &LinearView<u64>, prefixes: &LinearView<u64>, index: usize) -> usize {
    let total = flags.shape();
    let ones = prefixes[total - 1] as usize;
    let before = (prefixes[index] - flags[index]) as usize;
    if flags[index] == 0 { index - before } else { total - ones + before }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn scatter_keys<K: Numeric>(
    keys: &LinearView<K>, flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output: &mut LinearView<K, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < keys.shape() { output[position(flags, prefixes, index)] = keys[index]; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn scatter_pairs<K: Numeric, V: Numeric>(
    keys: &LinearView<K>, values: &LinearView<V>, flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output_keys: &mut LinearView<K, ReadWrite>, output_values: &mut LinearView<V, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < keys.shape() {
        let rank = position(flags, prefixes, index);
        output_keys[rank] = keys[index];
        output_values[rank] = values[index];
    }
}

pub(crate) fn check_bits<K: TensorElement>(begin: u32, end: u32) -> Result<(), RudaPrimitiveError> {
    if begin > end || end > core::mem::size_of::<K>() as u32 * 8 {
        return Err(RudaPrimitiveError::Configuration("invalid radix bit interval"));
    }
    Ok(())
}

pub(super) fn bit_prefixes<R: Runtime, K: TensorElement + RudaRadixKey>(
    keys: &RudaTensor<R>, flags: &RudaTensor<R>, bit: u32, descending: bool, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    let count = keys.meta.num_elements();
    let dim = RudaDim::new(keys.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
    unsafe {
        flags_kernel::launch_unchecked::<K, R>(
            &keys.client, grid, dim, address_type!(keys, flags),
            keys.clone().into_linear_view(), flags.clone().into_linear_view(), bit, descending,
        );
    }
    scan::inclusive_scan::<R, u64, RudaSum>(flags, RudaSumLaunch::new(), threads)
}

/// Stable LSD radix sort over `[begin_bit, end_bit)` of the ordered encoding.
/// Descending reverses the ordering, not the order of equal keys.
pub fn sort_keys<R: Runtime, K: TensorElement + RudaRadixKey>(
    keys: &RudaTensor<R>, begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    check_type::<R, K>(keys)?;
    check_bits::<K>(begin_bit, end_bit)?;
    let count = keys.meta.num_elements();
    if count < 2 || begin_bit == end_bit { return Ok(keys.clone()); }
    let buffers = [empty_like(keys), empty_like(keys)];
    let flags = empty_device_dtype(keys.client.clone(), keys.device.clone(), Shape::new([count]), DType::U64);
    let dim = RudaDim::new(keys.client.properties(), count);
    let mut source = keys.clone();
    for (pass, bit) in (begin_bit..end_bit).enumerate() {
        let prefixes = bit_prefixes::<R, K>(&source, &flags, bit, descending, threads)?;
        let output = buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
        unsafe {
            scatter_keys::launch_unchecked::<K, R>(
                &keys.client, grid, dim, address_type!(source, flags, prefixes, output),
                source.into_linear_view(), flags.clone().into_linear_view(), prefixes.into_linear_view(), output.clone().into_linear_view(),
            );
        }
        source = output;
    }
    Ok(source)
}

pub fn sort_pairs<R: Runtime, K: TensorElement + RudaRadixKey, V: TensorElement>(
    keys: &RudaTensor<R>, values: &RudaTensor<R>, begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError> {
    check_type::<R, K>(keys)?;
    check_type::<R, V>(values)?;
    check_bits::<K>(begin_bit, end_bit)?;
    let count = keys.meta.num_elements();
    if count != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if count < 2 || begin_bit == end_bit { return Ok((keys.clone(), values.clone())); }
    let key_buffers = [empty_like(keys), empty_like(keys)];
    let value_buffers = [empty_like(values), empty_like(values)];
    let flags = empty_device_dtype(keys.client.clone(), keys.device.clone(), Shape::new([count]), DType::U64);
    let dim = RudaDim::new(keys.client.properties(), count);
    let mut source_keys = keys.clone();
    let mut source_values = values.clone();
    for (pass, bit) in (begin_bit..end_bit).enumerate() {
        let prefixes = bit_prefixes::<R, K>(&source_keys, &flags, bit, descending, threads)?;
        let output_keys = key_buffers[pass % 2].clone();
        let output_values = value_buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
        unsafe {
            scatter_pairs::launch_unchecked::<K, V, R>(
                &keys.client, grid, dim, address_type!(source_keys, source_values, flags, prefixes, output_keys, output_values),
                source_keys.into_linear_view(), source_values.into_linear_view(),
                flags.clone().into_linear_view(), prefixes.into_linear_view(),
                output_keys.clone().into_linear_view(), output_values.clone().into_linear_view(),
            );
        }
        source_keys = output_keys;
        source_values = output_values;
    }
    Ok((source_keys, source_values))
}
