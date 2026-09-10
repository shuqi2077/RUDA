use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, element::TensorElement, layout::address_type};
use crate::collective::{RudaCompare, RudaCompareExpand};
use super::{RudaPrimitiveError, check_type, empty_like};

pub fn merge_sort_keys_into<R, K, C>(input: &RudaTensor<R>, output: &RudaTensor<R>, compare: C::RuntimeArg<R>)
    -> Result<(), RudaPrimitiveError>
where R: Runtime, K: TensorElement, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(output)?;
    if input.meta.num_elements() != output.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != output.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let sorted = merge_sort_keys::<R, K, C>(input, compare)?;
    super::iteration::copy_into::<R, K>(&sorted, output)
}

pub fn merge_sort_pairs_into<R, K, V, C>(keys: &RudaTensor<R>, values: &RudaTensor<R>,
    output_keys: &RudaTensor<R>, output_values: &RudaTensor<R>, compare: C::RuntimeArg<R>) -> Result<(), RudaPrimitiveError>
where R: Runtime, K: TensorElement, V: TensorElement, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(output_keys)?;
    check_type::<R, V>(output_values)?;
    if keys.meta.num_elements() != output_keys.meta.num_elements() || values.meta.num_elements() != output_values.meta.num_elements() {
        return Err(RudaPrimitiveError::Length);
    }
    if keys.device.to_id() != output_keys.device.to_id() || keys.device.to_id() != output_values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let (sorted_keys, sorted_values) = merge_sort_pairs::<R, K, V, C>(keys, values, compare)?;
    super::iteration::copy_into::<R, K>(&sorted_keys, output_keys)?;
    super::iteration::copy_into::<R, V>(&sorted_values, output_values)
}

#[ruda]
fn merge_rank<K: Numeric, C: RudaCompare<K>>(
    keys: &LinearView<K>, compare: &C, index: usize, run: usize,
) -> usize {
    let begin = index / (run * 2) * (run * 2);
    let middle = min(begin + run, keys.shape());
    let end = min(begin + run * 2, keys.shape());
    let left = index < middle;
    let own_begin = select(left, begin, middle);
    let other_begin = select(left, middle, begin);
    let mut low = other_begin;
    let mut high = select(left, end, middle);
    let key = keys[index];
    while low < high {
        let mid = low + (high - low) / 2;
        let advance = if left { compare.before(keys[mid], key) } else { !compare.before(key, keys[mid]) };
        if advance { low = mid + 1; } else { high = mid; }
    }
    begin + index - own_begin + low - other_begin
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn merge_keys<K: Numeric, C: RudaCompare<K> + LaunchArg>(
    input: &LinearView<K>, output: &mut LinearView<K, ReadWrite>,
    compare: &C, run: usize,
) {
    let index = ABSOLUTE_POS;
    if index >= input.shape() { terminate!(); }
    let rank = merge_rank::<K, C>(input, compare, index, run);
    output[rank] = input[index];
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn merge_pairs<K: Numeric, V: Numeric, C: RudaCompare<K> + LaunchArg>(
    keys: &LinearView<K>, values: &LinearView<V>,
    output_keys: &mut LinearView<K, ReadWrite>, output_values: &mut LinearView<V, ReadWrite>,
    compare: &C, run: usize,
) {
    let index = ABSOLUTE_POS;
    if index >= keys.shape() { terminate!(); }
    let rank = merge_rank::<K, C>(keys, compare, index, run);
    output_keys[rank] = keys[index];
    output_values[rank] = values[index];
}

/// Stable, out-of-place device merge sort. Empty/singleton input is unchanged.
pub fn merge_sort_keys<R, K, C>(
    keys: &RudaTensor<R>, compare: C::RuntimeArg<R>,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(keys)?;
    let count = keys.meta.num_elements();
    if count < 2 { return Ok(keys.clone()); }
    let buffers = [empty_like(keys), empty_like(keys)];
    let dim = RudaDim::new(keys.client.properties(), count);
    let mut source = keys.clone();
    let mut pass = 0;
    let mut run = 1usize;
    while run < count {
        let output = buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
        unsafe {
            merge_keys::launch_unchecked::<K, C, R>(
                &keys.client, grid, dim, address_type!(source, output),
                source.into_linear_view(), output.clone().into_linear_view(), compare.clone(), run,
            );
        }
        source = output;
        run = run.saturating_mul(2);
        pass += 1;
    }
    Ok(source)
}

/// Stable device merge sort of key/value pairs in logical flattened order.
pub fn merge_sort_pairs<R, K, V, C>(
    keys: &RudaTensor<R>, values: &RudaTensor<R>, compare: C::RuntimeArg<R>,
) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError>
where R: Runtime, K: TensorElement, V: TensorElement, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(keys)?;
    check_type::<R, V>(values)?;
    let count = keys.meta.num_elements();
    if count != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if count < 2 { return Ok((keys.clone(), values.clone())); }
    let key_buffers = [empty_like(keys), empty_like(keys)];
    let value_buffers = [empty_like(values), empty_like(values)];
    let dim = RudaDim::new(keys.client.properties(), count);
    let mut source_keys = keys.clone();
    let mut source_values = values.clone();
    let mut pass = 0;
    let mut run = 1usize;
    while run < count {
        let output_keys = key_buffers[pass % 2].clone();
        let output_values = value_buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
        unsafe {
            merge_pairs::launch_unchecked::<K, V, C, R>(
                &keys.client, grid, dim, address_type!(source_keys, source_values, output_keys, output_values),
                source_keys.into_linear_view(), source_values.into_linear_view(),
                output_keys.clone().into_linear_view(), output_values.clone().into_linear_view(),
                compare.clone(), run,
            );
        }
        source_keys = output_keys;
        source_values = output_values;
        run = run.saturating_mul(2);
        pass += 1;
    }
    Ok((source_keys, source_values))
}
