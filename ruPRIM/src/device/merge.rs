use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::Shape;
use crate::collective::{RudaCompare, RudaCompareExpand};
use super::{RudaPrimitiveError, check_type};

#[ruda]
fn insertion<K: Numeric, C: RudaCompare<K>>(
    input: &LinearView<K>, key: K, compare: &C, #[comptime] upper: bool,
) -> usize {
    let mut low = 0usize;
    let mut high = input.shape();
    while low < high {
        let middle = low + (high - low) / 2;
        let before = if upper { !compare.before(key, input[middle]) } else { compare.before(input[middle], key) };
        if before { low = middle + 1; } else { high = middle; }
    }
    low
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn keys_kernel<K: Numeric, C: RudaCompare<K> + LaunchArg>(
    left: &LinearView<K>, right: &LinearView<K>, output: &mut LinearView<K, ReadWrite>, compare: &C,
) {
    let index = ABSOLUTE_POS;
    if index < left.shape() {
        let key = left[index];
        output[index + insertion::<K, C>(right, key, compare, false)] = key;
    } else {
        let right_index = index - left.shape();
        if right_index < right.shape() {
            let key = right[right_index];
            output[right_index + insertion::<K, C>(left, key, compare, true)] = key;
        }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn pairs_kernel<K: Numeric, V: Numeric, C: RudaCompare<K> + LaunchArg>(
    left_keys: &LinearView<K>, left_values: &LinearView<V>,
    right_keys: &LinearView<K>, right_values: &LinearView<V>,
    output_keys: &mut LinearView<K, ReadWrite>, output_values: &mut LinearView<V, ReadWrite>, compare: &C,
) {
    let index = ABSOLUTE_POS;
    if index < left_keys.shape() {
        let key = left_keys[index];
        let rank = index + insertion::<K, C>(right_keys, key, compare, false);
        output_keys[rank] = key;
        output_values[rank] = left_values[index];
    } else {
        let right_index = index - left_keys.shape();
        if right_index < right_keys.shape() {
            let key = right_keys[right_index];
            let rank = right_index + insertion::<K, C>(left_keys, key, compare, true);
            output_keys[rank] = key;
            output_values[rank] = right_values[right_index];
        }
    }
}

/// Stable merge of two sorted sequences. Equivalent left keys precede right keys.
pub fn keys<R, K, C>(left: &RudaTensor<R>, right: &RudaTensor<R>, compare: C::RuntimeArg<R>) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, C: RudaCompare<K> + LaunchArg,
{
    check_type::<R, K>(left)?;
    check_type::<R, K>(right)?;
    if left.device.to_id() != right.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let count = left.meta.num_elements().checked_add(right.meta.num_elements())
        .ok_or(RudaPrimitiveError::Configuration("merged length overflow"))?;
    let output = empty_device_dtype(left.client.clone(), left.device.clone(), Shape::new([count]), left.dtype);
    if count == 0 { return Ok(output); }
    let dim = RudaDim::new(left.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&left.client, count, dim);
    unsafe {
        keys_kernel::launch_unchecked::<K, C, R>(
            &left.client, grid, dim, address_type!(left, right, output),
            left.clone().into_linear_view(), right.clone().into_linear_view(), output.clone().into_linear_view(), compare,
        );
    }
    Ok(output)
}

pub fn pairs<R, K, V, C>(
    left_keys: &RudaTensor<R>, left_values: &RudaTensor<R>, right_keys: &RudaTensor<R>, right_values: &RudaTensor<R>, compare: C::RuntimeArg<R>,
) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError>
where R: Runtime, K: TensorElement, V: TensorElement, C: RudaCompare<K> + LaunchArg,
{
    check_type::<R, K>(left_keys)?;
    check_type::<R, K>(right_keys)?;
    check_type::<R, V>(left_values)?;
    check_type::<R, V>(right_values)?;
    if left_keys.meta.num_elements() != left_values.meta.num_elements() || right_keys.meta.num_elements() != right_values.meta.num_elements() {
        return Err(RudaPrimitiveError::Length);
    }
    if left_keys.device.to_id() != right_keys.device.to_id() || left_keys.device.to_id() != left_values.device.to_id() || left_keys.device.to_id() != right_values.device.to_id() {
        return Err(RudaPrimitiveError::Device);
    }
    let count = left_keys.meta.num_elements().checked_add(right_keys.meta.num_elements())
        .ok_or(RudaPrimitiveError::Configuration("merged length overflow"))?;
    let output_keys = empty_device_dtype(left_keys.client.clone(), left_keys.device.clone(), Shape::new([count]), left_keys.dtype);
    let output_values = empty_device_dtype(left_keys.client.clone(), left_keys.device.clone(), Shape::new([count]), left_values.dtype);
    if count == 0 { return Ok((output_keys, output_values)); }
    let dim = RudaDim::new(left_keys.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&left_keys.client, count, dim);
    unsafe {
        pairs_kernel::launch_unchecked::<K, V, C, R>(
            &left_keys.client, grid, dim, address_type!(left_keys, left_values, right_keys, right_values, output_keys, output_values),
            left_keys.clone().into_linear_view(), left_values.clone().into_linear_view(),
            right_keys.clone().into_linear_view(), right_values.clone().into_linear_view(),
            output_keys.clone().into_linear_view(), output_values.clone().into_linear_view(), compare,
        );
    }
    Ok((output_keys, output_values))
}
