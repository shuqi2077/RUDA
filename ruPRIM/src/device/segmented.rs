use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand, RudaKeyEqual, RudaKeyEqualExpand};
use super::{RudaPrimitiveError, check_type, empty_like};

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn heads<K: Numeric, E: RudaKeyEqual<K> + LaunchArg>(
    keys: &LinearView<K>, output: &mut LinearView<u32, ReadWrite>, equal: &E,
) {
    let index = ABSOLUTE_POS;
    if index >= keys.shape() { terminate!(); }
    let mut head = true;
    if index > 0 { head = !equal.equal(keys[index - 1], keys[index]); }
    output[index] = u32::cast_from(head);
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn scan_step<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    input: &LinearView<T>, flags: &LinearView<u32>,
    output: &mut LinearView<T, ReadWrite>, output_flags: &mut LinearView<u32, ReadWrite>,
    op: &O, distance: usize,
) {
    let index = ABSOLUTE_POS;
    if index >= input.shape() { terminate!(); }
    let mut value = input[index];
    let mut head = flags[index];
    if index >= distance {
        if head == 0 { value = op.combine(input[index - distance], value); }
        head |= flags[index - distance];
    }
    output[index] = value;
    output_flags[index] = head;
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn exclusive_step<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    inclusive: &LinearView<T>, heads: &LinearView<u32>,
    output: &mut LinearView<T, ReadWrite>, initial: InputScalar, op: &O,
) {
    let index = ABSOLUTE_POS;
    if index >= output.shape() { terminate!(); }
    let mut value = initial.get::<T>();
    if index > 0 && heads[index] == 0 { value = op.combine(value, inclusive[index - 1]); }
    output[index] = value;
}

pub(crate) fn key_heads<R, K, E>(keys: &RudaTensor<R>, equal: E::RuntimeArg<R>) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, E: RudaKeyEqual<K> + LaunchArg,
{
    check_type::<R, K>(keys)?;
    let size = keys.meta.num_elements();
    let output = empty_device_dtype(keys.client.clone(), keys.device.clone(), Shape::new([size]), DType::U32);
    if size > 0 {
        let dim = RudaDim::new(keys.client.properties(), size);
        let grid = calculate_ruda_count_elemwise(&keys.client, size, dim);
        unsafe {
            heads::launch_unchecked::<K, E, R>(&keys.client, grid, dim, address_type!(keys, output),
                keys.clone().into_linear_view(), output.clone().into_linear_view(), equal);
        }
    }
    Ok(output)
}

/// Scan independent adjacent-key runs. `None` selects inclusive output;
/// `Some(initial)` selects exclusive output with a fresh seed for each run.
pub fn scan_by_key<R, K, T, O, E>(
    keys: &RudaTensor<R>, values: &RudaTensor<R>, op: O::RuntimeArg<R>,
    equal: E::RuntimeArg<R>, initial: Option<T>,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, T: TensorElement,
    O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
    E: RudaKeyEqual<K> + LaunchArg,
{
    check_type::<R, T>(values)?;
    if keys.meta.num_elements() != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let flags = key_heads::<R, K, E>(keys, equal)?;
    scan_by_heads::<R, T, O>(values, &flags, op, initial)
}

/// Scan runs described by U32 head flags. The first element implicitly starts
/// a run. Intermediate flags remain on the device; no scheduling-dependent
/// cross-block polling is used.
pub fn scan_by_heads<R, T, O>(
    values: &RudaTensor<R>, heads: &RudaTensor<R>, op: O::RuntimeArg<R>, initial: Option<T>,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    check_type::<R, T>(values)?;
    check_type::<R, u32>(heads)?;
    let size = values.meta.num_elements();
    if size != heads.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if values.device.to_id() != heads.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if size == 0 { return Ok(empty_like(values)); }
    let buffers = [empty_like(values), empty_like(values)];
    let flag_buffers = [empty_like(heads), empty_like(heads)];
    let mut source = values.clone();
    let mut source_flags = heads.clone();
    let dim = RudaDim::new(values.client.properties(), size);
    let mut distance = 1usize;
    let mut pass = 0;
    while distance < size {
        let output = buffers[pass % 2].clone();
        let output_flags = flag_buffers[pass % 2].clone();
        let grid = calculate_ruda_count_elemwise(&values.client, size, dim);
        unsafe {
            scan_step::launch_unchecked::<T, O, R>(
                &values.client, grid, dim, address_type!(source, source_flags, output, output_flags),
                source.into_linear_view(), source_flags.into_linear_view(), output.clone().into_linear_view(),
                output_flags.clone().into_linear_view(), op.clone(), distance,
            );
        }
        source = output;
        source_flags = output_flags;
        distance = distance.saturating_mul(2);
        pass += 1;
    }
    if let Some(initial) = initial {
        let output = empty_like(values);
        let grid = calculate_ruda_count_elemwise(&values.client, size, dim);
        unsafe {
            exclusive_step::launch_unchecked::<T, O, R>(
                &values.client, grid, dim, address_type!(source, heads, output),
                source.into_linear_view(), heads.clone().into_linear_view(), output.clone().into_linear_view(),
                InputScalar::new(initial, values.dtype), op,
            );
        }
        source = output;
    }
    Ok(source)
}
