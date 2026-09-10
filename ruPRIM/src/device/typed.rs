use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, element::TensorElement, allocation::empty_device_dtype, layout::address_type};
use ruda_core::tensor::{Shape, element::Element};
use crate::collective::{RudaBinaryOp, RudaCompare, RudaKeyEqual};
use super::{RudaPrimitiveError, check_type, transform::{self, RudaCast, RudaCastLaunch}};

/// Convert integral offset/count descriptors entirely on the device. Input
/// offsets are nonnegative and results must fit the chosen output type.
pub fn indices<R: Runtime, I: TensorElement + Int, O: TensorElement + Int>(input: &RudaTensor<R>)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
{
    transform::unary::<R, I, O, RudaCast>(input, RudaCastLaunch::new())
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn convert_prefix<I: Int, O: Int>(input: &LinearView<I>, count: &LinearView<u64>, output: &mut LinearView<O, ReadWrite>) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        if (index as u64) < count[0] { output[index] = O::cast_from(input[index]); }
    }
}

/// Convert only the device-counted valid prefix, never loading an uninitialized
/// capacity tail. Count must not exceed input length; converted values must fit O.
pub fn indices_prefix<R: Runtime, I: TensorElement + Int, O: TensorElement + Int>(input: &RudaTensor<R>, count: &RudaTensor<R>)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
{
    check_type::<R, I>(input)?;
    check_type::<R, u64>(count)?;
    if count.meta.num_elements() != 1 { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != count.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let length = input.meta.num_elements();
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([length]), <O as Element>::dtype());
    if length > 0 {
        let dim = RudaDim::new(input.client.properties(), length);
        let grid = calculate_ruda_count_elemwise(&input.client, length, dim);
        unsafe { convert_prefix::launch_unchecked::<I, O, R>(&input.client, grid, dim, address_type!(input, count, output),
            input.clone().into_linear_view(), count.clone().into_linear_view(), output.clone().into_linear_view()); }
    }
    Ok(output)
}

pub fn reduce<R, T, U, O>(input: &RudaTensor<R>, initial: U, op: O::RuntimeArg<R>, threads: u32)
    -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement, O: RudaBinaryOp<U> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    super::reduce::transform_reduce::<R, T, U, RudaCast, O>(input, RudaCastLaunch::new(), initial, op, threads)
}

pub fn reduce_by_key<R, K, T, U, O, E>(keys: &RudaTensor<R>, values: &RudaTensor<R>,
    op: O::RuntimeArg<R>, equal: E::RuntimeArg<R>, threads: u32) -> Result<super::reduce::RudaKeyReduction<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, T: TensorElement, U: TensorElement,
    O: RudaBinaryOp<U> + LaunchArg, O::RuntimeArg<R>: Clone, E: RudaKeyEqual<K> + LaunchArg,
{
    check_type::<R, K>(keys)?;
    if keys.meta.num_elements() != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let values = transform::unary::<R, T, U, RudaCast>(values, RudaCastLaunch::new())?;
    super::reduce::by_key::<R, K, U, O, E>(keys, &values, op, equal, threads)
}

/// initial=None is inclusive; Some(initial) is exclusive within each equal-key run.
pub fn scan_by_key<R, K, T, U, O, E>(keys: &RudaTensor<R>, values: &RudaTensor<R>,
    op: O::RuntimeArg<R>, equal: E::RuntimeArg<R>, initial: Option<U>) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, T: TensorElement, U: TensorElement,
    O: RudaBinaryOp<U> + LaunchArg, O::RuntimeArg<R>: Clone, E: RudaKeyEqual<K> + LaunchArg,
{
    check_type::<R, K>(keys)?;
    if keys.meta.num_elements() != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let values = transform::unary::<R, T, U, RudaCast>(values, RudaCastLaunch::new())?;
    super::segmented::scan_by_key::<R, K, U, O, E>(keys, &values, op, equal, initial)
}

/// Scan in accumulator type U, converting each input value before the first
/// operation. This does not reduce in T and convert only the final result.
pub fn scan<R, T, U, O>(input: &RudaTensor<R>, initial: Option<U>, exclusive: bool,
    op: O::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement, O: RudaBinaryOp<U> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    if exclusive && initial.is_none() { return Err(RudaPrimitiveError::Configuration("exclusive scan requires an initial value")); }
    let values = transform::unary::<R, T, U, RudaCast>(input, RudaCastLaunch::new())?;
    match (exclusive, initial) {
        (true, Some(initial)) => super::scan::exclusive_scan::<R, U, O>(&values, initial, op, threads),
        (false, Some(initial)) => super::scan::inclusive_scan_init::<R, U, O>(&values, initial, op, threads),
        _ => super::scan::inclusive_scan::<R, U, O>(&values, op, threads),
    }
}

pub fn segmented_reduce<R, T, U, I, O>(input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    initial: U, op: O::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement, I: TensorElement + Int, O: RudaBinaryOp<U> + LaunchArg,
{
    check_type::<R, I>(begins)?;
    check_type::<R, I>(ends)?;
    if begins.meta.num_elements() != ends.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != begins.device.to_id() || input.device.to_id() != ends.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let values = transform::unary::<R, T, U, RudaCast>(input, RudaCastLaunch::new())?;
    let begins = indices::<R, I, u64>(begins)?;
    let ends = indices::<R, I, u64>(ends)?;
    super::segments::reduce::<R, U, O>(&values, &begins, &ends, initial, op, threads)
}

pub fn segmented_scan_into<R, T, U, I, J, O>(input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    output_begins: &RudaTensor<R>, output: &RudaTensor<R>, initial: Option<U>, exclusive: bool,
    op: O::RuntimeArg<R>, threads: u32) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement, I: TensorElement + Int, J: TensorElement + Int, O: RudaBinaryOp<U> + LaunchArg,
{
    check_type::<R, I>(begins)?;
    check_type::<R, I>(ends)?;
    check_type::<R, J>(output_begins)?;
    check_type::<R, U>(output)?;
    let count = begins.meta.num_elements();
    if count != ends.meta.num_elements() || count != output_begins.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    for tensor in [begins, ends, output_begins, output] {
        if tensor.device.to_id() != input.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    if exclusive && initial.is_none() { return Err(RudaPrimitiveError::Configuration("exclusive scan requires an initial value")); }
    let values = transform::unary::<R, T, U, RudaCast>(input, RudaCastLaunch::new())?;
    let begins = indices::<R, I, u64>(begins)?;
    let ends = indices::<R, I, u64>(ends)?;
    let output_begins = indices::<R, J, u64>(output_begins)?;
    super::segments::scan_into::<R, U, O>(&values, &begins, &ends, &output_begins, output, initial, exclusive, op, threads)
}

pub fn segmented_sort_keys<R, K, I, C>(input: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    compare: C::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, I: TensorElement + Int, C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(input)?;
    if begins.meta.num_elements() != ends.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != begins.device.to_id() || input.device.to_id() != ends.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let begins = indices::<R, I, u64>(begins)?;
    let ends = indices::<R, I, u64>(ends)?;
    super::segmented_sort::sort_keys::<R, K, C>(input, &begins, &ends, compare, threads)
}

pub fn segmented_sort_pairs<R, K, V, I, C>(keys: &RudaTensor<R>, values: &RudaTensor<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    compare: C::RuntimeArg<R>, threads: u32) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError>
where R: Runtime, K: TensorElement, V: TensorElement, I: TensorElement + Int,
    C: RudaCompare<K> + LaunchArg, C::RuntimeArg<R>: Clone,
{
    check_type::<R, K>(keys)?;
    check_type::<R, V>(values)?;
    if begins.meta.num_elements() != ends.meta.num_elements() || keys.meta.num_elements() != values.meta.num_elements() {
        return Err(RudaPrimitiveError::Length);
    }
    for tensor in [values, begins, ends] {
        if tensor.device.to_id() != keys.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    let begins = indices::<R, I, u64>(begins)?;
    let ends = indices::<R, I, u64>(ends)?;
    super::segmented_sort::sort_pairs::<R, K, V, C>(keys, values, &begins, &ends, compare, threads)
}
