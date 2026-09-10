use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{TensorMetadata, element::Element};
use super::{RudaPrimitiveError, check_type};
use super::select::{RudaPredicate, RudaPredicateExpand};

pub use crate::collective::{RudaUnaryOp, RudaUnaryOpExpand};

#[ruda]
pub trait RudaBinaryTransform<T: RudaType, U: RudaType, V: RudaType>: RudaType {
    fn apply(&self, left: T, right: U) -> V;
}

#[ruda]
pub trait RudaGenerator<T: RudaType>: RudaType {
    fn generate(&mut self) -> T;
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn fill_kernel<T: Numeric>(output: &mut LinearView<T, ReadWrite>, value: InputScalar) {
    if ABSOLUTE_POS < output.shape() { output[ABSOLUTE_POS] = value.get::<T>(); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn generate_kernel<T: Numeric, G: RudaGenerator<T> + LaunchArg>(output: &mut LinearView<T, ReadWrite>, generator: &mut G) {
    if ABSOLUTE_POS < output.shape() { output[ABSOLUTE_POS] = generator.generate(); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn unary_if_kernel<T: Numeric, U: Numeric, O: RudaUnaryOp<T, U> + LaunchArg, P: RudaPredicate<T> + LaunchArg>(
    input: &LinearView<T>, output: &mut LinearView<U, ReadWrite>, op: &O, predicate: &P,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() {
        let value = input[index];
        if predicate.test(value) { output[index] = op.apply(value); }
    }
}

pub fn fill<R: Runtime, T: TensorElement>(output: &RudaTensor<R>, value: T) -> Result<(), RudaPrimitiveError> {
    check_type::<R, T>(output)?;
    let count = output.meta.num_elements();
    if count > 0 {
        let dim = RudaDim::new(output.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&output.client, count, dim);
        unsafe {
            fill_kernel::launch_unchecked::<T, R>(&output.client, grid, dim, address_type!(output),
                output.clone().into_linear_view(), InputScalar::new(value, output.dtype));
        }
    }
    Ok(())
}

pub fn generate<R, T, G>(output: &RudaTensor<R>, generator: G::RuntimeArg<R>) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: TensorElement, G: RudaGenerator<T> + LaunchArg,
{
    check_type::<R, T>(output)?;
    let count = output.meta.num_elements();
    if count > 0 {
        let dim = RudaDim::new(output.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&output.client, count, dim);
        unsafe {
            generate_kernel::launch_unchecked::<T, G, R>(&output.client, grid, dim, address_type!(output),
                output.clone().into_linear_view(), generator);
        }
    }
    Ok(())
}

/// Transform selected input elements, leaving other output elements unchanged.
/// Input/output may alias exactly for identical element types, but must not
/// partially overlap.
pub fn unary_if_into<R, T, U, O, P>(
    input: &RudaTensor<R>, output: &RudaTensor<R>, op: O::RuntimeArg<R>, predicate: P::RuntimeArg<R>,
) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement,
    O: RudaUnaryOp<T, U> + LaunchArg, P: RudaPredicate<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    check_type::<R, U>(output)?;
    let count = input.meta.num_elements();
    if count != output.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != output.device.to_id() { return Err(RudaPrimitiveError::Device); }
    if count > 0 {
        let dim = RudaDim::new(input.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            unary_if_kernel::launch_unchecked::<T, U, O, P, R>(&input.client, grid, dim, address_type!(input, output),
                input.clone().into_linear_view(), output.clone().into_linear_view(), op, predicate);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, RudaType, RudaLaunch)]
pub struct RudaCast;

impl<R: Runtime> Clone for RudaCastLaunch<R> {
    fn clone(&self) -> Self { Self::new() }
}

#[ruda]
impl<T: Numeric, U: Numeric> RudaUnaryOp<T, U> for RudaCast {
    fn apply(&self, value: T) -> U { U::cast_from(value) }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn unary_kernel<T: Numeric, U: Numeric, O: RudaUnaryOp<T, U> + LaunchArg>(
    input: &LinearView<T>, output: &mut LinearView<U, ReadWrite>, op: &O,
) {
    let index = ABSOLUTE_POS;
    if index < output.shape() { output[index] = op.apply(input[index]); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn binary_kernel<T: Numeric, U: Numeric, V: Numeric, O: RudaBinaryTransform<T, U, V> + LaunchArg>(
    left: &LinearView<T>, right: &LinearView<U>, output: &mut LinearView<V, ReadWrite>, op: &O,
) {
    let index = ABSOLUTE_POS;
    if index < output.shape() { output[index] = op.apply(left[index], right[index]); }
}

pub fn unary<R, T, U, O>(input: &RudaTensor<R>, op: O::RuntimeArg<R>) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement, O: RudaUnaryOp<T, U> + LaunchArg,
{
    check_type::<R, T>(input)?;
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), input.shape(), <U as Element>::dtype());
    let count = input.meta.num_elements();
    if count == 0 { return Ok(output); }
    let dim = RudaDim::new(input.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
    unsafe {
        unary_kernel::launch_unchecked::<T, U, O, R>(
            &input.client, grid, dim, address_type!(input, output),
            input.clone().into_linear_view(), output.clone().into_linear_view(), op,
        );
    }
    Ok(output)
}

pub fn binary<R, T, U, V, O>(left: &RudaTensor<R>, right: &RudaTensor<R>, op: O::RuntimeArg<R>) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement, V: TensorElement, O: RudaBinaryTransform<T, U, V> + LaunchArg,
{
    check_type::<R, T>(left)?;
    check_type::<R, U>(right)?;
    let count = left.meta.num_elements();
    if count != right.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if left.device.to_id() != right.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let output = empty_device_dtype(left.client.clone(), left.device.clone(), left.shape(), <V as Element>::dtype());
    if count == 0 { return Ok(output); }
    let dim = RudaDim::new(left.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&left.client, count, dim);
    unsafe {
        binary_kernel::launch_unchecked::<T, U, V, O, R>(
            &left.client, grid, dim, address_type!(left, right, output),
            left.clone().into_linear_view(), right.clone().into_linear_view(), output.clone().into_linear_view(), op,
        );
    }
    Ok(output)
}
