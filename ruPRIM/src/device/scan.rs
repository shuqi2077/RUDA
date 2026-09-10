use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{Shape, TensorMetadata};
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};
use super::{RudaPrimitiveError, check_type, empty_like, scan_threads};

/// Caller-owned output, including exact in-place scans. Inclusive scans may
/// optionally be seeded; exclusive scans always require an initial value.
pub fn scan_into<R, T, O>(input: &RudaTensor<R>, output: &RudaTensor<R>, initial: Option<T>,
    exclusive: bool, op: O::RuntimeArg<R>, threads: u32) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    check_type::<R, T>(output)?;
    if input.meta.num_elements() != output.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != output.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let scanned = match (exclusive, initial) {
        (true, None) => return Err(RudaPrimitiveError::Configuration("exclusive scan requires an initial value")),
        (true, Some(initial)) => exclusive_scan::<R, T, O>(input, initial, op, threads)?,
        (false, Some(initial)) => inclusive_scan_init::<R, T, O>(input, initial, op, threads)?,
        (false, None) => inclusive_scan::<R, T, O>(input, op, threads)?,
    };
    super::iteration::copy_into::<R, T>(&scanned, output)
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn tile_scan<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    input: &LinearView<T>,
    output: &mut LinearView<T, ReadWrite>,
    totals: &mut LinearView<T, ReadWrite>,
    op: &O,
    #[comptime] threads: usize,
) {
    let start = RUDA_POS as usize * threads;
    if start >= input.shape() { terminate!(); }
    let valid = min(threads, input.shape() - start);
    let index = start + UNIT_POS as usize;
    let mut local = Array::<T>::new(1usize);
    let mut result = Array::<T>::new(1usize);
    let mut scratch = SharedMemory::<T>::new(threads);
    let mut value = T::from_int(0);
    if index < input.shape() { value = input[index]; }
    local[0] = value;
    crate::block::inclusive_scan::<T, O>(&local, &mut result, &mut scratch, op, valid, threads, 1usize);
    if index < output.shape() { output[index] = result[0]; }
    if UNIT_POS == 0 { totals[RUDA_POS as usize] = scratch[valid - 1]; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn apply_prefix<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    output: &mut LinearView<T, ReadWrite>,
    totals: &LinearView<T>,
    op: &O,
    #[comptime] threads: usize,
) {
    let index = ABSOLUTE_POS;
    if index >= output.shape() { terminate!(); }
    let tile = index / threads;
    if tile > 0 { output[index] = op.combine(totals[tile - 1], output[index]); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn make_exclusive<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    inclusive: &LinearView<T>,
    output: &mut LinearView<T, ReadWrite>,
    initial: InputScalar,
    op: &O,
) {
    let index = ABSOLUTE_POS;
    if index >= output.shape() { terminate!(); }
    let mut value = initial.get::<T>();
    if index > 0 { value = op.combine(value, inclusive[index - 1]); }
    output[index] = value;
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn seed_inclusive<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    output: &mut LinearView<T, ReadWrite>, initial: InputScalar, op: &O,
) {
    if ABSOLUTE_POS < output.shape() { output[ABSOLUTE_POS] = op.combine(initial.get::<T>(), output[ABSOLUTE_POS]); }
}

pub fn inclusive_scan_init<R, T, O>(
    input: &RudaTensor<R>, initial: T, op: O::RuntimeArg<R>, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    let output = inclusive_scan::<R, T, O>(input, op.clone(), threads)?;
    let count = input.meta.num_elements();
    if count > 0 {
        let dim = RudaDim::new(input.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            seed_inclusive::launch_unchecked::<T, O, R>(&input.client, grid, dim, address_type!(output),
                output.clone().into_linear_view(), InputScalar::new(initial, input.dtype), op);
        }
    }
    Ok(output)
}

/// Inclusive scan of the tensor's logical flattened order. Each hierarchy
/// level is a separate kernel dispatch, requiring no inter-block spin waits.
/// `threads` selects the tile width and must meet the device's block limits.
pub fn inclusive_scan<R, T, O>(
    input: &RudaTensor<R>,
    op: O::RuntimeArg<R>,
    threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where
    R: Runtime,
    T: TensorElement,
    O: RudaBinaryOp<T> + LaunchArg,
    O::RuntimeArg<R>: Clone,
{
    check_type::<R, T>(input)?;
    scan_threads::<R, T>(input, threads)?;
    let output = empty_like(input);
    let count = input.meta.num_elements();
    if count == 0 { return Ok(output); }
    let tile_count = count.div_ceil(threads as usize);
    let totals = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([tile_count]), input.dtype);
    let dim = RudaDim::new_1d(threads);
    let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
    unsafe {
        tile_scan::launch_unchecked::<T, O, R>(
            &input.client, grid, dim, address_type!(input, output, totals),
            input.clone().into_linear_view(), output.clone().into_linear_view(),
            totals.clone().into_linear_view(), op.clone(), threads as usize,
        );
    }
    if tile_count > 1 {
        let prefixes = inclusive_scan::<R, T, O>(&totals, op.clone(), threads)?;
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            apply_prefix::launch_unchecked::<T, O, R>(
                &input.client, grid, dim, address_type!(output, prefixes),
                output.clone().into_linear_view(), prefixes.into_linear_view(), op,
                threads as usize,
            );
        }
    }
    Ok(output)
}

/// Exclusive scan seeded by `initial`, without mutating input or reading
/// intermediate results back to the host.
pub fn exclusive_scan<R, T, O>(
    input: &RudaTensor<R>,
    initial: T,
    op: O::RuntimeArg<R>,
    threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where
    R: Runtime,
    T: TensorElement,
    O: RudaBinaryOp<T> + LaunchArg,
    O::RuntimeArg<R>: Clone,
{
    let inclusive = inclusive_scan::<R, T, O>(input, op.clone(), threads)?;
    let output = empty_like(input);
    let count = input.meta.num_elements();
    if count == 0 { return Ok(output); }
    let dim = RudaDim::new_1d(threads);
    let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
    unsafe {
        make_exclusive::launch_unchecked::<T, O, R>(
            &input.client, grid, dim, address_type!(inclusive, output),
            inclusive.into_linear_view(), output.clone().into_linear_view(),
            InputScalar::new(initial, input.dtype), op,
        );
    }
    Ok(output)
}
