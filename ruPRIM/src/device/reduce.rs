use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{Shape, DType};
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand, RudaKeyEqual, RudaSum, RudaSumLaunch};
use super::{RudaPrimitiveError, check_type, scan_threads, empty_like, segmented, scan, transform::{self, RudaCast, RudaCastLaunch}};

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn tile_reduce<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    input: &LinearView<T>, output: &mut LinearView<T, ReadWrite>, op: &O,
    #[comptime] threads: usize,
) {
    let start = RUDA_POS as usize * threads;
    if start >= input.shape() { terminate!(); }
    let valid = min(threads, input.shape() - start);
    let index = start + UNIT_POS as usize;
    let mut value = T::from_int(0);
    if index < input.shape() { value = input[index]; }
    let mut local = Array::<T>::new(1usize);
    local[0] = value;
    let mut scratch = SharedMemory::<T>::new(threads);
    let result = crate::block::reduce::<T, O>(&local, &mut scratch, op, valid, threads, 1usize);
    if UNIT_POS == 0 { output[RUDA_POS as usize] = result; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn seed<T: Numeric, O: RudaBinaryOp<T> + LaunchArg>(
    input: &LinearView<T>, output: &mut LinearView<T, ReadWrite>, initial: InputScalar, op: &O,
) {
    if ABSOLUTE_POS == 0 {
        let mut value = initial.get::<T>();
        if input.shape() > 0 { value = op.combine(value, input[0]); }
        output[0] = value;
    }
}

/// Device-wide ordered reduction with an explicit initial value. Empty input
/// produces the initial value. Partial results use the same element type.
pub fn reduce<R, T, O>(input: &RudaTensor<R>, initial: T, op: O::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    check_type::<R, T>(input)?;
    scan_threads::<R, T>(input, threads)?;
    let mut source = input.clone();
    let dim = RudaDim::new_1d(threads);
    while source.meta.num_elements() > 1 {
        let count = source.meta.num_elements();
        let partial_count = count.div_ceil(threads as usize);
        let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([partial_count]), input.dtype);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            tile_reduce::launch_unchecked::<T, O, R>(
                &input.client, grid, dim, address_type!(source, output), source.into_linear_view(),
                output.clone().into_linear_view(), op.clone(), threads as usize,
            );
        }
        source = output;
    }
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([1]), input.dtype);
    unsafe {
        seed::launch_unchecked::<T, O, R>(
            &input.client, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), address_type!(source, output),
            source.into_linear_view(), output.clone().into_linear_view(), InputScalar::new(initial, input.dtype), op,
        );
    }
    Ok(output)
}

pub struct RudaKeyReduction<R: Runtime> {
    pub keys: RudaTensor<R>,
    pub aggregates: RudaTensor<R>,
    pub count: RudaTensor<R>,
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn compact_key_reduction<K: Numeric, T: Numeric>(
    keys: &LinearView<K>, scanned: &LinearView<T>, heads: &LinearView<u32>, prefixes: &LinearView<u64>,
    output_keys: &mut LinearView<K, ReadWrite>, aggregates: &mut LinearView<T, ReadWrite>, count: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut total = 0u64;
        if keys.shape() > 0 { total = prefixes[keys.shape() - 1]; }
        count[0] = total;
    }
    if index < keys.shape() {
        let rank = prefixes[index] as usize - 1;
        if heads[index] != 0 { output_keys[rank] = keys[index]; }
        let mut tail = index + 1 == keys.shape();
        if index + 1 < keys.shape() { tail = heads[index + 1] != 0; }
        if tail { aggregates[rank] = scanned[index]; }
    }
}

/// Reduce adjacent equivalent key runs without an identity element. The first
/// key represents each run; only `count` leading outputs are valid.
pub fn by_key<R, K, T, O, E>(
    keys: &RudaTensor<R>, values: &RudaTensor<R>, op: O::RuntimeArg<R>, equal: E::RuntimeArg<R>, threads: u32,
) -> Result<RudaKeyReduction<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, T: TensorElement,
    O: RudaBinaryOp<T> + LaunchArg, O::RuntimeArg<R>: Clone, E: RudaKeyEqual<K> + LaunchArg,
{
    check_type::<R, T>(values)?;
    if keys.meta.num_elements() != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let heads = segmented::key_heads::<R, K, E>(keys, equal)?;
    let flags = transform::unary::<R, u32, u64, RudaCast>(&heads, RudaCastLaunch::new())?;
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
    let scanned = segmented::scan_by_heads::<R, T, O>(values, &heads, op, None)?;
    let output_keys = empty_like(keys);
    let aggregates = empty_like(values);
    let count = empty_device_dtype(keys.client.clone(), keys.device.clone(), Shape::new([1]), DType::U64);
    let work = keys.meta.num_elements().max(1);
    let dim = RudaDim::new(keys.client.properties(), work);
    let grid = calculate_ruda_count_elemwise(&keys.client, work, dim);
    unsafe {
        compact_key_reduction::launch_unchecked::<K, T, R>(
            &keys.client, grid, dim, address_type!(keys, scanned, heads, prefixes, output_keys, aggregates, count),
            keys.clone().into_linear_view(), scanned.into_linear_view(), heads.into_linear_view(), prefixes.into_linear_view(),
            output_keys.clone().into_linear_view(), aggregates.clone().into_linear_view(), count.clone().into_linear_view(),
        );
    }
    Ok(RudaKeyReduction { keys: output_keys, aggregates, count })
}

pub fn transform_reduce<R, T, U, F, O>(
    input: &RudaTensor<R>, transform: F::RuntimeArg<R>, initial: U, op: O::RuntimeArg<R>, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, U: TensorElement, F: transform::RudaUnaryOp<T, U> + LaunchArg,
    O: RudaBinaryOp<U> + LaunchArg, O::RuntimeArg<R>: Clone,
{
    let transformed = transform::unary::<R, T, U, F>(input, transform)?;
    reduce::<R, U, O>(&transformed, initial, op, threads)
}
