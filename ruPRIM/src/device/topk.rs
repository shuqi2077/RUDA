use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaSum, RudaSumLaunch, radix::RudaRadixKey};
use super::{RudaPrimitiveError, check_type, scan_threads, scan, reduce};

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn initialise(
    candidates: &mut LinearView<u64, ReadWrite>, selected: &mut LinearView<u64, ReadWrite>,
    state: &mut LinearView<u64, ReadWrite>, k: usize,
) {
    if ABSOLUTE_POS < candidates.shape() { candidates[ABSOLUTE_POS] = 1; selected[ABSOLUTE_POS] = 0; }
    if ABSOLUTE_POS == 0 { state[0] = k as u64; state[1] = 0; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn bit_flags<K: RudaRadixKey>(
    keys: &LinearView<K>, candidates: &LinearView<u64>, flags: &mut LinearView<u64, ReadWrite>,
    #[comptime] bit: u32, #[comptime] largest: bool,
) {
    let index = ABSOLUTE_POS;
    if index < keys.shape() {
        let mut flag = 0u64;
        if candidates[index] != 0 {
            let one = ((K::ordered_bits(keys[index]) >> bit) & 1u64) != 0;
            flag = u64::cast_from(one == largest);
        }
        flags[index] = flag;
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn decision(total: &LinearView<u64>, state: &mut LinearView<u64, ReadWrite>) {
    if ABSOLUTE_POS == 0 {
        let accept = total[0] <= state[0];
        state[1] = u64::cast_from(accept);
        if accept { state[0] = state[0] - total[0]; }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn refine(
    flags: &LinearView<u64>, state: &LinearView<u64>,
    candidates: &mut LinearView<u64, ReadWrite>, selected: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < candidates.shape() {
        if candidates[index] != 0 {
            if state[1] != 0 {
                if flags[index] != 0 { selected[index] = 1; candidates[index] = 0; }
            } else {
                candidates[index] = flags[index];
            }
        }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn resolve_ties(
    candidates: &LinearView<u64>, prefixes: &LinearView<u64>, state: &LinearView<u64>,
    selected: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < selected.shape() {
        if candidates[index] != 0 && prefixes[index] <= state[0] { selected[index] = 1; }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn pack_keys<K: Numeric>(
    keys: &LinearView<K>, selected: &LinearView<u64>, prefixes: &LinearView<u64>, output: &mut LinearView<K, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < keys.shape() {
        if selected[index] != 0 { let rank = prefixes[index] as usize - 1; output[rank] = keys[index]; }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn pack_pairs<K: Numeric, V: Numeric>(
    keys: &LinearView<K>, values: &LinearView<V>, selected: &LinearView<u64>, prefixes: &LinearView<u64>,
    output: &mut LinearView<K, ReadWrite>, output_values: &mut LinearView<V, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index < keys.shape() {
        if selected[index] != 0 {
            let rank = prefixes[index] as usize - 1;
            output[rank] = keys[index];
            output_values[rank] = values[index];
        }
    }
}

fn selected_prefixes<R: Runtime, K: TensorElement + RudaRadixKey>(
    keys: &RudaTensor<R>, k: usize, largest: bool, threads: u32,
) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError> {
    let count = keys.meta.num_elements();
    let allocate = |len| empty_device_dtype(keys.client.clone(), keys.device.clone(), Shape::new([len]), DType::U64);
    let candidates = allocate(count);
    let selected = allocate(count);
    let flags = allocate(count);
    let state = allocate(2);
    let dim = RudaDim::new(keys.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
    unsafe {
        initialise::launch_unchecked::<R>(&keys.client, grid, dim, address_type!(candidates, selected, state),
            candidates.clone().into_linear_view(), selected.clone().into_linear_view(), state.clone().into_linear_view(), k);
    }
    for bit in (0..core::mem::size_of::<K>() as u32 * 8).rev() {
        let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
        unsafe {
            bit_flags::launch_unchecked::<K, R>(&keys.client, grid, dim, address_type!(keys, candidates, flags),
                keys.clone().into_linear_view(), candidates.clone().into_linear_view(), flags.clone().into_linear_view(), bit, largest);
        }
        let total = reduce::reduce::<R, u64, RudaSum>(&flags, 0, RudaSumLaunch::new(), threads)?;
        unsafe {
            decision::launch_unchecked::<R>(&keys.client, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), address_type!(total, state),
                total.into_linear_view(), state.clone().into_linear_view());
            let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
            refine::launch_unchecked::<R>(&keys.client, grid, dim, address_type!(flags, state, candidates, selected),
                flags.clone().into_linear_view(), state.clone().into_linear_view(), candidates.clone().into_linear_view(), selected.clone().into_linear_view());
        }
    }
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&candidates, RudaSumLaunch::new(), threads)?;
    let grid = calculate_ruda_count_elemwise(&keys.client, count, dim);
    unsafe {
        resolve_ties::launch_unchecked::<R>(&keys.client, grid, dim, address_type!(candidates, prefixes, state, selected),
            candidates.into_linear_view(), prefixes.into_linear_view(), state.into_linear_view(), selected.clone().into_linear_view());
    }
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&selected, RudaSumLaunch::new(), threads)?;
    Ok((selected, prefixes))
}

/// Select the largest or smallest K keys by radix refinement. K is capped at
/// input length. Results are not sorted; selected items and cutoff ties retain
/// input order. Selection state and counts remain on the device.
pub fn keys<R: Runtime, K: TensorElement + RudaRadixKey>(
    input: &RudaTensor<R>, k: usize, largest: bool, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    check_type::<R, K>(input)?;
    scan_threads::<R, u64>(input, threads)?;
    let count = input.meta.num_elements();
    let k = k.min(count);
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([k]), input.dtype);
    if k == 0 { return Ok(output); }
    let (selected, prefixes) = selected_prefixes::<R, K>(input, k, largest, threads)?;
    let dim = RudaDim::new(input.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
    unsafe {
        pack_keys::launch_unchecked::<K, R>(&input.client, grid, dim, address_type!(input, selected, prefixes, output),
            input.clone().into_linear_view(), selected.into_linear_view(), prefixes.into_linear_view(), output.clone().into_linear_view());
    }
    Ok(output)
}

pub fn pairs<R: Runtime, K: TensorElement + RudaRadixKey, V: TensorElement>(
    input: &RudaTensor<R>, values: &RudaTensor<R>, k: usize, largest: bool, threads: u32,
) -> Result<(RudaTensor<R>, RudaTensor<R>), RudaPrimitiveError> {
    check_type::<R, K>(input)?;
    check_type::<R, V>(values)?;
    scan_threads::<R, u64>(input, threads)?;
    let count = input.meta.num_elements();
    if count != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let k = k.min(count);
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([k]), input.dtype);
    let output_values = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([k]), values.dtype);
    if k == 0 { return Ok((output, output_values)); }
    let (selected, prefixes) = selected_prefixes::<R, K>(input, k, largest, threads)?;
    let dim = RudaDim::new(input.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
    unsafe {
        pack_pairs::launch_unchecked::<K, V, R>(
            &input.client, grid, dim, address_type!(input, values, selected, prefixes, output, output_values),
            input.clone().into_linear_view(), values.clone().into_linear_view(), selected.into_linear_view(), prefixes.into_linear_view(),
            output.clone().into_linear_view(), output_values.clone().into_linear_view(),
        );
    }
    Ok((output, output_values))
}
