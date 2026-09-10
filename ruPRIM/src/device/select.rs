use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaSum, RudaSumLaunch, RudaKeyEqual, RudaKeyEqualExpand};
use super::{RudaPrimitiveError, check_type, empty_like, scan};

#[ruda]
pub trait RudaPredicate<T: RudaType>: RudaType {
    fn test(&self, value: T) -> bool;
}

/// The valid output prefix is described by `count`, a one-element U64 device
/// tensor. Capacity is the input length; no count readback or host sync occurs.
pub struct RudaSelection<R: Runtime> {
    pub values: RudaTensor<R>,
    pub count: RudaTensor<R>,
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn predicate_flags<T: Numeric, P: RudaPredicate<T> + LaunchArg>(
    input: &LinearView<T>, flags: &mut LinearView<u64, ReadWrite>, predicate: &P,
) {
    let index = ABSOLUTE_POS;
    if index < input.shape() { flags[index] = u64::cast_from(predicate.test(input[index])); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn convert_flags<F: Numeric>(input: &LinearView<F>, flags: &mut LinearView<u64, ReadWrite>) {
    let index = ABSOLUTE_POS;
    if index < input.shape() { flags[index] = u64::cast_from(input[index] != F::from_int(0)); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn unique_flags<T: Numeric, E: RudaKeyEqual<T> + LaunchArg>(
    input: &LinearView<T>, flags: &mut LinearView<u64, ReadWrite>, equal: &E,
) {
    let index = ABSOLUTE_POS;
    if index >= input.shape() { terminate!(); }
    let mut head = true;
    if index > 0 { head = !equal.equal(input[index - 1], input[index]); }
    flags[index] = u64::cast_from(head);
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn scatter<T: Numeric>(
    input: &LinearView<T>, flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output: &mut LinearView<T, ReadWrite>, count: &mut LinearView<u64, ReadWrite>,
    #[comptime] partition: bool,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut total = 0u64;
        if input.shape() > 0 { total = prefixes[input.shape() - 1]; }
        count[0] = total;
    }
    if index < input.shape() {
        let prefix = prefixes[index] as usize;
        if flags[index] != 0 {
            output[prefix - 1] = input[index];
        } else {
            if partition {
                output[input.shape() - 1 - (index - prefix)] = input[index];
            }
        }
    }
}

pub(crate) fn compact<R: Runtime, T: TensorElement>(
    input: &RudaTensor<R>, flags: &RudaTensor<R>, threads: u32, partition: bool,
) -> Result<RudaSelection<R>, RudaPrimitiveError> {
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(flags, RudaSumLaunch::new(), threads)?;
    let values = empty_like(input);
    let count = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([1]), DType::U64);
    let work = input.meta.num_elements().max(1);
    let dim = RudaDim::new(input.client.properties(), work);
    let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
    unsafe {
        scatter::launch_unchecked::<T, R>(
            &input.client, grid, dim, address_type!(input, flags, prefixes, values, count),
            input.clone().into_linear_view(), flags.clone().into_linear_view(), prefixes.into_linear_view(),
            values.clone().into_linear_view(), count.clone().into_linear_view(), partition,
        );
    }
    Ok(RudaSelection { values, count })
}

/// Select or partition by explicit flags. Nonzero flags select an item.
/// Selection is stable; partition places rejected items in reverse input order.
pub fn flagged<R: Runtime, T: TensorElement, F: TensorElement>(
    input: &RudaTensor<R>, flags: &RudaTensor<R>, threads: u32, partition: bool,
) -> Result<RudaSelection<R>, RudaPrimitiveError> {
    check_type::<R, T>(input)?;
    check_type::<R, F>(flags)?;
    let size = input.meta.num_elements();
    if size != flags.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != flags.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let normalized = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([size]), DType::U64);
    if size > 0 {
        let dim = RudaDim::new(input.client.properties(), size);
        let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
        unsafe {
            convert_flags::launch_unchecked::<F, R>(
                &input.client, grid, dim, address_type!(flags, normalized),
                flags.clone().into_linear_view(), normalized.clone().into_linear_view(),
            );
        }
    }
    compact::<R, T>(input, &normalized, threads, partition)
}

pub fn select_if<R, T, P>(input: &RudaTensor<R>, predicate: P::RuntimeArg<R>, threads: u32, partition: bool) -> Result<RudaSelection<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, P: RudaPredicate<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    let size = input.meta.num_elements();
    let flags = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([size]), DType::U64);
    if size > 0 {
        let dim = RudaDim::new(input.client.properties(), size);
        let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
        unsafe {
            predicate_flags::launch_unchecked::<T, P, R>(
                &input.client, grid, dim, address_type!(input, flags),
                input.clone().into_linear_view(), flags.clone().into_linear_view(), predicate,
            );
        }
    }
    compact::<R, T>(input, &flags, threads, partition)
}

/// Keep the first item of every adjacent equal run; this does not sort input.
pub fn unique<R, T, E>(input: &RudaTensor<R>, equal: E::RuntimeArg<R>, threads: u32) -> Result<RudaSelection<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, E: RudaKeyEqual<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    let size = input.meta.num_elements();
    let flags = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([size]), DType::U64);
    if size > 0 {
        let dim = RudaDim::new(input.client.properties(), size);
        let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
        unsafe {
            unique_flags::launch_unchecked::<T, E, R>(
                &input.client, grid, dim, address_type!(input, flags),
                input.clone().into_linear_view(), flags.clone().into_linear_view(), equal,
            );
        }
    }
    compact::<R, T>(input, &flags, threads, false)
}

pub struct RudaPairSelection<R: Runtime> {
    pub keys: RudaTensor<R>,
    pub values: RudaTensor<R>,
    pub count: RudaTensor<R>,
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn scatter_unique_pairs<K: Numeric, V: Numeric>(
    keys: &LinearView<K>, values: &LinearView<V>, flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output_keys: &mut LinearView<K, ReadWrite>, output_values: &mut LinearView<V, ReadWrite>, count: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut total = 0u64;
        if keys.shape() > 0 { total = prefixes[keys.shape() - 1]; }
        count[0] = total;
    }
    if index < keys.shape() {
        if flags[index] != 0 {
            let rank = prefixes[index] as usize - 1;
            output_keys[rank] = keys[index];
            output_values[rank] = values[index];
        }
    }
}

/// Keep the first key/value pair of each adjacent equivalent key run.
pub fn unique_by_key<R, K, V, E>(
    keys: &RudaTensor<R>, values: &RudaTensor<R>, equal: E::RuntimeArg<R>, threads: u32,
) -> Result<RudaPairSelection<R>, RudaPrimitiveError>
where R: Runtime, K: TensorElement, V: TensorElement, E: RudaKeyEqual<K> + LaunchArg,
{
    check_type::<R, K>(keys)?;
    check_type::<R, V>(values)?;
    let size = keys.meta.num_elements();
    if size != values.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.device.to_id() != values.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let flags = empty_device_dtype(keys.client.clone(), keys.device.clone(), Shape::new([size]), DType::U64);
    let dim = RudaDim::new(keys.client.properties(), size.max(1));
    if size > 0 {
        let grid = calculate_ruda_count_elemwise(&keys.client, size, dim);
        unsafe {
            unique_flags::launch_unchecked::<K, E, R>(&keys.client, grid, dim, address_type!(keys, flags),
                keys.clone().into_linear_view(), flags.clone().into_linear_view(), equal);
        }
    }
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
    let output_keys = empty_like(keys);
    let output_values = empty_like(values);
    let count = empty_device_dtype(keys.client.clone(), keys.device.clone(), Shape::new([1]), DType::U64);
    let grid = calculate_ruda_count_elemwise(&keys.client, size.max(1), dim);
    unsafe {
        scatter_unique_pairs::launch_unchecked::<K, V, R>(
            &keys.client, grid, dim, address_type!(keys, values, flags, prefixes, output_keys, output_values, count),
            keys.clone().into_linear_view(), values.clone().into_linear_view(), flags.into_linear_view(), prefixes.into_linear_view(),
            output_keys.clone().into_linear_view(), output_values.clone().into_linear_view(), count.clone().into_linear_view(),
        );
    }
    Ok(RudaPairSelection { keys: output_keys, values: output_values, count })
}

/// Select data using a predicate applied to the separate flag sequence.
pub fn flagged_if<R, T, F, P>(
    input: &RudaTensor<R>, flags: &RudaTensor<R>, predicate: P::RuntimeArg<R>, threads: u32,
) -> Result<RudaSelection<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, F: TensorElement, P: RudaPredicate<F> + LaunchArg,
{
    check_type::<R, T>(input)?;
    check_type::<R, F>(flags)?;
    let size = input.meta.num_elements();
    if size != flags.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != flags.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let normalized = empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([size]), DType::U64);
    if size > 0 {
        let dim = RudaDim::new(input.client.properties(), size);
        let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
        unsafe {
            predicate_flags::launch_unchecked::<F, P, R>(&input.client, grid, dim, address_type!(flags, normalized),
                flags.clone().into_linear_view(), normalized.clone().into_linear_view(), predicate);
        }
    }
    compact::<R, T>(input, &normalized, threads, false)
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn write_selection<T: Numeric>(values: &LinearView<T>, count: &LinearView<u64>, output: &mut LinearView<T, ReadWrite>) {
    let index = ABSOLUTE_POS;
    if index < output.shape() {
        if (index as u64) < count[0] { output[index] = values[index]; }
    }
}

/// Write the valid selection prefix into an existing buffer, preserving its
/// tail. The output must have capacity for the selection and not partially
/// overlap its temporary values. Returns the device count without readback.
pub fn write_into<R: Runtime, T: TensorElement>(
    selection: RudaSelection<R>, output: &RudaTensor<R>,
) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    check_type::<R, T>(&selection.values)?;
    check_type::<R, T>(output)?;
    check_type::<R, u64>(&selection.count)?;
    if selection.count.meta.num_elements() != 1 || output.meta.num_elements() < selection.values.meta.num_elements() {
        return Err(RudaPrimitiveError::Length);
    }
    if output.device.to_id() != selection.values.device.to_id() || output.device.to_id() != selection.count.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let size = output.meta.num_elements();
    if size > 0 {
        let dim = RudaDim::new(output.client.properties(), size);
        let grid = calculate_ruda_count_elemwise(&output.client, size, dim);
        unsafe {
            write_selection::launch_unchecked::<T, R>(&output.client, grid, dim, address_type!((selection.values), (selection.count), output),
                selection.values.into_linear_view(), selection.count.clone().into_linear_view(), output.clone().into_linear_view());
        }
    }
    Ok(selection.count)
}

pub fn select_if_in_place<R, T, P>(input: &RudaTensor<R>, predicate: P::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, P: RudaPredicate<T> + LaunchArg,
{
    let selected = select_if::<R, T, P>(input, predicate, threads, false)?;
    write_into::<R, T>(selected, input)
}

pub fn flagged_if_in_place<R, T, F, P>(
    input: &RudaTensor<R>, flags: &RudaTensor<R>, predicate: P::RuntimeArg<R>, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, F: TensorElement, P: RudaPredicate<F> + LaunchArg,
{
    let selected = flagged_if::<R, T, F, P>(input, flags, predicate, threads)?;
    write_into::<R, T>(selected, input)
}

pub fn flagged_in_place<R: Runtime, T: TensorElement, F: TensorElement>(
    input: &RudaTensor<R>, flags: &RudaTensor<R>, threads: u32,
) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    let selected = flagged::<R, T, F>(input, flags, threads, false)?;
    write_into::<R, T>(selected, input)
}

pub fn unique_in_place<R, T, E>(input: &RudaTensor<R>, equal: E::RuntimeArg<R>, threads: u32) -> Result<RudaTensor<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, E: RudaKeyEqual<T> + LaunchArg,
{
    let selected = unique::<R, T, E>(input, equal, threads)?;
    write_into::<R, T>(selected, input)
}
