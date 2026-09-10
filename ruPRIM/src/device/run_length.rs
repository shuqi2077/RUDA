use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaKeyEqual, RudaSum, RudaSumLaunch};
use super::{RudaPrimitiveError, empty_like, segmented, scan, transform::{self, RudaCast, RudaCastLaunch}};

/// Device-resident run records. Only the prefix specified by `count` is valid.
/// Offsets, lengths and the single run count are U64 tensors.
pub struct RudaRuns<R: Runtime> {
    pub values: RudaTensor<R>,
    pub offsets: RudaTensor<R>,
    pub lengths: RudaTensor<R>,
    pub count: RudaTensor<R>,
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn scatter_heads<T: Numeric>(
    input: &LinearView<T>, heads: &LinearView<u32>, prefixes: &LinearView<u64>,
    values: &mut LinearView<T, ReadWrite>, offsets: &mut LinearView<u64, ReadWrite>, count: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut total = 0u64;
        if input.shape() > 0 { total = prefixes[input.shape() - 1]; }
        count[0] = total;
    }
    if index < input.shape() {
        if heads[index] != 0 {
            let rank = prefixes[index] as usize - 1;
            values[rank] = input[index];
            offsets[rank] = index as u64;
        }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn lengths_kernel(offsets: &LinearView<u64>, count: &LinearView<u64>, lengths: &mut LinearView<u64, ReadWrite>, input_length: usize) {
    let run = ABSOLUTE_POS;
    let runs = count[0] as usize;
    if run < runs {
        let mut end = input_length as u64;
        if run + 1 < runs { end = offsets[run + 1]; }
        lengths[run] = end - offsets[run];
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
pub(super) fn nontrivial_flags(lengths: &LinearView<u64>, count: &LinearView<u64>, flags: &mut LinearView<u64, ReadWrite>) {
    let index = ABSOLUTE_POS;
    if index < flags.shape() {
        let mut flag = 0u64;
        if index < count[0] as usize { flag = u64::cast_from(lengths[index] > 1); }
        flags[index] = flag;
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn compact_runs<T: Numeric>(
    values: &LinearView<T>, offsets: &LinearView<u64>, lengths: &LinearView<u64>,
    flags: &LinearView<u64>, prefixes: &LinearView<u64>,
    output_values: &mut LinearView<T, ReadWrite>, output_offsets: &mut LinearView<u64, ReadWrite>,
    output_lengths: &mut LinearView<u64, ReadWrite>, output_count: &mut LinearView<u64, ReadWrite>,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut total = 0u64;
        if flags.shape() > 0 { total = prefixes[flags.shape() - 1]; }
        output_count[0] = total;
    }
    if index < flags.shape() {
        if flags[index] != 0 {
            let rank = prefixes[index] as usize - 1;
            output_values[rank] = values[index];
            output_offsets[rank] = offsets[index];
            output_lengths[rank] = lengths[index];
        }
    }
}

fn allocate<R: Runtime>(input: &RudaTensor<R>) -> RudaRuns<R> {
    let size = input.meta.num_elements();
    let buffer = |count| empty_device_dtype(input.client.clone(), input.device.clone(), Shape::new([count]), DType::U64);
    RudaRuns { values: empty_like(input), offsets: buffer(size), lengths: buffer(size), count: buffer(1) }
}

/// Encode adjacent equal values. Output values are the first input value of
/// each run, including when a custom equality relation groups unequal bit patterns.
pub fn encode<R, T, E>(input: &RudaTensor<R>, equal: E::RuntimeArg<R>, threads: u32) -> Result<RudaRuns<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, E: RudaKeyEqual<T> + LaunchArg,
{
    let heads = segmented::key_heads::<R, T, E>(input, equal)?;
    let flags = transform::unary::<R, u32, u64, RudaCast>(&heads, RudaCastLaunch::new())?;
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
    let output = allocate(input);
    let work = input.meta.num_elements().max(1);
    let dim = RudaDim::new(input.client.properties(), work);
    let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
    unsafe {
        scatter_heads::launch_unchecked::<T, R>(
            &input.client, grid, dim, address_type!(input, heads, prefixes, (output.values), (output.offsets), (output.count)),
            input.clone().into_linear_view(), heads.into_linear_view(), prefixes.into_linear_view(),
            output.values.clone().into_linear_view(), output.offsets.clone().into_linear_view(), output.count.clone().into_linear_view(),
        );
        let grid = calculate_ruda_count_elemwise(&input.client, work, dim);
        lengths_kernel::launch_unchecked::<R>(
            &input.client, grid, dim, address_type!((output.offsets), (output.count), (output.lengths)),
            output.offsets.clone().into_linear_view(), output.count.clone().into_linear_view(),
            output.lengths.clone().into_linear_view(), input.meta.num_elements(),
        );
    }
    Ok(output)
}

/// Encode only runs longer than one element, preserving their input order.
pub fn non_trivial_runs<R, T, E>(input: &RudaTensor<R>, equal: E::RuntimeArg<R>, threads: u32) -> Result<RudaRuns<R>, RudaPrimitiveError>
where R: Runtime, T: TensorElement, E: RudaKeyEqual<T> + LaunchArg,
{
    let runs = encode::<R, T, E>(input, equal, threads)?;
    let size = input.meta.num_elements();
    let flags = empty_like(&runs.lengths);
    let dim = RudaDim::new(input.client.properties(), size.max(1));
    if size > 0 {
        let grid = calculate_ruda_count_elemwise(&input.client, size, dim);
        unsafe {
            nontrivial_flags::launch_unchecked::<R>(
                &input.client, grid, dim, address_type!((runs.lengths), (runs.count), flags),
                runs.lengths.clone().into_linear_view(), runs.count.clone().into_linear_view(), flags.clone().into_linear_view(),
            );
        }
    }
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
    let output = allocate(input);
    let grid = calculate_ruda_count_elemwise(&input.client, size.max(1), dim);
    unsafe {
        compact_runs::launch_unchecked::<T, R>(
            &input.client, grid, dim, address_type!((runs.values), (runs.offsets), (runs.lengths), flags, prefixes, (output.values), (output.offsets), (output.lengths), (output.count)),
            runs.values.into_linear_view(), runs.offsets.into_linear_view(), runs.lengths.into_linear_view(),
            flags.into_linear_view(), prefixes.into_linear_view(), output.values.clone().into_linear_view(),
            output.offsets.clone().into_linear_view(), output.lengths.clone().into_linear_view(), output.count.clone().into_linear_view(),
        );
    }
    Ok(output)
}
