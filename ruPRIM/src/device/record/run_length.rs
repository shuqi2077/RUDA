use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaKeyEqual, RudaSum, RudaSumLaunch};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::device::{RudaPrimitiveError, scan, transform::{self, RudaCast, RudaCastLaunch}};
use super::{RudaRecordBuffer, RudaRecordBytes, grouped};

pub struct RudaRecordRuns<R: Runtime, T: RudaRecord> {
    pub values: RudaRecordBuffer<R, T>,
    pub offsets: RudaTensor<R>,
    pub lengths: RudaTensor<R>,
    pub count: RudaTensor<R>,
}

fn allocate<R: Runtime, T: RudaRecord>(input: &RudaRecordBuffer<R, T>) -> Result<RudaRecordRuns<R, T>, RudaPrimitiveError> {
    let tensor = |count| empty_device_dtype(input.bytes.client.clone(), input.bytes.device.clone(), Shape::new([count]), DType::U64);
    Ok(RudaRecordRuns { values: input.empty(input.len())?, offsets: tensor(input.len()), lengths: tensor(input.len()), count: tensor(1) })
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn scatter_heads<T: RudaRecord>(input: &RudaRecordBytes, heads: &LinearView<u32>, prefixes: &LinearView<u64>,
    values: &mut RudaRecordBytes, offsets: &mut LinearView<u64, ReadWrite>, output_count: &mut LinearView<u64, ReadWrite>, count: usize,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut runs = 0u64;
        if count > 0 { runs = prefixes[count - 1]; }
        output_count[0] = runs;
    }
    if index < count {
        if heads[index] != 0 {
            let rank = prefixes[index] as usize - 1;
            <RudaRecordBytes as RudaWrite<T>>::write(values, rank, <RudaRecordBytes as RudaRead<T>>::read(input, index));
            offsets[rank] = index as u64;
        }
    }
}

pub fn encode<R, T, E>(input: &RudaRecordBuffer<R, T>, equal: E::RuntimeArg<R>, threads: u32)
    -> Result<RudaRecordRuns<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, E: RudaKeyEqual<T> + LaunchArg,
{
    let heads = grouped::key_heads::<R, T, E>(input, equal);
    let flags = transform::unary::<R, u32, u64, RudaCast>(&heads, RudaCastLaunch::new())?;
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
    let output = allocate(input)?;
    let work = input.len().max(1);
    let dim = RudaDim::new(input.client().properties(), work);
    let grid = calculate_ruda_count_elemwise(input.client(), work, dim);
    unsafe {
        scatter_heads::launch_unchecked::<T, R>(input.client(), grid, dim, input.view(), heads.into_linear_view(), prefixes.into_linear_view(),
            output.values.view(), output.offsets.clone().into_linear_view(), output.count.clone().into_linear_view(), input.len());
        let grid = calculate_ruda_count_elemwise(input.client(), work, dim);
        crate::device::run_length::lengths_kernel::launch_unchecked::<R>(input.client(), grid, dim, AddressType::U64,
            output.offsets.clone().into_linear_view(), output.count.clone().into_linear_view(), output.lengths.clone().into_linear_view(), input.len());
    }
    Ok(output)
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn compact<T: RudaRecord>(input: &RudaRecordBytes, offsets: &LinearView<u64>, lengths: &LinearView<u64>,
    flags: &LinearView<u64>, prefixes: &LinearView<u64>, output: &mut RudaRecordBytes,
    output_offsets: &mut LinearView<u64, ReadWrite>, output_lengths: &mut LinearView<u64, ReadWrite>,
    output_count: &mut LinearView<u64, ReadWrite>, capacity: usize,
) {
    let index = ABSOLUTE_POS;
    if index == 0 {
        let mut count = 0u64;
        if capacity > 0 { count = prefixes[capacity - 1]; }
        output_count[0] = count;
    }
    if index < capacity {
        if flags[index] != 0 {
            let rank = prefixes[index] as usize - 1;
            <RudaRecordBytes as RudaWrite<T>>::write(output, rank, <RudaRecordBytes as RudaRead<T>>::read(input, index));
            output_offsets[rank] = offsets[index];
            output_lengths[rank] = lengths[index];
        }
    }
}

pub fn nontrivial<R, T, E>(input: &RudaRecordBuffer<R, T>, equal: E::RuntimeArg<R>, threads: u32)
    -> Result<RudaRecordRuns<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, E: RudaKeyEqual<T> + LaunchArg,
{
    let runs = encode::<R, T, E>(input, equal, threads)?;
    let flags = empty_device_dtype(input.bytes.client.clone(), input.bytes.device.clone(), Shape::new([input.len()]), DType::U64);
    let work = input.len().max(1);
    let dim = RudaDim::new(input.client().properties(), work);
    let grid = calculate_ruda_count_elemwise(input.client(), work, dim);
    unsafe {
        crate::device::run_length::nontrivial_flags::launch_unchecked::<R>(input.client(), grid, dim, AddressType::U64,
            runs.lengths.clone().into_linear_view(), runs.count.clone().into_linear_view(), flags.clone().into_linear_view());
    }
    let prefixes = scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?;
    let output = allocate(input)?;
    let grid = calculate_ruda_count_elemwise(input.client(), work, dim);
    unsafe {
        compact::launch_unchecked::<T, R>(input.client(), grid, dim, runs.values.view(), runs.offsets.into_linear_view(), runs.lengths.into_linear_view(),
            flags.into_linear_view(), prefixes.into_linear_view(), output.values.view(), output.offsets.clone().into_linear_view(),
            output.lengths.clone().into_linear_view(), output.count.clone().into_linear_view(), input.len());
    }
    Ok(output)
}
