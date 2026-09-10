use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, element::TensorElement, layout::address_type};
use super::{RudaPrimitiveError, check_type};
use crate::collective::record::RudaRecord;

#[ruda]
pub trait RudaBatchRead<T: RudaType + 'static>: RudaType {
    fn len(&self, batch: usize) -> usize;
    fn read(&self, batch: usize, offset: usize) -> T;
}

#[ruda]
pub trait RudaBatchWrite<T: RudaType + 'static>: RudaType {
    fn write(&mut self, batch: usize, offset: usize, value: T);
}

#[derive(RudaType, RudaLaunch)]
pub struct RudaNativeBatchInput {
    pub addresses: LinearView<u64>,
    pub lengths: LinearView<u64>,
}

#[ruda]
impl<T: RudaRecord> RudaBatchRead<T> for RudaNativeBatchInput {
    fn len(&self, batch: usize) -> usize { self.lengths[batch] as usize }
    fn read(&self, batch: usize, offset: usize) -> T {
        let bytes = comptime![T::SIZE as u64];
        T::load(self.addresses[batch] + offset as u64 * bytes)
    }
}

#[derive(RudaType, RudaLaunch)]
pub struct RudaNativeBatchOutput {
    pub addresses: LinearView<u64>,
}

#[ruda]
impl<T: RudaRecord> RudaBatchWrite<T> for RudaNativeBatchOutput {
    fn write(&mut self, batch: usize, offset: usize, value: T) {
        let bytes = comptime![T::SIZE as u64];
        T::store(self.addresses[batch] + offset as u64 * bytes, value);
    }
}

/// Typed batched copy through device-resident native pointer tables.
///
/// # Safety
/// Tables name live, aligned allocations for the selected device and type T.
/// Input/output aliasing follows `batched`; allocations outlive queued work.
pub unsafe fn native_ranges<R: Runtime, T: RudaRecord>(
    sources: &RudaTensor<R>, destinations: &RudaTensor<R>, lengths: &RudaTensor<R>, threads: u32,
) -> Result<(), RudaPrimitiveError> {
    let batches = lengths.meta.num_elements();
    for table in [sources, destinations, lengths] {
        check_type::<R, u64>(table)?;
        if table.meta.num_elements() != batches { return Err(RudaPrimitiveError::Length); }
        if table.device.to_id() != lengths.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    let input = RudaNativeBatchInputLaunch::new(sources.clone().into_linear_view(), lengths.clone().into_linear_view());
    let output = RudaNativeBatchOutputLaunch::new(destinations.clone().into_linear_view());
    batched::<R, T, RudaNativeBatchInput, RudaNativeBatchOutput>(&sources.client, batches, input, output, threads, AddressType::U64)
}

#[derive(RudaType, RudaLaunch)]
pub struct RudaBatchInput<T: Numeric> {
    pub data: LinearView<T>,
    pub begins: LinearView<u64>,
    pub lengths: LinearView<u64>,
}

#[ruda]
impl<T: Numeric> RudaBatchRead<T> for RudaBatchInput<T> {
    fn len(&self, batch: usize) -> usize { self.lengths[batch] as usize }
    fn read(&self, batch: usize, offset: usize) -> T { self.data[self.begins[batch] as usize + offset] }
}

#[derive(RudaType, RudaLaunch)]
pub struct RudaBatchOutput<T: Numeric> {
    pub data: LinearView<T, ReadWrite>,
    pub begins: LinearView<u64>,
}

#[ruda]
impl<T: Numeric> RudaBatchWrite<T> for RudaBatchOutput<T> {
    fn write(&mut self, batch: usize, offset: usize, value: T) { let index = self.begins[batch] as usize + offset; self.data[index] = value; }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn batched_kernel<T: RudaType + 'static, I: RudaBatchRead<T> + LaunchArg, O: RudaBatchWrite<T> + LaunchArg>(
    input: &I, output: &mut O, batches: usize, #[comptime] threads: usize,
) {
    let batch = RUDA_POS as usize;
    if batch < batches {
        let length = input.len(batch);
        let mut offset = UNIT_POS as usize;
        while offset < length {
            output.write(batch, offset, input.read(batch, offset));
            offset += threads;
        }
    }
}

/// Copy device-described ranges through random-access reader/writer operations.
/// Readers may synthesize values or use device-resident indexing. Input ranges
/// may overlap each other, but output ranges must be disjoint from every other
/// output and input range. `address_type` must cover the operations' addresses.
pub fn batched<R, T, I, O>(
    client: &ComputeClient<R>, batches: usize, input: I::RuntimeArg<R>, output: O::RuntimeArg<R>,
    threads: u32, address_type: AddressType,
) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: RudaType + 'static, I: RudaBatchRead<T> + LaunchArg, O: RudaBatchWrite<T> + LaunchArg,
{
    let hardware = &client.properties().hardware;
    if threads == 0 || threads > hardware.max_units_per_ruda || threads > hardware.max_ruda_dim.0 {
        return Err(RudaPrimitiveError::Configuration("invalid batched-copy block size"));
    }
    if batches == 0 { return Ok(()); }
    let work = batches.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("batched-copy launch size overflow"))?;
    let dim = RudaDim::new_1d(threads);
    let grid = calculate_ruda_count_elemwise(client, work, dim);
    unsafe {
        batched_kernel::launch_unchecked::<T, I, O, R>(client, grid, dim,
            address_type.max(AddressType::from_len(work)), input, output, batches, threads as usize);
    }
    Ok(())
}

/// Batched ranges within two tensors; descriptors and lengths are U64 device
/// tensors. Range validity and non-aliasing follow `batched`'s contract.
pub fn ranges<R: Runtime, T: TensorElement>(
    input: &RudaTensor<R>, output: &RudaTensor<R>, input_begins: &RudaTensor<R>,
    output_begins: &RudaTensor<R>, lengths: &RudaTensor<R>, threads: u32,
) -> Result<(), RudaPrimitiveError> {
    check_type::<R, T>(input)?;
    check_type::<R, T>(output)?;
    let batches = lengths.meta.num_elements();
    for descriptor in [input_begins, output_begins, lengths] {
        check_type::<R, u64>(descriptor)?;
        if descriptor.meta.num_elements() != batches { return Err(RudaPrimitiveError::Length); }
        if descriptor.device.to_id() != input.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    if output.device.to_id() != input.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let source = RudaBatchInputLaunch::new(input.clone().into_linear_view(), input_begins.clone().into_linear_view(), lengths.clone().into_linear_view());
    let destination = RudaBatchOutputLaunch::new(output.clone().into_linear_view(), output_begins.clone().into_linear_view());
    batched::<R, T, RudaBatchInput<T>, RudaBatchOutput<T>>(&input.client, batches, source, destination,
        threads, address_type!(input, output, input_begins, output_begins, lengths))
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn tensor_kernel<T: Numeric>(input: &LinearView<T>, output: &mut LinearView<T, ReadWrite>) {
    if ABSOLUTE_POS < input.shape() { output[ABSOLUTE_POS] = input[ABSOLUTE_POS]; }
}

/// Copy equal-shaped multidimensional views with independent layouts/strides.
/// Input and output storage must not overlap.
pub fn tensor_into<R: Runtime, T: TensorElement>(input: &RudaTensor<R>, output: &RudaTensor<R>) -> Result<(), RudaPrimitiveError> {
    check_type::<R, T>(input)?;
    check_type::<R, T>(output)?;
    if input.meta.shape() != output.meta.shape() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != output.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let count = input.meta.num_elements();
    if count > 0 {
        let dim = RudaDim::new(input.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            tensor_kernel::launch_unchecked::<T, R>(&input.client, grid, dim, address_type!(input, output),
                input.clone().into_linear_view(), output.clone().into_linear_view());
        }
    }
    Ok(())
}
