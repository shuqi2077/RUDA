use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, element::TensorElement, layout::address_type};
use super::{RudaPrimitiveError, check_type, empty_like};

#[ruda]
pub trait RudaBulkOp: RudaType {
    fn apply(&self, index: usize);
}

#[ruda]
pub trait RudaForEachOp<T: Numeric>: RudaType {
    fn apply(&self, value: &mut T, index: usize);
}

#[ruda]
pub trait RudaCoordinateOp: RudaType {
    fn apply(&self, linear_index: usize, coordinates: &Array<usize>);
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn extents_kernel<O: RudaCoordinateOp + LaunchArg>(
    count: usize, op: &O, #[comptime] extents: Vec<usize>, #[comptime] column_major: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let rank = comptime![extents.len()];
        let mut coordinates = Array::<usize>::new(rank);
        let mut remaining = index;
        #[unroll]
        for step in 0..rank {
            let axis = comptime![if column_major { step } else { rank - 1 - step }];
            let size = comptime![extents[axis]];
            coordinates[axis] = remaining % size;
            remaining /= size;
        }
        op.apply(index, &coordinates);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn bulk_kernel<O: RudaBulkOp + LaunchArg>(count: usize, op: &O) {
    if ABSOLUTE_POS < count { op.apply(ABSOLUTE_POS); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn for_each_kernel<T: Numeric, O: RudaForEachOp<T> + LaunchArg>(
    input: &mut LinearView<T, ReadWrite>, op: &O, count: usize, #[comptime] copy: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut value = input[index];
        op.apply(&mut value, index);
        if !copy { input[index] = value; }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn copy_kernel<T: Numeric>(input: &LinearView<T>, output: &mut LinearView<T, ReadWrite>) {
    let index = ABSOLUTE_POS;
    if index < output.shape() { output[index] = input[index]; }
}

/// Copy into caller-owned output, including exact in-place aliasing. Partial
/// overlap is not allowed. Both tensors describe the same logical item count.
pub fn copy_into<R: Runtime, T: TensorElement>(input: &RudaTensor<R>, output: &RudaTensor<R>) -> Result<(), RudaPrimitiveError> {
    check_type::<R, T>(input)?;
    check_type::<R, T>(output)?;
    if input.meta.num_elements() != output.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if input.device.to_id() != output.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let count = input.meta.num_elements();
    if count > 0 {
        let dim = RudaDim::new(input.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            copy_kernel::launch_unchecked::<T, R>(&input.client, grid, dim, address_type!(input, output),
                input.clone().into_linear_view(), output.clone().into_linear_view());
        }
    }
    Ok(())
}

/// Invoke a device operation once per index. Side effects are carried by the
/// operation's launched device arguments, not a host callback.
pub fn bulk<R: Runtime, O: RudaBulkOp + LaunchArg>(
    client: &ComputeClient<R>, count: usize, op: O::RuntimeArg<R>, address_type: AddressType,
) {
    if count == 0 { return; }
    let dim = RudaDim::new(client.properties(), count);
    let grid = calculate_ruda_count_elemwise(client, count, dim);
    unsafe {
        bulk_kernel::launch_unchecked::<O, R>(client, grid, dim, address_type.max(AddressType::from_len(count)), count, op);
    }
}

/// Apply an operation to each element. With `copy = true`, mutations of the
/// local element are not written back; other operation side effects still run.
pub fn for_each<R, T, O>(input: &RudaTensor<R>, op: O::RuntimeArg<R>, copy: bool) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaForEachOp<T> + LaunchArg,
{
    for_each_n::<R, T, O>(input, input.meta.num_elements(), op, copy)
}

pub fn for_each_n<R, T, O>(input: &RudaTensor<R>, count: usize, op: O::RuntimeArg<R>, copy: bool) -> Result<(), RudaPrimitiveError>
where R: Runtime, T: TensorElement, O: RudaForEachOp<T> + LaunchArg,
{
    check_type::<R, T>(input)?;
    if count > input.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if count == 0 { return Ok(()); }
    let dim = RudaDim::new(input.client.properties(), count);
    let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
    unsafe {
        for_each_kernel::launch_unchecked::<T, O, R>(&input.client, grid, dim, address_type!(input),
            input.clone().into_linear_view(), op, count, copy);
    }
    Ok(())
}

/// Visit row-major or column-major extents, passing both the linear iteration
/// index and coordinates. Rank zero has one iteration; any zero extent has none.
/// Address type must also cover buffers captured by the operation.
pub fn for_each_in_extents<R: Runtime, O: RudaCoordinateOp + LaunchArg>(
    client: &ComputeClient<R>, extents: &[usize], column_major: bool, op: O::RuntimeArg<R>, address_type: AddressType,
) -> Result<(), RudaPrimitiveError> {
    if extents.contains(&0) { return Ok(()); }
    let count = extents.iter().try_fold(1usize, |count, &size| count.checked_mul(size))
        .ok_or(RudaPrimitiveError::Configuration("extent product overflows address space"))?;
    let dim = RudaDim::new(client.properties(), count);
    let grid = calculate_ruda_count_elemwise(client, count, dim);
    unsafe {
        extents_kernel::launch_unchecked::<O, R>(client, grid, dim, address_type.max(AddressType::from_len(count)),
            count, op, extents.to_vec(), column_major);
    }
    Ok(())
}

/// Copy logical tensor elements into distinct device storage.
pub fn copy<R: Runtime, T: TensorElement>(input: &RudaTensor<R>) -> Result<RudaTensor<R>, RudaPrimitiveError> {
    check_type::<R, T>(input)?;
    let output = empty_like(input);
    let count = input.meta.num_elements();
    if count > 0 {
        let dim = RudaDim::new(input.client.properties(), count);
        let grid = calculate_ruda_count_elemwise(&input.client, count, dim);
        unsafe {
            copy_kernel::launch_unchecked::<T, R>(&input.client, grid, dim, address_type!(input, output),
                input.clone().into_linear_view(), output.clone().into_linear_view());
        }
    }
    Ok(output)
}
