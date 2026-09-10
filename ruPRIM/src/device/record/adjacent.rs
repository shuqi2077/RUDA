use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};
use crate::collective::record::{RudaRecord, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::device::RudaPrimitiveError;
use super::{RudaRecordBuffer, RudaRecordBytes};

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn difference_kernel<T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg>(
    input: &RudaRecordBytes, output: &mut RudaRecordBytes, op: &O, count: usize, #[comptime] right: bool,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        let mut value = <RudaRecordBytes as RudaRead<T>>::read(input, index);
        if right {
            if index + 1 < count { value = op.combine(value, <RudaRecordBytes as RudaRead<T>>::read(input, index + 1)); }
        } else if index > 0 { value = op.combine(value, <RudaRecordBytes as RudaRead<T>>::read(input, index - 1)); }
        <RudaRecordBytes as RudaWrite<T>>::write(output, index, value);
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "u64")]
fn copy_kernel<T: RudaRecord>(input: &RudaRecordBytes, output: &mut RudaRecordBytes, count: usize) {
    let index = ABSOLUTE_POS;
    if index < count { <RudaRecordBytes as RudaWrite<T>>::write(output, index, <RudaRecordBytes as RudaRead<T>>::read(input, index)); }
}

/// Apply op(current, neighbour); preserve the unpaired edge record.
pub fn difference<R, T, O>(input: &RudaRecordBuffer<R, T>, op: O::RuntimeArg<R>, right: bool, in_place: bool)
    -> Result<RudaRecordBuffer<R, T>, RudaPrimitiveError>
where R: Runtime, T: RudaRecord, O: RudaBinaryOp<T> + LaunchArg,
{
    let result = input.empty(input.len())?;
    if !input.is_empty() {
        let dim = RudaDim::new(input.client().properties(), input.len());
        let grid = calculate_ruda_count_elemwise(input.client(), input.len(), dim);
        unsafe {
            difference_kernel::launch_unchecked::<T, O, R>(input.client(), grid.clone(), dim,
                input.view(), result.view(), op, input.len(), right);
            if in_place {
                copy_kernel::launch_unchecked::<T, R>(input.client(), grid, dim, result.view(), input.view(), input.len());
            }
        }
    }
    Ok(if in_place { input.clone() } else { result })
}
