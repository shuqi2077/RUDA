use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::RudaBinaryOp;

/// Reduce one item per batch per lane. Outputs are distributed over logical
/// lanes in blocked or striped order; padded output slots are not written.
/// Input length is `batches`, output capacity is ceil(batches / width).
#[ruda]
pub fn reduce<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    input: &Array<T>, output: &mut Array<T>, op: &O,
    #[comptime] width: u32, #[comptime] batches: usize, #[comptime] striped: bool,
) {
    let lane = (UNIT_POS_PLANE % width) as usize;
    let per_lane = (batches + width as usize - 1) / width as usize;
    #[unroll]
    for batch in 0..batches {
        let value = crate::warp::reduce::<T, O>(input[batch], op, width, width);
        let owner = if striped { batch % width as usize } else { batch / per_lane };
        let slot = if striped { batch / width as usize } else { batch % per_lane };
        if lane == owner { output[slot] = value; }
    }
}
