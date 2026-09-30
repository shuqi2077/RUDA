use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn clear_counts(counts: &mut Array<u32>) {
    if ABSOLUTE_POS < counts.len() {
        counts[ABSOLUTE_POS] = 0;
    }
}

#[ruda(launch)]
pub(crate) fn count_routes(
    indices: &Array<u32>,
    counts: &mut Array<Atomic<u32>>,
    ranks: &mut Array<u32>,
) {
    let slot = ABSOLUTE_POS;
    if slot >= indices.len() {
        terminate!();
    }
    ranks[slot] = counts[indices[slot] as usize].fetch_add(1u32);
}

#[ruda(launch)]
pub(crate) fn prefix(counts: &Array<u32>, offsets: &mut Array<u32>) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    let mut expert = 0usize;
    let mut total = 0u32;
    offsets[0] = 0;
    while expert < counts.len() {
        total += counts[expert];
        offsets[expert + 1] = total;
        expert += 1;
    }
}

#[ruda(launch)]
pub(crate) fn scatter_routes(
    indices: &Array<u32>,
    ranks: &Array<u32>,
    offsets: &Array<u32>,
    sorted_slots: &mut Array<u32>,
    slot_rows: &mut Array<u32>,
    row_experts: &mut Array<u32>,
) {
    let slot = ABSOLUTE_POS;
    if slot >= indices.len() {
        terminate!();
    }
    let expert = indices[slot];
    let row = offsets[expert as usize] + ranks[slot];
    sorted_slots[row as usize] = slot as u32;
    slot_rows[slot] = row;
    row_experts[row as usize] = expert;
}

#[ruda(launch)]
pub(crate) fn gather<F: Float>(
    input: &Array<F>,
    sorted_slots: &Array<u32>,
    output: &mut Array<F>,
    hidden: u32,
    top_k: u32,
    #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= output.len() {
        terminate!();
    }
    let row = position / hidden as usize;
    let token = sorted_slots[row] as usize / top_k as usize;
    output[position] = input[token * hidden as usize + position % hidden as usize];
}

#[ruda(launch)]
pub(crate) fn combine<F: Float, W: Float>(
    expert_output: &Array<F>,
    weights: &Array<W>,
    indices: &Array<u32>,
    slot_rows: &Array<u32>,
    output: &mut Array<F>,
    width: u32,
    top_k: u32,
    #[define(F)] _dtype: StorageType,
    #[define(W)] _weight_dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= output.len() {
        terminate!();
    }
    let token = position / width as usize;
    let column = position % width as usize;
    let k = top_k as usize;
    let mut sum = F::cast_from(0.0f32);
    let mut previous_id = 0u32;
    let mut count = 0usize;
    while count < k {
        let mut found = false;
        let mut next_id = 0u32;
        let mut next_slot = 0usize;
        let mut slot = 0usize;
        while slot < k {
            let expert = indices[token * k + slot];
            if (count == 0 || expert > previous_id) && (!found || expert < next_id) {
                next_id = expert;
                next_slot = slot;
                found = true;
            }
            slot += 1;
        }
        let assignment = token * k + next_slot;
        let row = slot_rows[assignment] as usize;
        let weighted = F::cast_from(
            f32::cast_from(expert_output[row * width as usize + column])
                * f32::cast_from(weights[assignment]),
        );
        sum = F::cast_from(f32::cast_from(sum) + f32::cast_from(weighted));
        previous_id = next_id;
        count += 1;
    }
    output[position] = sum;
}

/// Gradient from combined token output back to expert-row outputs. `sorted_slots`
/// maps each contiguous expert row back to its original token/top-k assignment.
#[ruda(launch)]
pub(crate) fn combine_backward_values<F: Float, W: Float>(
    grad: &Array<F>, weights: &Array<W>, sorted_slots: &Array<u32>, out: &mut Array<F>,
    width:u32, top_k:u32, #[define(F)] _dtype:StorageType, #[define(W)] _weight_dtype:StorageType,
) {
    let i=ABSOLUTE_POS;
    if i>=out.len(){terminate!();}
    let row=i/width as usize; let col=i%width as usize;
    let assignment=sorted_slots[row] as usize;
    let token=assignment/top_k as usize;
    out[i]=F::cast_from(f32::cast_from(grad[token*width as usize+col])*f32::cast_from(weights[assignment]));
}

/// FP32 gradient of selected routing weights. One assignment owns one reduction,
/// so no atomics are required and the result can feed a later router backward.
#[ruda(launch)]
pub(crate) fn combine_backward_weights<F: Float>(
    grad:&Array<F>, expert_output:&Array<F>, slot_rows:&Array<u32>, out:&mut Array<f32>,
    width:u32, top_k:u32, #[define(F)] _dtype:StorageType,
) {
    let assignment=ABSOLUTE_POS;
    if assignment>=out.len(){terminate!();}
    let token=assignment/top_k as usize;
    let row=slot_rows[assignment] as usize;
    let mut sum=0.0f32; let mut col=0usize;
    while col<width as usize {
        sum=fma(f32::cast_from(grad[token*width as usize+col]),
                f32::cast_from(expert_output[row*width as usize+col]),sum);
        col+=1;
    }
    out[assignment]=sum;
}

/// Copy-dispatch backward: sum expert-row input gradients back to each token.
/// Each output element has one owner; fixed slot order and FP32 accumulation.
/// Routing weights do NOT enter this derivative of the unweighted copy.
#[ruda(launch)]
pub(crate) fn dispatch_backward<F: Float>(grad_rows: &Array<F>, slot_rows: &Array<u32>,
    out: &mut Array<F>, width: u32, top_k: u32, #[define(F)] _dtype: StorageType)
{
    let pos = ABSOLUTE_POS;
    if pos >= out.len() { terminate!(); }
    let token = pos / width as usize; let col = pos % width as usize;
    let mut sum = 0.0f32;
    for slot in 0..top_k as usize {
        let row = slot_rows[token * top_k as usize + slot] as usize;
        sum += f32::cast_from(grad_rows[row * width as usize + col]);
    }
    out[pos] = F::cast_from(sum);
}

/// One hardware plane per selected routing weight instead of one serial thread.
/// Explicit opt-in: reduction order differs; small widths may not benefit.
#[ruda(launch)]
pub(crate) fn combine_backward_weights_plane<F: Float>(
    grad: &Array<F>, expert_output: &Array<F>, slot_rows: &Array<u32>, out: &mut Array<f32>,
    width: u32, top_k: u32, #[comptime] lanes: usize, #[define(F)] _dtype: StorageType)
{
    let assignment = RUDA_POS_X as usize; let lane = UNIT_POS_X as usize;
    let token = assignment / top_k as usize;
    let row = slot_rows[assignment] as usize;
    let mut sum = 0.0f32; let mut col = lane;
    while col < width as usize {
        sum = fma(f32::cast_from(grad[token * width as usize + col]),
                  f32::cast_from(expert_output[row * width as usize + col]), sum);
        col += lanes;
    }
    let result = plane_sum(sum);
    if lane == 0 { out[assignment] = result; }
}
