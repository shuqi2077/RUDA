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
