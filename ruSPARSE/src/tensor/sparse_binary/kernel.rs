use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn sum(
    a_offsets: &Array<u32>,
    a_indices: &Array<u32>,
    a_values: &Array<f32>,
    b_offsets: &Array<u32>,
    b_indices: &Array<u32>,
    b_values: &Array<f32>,
    rows: &Array<u32>,
    columns: &Array<u32>,
    output: &mut Array<f32>,
    nnz: u32,
    a_base: u32,
    b_base: u32,
    alpha: f32,
    beta: f32,
    #[comptime] _source: String,
) {
    let entry = ABSOLUTE_POS / 32;
    let lane = (UNIT_POS % 32) as usize;
    if entry < nnz as usize {
        let row = rows[entry] as usize;
        let column = columns[entry] - a_base;
        let mut sum = 0f32;
        let mut cursor = (a_offsets[row] - a_base) as usize + lane;
        let end = (a_offsets[row + 1] - a_base) as usize;
        while cursor < end {
            if a_indices[cursor] - a_base == column {
                sum = fma(alpha, a_values[cursor], sum);
            }
            cursor += 32;
        }
        let mut cursor = (b_offsets[row] - b_base) as usize + lane;
        let end = (b_offsets[row + 1] - b_base) as usize;
        while cursor < end {
            if b_indices[cursor] - b_base == column {
                sum = fma(beta, b_values[cursor], sum);
            }
            cursor += 32;
        }
        sum = plane_sum(sum);
        if lane == 0 {
            output[entry] = sum;
        }
    }
}

#[ruda(launch)]
pub(crate) fn product(
    a_offsets: &Array<u32>,
    a_indices: &Array<u32>,
    a_values: &Array<f32>,
    b_offsets: &Array<u32>,
    b_indices: &Array<u32>,
    b_values: &Array<f32>,
    rows: &Array<u32>,
    columns: &Array<u32>,
    output: &mut Array<f32>,
    nnz: u32,
    a_base: u32,
    b_base: u32,
    alpha: f32,
    #[comptime] _source: String,
    output_base: u32,
) {
    let entry = ABSOLUTE_POS / 32;
    let lane = (UNIT_POS % 32) as usize;
    if entry < nnz as usize {
        let row = rows[entry] as usize;
        let column = columns[entry] - output_base;
        let mut sum = 0f32;
        let mut cursor = (a_offsets[row] - a_base) as usize + lane;
        let end = (a_offsets[row + 1] - a_base) as usize;
        while cursor < end {
            let inner = (a_indices[cursor] - a_base) as usize;
            let value = a_values[cursor];
            let mut other = (b_offsets[inner] - b_base) as usize;
            let other_end = (b_offsets[inner + 1] - b_base) as usize;
            while other < other_end {
                if b_indices[other] - b_base == column {
                    sum = fma(value, b_values[other], sum);
                }
                other += 1;
            }
            cursor += 32;
        }
        sum = plane_sum(sum);
        if lane == 0 {
            output[entry] = alpha * sum;
        }
    }
}
