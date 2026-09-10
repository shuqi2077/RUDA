use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn csrmv(
    offsets: &Array<u32>,
    indices: &Array<u32>,
    values: &Array<f32>,
    x: &Array<f32>,
    y: &Array<f32>,
    output: &mut Array<f32>,
    rows: u32,
    base: u32,
    alpha: f32,
    beta: f32,
    #[comptime] _source: String,
) {
    let rows = rows as usize;
    let row = ABSOLUTE_POS / 32;
    let lane = UNIT_POS % 32;
    if row < rows {
        let start = offsets[row as usize] - base;
        let end = offsets[row as usize + 1] - base;
        let mut sum = 0f32;
        let mut entry = start as usize + lane as usize;
        while entry < end as usize {
            sum = fma(values[entry], x[(indices[entry] - base) as usize], sum);
            entry += 32;
        }
        sum = plane_sum(sum);
        if lane == 0 {
            output[row as usize] = fma(beta, y[row as usize], alpha * sum);
        }
    }
}

#[ruda(launch)]
pub(super) fn csrmm(
    offsets: &Array<u32>,
    indices: &Array<u32>,
    values: &Array<f32>,
    b: &Array<f32>,
    c: &Array<f32>,
    output: &mut Array<f32>,
    rows: u32,
    columns: u32,
    base: u32,
    row_stride: u32,
    column_stride: u32,
    alpha: f32,
    beta: f32,
    #[comptime] _source: String,
) {
    let rows = rows as usize;
    let columns = columns as usize;
    let row_stride = row_stride as usize;
    let column_stride = column_stride as usize;
    let element = ABSOLUTE_POS / 32;
    let lane = UNIT_POS % 32;
    if element < rows * columns {
        let row = element / columns;
        let column = element % columns;
        let start = offsets[row as usize] - base;
        let end = offsets[row as usize + 1] - base;
        let mut sum = 0f32;
        let mut entry = start as usize + lane as usize;
        while entry < end as usize {
            let k = (indices[entry] - base) as usize;
            sum = fma(
                values[entry],
                b[k * columns as usize + column as usize],
                sum,
            );
            entry += 32;
        }
        sum = plane_sum(sum);
        if lane == 0 {
            output[(row * row_stride + column * column_stride) as usize] =
                fma(beta, c[element as usize], alpha * sum);
        }
    }
}
