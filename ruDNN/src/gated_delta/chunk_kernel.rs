use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn prefix(
    decay: &Array<f32>, cumulative: &mut Array<f32>,
    sequence: u32, start: u32, #[comptime] chunk: usize,
    #[comptime] tile: usize, #[comptime] _source: String,
) {
    let head = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let threads = RUDA_DIM as usize;
    let mut values = SharedMemory::<f32>::new(tile);
    let mut carry = 0f32;
    let mut offset = 0usize;
    while offset < chunk {
        let mut i = lane;
        while i < tile {
            let token = start as usize + offset + i;
            let mut value = 0f32;
            if offset + i < chunk && token < sequence as usize {
                value = decay[head * sequence as usize + token];
            }
            if i == 0 { value += carry; }
            values[i] = value;
            i += threads;
        }
        sync_ruda();
        let mut stride = 1usize;
        while stride < tile {
            let target = (lane / stride) * (2 * stride) + stride + lane % stride;
            if target < tile {
                values[target] += values[(lane / stride) * (2 * stride) + stride - 1];
            }
            sync_ruda();
            stride *= 2;
        }
        i = lane;
        while i < tile {
            if offset + i < chunk { cumulative[head * chunk + offset + i] = values[i]; }
            i += threads;
        }
        carry = values[tile - 1];
        sync_ruda();
        offset += tile;
    }
}

#[ruda(launch)]
pub(crate) fn prepare<F: Float>(
    input: &Array<F>, beta: &Array<F>, cumulative: &Array<f32>,
    output: &mut Array<f32>, sequence: u32, start: u32, chunk: u32, width: u32,
    scale: f32, #[comptime] mode: u32, #[comptime] _source: String,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    let w = width as usize;
    let c = chunk as usize;
    let row = i / w;
    let head = row / c;
    let token = start as usize + row % c;
    let mut value = 0f32;
    if token < sequence as usize {
        let source_row = head * sequence as usize + token;
        value = f32::cast_from(input[source_row * w + i % w]) * scale;
        if comptime!(mode == 1 || mode == 2) { value *= f32::cast_from(beta[source_row]); }
        if comptime!(mode == 2 || mode == 3) { value *= cumulative[row].exp(); }
        if comptime!(mode == 4) {
            value *= (cumulative[head * c + c - 1] - cumulative[row]).exp();
        }
    }
    output[i] = value;
}

#[ruda(launch)]
pub(crate) fn decay_mask(
    products: &Array<f32>, cumulative: &Array<f32>, output: &mut Array<f32>,
    chunk: u32, #[comptime] strict: bool, #[comptime] _source: String,
) {
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    let c = chunk as usize;
    let head = i / (c * c);
    let row = i / c % c;
    let col = i % c;
    let mut value = 0f32;
    if row >= col {
        value = products[i] * (cumulative[head * c + row] - cumulative[head * c + col]).exp();
    }
    if comptime!(strict) {
        value = -value;
        if row <= col { value = 0f32; }
    }
    output[i] = value;
}

#[ruda(launch)]
pub(crate) fn triangular_inverse(
    input: &Array<f32>, output: &mut Array<f32>,
    #[comptime] chunk: usize, #[comptime] _source: String,
) {
    let base = RUDA_POS_X as usize * chunk * chunk;
    let lane = UNIT_POS as usize;
    let threads = RUDA_DIM as usize;
    let mut matrix = SharedMemory::<f32>::new(chunk * chunk);
    let mut row = SharedMemory::<f32>::new(chunk);
    let mut i = lane;
    while i < chunk * chunk {
        matrix[i] = input[base + i];
        i += threads;
    }
    sync_ruda();
    for r in 1..chunk {
        let mut col = lane;
        while col < r {
            row[col] = matrix[r * chunk + col];
            col += threads;
        }
        sync_ruda();
        col = lane;
        while col < r {
            let mut sum0 = 0f32;
            let mut sum1 = 0f32;
            let mut sum2 = 0f32;
            let mut sum3 = 0f32;
            let mut k = 0usize;
            while k + 3 < r {
                sum0 += row[k] * matrix[k * chunk + col];
                sum1 += row[k + 1] * matrix[(k + 1) * chunk + col];
                sum2 += row[k + 2] * matrix[(k + 2) * chunk + col];
                sum3 += row[k + 3] * matrix[(k + 3) * chunk + col];
                k += 4;
            }
            if k < r { sum0 += row[k] * matrix[k * chunk + col]; }
            if k + 1 < r { sum1 += row[k + 1] * matrix[(k + 1) * chunk + col]; }
            if k + 2 < r { sum2 += row[k + 2] * matrix[(k + 2) * chunk + col]; }
            let sum = ((sum0 + sum1) + sum2) + sum3;
            matrix[r * chunk + col] = row[col] + sum;
            col += threads;
        }
        sync_ruda();
    }
    i = lane;
    while i < chunk * chunk {
        let mut value = matrix[i];
        if i / chunk == i % chunk { value += 1f32; }
        output[base + i] = value;
        i += threads;
    }
}

#[ruda(launch)]
pub(crate) fn subtract(
    lhs: &Array<f32>, rhs: &Array<f32>, output: &mut Array<f32>,
    #[comptime] _source: String,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() { output[i] = lhs[i] - rhs[i]; }
}

#[ruda(launch)]
pub(crate) fn write_output<F: Float>(
    inter: &Array<f32>, intra: &Array<f32>, output: &mut Array<F>,
    sequence: u32, start: u32, chunk: u32, width: u32,
    #[comptime] _source: String, #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS;
    if i >= inter.len() { terminate!(); }
    let c = chunk as usize;
    let w = width as usize;
    let row = i / w;
    let token = start as usize + row % c;
    if token < sequence as usize {
        output[(row / c * sequence as usize + token) * w + i % w] = F::cast_from(inter[i] + intra[i]);
    }
}

#[ruda(launch)]
pub(crate) fn update_state(
    initial: &Array<f32>, update: &Array<f32>, cumulative: &Array<f32>,
    output: &mut Array<f32>, chunk: u32, state_size: u32, #[comptime] _source: String,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let head = i / state_size as usize;
        output[i] = initial[i] * cumulative[(head + 1) * chunk as usize - 1].exp() + update[i];
    }
}
