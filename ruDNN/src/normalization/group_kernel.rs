use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn statistics<F: Float>(input: &Array<F>, mean: &mut Array<f32>, rstd: &mut Array<f32>,
    width: u32, epsilon: f32, #[comptime] _source: String, #[define(F)] _storage: StorageType) {
    let (mu, inverse) = crate::normalization::kernel::row_statistics(input, width, epsilon);
    if UNIT_POS == 0 { mean[RUDA_POS_X as usize] = mu; rstd[RUDA_POS_X as usize] = inverse; }
}

#[ruda]
fn normalized<F: Float>(input: &Array<F>, mean: &Array<f32>, rstd: &Array<f32>, index: usize, width: u32) -> f32 {
    let row = index / width as usize;
    (f32::cast_from(input[index]) - mean[row]) * rstd[row]
}

#[ruda(launch)]
pub(super) fn forward_plain<F: Float>(input: &Array<F>, mean: &Array<f32>, rstd: &Array<f32>,
    output: &mut Array<F>, width: u32, #[define(F)] _storage: StorageType) {
    let i = ABSOLUTE_POS as usize;
    if i < output.len() { output[i] = F::cast_from(normalized(input, mean, rstd, i, width)); }
}

#[ruda(launch)]
pub(super) fn forward_weight<F: Float, W: Float>(input: &Array<F>, gamma: &Array<W>, mean: &Array<f32>, rstd: &Array<f32>,
    output: &mut Array<F>, width: u32, channels: u32, spatial: u32, #[define(F, W)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i < output.len() {
        let channel = (i / spatial as usize) % channels as usize;
        output[i] = F::cast_from(normalized(input, mean, rstd, i, width) * f32::cast_from(gamma[channel]));
    }
}

#[ruda(launch)]
pub(super) fn forward_bias<F: Float, B: Float>(input: &Array<F>, beta: &Array<B>, mean: &Array<f32>, rstd: &Array<f32>,
    output: &mut Array<F>, width: u32, channels: u32, spatial: u32, #[define(F, B)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i < output.len() {
        let channel = (i / spatial as usize) % channels as usize;
        output[i] = F::cast_from(normalized(input, mean, rstd, i, width) + f32::cast_from(beta[channel]));
    }
}

#[ruda(launch)]
pub(super) fn forward_affine<F: Float, W: Float, B: Float>(input: &Array<F>, gamma: &Array<W>, beta: &Array<B>,
    mean: &Array<f32>, rstd: &Array<f32>, output: &mut Array<F>, width: u32, channels: u32, spatial: u32,
    #[define(F, W, B)] _types: [StorageType; 3]) {
    let i = ABSOLUTE_POS as usize;
    if i < output.len() {
        let channel = (i / spatial as usize) % channels as usize;
        let value = normalized(input, mean, rstd, i, width) * f32::cast_from(gamma[channel]);
        output[i] = F::cast_from(value + f32::cast_from(beta[channel]));
    }
}

#[ruda(launch)]
pub(super) fn input_plain<F: Float, G: Float>(input: &Array<F>, grad: &Array<G>, mean: &Array<f32>, rstd: &Array<f32>,
    output: &mut Array<F>, width: u32, #[define(F, G)] _types: [StorageType; 2]) {
    let base = RUDA_POS_X as usize * width as usize;
    let mut first = 0f32;
    let mut second = 0f32;
    let mut column = UNIT_POS as usize;
    while column < width as usize {
        let dy = f32::cast_from(grad[base + column]);
        first += dy; second += dy * normalized(input, mean, rstd, base + column, width);
        if width as usize - column <= RUDA_DIM as usize { break; }
        column += RUDA_DIM as usize;
    }
    let (first, second) = crate::normalization::kernel::block_sum_pair(first, second);
    let inverse = rstd[RUDA_POS_X as usize];
    column = UNIT_POS as usize;
    while column < width as usize {
        let value = f32::cast_from(grad[base + column]) - first / width as f32
            - normalized(input, mean, rstd, base + column, width) * (second / width as f32);
        output[base + column] = F::cast_from(inverse * value);
        if width as usize - column <= RUDA_DIM as usize { break; }
        column += RUDA_DIM as usize;
    }
}

#[ruda(launch)]
pub(super) fn input_weighted<F: Float, W: Float, G: Float>(input: &Array<F>, gamma: &Array<W>, grad: &Array<G>,
    mean: &Array<f32>, rstd: &Array<f32>, output: &mut Array<F>, width: u32, channels: u32, spatial: u32,
    #[define(F, W, G)] _types: [StorageType; 3]) {
    let base = RUDA_POS_X as usize * width as usize;
    let mut first = 0f32;
    let mut second = 0f32;
    let mut column = UNIT_POS as usize;
    while column < width as usize {
        let i = base + column;
        let channel = (i / spatial as usize) % channels as usize;
        let dy = f32::cast_from(grad[i]) * f32::cast_from(gamma[channel]);
        first += dy; second += dy * normalized(input, mean, rstd, i, width);
        if width as usize - column <= RUDA_DIM as usize { break; }
        column += RUDA_DIM as usize;
    }
    let (first, second) = crate::normalization::kernel::block_sum_pair(first, second);
    let inverse = rstd[RUDA_POS_X as usize];
    column = UNIT_POS as usize;
    while column < width as usize {
        let i = base + column;
        let channel = (i / spatial as usize) % channels as usize;
        let dy = f32::cast_from(grad[i]) * f32::cast_from(gamma[channel]);
        output[i] = F::cast_from(inverse * (dy - first / width as f32
            - normalized(input, mean, rstd, i, width) * (second / width as f32)));
        if width as usize - column <= RUDA_DIM as usize { break; }
        column += RUDA_DIM as usize;
    }
}

#[ruda(launch)]
pub(super) fn weight_partial<F: Float, G: Float>(input: &Array<F>, grad: &Array<G>, mean: &Array<f32>, rstd: &Array<f32>,
    output: &mut Array<f32>, width: u32, channels: u32, spatial: u32, parts: u32,
    #[define(F, G)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let channel = i % channels as usize;
    let rows = grad.len() / channels as usize;
    let mut row = i / channels as usize;
    let mut sum = 0f32;
    while row < rows {
        let index = (row / spatial as usize * channels as usize + channel) * spatial as usize + row % spatial as usize;
        sum += f32::cast_from(grad[index]) * normalized(input, mean, rstd, index, width);
        if rows - row <= parts as usize { break; }
        row += parts as usize;
    }
    output[i] = sum;
}

#[ruda(launch)]
pub(super) fn bias_partial<G: Float>(grad: &Array<G>, output: &mut Array<f32>, channels: u32, spatial: u32, parts: u32,
    #[define(G)] _storage: StorageType) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let channel = i % channels as usize;
    let rows = grad.len() / channels as usize;
    let mut row = i / channels as usize;
    let mut sum = 0f32;
    while row < rows {
        let index = (row / spatial as usize * channels as usize + channel) * spatial as usize + row % spatial as usize;
        sum += f32::cast_from(grad[index]);
        if rows - row <= parts as usize { break; }
        row += parts as usize;
    }
    output[i] = sum;
}

#[ruda(launch)]
pub(super) fn affine_partial<F: Float, G: Float>(input: &Array<F>, grad: &Array<G>, mean: &Array<f32>, rstd: &Array<f32>,
    weight: &mut Array<f32>, bias: &mut Array<f32>, width: u32, channels: u32, spatial: u32, parts: u32,
    #[define(F, G)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i >= weight.len() { terminate!(); }
    let channel = i % channels as usize;
    let rows = grad.len() / channels as usize;
    let mut row = i / channels as usize;
    let mut weight_sum = 0f32;
    let mut bias_sum = 0f32;
    while row < rows {
        let index = (row / spatial as usize * channels as usize + channel) * spatial as usize + row % spatial as usize;
        let dy = f32::cast_from(grad[index]);
        weight_sum += dy * normalized(input, mean, rstd, index, width); bias_sum += dy;
        if rows - row <= parts as usize { break; }
        row += parts as usize;
    }
    weight[i] = weight_sum; bias[i] = bias_sum;
}
