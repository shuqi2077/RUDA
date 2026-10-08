use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[derive(RudaType, Clone)]
struct Moments {
    mean: f32,
    m2: f32,
    count: f32,
}

#[ruda]
fn combine(left: Moments, right: Moments) -> Moments {
    let count = left.count + right.count;
    let mut mean = 0f32;
    let mut m2 = 0f32;
    if count > 0.0 {
        let inverse = count.recip();
        let left_fraction = left.count * inverse;
        let right_fraction = right.count * inverse;
        let delta = left.mean - right.mean;
        mean = fma(right_fraction, right.mean, left_fraction * left.mean);
        m2 = fma(
            delta * delta * right.count,
            left_fraction,
            left.m2 + right.m2,
        );
    }
    Moments { mean, m2, count }
}

#[ruda]
pub(super) fn row_statistics<F: Float>(
    input: &Array<F>,
    width: u32,
    epsilon: f32,
) -> (f32, f32) {
    let row = RUDA_POS_X as usize;
    let width = width as usize;
    let thread = UNIT_POS as usize;
    let threads = RUDA_DIM as usize;
    let mut moments = Moments {
        mean: 0.0,
        m2: 0.0,
        count: 0.0,
    };
    let mut group = thread * 4;
    while group < width {
        #[unroll]
        for lane in 0usize..4usize {
            let column = group + lane;
            if column < width {
                let value = f32::cast_from(input[row * width + column]);
                let delta = value - moments.mean;
                moments.count += 1.0;
                moments.mean = fma(delta, moments.count.recip(), moments.mean);
                moments.m2 = fma(delta, value - moments.mean, moments.m2);
            }
        }
        if width - group <= threads * 4 {
            break;
        }
        group += threads * 4;
    }
    let mut offset = PLANE_DIM / 2;
    while offset > 0 {
        let other = Moments {
            mean: plane_shuffle_down(moments.mean, offset),
            m2: plane_shuffle_down(moments.m2, offset),
            count: plane_shuffle_down(moments.count, offset),
        };
        let combined = combine(moments.clone(), other);
        moments.mean = combined.mean;
        moments.m2 = combined.m2;
        moments.count = combined.count;
        offset /= 2;
    }
    let mut means = SharedMemory::<f32>::new(4usize);
    let mut variances = SharedMemory::<f32>::new(4usize);
    let mut counts = SharedMemory::<f32>::new(4usize);
    let warp = UNIT_POS_Y as usize;
    offset = 2;
    while offset > 0 {
        if UNIT_POS_X == 0 && warp >= offset as usize && warp < 2usize * offset as usize {
            let slot = warp - offset as usize;
            means[slot] = moments.mean;
            variances[slot] = moments.m2;
            counts[slot] = moments.count;
        }
        sync_ruda();
        if UNIT_POS_X == 0 && warp < offset as usize {
            let combined = combine(
                moments.clone(),
                Moments {
                    mean: means[warp],
                    m2: variances[warp],
                    count: counts[warp],
                },
            );
            moments.mean = combined.mean;
            moments.m2 = combined.m2;
            moments.count = combined.count;
        }
        sync_ruda();
        offset /= 2;
    }
    if UNIT_POS == 0 {
        means[0] = moments.mean;
        variances[0] = moments.m2 / width as f32;
    }
    sync_ruda();
    let mean = means[0];
    let inverse_std = (variances[0] + epsilon).inverse_sqrt();
    (mean, inverse_std)
}

#[ruda(launch)]
pub(crate) fn layer_norm<F: Float>(
    input: &Array<F>, gamma: &Array<f32>, beta: &Array<f32>, output: &mut Array<F>,
    width: u32, epsilon: f32,
    #[comptime] has_beta: bool,
    #[comptime] _source: String,
    #[define(F)] _dtype: StorageType,
) {
    let (mean, inverse_std) = row_statistics(input, width, epsilon);
    let row = RUDA_POS_X as usize;
    let width = width as usize;
    let thread = UNIT_POS as usize;
    let threads = RUDA_DIM as usize;
    let mut column = thread;
    while column < width {
        let value = f32::cast_from(input[row * width + column]);
        let normalized = inverse_std * (value - mean);
        let mut result = gamma[column] * normalized;
        if has_beta {
            result = fma(gamma[column], normalized, beta[column]);
        }
        output[row * width + column] = F::cast_from(result);
        if width - column <= threads {
            break;
        }
        column += threads;
    }
}

#[ruda(launch)]
pub(crate) fn layer_norm_training<F: Float, W: Float, B: Float>(
    input: &Array<F>, gamma: &Array<W>, beta: &Array<B>, output: &mut Array<F>,
    mean: &mut Array<f32>, rstd: &mut Array<f32>, width: u32, epsilon: f32,
    #[comptime] has_beta: bool,
    #[define(F, W, B)] _types: [StorageType; 3],
) {
    let (mu, inverse_std) = row_statistics(input, width, epsilon);
    let row = RUDA_POS_X as usize;
    if UNIT_POS == 0 { mean[row] = mu; rstd[row] = inverse_std; }
    let width = width as usize;
    let mut column = UNIT_POS as usize;
    while column < width {
        let normalized = inverse_std * (f32::cast_from(input[row * width + column]) - mu);
        let mut value = f32::cast_from(gamma[column]) * normalized;
        if has_beta { value = fma(f32::cast_from(gamma[column]), normalized, f32::cast_from(beta[column])); }
        output[row * width + column] = F::cast_from(value);
        if width - column <= RUDA_DIM as usize { break; }
        column += RUDA_DIM as usize;
    }
}

#[ruda]
pub(super) fn block_sum_pair(first: f32, second: f32) -> (f32, f32) {
    let mut first_parts = SharedMemory::<f32>::new(4usize);
    let mut second_parts = SharedMemory::<f32>::new(4usize);
    let first = plane_sum(first);
    let second = plane_sum(second);
    if UNIT_POS_X == 0 {
        first_parts[UNIT_POS_Y as usize] = first;
        second_parts[UNIT_POS_Y as usize] = second;
    }
    sync_ruda();
    if UNIT_POS == 0 {
        let mut first_total = 0.0f32;
        let mut second_total = 0.0f32;
        #[unroll]
        for plane in 0usize..4usize {
            first_total += first_parts[plane];
            second_total += second_parts[plane];
        }
        first_parts[0] = first_total;
        second_parts[0] = second_total;
    }
    sync_ruda();
    (first_parts[0], second_parts[0])
}

#[ruda(launch)]
pub(crate) fn layer_norm_input_backward<F: Float, W: Float, G: Float>(
    input: &Array<F>, gamma: &Array<W>, grad: &Array<G>, mean: &Array<f32>, rstd: &Array<f32>,
    output: &mut Array<F>, width: u32,
    #[define(F, W, G)] _types: [StorageType; 3],
) {
    let row = RUDA_POS_X as usize;
    let width = width as usize;
    let base = row * width;
    let mu = mean[row];
    let inv = rstd[row];
    let mut sum_grad = 0.0f32;
    let mut sum_product = 0.0f32;
    let mut column = UNIT_POS as usize;
    while column < width {
        let scaled = f32::cast_from(grad[base + column]) * f32::cast_from(gamma[column]);
        let normalized = (f32::cast_from(input[base + column]) - mu) * inv;
        sum_grad += scaled;
        sum_product += scaled * normalized;
        if width - column <= RUDA_DIM as usize { break; }
        column += RUDA_DIM as usize;
    }
    let (sum_grad, sum_product) = block_sum_pair(sum_grad, sum_product);
    let average_grad = sum_grad / width as f32;
    let average_product = sum_product / width as f32;
    column = UNIT_POS as usize;
    while column < width {
        let scaled = f32::cast_from(grad[base + column]) * f32::cast_from(gamma[column]);
        let normalized = (f32::cast_from(input[base + column]) - mu) * inv;
        output[base + column] = F::cast_from(inv * (scaled - average_grad - normalized * average_product));
        if width - column <= RUDA_DIM as usize { break; }
        column += RUDA_DIM as usize;
    }
}

#[ruda(launch)]
pub(crate) fn layer_norm_affine_partial<F: Float, G: Float>(
    input: &Array<F>, grad: &Array<G>, mean: &Array<f32>, rstd: &Array<f32>,
    weight_parts: &mut Array<f32>, bias_parts: &mut Array<f32>, width: u32, parts: u32,
    #[define(F, G)] _types: [StorageType; 2],
) {
    let position = ABSOLUTE_POS as usize;
    if position >= weight_parts.len() { terminate!(); }
    let width = width as usize;
    let parts = parts as usize;
    let column = position % width;
    let mut row = position / width;
    let mut weight_sum = 0.0f32;
    let mut bias_sum = 0.0f32;
    while row < mean.len() {
        let value = f32::cast_from(grad[row * width + column]);
        let normalized = (f32::cast_from(input[row * width + column]) - mean[row]) * rstd[row];
        weight_sum += value * normalized;
        bias_sum += value;
        if mean.len() - row <= parts { break; }
        row += parts;
    }
    weight_parts[position] = weight_sum;
    bias_parts[position] = bias_sum;
}

#[ruda(launch)]
pub(crate) fn layer_norm_affine_merge<W: Float>(
    weight_parts: &Array<f32>, bias_parts: &Array<f32>,
    weight: &mut Array<W>, bias: &mut Array<f32>, parts: u32,
    #[define(W)] _storage: StorageType,
) {
    let column = ABSOLUTE_POS as usize;
    if column >= weight.len() { terminate!(); }
    let mut weight_sum = 0.0f32;
    let mut bias_sum = 0.0f32;
    for part in 0..parts as usize {
        weight_sum += weight_parts[part * weight.len() + column];
        bias_sum += bias_parts[part * bias.len() + column];
    }
    weight[column] = W::cast_from(weight_sum);
    bias[column] = bias_sum;
}

#[ruda(launch)]
pub(crate) fn layer_norm_weight_partial<F: Float, G: Float>(
    input: &Array<F>, grad: &Array<G>, mean: &Array<f32>, rstd: &Array<f32>,
    output: &mut Array<f32>, width: u32, parts: u32,
    #[define(F, G)] _types: [StorageType; 2],
) {
    let position = ABSOLUTE_POS as usize;
    if position >= output.len() { terminate!(); }
    let width = width as usize;
    let column = position % width;
    let mut row = position / width;
    let mut sum = 0.0f32;
    while row < mean.len() {
        let normalized = (f32::cast_from(input[row * width + column]) - mean[row]) * rstd[row];
        sum += f32::cast_from(grad[row * width + column]) * normalized;
        if mean.len() - row <= parts as usize { break; }
        row += parts as usize;
    }
    output[position] = sum;
}

#[ruda(launch)]
pub(crate) fn layer_norm_bias_partial<G: Float>(
    grad: &Array<G>, output: &mut Array<f32>, width: u32, rows: u32, parts: u32,
    #[define(G)] _storage: StorageType,
) {
    let position = ABSOLUTE_POS as usize;
    if position >= output.len() { terminate!(); }
    let width = width as usize;
    let column = position % width;
    let mut row = position / width;
    let mut sum = 0.0f32;
    while row < rows as usize {
        sum += f32::cast_from(grad[row * width + column]);
        if rows as usize - row <= parts as usize { break; }
        row += parts as usize;
    }
    output[position] = sum;
}
