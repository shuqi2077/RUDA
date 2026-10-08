use super::{NormalizationError, RudaTensor, Runtime};
use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{allocation::empty_device_contiguous_dtype, contiguous::into_contiguous};

/// Last-axis RMSNorm with FP32 statistics and affine arithmetic, then one output cast.
/// Input/output are F32, F16 or BF16; gamma is a same-device F32 feature vector.
pub fn rms_norm<R: Runtime>(
    input: RudaTensor<R>, gamma: RudaTensor<R>, epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    let shape = input.meta.shape();
    let width = *shape.last().ok_or(NormalizationError("RMSNorm requires an axis"))?;
    let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if width == 0 || width > u32::MAX as usize
        || elements.is_none_or(|n| n > u32::MAX as usize)
        || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        || input.qparams.is_some() || !epsilon.is_finite() || epsilon <= 0.0
    {
        return Err(NormalizationError("invalid RMSNorm shape, dtype or epsilon"));
    }
    if gamma.meta.shape()[..] != [width] || gamma.dtype != DType::F32
        || gamma.qparams.is_some() || gamma.device.to_id() != input.device.to_id()
    {
        return Err(NormalizationError("RMSNorm gamma must be a same-device F32 feature vector"));
    }
    let plane = input.client.properties().hardware.plane_size_max;
    let maximum = input.client.properties().hardware.max_ruda_dim.0;
    if !plane.is_power_of_two() || plane > maximum {
        return Err(NormalizationError("RMSNorm requires a power-of-two plane"));
    }
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape.clone().into(), input.dtype);
    let rows = elements.unwrap() / width;
    if rows == 0 { return Ok(output); }
    let (threads, vectorized) = launch_parameters(width, rows, plane, maximum);
    let client = input.client.clone();
    let dtype = input.dtype;
    row_rms_norm::launch::<R>(
        &client, RudaCount::Static(rows as u32, 1, 1), RudaDim::new_1d(threads),
        into_contiguous(input).into_array_arg(), into_contiguous(gamma).into_array_arg(),
        output.clone().into_array_arg(), width as u32, epsilon, threads, vectorized,
        include_str!("rms.rs").to_owned(), dtype.into(),
    );
    Ok(output)
}

pub(super) fn launch_parameters(width: usize, rows: usize, plane: u32, maximum: u32) -> (u32, bool) {
    let vectorized = width >= 128;
    let floor_power = |n: usize| 1usize << n.ilog2();
    let columns = floor_power(if vectorized { width / 4 } else { width }).min(512);
    let lanes = columns.min(plane as usize);
    let row_groups = floor_power(rows.max(1)).min(512 / lanes);
    let threads = columns.min(512 / row_groups).min(maximum as usize).max(plane as usize);
    (floor_power(threads) as u32, vectorized)
}

#[ruda]
fn row_statistics<F: Float>(input: &Array<F>, width: u32, epsilon: f32,
    #[comptime] threads: u32, #[comptime] vectorized: bool) -> f32 {
    let width = width as usize;
    let base = RUDA_POS_X as usize * width;
    let lane = UNIT_POS as usize;
    let step = threads as usize;
    let mut sums = Array::<f32>::new(4usize);
    #[unroll]
    for i in 0usize..4usize { sums[i] = 0.0; }
    if comptime!(vectorized) {
        let end = width / 4 * 4;
        let mut column = lane * 4;
        while column < end {
            #[unroll]
            for i in 0usize..4usize {
                let value = f32::cast_from(input[base + column + i]);
                sums[i] += value * value;
            }
            if end - column <= step * 4 { break; }
            column += step * 4;
        }
        if lane < width - end {
            let value = f32::cast_from(input[base + end + lane]);
            sums[0] += value * value;
        }
    } else {
        let mut column = lane;
        while column < width {
            #[unroll]
            for i in 0usize..4usize {
                if i * step < width - column {
                    let index = column + i * step;
                    let value = f32::cast_from(input[base + index]);
                    sums[i] += value * value;
                }
            }
            if width - column <= step * 4 { break; }
            column += step * 4;
        }
    }
    let sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
    (reduce_row(sum, width as f32, threads) + epsilon).inverse_sqrt()
}

#[ruda]
fn reduce_row(value: f32, divisor: f32, #[comptime] threads: u32) -> f32 {
    let mut sum = value;
    let lane = UNIT_POS as usize;
    let mut shared = SharedMemory::<f32>::new(threads as usize);
    shared[lane] = sum;
    let mut offset = RUDA_DIM / 2;
    while offset >= PLANE_DIM {
        sync_ruda();
        if lane < offset as usize {
            sum += shared[lane + offset as usize];
            shared[lane] = sum;
        }
        offset /= 2;
    }
    sync_ruda();
    offset = PLANE_DIM / 2;
    while offset > 0 {
        sum += plane_shuffle_down(sum, offset);
        offset /= 2;
    }
    if lane == 0 { shared[0] = sum / divisor; }
    sync_ruda();
    shared[0]
}

#[ruda(launch)]
fn row_rms_norm<F: Float>(
    input: &Array<F>, gamma: &Array<f32>, output: &mut Array<F>, width: u32, epsilon: f32,
    #[comptime] threads: u32, #[comptime] vectorized: bool, #[comptime] _source: String,
    #[define(F)] _dtype: StorageType,
) {
    let inverse = row_statistics(input, width, epsilon, threads, vectorized);
    let width = width as usize;
    let base = RUDA_POS_X as usize * width;
    let lane = UNIT_POS as usize;
    let step = threads as usize;
    let mut column = lane;
    while column < width {
        let value = f32::cast_from(input[base + column]);
        output[base + column] = F::cast_from((value * inverse) * gamma[column]);
        if width - column <= step { break; }
        column += step;
    }
}

#[ruda(launch)]
pub(super) fn training_forward<F: Float, W: Float>(
    input: &Array<F>, gamma: &Array<W>, output: &mut Array<F>, rstd: &mut Array<f32>,
    width: u32, epsilon: f32, #[comptime] threads: u32, #[comptime] vectorized: bool,
    #[define(F, W)] _types: [StorageType; 2],
) {
    let inverse = row_statistics(input, width, epsilon, threads, vectorized);
    let row = RUDA_POS_X as usize;
    if UNIT_POS == 0 { rstd[row] = inverse; }
    let width = width as usize;
    let base = row * width;
    let mut column = UNIT_POS as usize;
    while column < width {
        let value = f32::cast_from(input[base + column]);
        output[base + column] = F::cast_from((value * inverse) * f32::cast_from(gamma[column]));
        if width - column <= threads as usize { break; }
        column += threads as usize;
    }
}

#[ruda(launch)]
pub(super) fn input_backward<F: Float, W: Float, G: Float>(
    input: &Array<F>, gamma: &Array<W>, grad: &Array<G>, rstd: &Array<f32>, output: &mut Array<F>,
    width: u32, #[comptime] threads: u32,
    #[define(F, W, G)] _types: [StorageType; 3],
) {
    let row = RUDA_POS_X as usize;
    let inverse = rstd[row];
    let width = width as usize;
    let base = row * width;
    let mut product = 0.0f32;
    let mut column = UNIT_POS as usize;
    while column < width {
        let normalized = f32::cast_from(input[base + column]) * inverse;
        let scaled_grad = f32::cast_from(grad[base + column]) * f32::cast_from(gamma[column]);
        product += scaled_grad * normalized;
        if width - column <= threads as usize { break; }
        column += threads as usize;
    }
    let average_product = reduce_row(product, width as f32, threads);
    column = UNIT_POS as usize;
    while column < width {
        let normalized = f32::cast_from(input[base + column]) * inverse;
        let scaled_grad = f32::cast_from(grad[base + column]) * f32::cast_from(gamma[column]);
        output[base + column] = F::cast_from(inverse * (scaled_grad - normalized * average_product));
        if width - column <= threads as usize { break; }
        column += threads as usize;
    }
}

#[ruda(launch)]
pub(super) fn weight_partial<F: Float, G: Float>(
    input: &Array<F>, grad: &Array<G>, rstd: &Array<f32>, output: &mut Array<f32>,
    width: u32, parts: u32, #[define(F, G)] _types: [StorageType; 2],
) {
    let position = ABSOLUTE_POS as usize;
    if position >= output.len() { terminate!(); }
    let width = width as usize;
    let column = position % width;
    let mut row = position / width;
    let mut sum = 0.0f32;
    while row < rstd.len() {
        let normalized = f32::cast_from(input[row * width + column]) * rstd[row];
        sum += f32::cast_from(grad[row * width + column]) * normalized;
        if rstd.len() - row <= parts as usize { break; }
        row += parts as usize;
    }
    output[position] = sum;
}

#[ruda(launch)]
pub(super) fn weight_merge<W: Float>(partial: &Array<f32>, output: &mut Array<W>, parts: u32,
    #[define(W)] _storage: StorageType) {
    let column = ABSOLUTE_POS as usize;
    if column >= output.len() { terminate!(); }
    let mut sum = 0.0f32;
    for part in 0..parts as usize { sum += partial[part * output.len() + column]; }
    output[column] = W::cast_from(sum);
}
