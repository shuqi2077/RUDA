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

#[ruda(launch)]
pub(crate) fn layer_norm<F: Float>(
    input: &Array<F>,
    gamma: &Array<f32>,
    beta: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    epsilon: f32,
    #[comptime] has_beta: bool,
    #[comptime] _source: String,
    #[define(F)] _dtype: StorageType,
) {
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
