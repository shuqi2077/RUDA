//! One full hardware plane per token; no dense probability workspace or atomics.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn probability(x: f32, maximum: f32, total: f32, #[comptime] softmax: bool) -> f32 {
    if comptime!(softmax) { (x - maximum).exp() / total }
    else {
        // Avoid exp(-x) overflow for negative sigmoid logits.
        let mut p = 0.0f32;
        if x >= 0.0 { p = 1.0 / (1.0 + (-x).exp()); }
        else { let e = x.exp(); p = e / (1.0 + e); }
        p
    }
}

#[ruda]
fn softmax_stats<F: Float>(logits: &Array<F>, row: usize, lane: usize,
    #[comptime] experts: usize, #[comptime] lanes: usize, #[comptime] softmax: bool) -> (f32, f32)
{
    let mut maximum = 0.0f32; let mut total = 1.0f32;
    if comptime!(softmax) {
        let mut local_max = f32::cast_from(f32::NEG_INFINITY);
        let mut nan = f32::cast_from(0.0f32);
        let mut j = lane;
        while j < experts {
            let x = f32::cast_from(logits[row * experts + j]);
            if x != x { nan = 1.0; }
            local_max = f32::max(local_max, x); j += lanes;
        }
        maximum = plane_max(local_max);
        if plane_sum(nan) != 0.0 { maximum = f32::NAN; }
        let mut local_sum = 0.0f32; j = lane;
        while j < experts { local_sum += (f32::cast_from(logits[row * experts + j]) - maximum).exp(); j += lanes; }
        total = plane_sum(local_sum);
    }
    (maximum, total)
}

#[ruda(launch)]
pub(crate) fn weights<F: Float, I: Int>(logits: &Array<F>, ids: &Array<I>, output: &mut Array<f32>,
    #[comptime] experts: usize, #[comptime] k: usize, #[comptime] lanes: usize,
    #[comptime] softmax: bool, #[comptime] renormalize: bool, scale: f32,
    #[define(F, I)] _dtypes: [StorageType; 2])
{
    let row = RUDA_POS_X as usize; let lane = UNIT_POS_X as usize;
    let (maximum, total) = softmax_stats(logits, row, lane, experts, lanes, softmax);
    let mut sum = 0.0f32; let mut invalid = f32::cast_from(0.0f32); let mut slot = lane;
    while slot < k {
        // Bounds are checked in 64 bits BEFORE narrowing or address arithmetic.
        let expert = i64::cast_from(ids[row * k + slot]);
        if expert < 0i64 || expert >= experts as i64 { invalid = 1.0; }
        else { sum += probability(f32::cast_from(logits[row * experts + expert as usize]), maximum, total, softmax); }
        slot += lanes;
    }
    let selected_sum = plane_sum(sum); let bad = plane_sum(invalid);
    slot = lane;
    while slot < k {
        let mut w = f32::cast_from(f32::NAN);
        if bad == 0.0 {
            let expert = i64::cast_from(ids[row * k + slot]) as usize;
            w = probability(f32::cast_from(logits[row * experts + expert]), maximum, total, softmax);
            if comptime!(renormalize) { w /= selected_sum; }
            w *= scale;
        }
        output[row * k + slot] = w; slot += lanes;
    }
}

#[ruda(launch)]
pub(crate) fn backward<F: Float, I: Int>(logits: &Array<F>, ids: &Array<I>, grad: &Array<f32>, output: &mut Array<F>,
    #[comptime] experts: usize, #[comptime] k: usize, #[comptime] lanes: usize,
    #[comptime] softmax: bool, #[comptime] renormalize: bool, scale: f32,
    #[define(F, I)] _dtypes: [StorageType; 2])
{
    let row = RUDA_POS_X as usize; let lane = UNIT_POS_X as usize;
    let (maximum, total) = softmax_stats(logits, row, lane, experts, lanes, softmax);
    let mut sum = 0.0f32; let mut dot = 0.0f32; let mut invalid = f32::cast_from(0.0f32); let mut slot = lane;
    while slot < k {
        let expert = i64::cast_from(ids[row * k + slot]);
        if expert < 0i64 || expert >= experts as i64 { invalid = 1.0; }
        else {
            let p = probability(f32::cast_from(logits[row * experts + expert as usize]), maximum, total, softmax);
            sum += p; dot += p * grad[row * k + slot];
        }
        slot += lanes;
    }
    let selected_sum = plane_sum(sum); let weighted_grad = plane_sum(dot); let bad = plane_sum(invalid);
    let mut expert = lane;
    while expert < experts {
        let mut dx = f32::cast_from(f32::NAN);
        if bad == 0.0 {
            let p = probability(f32::cast_from(logits[row * experts + expert]), maximum, total, softmax);
            let mut g = 0.0f32; let mut hits = 0.0f32; let mut s = 0usize;
            while s < k {
                if i64::cast_from(ids[row * k + s]) == expert as i64 {
                    g += grad[row * k + s]; hits += 1.0;
                }
                s += 1;
            }
            if comptime!(renormalize) {
                let dp = scale * (g - hits * (weighted_grad / selected_sum)) / selected_sum;
                if comptime!(softmax) { dx = p * dp; }
                else { dx = p * (1.0 - p) * dp; }
            } else {
                if comptime!(softmax) {
                    // Includes nonselected experts' gradients through the full denominator.
                    dx = scale * p * (g - weighted_grad);
                } else { dx = scale * p * (1.0 - p) * g; }
            }
        }
        output[row * experts + expert] = F::cast_from(dx); expert += lanes;
    }
}
