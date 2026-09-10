// SPDX-License-Identifier: Apache-2.0
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

// Fixed power-of-two block; every lane participates in every barrier, including
// inactive tail lanes. No float atomics or warp-size assumption is required.
// A partial triple is (scale, sumsq, bad), representing scale^2 * sumsq.
// Never form g*g in FP32: finite g near 1e30 must NOT become an overflow flag.
#[ruda(launch)]
pub(super) fn scaled_l2<G: Float>(
    input: &Array<G>, output: &mut Array<f32>, inverse_scale: f32,
    #[comptime] partial_input: bool, #[comptime] _source: String,
    #[define(G)] _dtype: StorageType,
) {
    let lane = UNIT_POS as usize;
    let mut scales = SharedMemory::<f32>::new(256usize);
    let mut sums = SharedMemory::<f32>::new(256usize);
    let mut bads = SharedMemory::<u32>::new(256usize);
    let mut scale = 0.0f32;
    let mut sum = 0.0f32;
    let mut bad = 0u32;
    let mut count = input.len();
    if comptime!(partial_input) { count /= 3; }
    let mut index = ABSOLUTE_POS;
    let stride = RUDA_COUNT_X as usize * RUDA_DIM as usize;
    while index < count {
        let mut next_scale = 0.0f32;
        let mut next_sum = 0.0f32;
        if comptime!(partial_input) {
            next_scale = f32::cast_from(input[index * 3]);
            next_sum = f32::cast_from(input[index * 3 + 1]);
            if f32::cast_from(input[index * 3 + 2]) != 0.0 { bad = 1; }
        } else {
            let raw = f32::cast_from(input[index]);
            let value = raw * inverse_scale;
            if raw.is_nan() || raw.is_inf() || value.is_nan() || value.is_inf() {
                bad = 1;
            } else {
                next_scale = value.abs();
                if next_scale > 0.0 { next_sum = 1.0; }
            }
        }
        if next_scale > scale {
            let ratio = scale / next_scale;
            sum = next_sum + sum * (ratio * ratio);
            scale = next_scale;
        } else if next_scale > 0.0 {
            let ratio = next_scale / scale;
            sum += next_sum * (ratio * ratio);
        }
        index += stride;
    }
    scales[lane] = scale;
    sums[lane] = sum;
    bads[lane] = bad;
    sync_ruda();
    let mut offset = 128usize;
    while offset > 0 {
        if lane < offset {
            let a = scales[lane];
            let b = scales[lane + offset];
            let sa = sums[lane];
            let sb = sums[lane + offset];
            if b > a {
                let ratio = a / b;
                scales[lane] = b;
                sums[lane] = sb + sa * (ratio * ratio);
            } else if b > 0.0 {
                let ratio = b / a;
                sums[lane] = sa + sb * (ratio * ratio);
            }
            let combined_bad = bads[lane] | bads[lane + offset];
            bads[lane] = combined_bad;
        }
        sync_ruda();
        offset /= 2;
    }
    if lane == 0 {
        let base = RUDA_POS_X as usize * 3;
        output[base] = scales[0];
        output[base + 1] = sums[0];
        output[base + 2] = f32::cast_from(bads[0]);
    }
}
