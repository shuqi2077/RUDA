// SPDX-License-Identifier: Apache-2.0
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn adamw<G: Float>(
    parameters: &Array<f32>, gradients: &Array<G>,
    old_first: &Array<f32>, old_second: &Array<f32>, old_maximum: &Array<f32>,
    parameters_out: &mut Array<f32>, first_out: &mut Array<f32>,
    second_out: &mut Array<f32>, maximum_out: &mut Array<f32>,
    learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32,
    decay_multiplier: f32, inverse_bias1: f32, inverse_bias2: f32, inverse_scale: f32, clip_multiplier: f32,
    #[comptime] initialized: bool, #[comptime] amsgrad: bool, #[comptime] maximize: bool, #[comptime] clipped: bool,
    #[comptime] _source: String, #[define(G)] _gradient_dtype: StorageType,
) {
    let i = ABSOLUTE_POS;
    if i >= parameters_out.len() { terminate!(); }
    let mut gradient = f32::cast_from(gradients[i]) * inverse_scale;
    if comptime!(clipped) { gradient *= clip_multiplier; }
    if comptime!(maximize) { gradient = -gradient; }
    let mut old_m = 0.0f32;
    let mut old_v = 0.0f32;
    if comptime!(initialized) {
        old_m = old_first[i];
        old_v = old_second[i];
    }
    let first = beta1 * old_m + (1.0 - beta1) * gradient;
    let second = beta2 * old_v + (1.0 - beta2) * (gradient * gradient);
    first_out[i] = first;
    second_out[i] = second;
    let mut variance = second;
    if comptime!(amsgrad) {
        let mut previous_maximum = 0.0f32;
        if comptime!(initialized) { previous_maximum = old_maximum[i]; }
        // Explicit NaN propagation rather than backend-dependent max(NaN, x).
        if previous_maximum.is_nan() || second.is_nan() {
            variance = previous_maximum + second;
        } else {
            variance = f32::max(previous_maximum, second);
        }
        maximum_out[i] = variance;
    }
    let numerator = first * inverse_bias1;
    let denominator = (variance * inverse_bias2).sqrt() + epsilon;
    let update = numerator / denominator;
    parameters_out[i] = parameters[i] * decay_multiplier - learning_rate * update;
}
