// SPDX-License-Identifier: Apache-2.0
//! Benchmark-only decomposed baseline: 4 kernels (5 with AMSGrad), not an estimate
//! of existing high-level AdamW which may already be fused by another backend.
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::{calculate_ruda_count_elemwise, prelude::*}, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype}};
use ruda_optim::fused_adamw::{AdamWOptions, AdamWState, AdamWUpdate, StepControl};

pub fn staged_step<R: Runtime>(p: &RudaTensor<R>, g: &RudaTensor<R>, s: Option<&AdamWState<R>>, o: &AdamWOptions) -> AdamWUpdate<R> {
    assert_eq!(p.dtype, DType::F32);
    assert_eq!(p.meta.shape(), g.meta.shape());
    assert!(p.is_contiguous() && g.is_contiguous() && p.client.same_execution_queue(&g.client));
    let n = p.meta.num_elements();
    assert!(n > 0, "benchmark uses nonempty inputs");
    let c = o.prepare_step(s.map_or(0, AdamWState::step), StepControl::default()).unwrap();
    let alloc = || empty_device_contiguous_dtype(p.client.clone(), p.device.clone(), p.meta.shape().clone(), DType::F32);
    let m = alloc();
    let v = alloc();
    let maximum = o.amsgrad.then(alloc);
    let delta = alloc();
    let output = alloc();
    let old_m = s.map_or(p, AdamWState::first_moment);
    let old_v = s.map_or(p, AdamWState::second_moment);
    let dim = RudaDim::new(p.client.properties(), n);
    let count = calculate_ruda_count_elemwise(&p.client, n, dim);
    // No implicit fusion: each function below is an explicit kernel submission.
    moment::launch::<R>(&p.client, count.clone(), dim,
        g.clone().into_array_arg(), old_m.clone().into_array_arg(), m.clone().into_array_arg(),
        o.beta1, false, o.maximize, s.is_some(), include_str!("staged.rs").to_owned(), g.dtype.into());
    moment::launch::<R>(&p.client, count.clone(), dim,
        g.clone().into_array_arg(), old_v.clone().into_array_arg(), v.clone().into_array_arg(),
        o.beta2, true, o.maximize, s.is_some(), include_str!("staged.rs").to_owned(), g.dtype.into());
    let variance = if let Some(maximum) = &maximum {
        maximum_kernel::launch::<R>(&p.client, count.clone(), dim,
            v.clone().into_array_arg(), s.and_then(AdamWState::max_second_moment).unwrap_or(p).clone().into_array_arg(),
            maximum.clone().into_array_arg(), s.is_some(), include_str!("staged.rs").to_owned());
        maximum
    } else { &v };
    direction::launch::<R>(&p.client, count.clone(), dim,
        m.clone().into_array_arg(), variance.clone().into_array_arg(), delta.clone().into_array_arg(),
        c.inverse_bias1, c.inverse_bias2, o.epsilon, include_str!("staged.rs").to_owned());
    update::launch::<R>(&p.client, count, dim,
        p.clone().into_array_arg(), delta.into_array_arg(), output.clone().into_array_arg(),
        o.learning_rate, c.decay_multiplier, include_str!("staged.rs").to_owned());
    AdamWUpdate { parameters: output, state: Some(AdamWState::from_parts(c.step, m, v, maximum).unwrap()), updated: true }
}

#[ruda(launch)]
fn moment<G: Float>(gradient: &Array<G>, previous: &Array<f32>, output: &mut Array<f32>, beta: f32,
    #[comptime] square: bool, #[comptime] maximize: bool, #[comptime] initialized: bool,
    #[comptime] _source: String, #[define(G)] _dtype: StorageType) {
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    let mut g = f32::cast_from(gradient[i]);
    if comptime!(maximize) { g = -g; }
    if comptime!(square) { g = g * g; }
    let mut old = 0.0f32;
    if comptime!(initialized) { old = previous[i]; }
    output[i] = beta * old + (1.0 - beta) * g;
}
#[ruda(launch)]
fn maximum_kernel(v: &Array<f32>, previous: &Array<f32>, output: &mut Array<f32>,
    #[comptime] initialized: bool, #[comptime] _source: String) {
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    let mut old = 0.0f32;
    if comptime!(initialized) { old = previous[i]; }
    if old.is_nan() || v[i].is_nan() { output[i] = old + v[i]; }
    else { output[i] = f32::max(old, v[i]); }
}
#[ruda(launch)]
fn direction(m: &Array<f32>, v: &Array<f32>, output: &mut Array<f32>, inv1: f32, inv2: f32, eps: f32,
    #[comptime] _source: String) {
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    output[i] = (m[i] * inv1) / ((v[i] * inv2).sqrt() + eps);
}
#[ruda(launch)]
fn update(p: &Array<f32>, delta: &Array<f32>, output: &mut Array<f32>, lr: f32, decay: f32,
    #[comptime] _source: String) {
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    output[i] = p[i] * decay - lr * delta[i];
}
