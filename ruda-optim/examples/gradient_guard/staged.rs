// SPDX-License-Identifier: Apache-2.0
//! Explicit baseline for prevalidated benchmark fixtures, NOT a public optimizer.
//! Same statistics/FP32 clip decision, but materializes one F32 gradient buffer
//! and launches a separate unscale+clip kernel before the original AdamW path.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::{dsl::{prelude::*, calculate_ruda_count_elemwise}, tensor::allocation::empty_device_contiguous_dtype};
use ruda_core::tensor::DType;
use ruda_optim::fused_adamw::{
    AdamWEntry, AdamWOptions, AdamWUpdate, GuardedAdamWUpdate, SkipReason, StepControl,
    adamw_step, gradient_stats_sync,
    gradient_norm::{GradientGuardOptions, GradientGuardError},
};

#[ruda(launch)]
fn materialize<G: Float>(
    input: &Array<G>, output: &mut Array<f32>, inverse_scale: f32, clip: f32,
    #[comptime] _source: String, #[define(G)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS;
    if i >= output.len() { terminate!(); }
    let unscaled = f32::cast_from(input[i]) * inverse_scale;
    output[i] = unscaled * clip;
}

pub fn step<R: Runtime>(
    entries: &[AdamWEntry<'_, R>], options: &AdamWOptions,
    control: StepControl, guard: GradientGuardOptions,
) -> Result<GuardedAdamWUpdate<R>, GradientGuardError> {
    // Tests/benchmark deliberately supply nonempty, contiguous, same-queue inputs.
    assert!(!entries.is_empty() && !control.skip_update);
    options.validate()?; control.validate()?; guard.validate()?;
    let gradients: Vec<_> = entries.iter().map(|e| e.gradients).collect();
    let statistics = gradient_stats_sync(&gradients, control.gradient_scale)?;
    let decision = statistics.stats.decision(guard)?;
    if decision.skip_update {
        return Ok(GuardedAdamWUpdate {
            updates: entries.iter().map(|e| AdamWUpdate { parameters: e.parameters.clone(), state: e.state.cloned(), updated: false }).collect(),
            statistics: Some(statistics), clip_multiplier: 1.0, skip_reason: Some(SkipReason::NonFinite),
        });
    }
    let mut updates = Vec::with_capacity(entries.len());
    for e in entries {
        let g = e.gradients;
        let n = g.meta.shape().iter().product::<usize>();
        let out = empty_device_contiguous_dtype(g.client.clone(), g.device.clone(), g.meta.shape().clone(), DType::F32);
        if n > 0 {
            let dim = RudaDim::new(g.client.properties(), n);
            materialize::launch::<R>(
                &g.client, calculate_ruda_count_elemwise(&g.client, n, dim), dim,
                g.clone().into_array_arg(), out.clone().into_array_arg(), control.gradient_scale.recip(),
                decision.clip_multiplier, include_str!("staged.rs").to_owned(), g.dtype.into(),
            );
        }
        updates.push(adamw_step(e.parameters, &out, e.state, options, StepControl::default())?);
    }
    Ok(GuardedAdamWUpdate { updates, statistics: Some(statistics), clip_multiplier: decision.clip_multiplier, skip_reason: None })
}
