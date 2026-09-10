// SPDX-License-Identifier: Apache-2.0
//! CPU oracle, called explicitly. The device implementation never dispatches here.
//!
//! Arithmetic is evaluated in FP64 and persistent state is rounded to FP32 after
//! every step. Device FP32 reassociation/FMA is compared with a tolerance, not bit
//! equality. Half-precision gradients must first be rounded to their storage type
//! and then converted to FP32 by the test caller.
use super::{AdamWOptions, FusedAdamWError, StepControl};

/// Explicit host state for independent correctness tests and fixtures.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceState {
    /// Number of completed updates.
    pub step: u64,
    /// FP32 first moments.
    pub first: Vec<f32>,
    /// FP32 second moments.
    pub second: Vec<f32>,
    /// FP32 running maximum, present iff AMSGrad is selected.
    pub maximum: Option<Vec<f32>>,
}

/// Result of the independent CPU oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceUpdate {
    /// Updated parameters, or a copy of the original for an empty/skipped call.
    pub parameters: Vec<f32>,
    /// Empty/skipped calls preserve an absent state.
    pub state: Option<ReferenceState>,
    /// Whether a mathematical update was performed.
    pub updated: bool,
}

/// Compute the reference formula without modifying any input.
pub fn adamw_reference(
    parameters: &[f32],
    gradients: &[f32],
    state: Option<&ReferenceState>,
    options: &AdamWOptions,
    control: StepControl,
) -> Result<ReferenceUpdate, FusedAdamWError> {
    options.validate()?;
    control.validate()?;
    if parameters.len() != gradients.len() {
        return Err(FusedAdamWError::ShapeMismatch("gradient"));
    }
    if let Some(s) = state {
        if s.step == 0 || s.maximum.is_some() != options.amsgrad {
            return Err(FusedAdamWError::InvalidState("step/AMSGrad mode"));
        }
        if s.first.len() != parameters.len() || s.second.len() != parameters.len()
            || s.maximum.as_ref().is_some_and(|x| x.len() != parameters.len())
        {
            return Err(FusedAdamWError::ShapeMismatch("moment"));
        }
    }
    if parameters.is_empty() || control.skip_update {
        return Ok(ReferenceUpdate { parameters: parameters.to_vec(), state: state.cloned(), updated: false });
    }
    let previous_step = state.map_or(0, |s| s.step);
    let step = previous_step.checked_add(1).ok_or(FusedAdamWError::StepOverflow)?;
    let b1 = options.beta1 as f64;
    let b2 = options.beta2 as f64;
    // Independent of the production exponentiation helper.
    let correction1 = 1.0 - b1.powf(step as f64);
    let correction2 = 1.0 - b2.powf(step as f64);
    let lr = options.learning_rate as f64;
    let decay = 1.0 - lr * options.weight_decay as f64;
    let mut result = ReferenceUpdate {
        parameters: Vec::with_capacity(parameters.len()),
        state: Some(ReferenceState {
            step, first: Vec::with_capacity(parameters.len()), second: Vec::with_capacity(parameters.len()),
            maximum: options.amsgrad.then(|| Vec::with_capacity(parameters.len())),
        }),
        updated: true,
    };
    let next = result.state.as_mut().expect("state initialized above");
    for i in 0..parameters.len() {
        let mut g = gradients[i] as f64 / control.gradient_scale as f64;
        if options.maximize { g = -g; }
        let m = (b1 * state.map_or(0.0, |s| s.first[i] as f64) + (1.0 - b1) * g) as f32;
        let v = (b2 * state.map_or(0.0, |s| s.second[i] as f64) + (1.0 - b2) * g * g) as f32;
        next.first.push(m);
        next.second.push(v);
        let used_v = if let Some(maximum) = next.maximum.as_mut() {
            let old = state.and_then(|s| s.maximum.as_ref()).map_or(0.0, |s| s[i]);
            let value = if old.is_nan() || v.is_nan() { f32::NAN } else { old.max(v) };
            maximum.push(value);
            value
        } else { v };
        let denominator = (used_v as f64 / correction2).sqrt() + options.epsilon as f64;
        result.parameters.push((parameters[i] as f64 * decay - lr * (m as f64 / correction1) / denominator) as f32);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f32, b: f32) { assert!((a - b).abs() <= 3e-6 * b.abs().max(1.0), "{a} != {b}"); }

    #[test]
    fn first_step_known_values_and_epsilon_outside_sqrt() {
        let o = AdamWOptions { learning_rate: 0.1, beta1: 0.0, beta2: 0.0, epsilon: 0.5, weight_decay: 0.2, ..Default::default() };
        let r = adamw_reference(&[2.0, -1.0, 0.0], &[1.0, -2.0, 0.0], None, &o, StepControl::default()).unwrap();
        close(r.parameters[0], 2.0 * 0.98 - 0.1 / 1.5);
        close(r.parameters[1], -0.98 + 0.2 / 2.5);
        close(r.parameters[2], 0.0);
        assert_eq!(r.state.unwrap().first, [1.0, -2.0, 0.0]);
    }
    #[test]
    fn scale_and_maximize_do_not_change_decay_direction() {
        let o = AdamWOptions { maximize: true, ..Default::default() };
        let scaled = adamw_reference(&[1.0], &[128.0], None, &o, StepControl { gradient_scale: 128.0, skip_update: false }).unwrap();
        let plain = adamw_reference(&[1.0], &[-1.0], None, &AdamWOptions { maximize: false, ..o }, StepControl::default()).unwrap();
        assert_eq!(scaled, plain);
    }
    #[test]
    fn skip_is_transactional_even_for_nan_gradients() {
        let o = AdamWOptions { amsgrad: true, ..Default::default() };
        let first = adamw_reference(&[1.0, 2.0], &[0.1, -0.3], None, &o, StepControl::default()).unwrap();
        let snapshot = first.clone();
        let skipped = adamw_reference(&first.parameters, &[f32::NAN, f32::INFINITY], first.state.as_ref(), &o, StepControl { skip_update: true, ..Default::default() }).unwrap();
        assert_eq!(skipped.parameters, snapshot.parameters);
        assert_eq!(skipped.state, snapshot.state);
        assert!(!skipped.updated);
        assert_eq!(first, snapshot);
    }
    #[test]
    fn empty_and_first_skip_keep_state_absent() {
        let o = AdamWOptions::default();
        assert_eq!(adamw_reference(&[], &[], None, &o, StepControl::default()).unwrap().state, None);
        let r = adamw_reference(&[1.0], &[f32::NAN], None, &o, StepControl { skip_update: true, ..Default::default() }).unwrap();
        assert_eq!(r.parameters, [1.0]);
        assert!(r.state.is_none());
    }
    #[test]
    fn zero_learning_rate_still_updates_moments() {
        let o = AdamWOptions { learning_rate: 0.0, ..Default::default() };
        let r = adamw_reference(&[1.0], &[3.0], None, &o, StepControl::default()).unwrap();
        assert_eq!(r.parameters, [1.0]);
        let s = r.state.unwrap();
        assert_eq!(s.step, 1);
        assert!(s.first[0] > 0.0 && s.second[0] > 0.0);
    }
    #[test]
    fn amsgrad_preserves_historical_maximum() {
        let o = AdamWOptions { beta2: 0.5, amsgrad: true, ..Default::default() };
        let a = adamw_reference(&[1.0], &[4.0], None, &o, StepControl::default()).unwrap();
        let b = adamw_reference(&a.parameters, &[0.0], a.state.as_ref(), &o, StepControl::default()).unwrap();
        let s = b.state.unwrap();
        assert_eq!(s.second, [4.0]);
        assert_eq!(s.maximum.unwrap(), [8.0]);
    }
    #[test]
    fn restored_state_matches_uninterrupted_steps() {
        let o = AdamWOptions { amsgrad: true, ..Default::default() };
        let first = adamw_reference(&[0.5, -0.1], &[0.3, 0.7], None, &o, StepControl::default()).unwrap();
        let snapshot = first.clone();
        let next = adamw_reference(&first.parameters, &[-0.1, 0.2], first.state.as_ref(), &o, StepControl::default()).unwrap();
        let resumed = adamw_reference(&snapshot.parameters, &[-0.1, 0.2], snapshot.state.as_ref(), &o, StepControl::default()).unwrap();
        assert_eq!(next, resumed);
        assert_eq!(next.state.unwrap().step, 2);
    }
    #[test]
    fn invalid_state_and_lengths_reject() {
        let o = AdamWOptions::default();
        assert!(adamw_reference(&[1.0], &[], None, &o, StepControl::default()).is_err());
        let s = ReferenceState { step: 0, first: vec![0.0], second: vec![0.0], maximum: None };
        assert!(adamw_reference(&[1.0], &[0.0], Some(&s), &o, StepControl::default()).is_err());
        let s = ReferenceState { step: 1, ..s };
        assert!(adamw_reference(&[1.0], &[0.0], Some(&s), &AdamWOptions { amsgrad: true, ..o }, StepControl::default()).is_err());
        let s = ReferenceState { step: u64::MAX, ..s };
        assert_eq!(adamw_reference(&[1.0], &[0.0], Some(&s), &o, StepControl::default()).unwrap_err(), FusedAdamWError::StepOverflow);
    }
}
