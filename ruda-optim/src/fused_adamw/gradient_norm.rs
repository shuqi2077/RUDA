// SPDX-License-Identifier: Apache-2.0
//! Local optimizer-group L2 diagnostics and non-finite policy.
//!
//! The norm covers the supplied tensors as if concatenated, AFTER FP32 unscale.
//! It is not a distributed/FSDP norm: sharded callers must arrange their own
//! collective and replicated-gradient accounting. No tensor is modified here.
use super::{FusedAdamWError, StepControl};
use core::fmt;

/// Failure policy for NaN/Inf in raw OR unscaled gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonFinitePolicy {
    /// Skip the whole local optimizer group, retaining parameters and moments.
    Skip,
    /// Return an error before any optimizer-update kernel is submitted.
    Error,
}

/// Opt-in norm-clipping contract, deliberately separate from loss scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientGuardOptions {
    /// L2 limit across ALL selected tensors. `None` checks finiteness only.
    /// Zero is allowed: it zeroes the effective gradient, not weight decay.
    pub max_norm: Option<f32>,
    /// Positive finite stabilizer in `min(max_norm / (norm + epsilon), 1)`.
    pub epsilon: f32,
    /// Action on a non-finite gradient. Existing optimizer state is not inspected.
    pub nonfinite: NonFinitePolicy,
}
impl Default for GradientGuardOptions {
    fn default() -> Self {
        Self { max_norm: Some(1.0), epsilon: 1e-6, nonfinite: NonFinitePolicy::Skip }
    }
}
impl GradientGuardOptions {
    /// Check options without device work.
    pub fn validate(&self) -> Result<(), GradientGuardError> {
        if self.max_norm.is_some_and(|x| !x.is_finite() || x < 0.0) {
            return Err(GradientGuardError::InvalidOption("max_norm"));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(GradientGuardError::InvalidOption("epsilon"));
        }
        Ok(())
    }
}

/// Host validation, readback or numerical-policy error. Runtime allocation and
/// launch failures retain the runtime's existing error/panic contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientGuardError {
    /// Existing optimizer/tensor contract error.
    Optimizer(FusedAdamWError),
    /// Invalid clipping or launch configuration.
    InvalidOption(&'static str),
    /// The target does not support this reduction configuration.
    UnsupportedDevice(&'static str),
    /// The selected policy rejects non-finite gradients.
    NonFiniteGradients,
    /// Failed device synchronization/readback; no update was submitted.
    Readback(String),
    /// Invalid summary output or invalid summary aggregation.
    InvalidSummary(&'static str),
}
impl From<FusedAdamWError> for GradientGuardError {
    fn from(error: FusedAdamWError) -> Self { Self::Optimizer(error) }
}
impl fmt::Display for GradientGuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Optimizer(e) => write!(f, "{e}"),
            Self::InvalidOption(s) => write!(f, "invalid gradient guard option: {s}"),
            Self::UnsupportedDevice(s) => write!(f, "unsupported gradient reduction device: {s}"),
            Self::NonFiniteGradients => f.write_str("non-finite raw or FP32-unscaled gradient"),
            Self::Readback(s) => write!(f, "gradient summary readback failed: {s}"),
            Self::InvalidSummary(s) => write!(f, "invalid gradient summary: {s}"),
        }
    }
}
impl std::error::Error for GradientGuardError {}

/// Read-only statistics of a selected local group. FP64 host accumulation avoids
/// squaring a large FP32 gradient in FP32; device summaries use scaled sumsq.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientStats {
    sum_squares: f64,
    max_abs: f32,
    all_finite: bool,
    elements: u64,
    tensors: usize,
    gradient_scale: f32,
}
impl GradientStats {
    /// Norm of FP32-unscaled gradients; infinity if ANY raw/unscaled value is bad.
    pub fn total_norm(&self) -> f64 {
        if self.all_finite { self.sum_squares.sqrt() } else { f64::INFINITY }
    }
    /// Largest finite magnitude (non-finite values are excluded).
    pub fn max_finite_abs(&self) -> f32 { self.max_abs }
    /// Whether raw and FP32-unscaled gradients are all finite.
    pub fn all_finite(&self) -> bool { self.all_finite }
    /// Total elements, including bad ones. Empty tensors contribute zero.
    pub fn elements(&self) -> u64 { self.elements }
    /// Number of supplied tensors, including empty ones.
    pub fn tensors(&self) -> usize { self.tensors }
    /// The divisor used before measuring the norm.
    pub fn gradient_scale(&self) -> f32 { self.gradient_scale }

    /// Produce a single common coefficient for the entire selected group.
    /// Clipping uses FP64 norm arithmetic and rounds the coefficient to FP32.
    /// A coefficient too small for FP32 rounds to zero; it never amplifies.
    pub fn decision(&self, options: GradientGuardOptions) -> Result<GradientDecision, GradientGuardError> {
        options.validate()?;
        if !self.all_finite {
            return match options.nonfinite {
                NonFinitePolicy::Skip => Ok(GradientDecision { clip_multiplier: 1.0, skip_update: true }),
                NonFinitePolicy::Error => Err(GradientGuardError::NonFiniteGradients),
            };
        }
        let clip_multiplier = options.max_norm.map_or(1.0, |limit| {
            ((limit as f64 / (self.total_norm() + options.epsilon as f64)).min(1.0)) as f32
        });
        Ok(GradientDecision { clip_multiplier, skip_update: false })
    }

    pub(super) fn empty(gradient_scale: f32) -> Result<Self, GradientGuardError> {
        StepControl { gradient_scale, skip_update: false }.validate()?;
        Ok(Self { sum_squares: 0.0, max_abs: 0.0, all_finite: true, elements: 0, tensors: 0, gradient_scale })
    }

    // Summary: [scale, sum((g/scale)^2), any_bad]. One tensor per summary.
    pub(super) fn add_device_summary(&mut self, summary: &[f32], elements: usize) -> Result<(), GradientGuardError> {
        if summary.len() != 3 || !summary[0].is_finite() || summary[0] < 0.0
            || !summary[1].is_finite() || summary[1] < 0.0
            || (summary[2] != 0.0 && summary[2] != 1.0)
            || (summary[0] == 0.0 && summary[1] != 0.0)
            || (summary[0] > 0.0 && summary[1] < 1.0)
        {
            return Err(GradientGuardError::InvalidSummary("expected finite scaled sumsq and a 0/1 flag"));
        }
        let scale = summary[0] as f64;
        let addition = scale * scale * summary[1] as f64;
        let sum = self.sum_squares + addition;
        if !sum.is_finite() { return Err(GradientGuardError::InvalidSummary("FP64 sumsq overflow")); }
        let count = self.elements.checked_add(elements as u64)
            .ok_or(GradientGuardError::InvalidSummary("element count overflow"))?;
        let tensors = self.tensors.checked_add(1)
            .ok_or(GradientGuardError::InvalidSummary("tensor count overflow"))?;
        self.sum_squares = sum;
        self.max_abs = self.max_abs.max(summary[0]);
        self.all_finite &= summary[2] == 0.0;
        self.elements = count;
        self.tensors = tensors;
        Ok(())
    }
}

/// Host decision; applying it to different/stale gradients is the caller's error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientDecision {
    /// A common `[0, 1]` multiplier applied AFTER unscale, inside AdamW.
    pub clip_multiplier: f32,
    /// No weight decay, moment update or step increment is permitted when true.
    pub skip_update: bool,
}

/// Independently computed CPU reference. Gradients must first be rounded to their
/// storage format by the caller. This is an explicit test oracle, NOT a fallback.
/// Unscale intentionally uses the same FP32 reciprocal/multiply as AdamW, while
/// sumsq uses FP64 and a different reduction order than the device implementation.
pub fn gradient_stats_reference(gradients: &[&[f32]], gradient_scale: f32) -> Result<GradientStats, GradientGuardError> {
    let mut result = GradientStats::empty(gradient_scale)?;
    let inverse = gradient_scale.recip();
    for tensor in gradients {
        result.elements = result.elements.checked_add(tensor.len() as u64)
            .ok_or(GradientGuardError::InvalidSummary("element count overflow"))?;
        result.tensors += 1;
        for &raw in *tensor {
            let value = raw * inverse;
            if !raw.is_finite() || !value.is_finite() {
                result.all_finite = false;
                continue;
            }
            let value64 = value as f64;
            result.sum_squares += value64 * value64;
            result.max_abs = result.max_abs.max(value.abs());
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn norm(xs: &[&[f32]]) -> GradientStats { gradient_stats_reference(xs, 1.0).unwrap() }
    #[test]
    fn concatenated_norm_not_per_tensor_clipping() {
        let s = norm(&[&[3.0], &[4.0]]);
        assert_eq!(s.total_norm(), 5.0);
        assert_eq!((s.elements(), s.tensors()), (2, 2));
        let c = s.decision(GradientGuardOptions { max_norm: Some(1.0), ..Default::default() }).unwrap();
        assert!((c.clip_multiplier - 1.0 / (5.0 + 1e-6)).abs() < 1e-7);
    }
    #[test]
    fn scaled_norm_measured_after_unscale() {
        assert_eq!(gradient_stats_reference(&[&[384.0, 512.0]], 128.0).unwrap().total_norm(), 5.0);
    }
    #[test]
    fn finite_large_norm_does_not_overflow_when_fp32_square_would() {
        let x = 1e30f32;
        let s = norm(&[&[x, -x]]);
        assert!(s.total_norm().is_finite());
        assert!((s.total_norm() / x as f64 - 2.0f64.sqrt()).abs() < 1e-12);
    }
    #[test]
    fn small_normal_gradients_do_not_square_to_zero() {
        let x = 1e-30f32;
        assert!(norm(&[&[x, x]]).total_norm() > 0.0);
    }
    #[test]
    fn zero_and_empty_have_valid_summaries() {
        for s in [norm(&[]), norm(&[&[]]), norm(&[&[0.0, -0.0], &[]])] {
            assert!(s.all_finite()); assert_eq!(s.total_norm(), 0.0);
            assert_eq!(s.decision(Default::default()).unwrap().clip_multiplier, 1.0);
        }
    }
    #[test]
    fn each_nonfinite_kind_causes_whole_group_skip() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let s = norm(&[&[3.0, 4.0], &[bad]]);
            assert!(!s.all_finite()); assert_eq!(s.total_norm(), f64::INFINITY);
            assert!(s.decision(Default::default()).unwrap().skip_update);
            assert_eq!(s.decision(GradientGuardOptions { nonfinite: NonFinitePolicy::Error, ..Default::default() }), Err(GradientGuardError::NonFiniteGradients));
        }
    }
    #[test]
    fn finite_raw_value_can_overflow_on_unscale() {
        let s = gradient_stats_reference(&[&[f32::MAX]], 0.5).unwrap();
        assert!(!s.all_finite());
    }
    #[test]
    fn zero_limit_clips_but_does_not_skip() {
        let d = norm(&[&[3.0, 4.0]]).decision(GradientGuardOptions { max_norm: Some(0.0), ..Default::default() }).unwrap();
        assert_eq!(d.clip_multiplier, 0.0); assert!(!d.skip_update);
    }
    #[test]
    fn disabled_clipping_still_checks_finite() {
        let o = GradientGuardOptions { max_norm: None, ..Default::default() };
        assert_eq!(norm(&[&[1e30]]).decision(o).unwrap().clip_multiplier, 1.0);
        assert!(norm(&[&[f32::NAN]]).decision(o).unwrap().skip_update);
    }
    #[test]
    fn tiny_gradients_are_never_amplified() {
        assert_eq!(norm(&[&[1e-20]]).decision(Default::default()).unwrap().clip_multiplier, 1.0);
    }
    #[test]
    fn malformed_options_and_scales_reject() {
        for x in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(GradientGuardOptions { max_norm: Some(x), ..Default::default() }.validate().is_err());
        }
        for x in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(GradientGuardOptions { epsilon: x, ..Default::default() }.validate().is_err());
            assert!(gradient_stats_reference(&[], x).is_err());
        }
        assert!(gradient_stats_reference(&[], f32::from_bits(1)).is_err());
    }
    #[test]
    fn device_summaries_combine_stably() {
        let mut s = GradientStats::empty(1.0).unwrap();
        s.add_device_summary(&[3.0, 1.0, 0.0], 1).unwrap();
        s.add_device_summary(&[4.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(s, norm(&[&[3.0], &[4.0]]));
    }
    #[test]
    fn bad_summary_is_rejected_without_partial_host_commit() {
        let mut s = norm(&[&[3.0]]);
        let old = s;
        for v in [[f32::NAN, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 2.0], [1.0, 0.0, 0.0]] {
            assert!(s.add_device_summary(&v, 1).is_err()); assert_eq!(s, old);
        }
    }
    #[test]
    fn device_nonfinite_flag_propagates() {
        let mut s = GradientStats::empty(1.0).unwrap();
        s.add_device_summary(&[0.0, 0.0, 1.0], 7).unwrap();
        assert!(!s.all_finite()); assert_eq!(s.elements(), 7);
    }
}
