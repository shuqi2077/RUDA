// SPDX-License-Identifier: Apache-2.0
use core::fmt;

/// Host-side contract failures. Device execution failures retain runtime semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusedAdamWError {
    /// Invalid optimizer hyperparameter or gradient scale.
    InvalidOption(&'static str),
    /// Invalid shape or metadata, non-contiguous input, or undersized storage.
    InvalidTensor(&'static str),
    /// Parameter, gradient or moment shapes do not agree exactly (no broadcasting).
    ShapeMismatch(&'static str),
    /// All operands must have the same device and submission queue.
    DifferentExecutionQueue(&'static str),
    /// State cannot be used with this optimizer configuration.
    InvalidState(&'static str),
    /// An update would overflow the completed-step counter.
    StepOverflow,
}

impl fmt::Display for FusedAdamWError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOption(s) => write!(f, "invalid fused AdamW option: {s}"),
            Self::InvalidTensor(s) => write!(f, "invalid fused AdamW tensor: {s}"),
            Self::ShapeMismatch(s) => write!(f, "fused AdamW shape mismatch: {s}"),
            Self::DifferentExecutionQueue(s) => write!(f, "fused AdamW queue mismatch: {s}"),
            Self::InvalidState(s) => write!(f, "invalid fused AdamW state: {s}"),
            Self::StepOverflow => f.write_str("fused AdamW step counter overflow"),
        }
    }
}
impl std::error::Error for FusedAdamWError {}

/// AdamW/AMSGrad hyperparameters. Arithmetic and optimizer state use FP32.
///
/// Defaults match this repository's existing `AdamWConfig`, not PyTorch defaults.
/// Cautious weight decay, differentiable steps and per-element learning rates are
/// intentionally not included in this opt-in path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWOptions {
    /// Nonnegative, finite learning rate.
    pub learning_rate: f32,
    /// First-moment decay in `[0, 1)`.
    pub beta1: f32,
    /// Second-moment decay in `[0, 1)`.
    pub beta2: f32,
    /// Positive, finite epsilon, added **outside** the bias-corrected square root.
    pub epsilon: f32,
    /// Nonnegative, finite decoupled weight decay.
    pub weight_decay: f32,
    /// Keep the running maximum of the uncorrected second moment.
    pub amsgrad: bool,
    /// Negate the unscaled gradient, but do not negate weight decay.
    pub maximize: bool,
}

impl Default for AdamWOptions {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-5,
            weight_decay: 1e-4,
            amsgrad: false,
            maximize: false,
        }
    }
}

/// Per-call controls supplied by the training loop, without implicit readback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepControl {
    /// Gradients are divided by this positive finite loss scale inside the kernel.
    pub gradient_scale: f32,
    /// A caller-detected overflow/cancelled step: do not launch, allocate or advance.
    /// This flag is a host value; this module does not implement a GradScaler.
    pub skip_update: bool,
}
impl Default for StepControl {
    fn default() -> Self {
        Self { gradient_scale: 1.0, skip_update: false }
    }
}

/// Uniform coefficients computed once on the host instead of per parameter.
///
/// Exposed for benchmark baselines and external integrations. Construct via
/// [`AdamWOptions::prepare_step`]; never reuse across different completed steps.
#[derive(Debug, Clone, Copy)]
pub struct StepCoefficients {
    /// Completed steps after the requested update (one-based).
    pub step: u64,
    /// Reciprocal gradient scale.
    pub inverse_gradient_scale: f32,
    /// Reciprocal first-moment bias correction.
    pub inverse_bias1: f32,
    /// Reciprocal second-moment bias correction.
    pub inverse_bias2: f32,
    /// `1 - learning_rate * weight_decay`, evaluated in FP32.
    pub decay_multiplier: f32,
}

impl AdamWOptions {
    /// Validate without allocating, compiling, or submitting device work.
    pub fn validate(&self) -> Result<(), FusedAdamWError> {
        for (name, value) in [("learning_rate", self.learning_rate), ("weight_decay", self.weight_decay)] {
            if !value.is_finite() || value < 0.0 {
                return Err(FusedAdamWError::InvalidOption(name));
            }
        }
        for (name, value) in [("beta1", self.beta1), ("beta2", self.beta2)] {
            if !value.is_finite() || !(0.0..1.0).contains(&value) {
                return Err(FusedAdamWError::InvalidOption(name));
            }
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(FusedAdamWError::InvalidOption("epsilon"));
        }
        if !(self.learning_rate * self.weight_decay).is_finite() {
            return Err(FusedAdamWError::InvalidOption("learning_rate * weight_decay overflows"));
        }
        Ok(())
    }

    /// Prepare an actual (non-skipped) update from the number of completed steps.
    /// Bias powers use integer exponentiation in FP64, then coefficients round to
    /// FP32. No cast of a large `u64` step to signed `i32` is performed.
    pub fn prepare_step(
        &self,
        completed_steps: u64,
        control: StepControl,
    ) -> Result<StepCoefficients, FusedAdamWError> {
        self.validate()?;
        control.validate()?;
        if control.skip_update {
            return Err(FusedAdamWError::InvalidOption("cannot prepare a skipped step"));
        }
        let step = completed_steps.checked_add(1).ok_or(FusedAdamWError::StepOverflow)?;
        Ok(StepCoefficients {
            step,
            inverse_gradient_scale: control.gradient_scale.recip(),
            inverse_bias1: (1.0 / (1.0 - pow_u64(self.beta1 as f64, step))) as f32,
            inverse_bias2: (1.0 / (1.0 - pow_u64(self.beta2 as f64, step))) as f32,
            decay_multiplier: 1.0 - self.learning_rate * self.weight_decay,
        })
    }
}
impl StepControl {
    /// Validate the scale even when skipping an update; no device read is performed.
    pub fn validate(&self) -> Result<(), FusedAdamWError> {
        if !self.gradient_scale.is_finite() || self.gradient_scale <= 0.0
            || !self.gradient_scale.recip().is_finite()
        {
            return Err(FusedAdamWError::InvalidOption("gradient_scale"));
        }
        Ok(())
    }
}

fn pow_u64(mut base: f64, mut exponent: u64) -> f64 {
    let mut value = 1.0;
    while exponent != 0 {
        if exponent & 1 != 0 { value *= base; }
        exponent >>= 1;
        if exponent != 0 { base *= base; }
    }
    value
}

/// Conservative limit: all FP32 tensor byte ranges also fit in 32 bits.
pub(crate) const MAX_ELEMENTS: usize = u32::MAX as usize / 4;

/// Validate each dimension even if an earlier dimension is zero.
pub(crate) fn checked_elements(shape: &[usize]) -> Result<usize, FusedAdamWError> {
    if shape.iter().any(|&n| n > u32::MAX as usize) {
        return Err(FusedAdamWError::InvalidTensor("dimension exceeds u32"));
    }
    let size = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(FusedAdamWError::InvalidTensor("element count overflow"))?;
    if size > MAX_ELEMENTS {
        return Err(FusedAdamWError::InvalidTensor("FP32 byte range exceeds u32"));
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_first_step() {
        let o = AdamWOptions::default();
        let c = o.prepare_step(0, StepControl::default()).unwrap();
        assert_eq!(c.step, 1);
        assert!((c.inverse_bias1 - 1.0 / (1.0 - o.beta1)).abs() < 1e-5);
        assert!((c.inverse_bias2 - 1.0 / (1.0 - o.beta2)).abs() < 1e-3);
    }
    #[test]
    fn option_validation() {
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert!(AdamWOptions { learning_rate: x, ..Default::default() }.validate().is_err());
            assert!(AdamWOptions { weight_decay: x, ..Default::default() }.validate().is_err());
        }
        for x in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(AdamWOptions { epsilon: x, ..Default::default() }.validate().is_err());
        }
        for x in [-0.1, 1.0, 1.1, f32::NAN, f32::INFINITY] {
            assert!(AdamWOptions { beta1: x, ..Default::default() }.validate().is_err());
            assert!(AdamWOptions { beta2: x, ..Default::default() }.validate().is_err());
        }
        assert!(AdamWOptions { learning_rate: f32::MAX, weight_decay: 2.0, ..Default::default() }.validate().is_err());
    }
    #[test]
    fn scale_validation() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            assert!(StepControl { gradient_scale: scale, skip_update: false }.validate().is_err());
        }
        assert_eq!(AdamWOptions::default().prepare_step(0, StepControl {
            gradient_scale: 128.0, skip_update: false,
        }).unwrap().inverse_gradient_scale, 1.0 / 128.0);
    }
    #[test]
    fn step_counter_and_large_steps() {
        let o = AdamWOptions::default();
        assert_eq!(o.prepare_step(u64::MAX, StepControl::default()).unwrap_err(), FusedAdamWError::StepOverflow);
        let c = o.prepare_step(1u64 << 40, StepControl::default()).unwrap();
        assert_eq!(c.step, (1u64 << 40) + 1);
        assert_eq!(c.inverse_bias1, 1.0);
        assert_eq!(c.inverse_bias2, 1.0);
    }
    #[test]
    fn zero_betas_are_valid() {
        let c = AdamWOptions { beta1: 0.0, beta2: 0.0, ..Default::default() }
            .prepare_step(0, StepControl::default()).unwrap();
        assert_eq!(c.inverse_bias1, 1.0);
        assert_eq!(c.inverse_bias2, 1.0);
    }
    #[test]
    fn shape_bounds() {
        assert_eq!(checked_elements(&[]).unwrap(), 1);
        assert_eq!(checked_elements(&[0, 64]).unwrap(), 0);
        assert_eq!(checked_elements(&[3, 257]).unwrap(), 771);
        assert!(checked_elements(&[MAX_ELEMENTS, 2]).is_err());
        if usize::BITS > 32 {
            assert!(checked_elements(&[0, u32::MAX as usize + 1]).is_err());
            assert!(checked_elements(&[usize::MAX, 2]).is_err());
        }
    }
}
