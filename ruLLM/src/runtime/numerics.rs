use super::{RuntimeError, RuntimeErrorKind};

/// Strict logits validation: -infinity is an allowed mask; NaN, +infinity and
/// an entirely masked row are errors. Ties select the smallest token index.
pub fn checked_argmax(logits: &[f32]) -> Result<usize, RuntimeError> {
    let mut best = None;
    for (i, &x) in logits.iter().enumerate() {
        if x.is_nan() || x == f32::INFINITY {
            return Err(RuntimeError::new(RuntimeErrorKind::Numerical, "logits contain NaN or positive infinity"));
        }
        if x.is_finite() && best.is_none_or(|(_, maximum)| x > maximum) { best = Some((i, x)); }
    }
    best.map(|(i, _)| i).ok_or_else(|| RuntimeError::new(RuntimeErrorKind::Numerical, "logits are empty or entirely masked"))
}

/// Host validation reference, not a GPU softmax implementation. Widen before
/// subtraction so opposite-sign near-f32::MAX values cannot overflow in f32.
pub fn stable_softmax(logits: &[f32]) -> Result<Vec<f64>, RuntimeError> {
    let maximum = logits[checked_argmax(logits)?] as f64;
    let mut probabilities: Vec<_> = logits.iter().map(|&x| (x as f64 - maximum).exp()).collect();
    let sum: f64 = probabilities.iter().sum();
    for p in &mut probabilities { *p /= sum; }
    Ok(probabilities)
}

/// Scaled RMS reference avoids squaring large finite f64 values. The epsilon
/// must be positive and finite; zero vectors remain zero.
pub fn stable_rms_norm(input: &[f64], gamma: &[f64], epsilon: f64) -> Result<Vec<f64>, RuntimeError> {
    if input.is_empty() || input.len() != gamma.len() || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(RuntimeError::invalid("invalid RMSNorm width, gamma or epsilon"));
    }
    if input.iter().chain(gamma).any(|x| !x.is_finite()) {
        return Err(RuntimeError::new(RuntimeErrorKind::Numerical, "non-finite RMSNorm input"));
    }
    let scale = input.iter().fold(epsilon.sqrt(), |s, x| s.max(x.abs()));
    let sum: f64 = input.iter().map(|x| (x / scale).powi(2)).sum();
    let denominator = (sum / input.len() as f64 + (epsilon / scale) / scale).sqrt();
    let output: Vec<_> = input.iter().zip(gamma).map(|(x, g)| ((x / scale) / denominator) * g).collect();
    if output.iter().any(|x| !x.is_finite()) {
        return Err(RuntimeError::new(RuntimeErrorKind::Numerical, "RMSNorm result is non-finite"));
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
pub struct NumericalTolerance { pub absolute: f64, pub relative: f64 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericalComparison { pub values: usize, pub max_absolute_error: f64 }
impl NumericalTolerance {
    /// Criterion: |actual-reference| <= absolute + relative * |reference|.
    /// Non-finite values fail even when both arrays contain the same NaN/Inf.
    pub fn compare(self, actual: &[f64], reference: &[f64]) -> Result<NumericalComparison, RuntimeError> {
        if !self.absolute.is_finite() || !self.relative.is_finite() || self.absolute < 0.0 || self.relative < 0.0
            || actual.len() != reference.len() || actual.is_empty() {
            return Err(RuntimeError::invalid("invalid tolerance or comparison dimensions"));
        }
        let mut maximum = 0.0f64;
        for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
            if !a.is_finite() || !r.is_finite() {
                return Err(RuntimeError::new(RuntimeErrorKind::Numerical, format!("non-finite comparison value at {i}")));
            }
            let error = (a - r).abs();
            // Divide by a scale to avoid overflow both in a-r and r*relative.
            let scale = a.abs().max(r.abs()).max(1.0);
            let normalized_error = (a / scale - r / scale).abs();
            let limit = self.absolute / scale + self.relative * (r.abs() / scale);
            let direct_limit = self.absolute + self.relative * r.abs();
            let exceeds = if error.is_finite() && direct_limit.is_finite() {
                error > direct_limit
            } else {
                normalized_error > limit
            };
            if exceeds {
                return Err(RuntimeError::new(RuntimeErrorKind::Numerical, format!("numerical tolerance exceeded at {i}")));
            }
            maximum = maximum.max(error);
        }
        Ok(NumericalComparison { values: actual.len(), max_absolute_error: maximum })
    }
}
