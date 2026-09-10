//! Numerical validation shared by tensor and graph autotuning, independent of a GPU backend.
use alloc::{format, string::String};

/// Check finite values using |actual-reference| <= atol + rtol*|reference|.
/// Scaling before subtraction avoids overflow for large, finite values of opposite sign.
pub fn validate_finite_values(
    expected: impl IntoIterator<Item = f64>, actual: impl IntoIterator<Item = f64>,
    absolute: f64, relative: f64,
) -> Result<(), String> {
    if !absolute.is_finite() || absolute < 0.0 || !relative.is_finite() || relative < 0.0 {
        return Err("invalid numerical tolerance".into());
    }
    let mut expected = expected.into_iter(); let mut actual = actual.into_iter(); let mut index = 0usize;
    loop {
        match (expected.next(), actual.next()) {
            (None, None) => return Ok(()),
            (Some(a), Some(b)) => {
                if !a.is_finite() || !b.is_finite() { return Err(format!("non-finite autotune output at element {index}")); }
                let scale = a.abs().max(b.abs()).max(1.0);
                let delta = (a/scale - b/scale).abs();
                let limit = absolute/scale + relative*(a.abs()/scale);
                if (absolute == 0.0 && relative == 0.0 && a != b) || delta > limit { return Err(format!("autotune output mismatch at element {index}: reference={a}, actual={b}")); }
            }
            _ => return Err("autotune output element counts differ".into()),
        }
        index = index.checked_add(1).ok_or_else(|| String::from("validation element count overflow"))?;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn valid_values() { assert!(validate_finite_values([1., 0.], [1.00001, 0.00001], 1e-4, 1e-3).is_ok()); }
    #[test] fn nan_is_never_equivalent() { assert!(validate_finite_values([f64::NAN], [f64::NAN], 1., 1.).is_err()); }
    #[test] fn infinity_is_rejected() { assert!(validate_finite_values([f64::INFINITY], [f64::INFINITY], 0., 0.).is_err()); }
    #[test] fn opposite_extremes_do_not_pass_via_inf_comparison() { assert!(validate_finite_values([f64::MAX], [-f64::MAX], 0., 1e-3).is_err()); }
    #[test] fn equal_extremes_pass() { assert!(validate_finite_values([f64::MAX], [f64::MAX], 0., 0.).is_ok()); }
    #[test] fn exact_comparison_retains_subnormals() { assert!(validate_finite_values([f64::from_bits(1)], [0.], 0., 0.).is_err()); }
    #[test] fn lengths_must_match() { assert!(validate_finite_values([1.,2.], [1.], 0., 0.).is_err()); }
}
