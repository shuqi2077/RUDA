// SPDX-License-Identifier: Apache-2.0
use crate::{
    MatrixView, SolverError, Tolerance
};
pub(crate) fn zeros(n: usize) -> Result<Vec<f64>, SolverError> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| SolverError::Allocation)?;
    out.resize(n, 0.0);
    Ok(out)
}
pub(crate) fn finite(x: &[f64]) -> Result<(), SolverError> {
    for (index, v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(SolverError::NonFinite {
                index
            });
        }
    }
    Ok(())
}
pub(crate) fn checked(x: f64, where_: &'static str) -> Result<f64, SolverError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(SolverError::Arithmetic(where_))
    }
}
/// Scaled sum of squares, avoiding overflow from squaring large finite entries.
/// Returns an error if the norm itself exceeds the representable FP64 range.
pub fn l2_norm(x: &[f64]) -> Result<f64, SolverError> {
    norm_iter(x.iter().copied())
}
pub(crate) fn norm_iter(it: impl Iterator<Item=f64>) -> Result<f64, SolverError> {
    let (mut scale, mut sum) = (0.0f64, 1.0f64);
    for (index, value) in it.enumerate() {
        if !value.is_finite() {
            return Err(SolverError::NonFinite {
                index
            });
        }
        let a = value.abs();
        if a != 0.0 {
            if scale < a {
                let q = scale/a;
                sum = 1.0 + sum*q*q;
                scale = a;
            }
            else {
                let q = a/scale;
                sum += q*q;
            }
        }
    }
    checked(scale * sum.sqrt(), "Euclidean norm")
}
/// Compensated sum with explicit failure on unrepresentable intermediates.
/// It does not claim correctly-rounded arbitrary-precision dot products.
pub(crate) fn sum_iter(it: impl Iterator<Item=f64>) -> Result<f64, SolverError> {
    let (mut sum, mut correction) = (0.0f64, 0.0f64);
    for value in it {
        checked(value, "dot-product term")?;
        let t = checked(sum + value, "dot-product sum")?;
        let increment = if sum.abs() >= value.abs() {
            (sum-t)+value
        } else {
            (value-t)+sum
        };
        correction = checked(correction + increment, "dot-product compensation")?;
        sum = t;
    }
    checked(sum + correction, "compensated sum")
}
pub(crate) fn dot(x: &[f64], y: &[f64]) -> Result<f64, SolverError> {
    if x.len()!=y.len() {
        return Err(SolverError::Shape("dot-product lengths"));
    }
    sum_iter(x.iter().zip(y).map(|(x, y)| x*y))
}
pub(crate) fn symmetric(a: MatrixView<'_>, tolerance: Tolerance) -> Result<(), SolverError> {
    let n = a.square()?;
    tolerance.validate()?;
    for i in 0..n {
        for j in 0..i {
            let (x, y) = (a.at(i, j), a.at(j, i));
            // Scale each pair first so opposite, large finite values do not overflow.
            let scale = x.abs().max(y.abs());
            if scale != 0.0 && (x/scale-y/scale).abs() > (tolerance.absolute/scale).max(tolerance.relative) {
                return Err(SolverError::NotSymmetric {
                    row: i, column: j
                });
            }
        }
    }
    Ok(())
}
/// `||A*x-b||_2 / ||b||_2`; for a zero RHS returns the absolute residual norm.
/// This is a residual diagnostic, NOT a forward-error or condition estimate.
pub fn relative_residual(a: MatrixView<'_>, x: &[f64], b: &[f64]) -> Result<f64, SolverError> {
    if x.len()!=a.columns() || b.len()!=a.rows() {
        return Err(SolverError::Shape("residual dimensions"));
    }
    finite(x)?;
    finite(b)?;
    let mut residual = zeros(b.len())?;
    for i in 0..a.rows() {
        residual[i] = sum_iter(std::iter::once(-b[i]).chain((0..a.columns()).map(|j| a.at(i, j)*x[j])))?;
    }
    let (r, n)=(l2_norm(&residual)?, l2_norm(b)?);
    checked(if n == 0.0 {
        r
    } else {
        r/n
    }, "relative residual")
}
