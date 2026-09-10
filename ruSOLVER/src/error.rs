// SPDX-License-Identifier: Apache-2.0
use std::{
    error::Error, fmt
};
/// Absolute/relative cutoff: `max(absolute, relative * scale)`.
/// Both may be zero for exact-zero checks. Negative/nonfinite inputs are rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    pub absolute: f64, pub relative: f64
}
impl Default for Tolerance {
    fn default() -> Self {
        Self {
            absolute: 0.0, relative: 1e-12
        }
    }
}
impl Tolerance {
    pub const EXACT: Self = Self {
        absolute: 0.0, relative: 0.0
    };
    pub fn validate(self) -> Result<(), SolverError> {
        if !self.absolute.is_finite() || !self.relative.is_finite()
        || self.absolute < 0.0 || self.relative < 0.0 {
            Err(SolverError::InvalidOption("tolerance must be finite and nonnegative"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn threshold(self, scale: f64) -> Result<f64, SolverError> {
        self.validate()?;
        let value = self.absolute.max(self.relative * scale);
        if !scale.is_finite() || scale < 0.0 || !value.is_finite() {
            Err(SolverError::Arithmetic("tolerance scale overflow"))
        } else {
            Ok(value)
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum SolverError {
    Shape(&'static str),
    WorkspaceLimit { required: usize, limit: usize },
    Communication(String),
    SizeOverflow,
    Allocation,
    NonFinite {
        index: usize
    },
    InvalidOption(&'static str),
    Singular {
        index: usize, pivot: f64, threshold: f64
    },
    NotSymmetric {
        row: usize, column: usize
    },
    NotPositiveDefinite {
        index: usize, pivot: f64
    },
    RankDeficient {
        rank: usize, columns: usize
    },
    Arithmetic(&'static str),
    Breakdown(&'static str),
    NonConvergence {
        iterations: usize, residual: f64
    },
    Operator(&'static str),
}
impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceLimit { required, limit } => write!(f, "workspace fill {required} exceeds limit {limit}"),
            Self::Communication(s) => write!(f, "solver communication error: {s}"),
            Self::Shape(s) => write!(f, "invalid matrix/vector shape: {s}"),
            Self::SizeOverflow => write!(f, "matrix size or address range overflows usize"),
            Self::Allocation => write!(f, "numerical workspace allocation failed"),
            Self::NonFinite {
                index
            } => write!(f, "nonfinite input at logical index {index}"),
            Self::InvalidOption(s) => write!(f, "invalid solver option: {s}"),
            Self::Singular {
                index, pivot, threshold
            } => write!(f, "singular/numerically singular pivot {index}: |{pivot}| <= {threshold}"),
            Self::NotSymmetric {
                row, column
            } => write!(f, "matrix differs across ({row},{column}) and its transpose"),
            Self::NotPositiveDefinite {
                index, pivot
            } => write!(f, "nonpositive Cholesky pivot {index}: {pivot}"),
            Self::RankDeficient {
                rank, columns
            } => write!(f, "numerical column rank {rank} < {columns}; this QR path requires full column rank; use Svd for a minimum-norm solve"),
            Self::Arithmetic(s) => write!(f, "nonfinite/unsupported numerical intermediate: {s}"),
            Self::Breakdown(s) => write!(f, "iterative solver breakdown: {s}"),
            Self::NonConvergence {
                iterations, residual
            } => write!(f, "not converged after {iterations} iterations; residual {residual}"),
            Self::Operator(s) => write!(f, "linear operator error: {s}"),
        }
    }
}
impl Error for SolverError {
}
