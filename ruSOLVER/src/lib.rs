// SPDX-License-Identifier: Apache-2.0
//! Numerical linear algebra with explicit execution and failure contracts.
//!
//! Default: dependency-free, synchronous host FP64. No GPU tensor is implicitly
//! downloaded and no existing ruda-tensor LU entry point is replaced.
//! Optional `tensor`: native FP32 small batched LU/QR/Cholesky/eigen/CG kernels.
//! `sparse` adapts ruSPARSE CSR; `collective` adapts existing ruCCL host TCP.
//! Host SVD, complex decompositions, sparse LU and reverse-mode pullbacks are
//! dependency-free. These APIs report limitations rather than densifying or
//! changing the execution backend silently.
//!
//! This is an experimental numerical baseline, not LAPACK/cuSOLVER API parity.
//! Dense inputs are nonempty, finite, row-major matrices; views also permit
//! explicit positive strides and transposition. Numerical rank/pivot decisions
//! use the supplied tolerance and do not estimate a condition number.
mod error;
mod matrix;
mod numerics;
mod lu;
mod cholesky;
mod qr;
mod eigen;
mod iterative;
pub use error::{
    SolverError, Tolerance
};
pub use matrix::{
    Matrix, MatrixView
};
pub use numerics::{
    l2_norm, relative_residual
};
pub use lu::Lu;
pub use cholesky::{
    Cholesky, CholeskyOptions
};
pub use qr::{
    Qr, LeastSquares
};
pub use eigen::{
    symmetric_eigen, EigenOptions, SymmetricEigen
};
pub use iterative::{
    conjugate_gradient, CgOptions, CgReport, IterativeStatus,
    IdentityPreconditioner, JacobiPreconditioner, LinearOperator, Preconditioner
};
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "tensor")]
pub mod tensor;
#[cfg(test)]
mod tests;

mod svd;
pub use svd::{Svd,SvdOptions};
pub mod complex;
pub mod sparse_direct;
pub mod distributed;
pub mod adjoint;
#[cfg(test)]mod advanced_tests;

#[cfg(test)]mod advanced_fixtures;
