//! Portable SIMD kernels using macerator.
//!
//! Replaces platform-specific implementations (neon.rs) with a single
//! portable implementation that auto-dispatches to NEON/AVX2/SSE/SIMD128/scalar.

use macerator::{Scalar, Simd, VBitAnd, VBitOr, VBitXor, vload_unaligned, vstore_unaligned};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Threshold for parallel execution (elements).
/// For memory-bound operations, parallelism helps when data exceeds L3 cache.
#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4 * 1024 * 1024;

#[cfg(feature = "rayon")]
const CHUNK_SIZE: usize = 4096;

mod binary_ops;
pub use binary_ops::*;

mod unary_ops;
pub use unary_ops::*;

mod comparison_ops;
pub use comparison_ops::*;

mod boolean_ops;
pub use boolean_ops::*;

mod mask_ops;
pub use mask_ops::*;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
