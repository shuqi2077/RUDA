//! Ruda standard library.

mod fast_math;
mod reinterpret_slice;
mod swizzle;

pub use fast_math::*;
pub use reinterpret_slice::*;
pub use swizzle::*;

mod trigonometry;
pub use trigonometry::*;

/// Quantization functionality required in views
pub mod quant;
pub mod tensor;

/// Event utilities.
pub mod event;

#[cfg(feature = "library-tests")]
pub mod tests;

#[cfg(feature = "library-tests")]
pub use crate::{testgen_event, testgen, testgen_reinterpret_slice, testgen_tensor_identity, testgen_trigonometry, testgen_quantized_view};
