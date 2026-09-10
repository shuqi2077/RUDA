#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Ruda device tensor backend and domain-library dispatch.

#[macro_use]
extern crate derive_new;
extern crate alloc;

/// Backend implementations of tensor operations.
pub mod dispatch;
/// Scalar element contracts for device tensor operations.
pub mod element;
mod backend;
mod runtime;

pub use backend::DeviceBackend;
pub use runtime::DeviceRuntime;
pub use element::{BoolElement, TensorElement, FloatElement, IntElement};
pub use ruda_kernel::tensor::RudaTensor;

#[cfg(feature = "fusion")]
/// Device backend integration with tensor fusion.
pub mod fusion;

/// NVIDIA CUDA device tensor adapter.
#[cfg(feature = "cuda")]
pub mod cuda;
