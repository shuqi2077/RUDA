#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Ruda device tensor backend and domain-library dispatch.

#[macro_use]
extern crate derive_new;
extern crate alloc;

mod backend;
/// Backend implementations of tensor operations.
pub mod dispatch;
/// Scalar element contracts for device tensor operations.
pub mod element;
mod runtime;

pub use backend::DeviceBackend;
pub use element::{BoolElement, FloatElement, IntElement, TensorElement};
pub use runtime::DeviceRuntime;
/// Native inference operators over the same RudaTensor/runtime as this backend.
#[cfg(feature = "native-inference")]
pub mod inference {
    pub use rublas::tensor_float_quant as quantization;
    pub use rudnn::inference::*;
    pub use ruprim::{bitpack, topk};
    pub use rurand::tensor::sampling;
}
pub use ruda_kernel::tensor::RudaTensor;

#[cfg(feature = "fusion")]
/// Device backend integration with tensor fusion.
pub mod fusion;

/// NVIDIA CUDA device tensor adapter.
#[cfg(feature = "cuda")]
pub mod cuda;
