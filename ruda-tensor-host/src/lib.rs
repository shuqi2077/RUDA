#![cfg_attr(not(feature = "std"), no_std)]

//! # ruda-tensor-host
//!
//! A fast, portable CPU tensor backend for Ruda.
//!
//! ## Features
//!
//! - Pure Rust (no C dependencies)
//! - f16/bf16 support
//! - SIMD acceleration via macerator (NEON, AVX2/AVX-512/SSE, SIMD128, scalar fallback)
//! - Zero-copy tensor views
//! - Thread-safe by design
//!
//! ## Usage
//!
//! ```ignore
//! use ruda_tensor_host::Host;
//! use ruda_tensor::api::Tensor;
//!
//! let tensor: Tensor<Host, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default());
//! ```

extern crate alloc;

#[cfg(all(not(target_has_atomic = "ptr"), not(feature = "critical-section")))]
compile_error!(
    "This target lacks atomic CAS support. Enable the `critical-section` feature: \
     ruda-tensor-host = { ..., features = [\"critical-section\"] }"
);

mod backend;
mod layout;
mod qtensor;
mod strided_index;
mod tensor;

#[doc(hidden)]
pub mod ops;

#[doc(hidden)]
pub mod simd;

pub use backend::{Host, HostDevice};
pub use layout::Layout;
pub use qtensor::HostQTensor;
pub use tensor::HostTensor;
