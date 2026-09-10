#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "256"]

//! Ruda optimizer updates, gradient state, clipping and learning-rate schedules.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Optimizer module.
mod optim;
pub use optim::*;

/// Opt-in single-kernel AdamW/AMSGrad and explicit numerical reference.
#[cfg(feature = "fused-adamw")]
pub mod fused_adamw;

/// Gradient clipping module.
pub mod grad_clipping;

/// Learning rate scheduler module.
#[cfg(feature = "std")]
pub mod lr_scheduler;

/// Combined model, optimizer, scheduler and accumulated-gradient records.
#[cfg(feature = "std")]
pub mod training;

/// Type alias for the learning rate.
///
/// LearningRate also implements [learning rate scheduler](crate::lr_scheduler::LrScheduler) so it
/// can be used for constant learning rate.
pub type LearningRate = f64; // We could potentially change the type.

/// Backend for test cases
#[cfg(all(
    test,
    not(feature = "test-tch"),
    not(feature = "test-wgpu"),
    not(feature = "test-cuda"),
    not(feature = "test-rocm")
))]
pub type TestBackend = ruda_tensor_host::Host;

#[cfg(all(test, feature = "test-tch"))]
/// Backend for test cases
pub type TestBackend = ruda_tensor_tch::LibTorch<f32>;

#[cfg(all(test, feature = "test-wgpu"))]
/// Backend for test cases
pub type TestBackend = ruda_tensor_wgpu::Wgpu;

#[cfg(all(test, feature = "test-cuda"))]
/// Backend for test cases
pub type TestBackend = ruda_tensor_device::cuda::Cuda;

#[cfg(all(test, feature = "test-rocm"))]
/// Backend for test cases
pub type TestBackend = ruda_tensor_rocm::Rocm;

/// Backend for autodiff test cases
#[cfg(test)]
pub type TestAutodiffBackend = ruda_autodiff::Autodiff<TestBackend>;

#[cfg(all(test, feature = "test-memory-checks"))]
mod tests {
    ruda_fusion::memory_checks!();
}
