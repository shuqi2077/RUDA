#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "256"]

//! Ruda neural network layers, activation modules and losses.

/// Loss module
pub mod loss;

/// Neural network modules implementations.
pub mod modules;
pub use modules::*;

pub mod activation;
pub use activation::{
    celu::*, elu::*, gelu::*, glu::*, hard_shrink::*, hard_sigmoid::*, leaky_relu::*, prelu::*,
    relu::*, selu::*, shrink::*, sigmoid::*, soft_shrink::*, softplus::*, softsign::*, swiglu::*,
    tanh::*, thresholded_relu::*,
};

mod padding;
pub use padding::*;

// For backward compat, `ruda_nn::Initializer`
pub use ruda_model::module::Initializer;

extern crate alloc;

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
