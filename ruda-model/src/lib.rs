#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "135"]

//! Ruda model configuration, modules, parameters, records and data loading.

#[macro_use]
extern crate derive_new;

/// Re-export serde for proc macros.
pub use serde;

/// The configuration module.
pub mod config;

/// Data module.
#[cfg(feature = "std")]
pub mod data;

/// Module for the neural network module.
pub mod module;

/// Module for the recorder.
pub mod record;

/// Module for the tensor.
pub mod tensor;
// Tensor at root: `ruda::Tensor`
pub use tensor::Tensor;

extern crate alloc;
extern crate self as ruda_model;

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

#[cfg(test)]
mod test_utils {
    use crate::module::Module;
    use crate::module::Param;
    use ruda_tensor::api::Tensor;
    use ruda_tensor::api::backend::Backend;

    /// Simple linear module.
    #[derive(Module, Debug)]
    pub struct SimpleLinear<B: Backend> {
        pub weight: Param<Tensor<B, 2>>,
        pub bias: Option<Param<Tensor<B, 1>>>,
    }

    impl<B: Backend> SimpleLinear<B> {
        pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
            let weight = Tensor::random(
                [out_features, in_features],
                ruda_tensor::api::Distribution::Default,
                device,
            );
            let bias = Tensor::random([out_features], ruda_tensor::api::Distribution::Default, device);

            Self {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        }
    }
}

pub mod prelude {
    //! Structs and macros used by most projects. Add `use
    //! ruda::prelude::*` to your code to quickly get started with
    //! Ruda.
    pub use crate::{
        config::Config,
        module::Module,
        tensor::{
            Bool, Device, ElementConversion, Float, Int, Shape, SliceArg, Tensor, TensorData,
            backend::Backend, cast::ToElement, s,
        },
    };
    pub use ruda_core::device::Device as DeviceOps;
}
