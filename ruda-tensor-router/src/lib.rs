#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "138"]

//! Ruda multi-backend tensor router.

mod backend;
mod bridge;
mod channel;
mod client;
mod ops;
mod runner;
mod tensor;
mod types;

pub use backend::*;
pub use bridge::*;
pub use channel::*;
pub use client::*;
pub use runner::*;
pub use tensor::*;
pub use types::*;

/// A local channel with a simple byte bridge between backends.
/// It transfers tensors between backends via the underlying [tensor data](ruda_tensor::TensorData).
pub type DirectByteChannel<Backends> = DirectChannel<Backends, ByteBridge<Backends>>;

/// Router backend.
///
/// # Example
///
/// ```ignore
/// type MyBackend = Router<(Flex, Wgpu)>;
/// ```
pub type Router<Backends> = BackendRouter<DirectByteChannel<Backends>>;

extern crate alloc;

#[cfg(test)]
#[allow(unused)]
mod tests {
    use crate::BackendRouter;
    use crate::DirectByteChannel;

    pub type TestBackend1 = ruda_tensor_host::Host;
    pub type TestBackend2 = ruda_tensor_wgpu::Wgpu<f32, i32>;
    pub type TestBackend = BackendRouter<DirectByteChannel<(TestBackend1, TestBackend2)>>;
}
