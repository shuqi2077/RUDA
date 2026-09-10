//! CPU tensor storage, layouts and strided access shared by host implementations.

pub mod cast;
pub mod layout;
pub mod storage;
pub mod strided_index;

pub use layout::Layout;
pub use storage::HostTensor;

#[cfg(feature = "tensor-host-parallel")]
pub mod parallel;

pub mod dtype;
