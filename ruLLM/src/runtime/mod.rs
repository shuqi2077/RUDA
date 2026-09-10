//! Host-side admission, capability checks, bounded replica workers and diagnostics.
//!
//! This module has no CUDA/HIP dependency and can be tested directly with rustc.
//! Capability declarations and byte budgets do not allocate device memory or
//! implement kernels. GPU adapters remain responsible for probing capabilities,
//! selecting the current context and fencing device work before returning it.
#![forbid(unsafe_code)]

mod capabilities;
mod error;
mod memory;
mod numerics;
mod performance;
mod shape;
mod worker;

pub use capabilities::*;
pub use error::*;
pub use memory::*;
pub use numerics::*;
pub use performance::*;
pub use shape::*;
pub use worker::*;

#[cfg(test)]
mod tests;
