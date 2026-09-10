#[macro_use]
extern crate derive_new;

extern crate alloc;

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests;

pub mod compilation;
pub mod execution;
pub mod device;
pub mod memory;
pub mod runtime;

pub use device::CpuDevice;
pub use runtime::*;
