#![no_std]
//! Ruda GPU kernel API and portable runtime.

extern crate self as ruda;

pub use ruda_runtime::*;

#[cfg(feature = "kernel")]
pub use ruda_kernel::dsl;
#[cfg(feature = "kernel")]
pub use ruda_kernel::dsl::prelude;

#[cfg(feature = "cuda")]
pub use ruda_driver_cuda as cuda;
#[cfg(feature = "hip")]
pub use ruda_driver_hip as hip;
#[cfg(feature = "wgpu")]
pub use ruda_driver_wgpu as wgpu;



