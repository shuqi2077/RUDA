#![cfg_attr(
    not(any(feature = "std", feature = "kernel-ir", feature = "quantization-kernels", feature = "library")),
    no_std
)]
//! Ruda Rust kernel DSL and device tensor operations.

extern crate self as ruda_kernel;

#[cfg(any(feature = "quantization", feature = "frontend", feature = "library"))]
extern crate alloc;

#[cfg(all(any(feature = "frontend-std", feature = "source-template"), not(any(feature = "std", feature = "kernel-ir", feature = "quantization-kernels", feature = "library"))))]
extern crate std;

#[cfg(feature = "frontend")]
#[macro_use]
extern crate derive_new;




#[cfg(feature = "frontend")]
#[allow(unsafe_code)]
pub mod dsl;

#[cfg(feature = "library")]
#[allow(unsafe_code)]
pub mod library;

#[cfg(feature = "kernel-ir")]
#[allow(unsafe_code)]
pub mod tiling;

#[cfg(feature = "quantization")]
#[allow(unsafe_code)]
pub mod quantization;

#[cfg(feature = "device-tensor")]
#[allow(unsafe_code)]
pub mod tensor;

/// Host source templates and compiled Kernel task integration.
#[cfg(feature = "source-template")]
pub mod template;
