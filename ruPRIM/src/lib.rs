//! Ruda parallel primitives.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

#[cfg(feature = "kernel-ir")]
pub mod collective;

#[cfg(feature = "kernel-ir")]
pub mod warp;

#[cfg(feature = "kernel-ir")]
pub mod block;

#[cfg(feature = "device-primitives")]
#[allow(unsafe_code)]
pub mod device;



#[cfg(feature = "kernel-ir")]
#[allow(unsafe_code)]
pub mod reduce;

#[cfg(feature = "elementwise")]
#[allow(unsafe_code)]
pub mod elementwise;

#[cfg(feature = "indexing")]
#[allow(unsafe_code)]
pub mod indexing;

#[cfg(feature = "tensor-scan")]
#[allow(unsafe_code)]
pub mod scan;
