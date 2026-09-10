//! Ruda compiler frontends and target backends.

#[cfg(feature = "cpp")]
#[macro_use]
extern crate derive_new;





#[cfg(feature = "ptx")]
pub mod ptx;



#[cfg(feature = "optimizer")]
#[allow(unsafe_code)]
pub mod optimizer;

#[cfg(feature = "cpp")]
#[allow(unsafe_code)]
pub mod cpp;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "mlir")]
#[allow(unsafe_code)]
pub mod mlir;

#[cfg(feature = "spirv")]
#[allow(unsafe_code)]
pub mod spirv;
