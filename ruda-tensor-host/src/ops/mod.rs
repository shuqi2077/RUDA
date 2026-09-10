//! Backend operations implementations.

pub(crate) use ruda_core::tensor::host::dtype::{INDEX_DTYPE, float_storage_as_f32};

#[cfg(feature = "rayon")]
pub(crate) use ruda_core::tensor::host::parallel::{PARALLEL_THRESHOLD, SendMutPtr};

pub mod activation;
pub mod attention;
pub mod binary;
mod bool;
pub mod cat;
pub mod comparison;
pub mod conv;
pub mod conv_transpose;
pub mod cumulative;
pub mod deform_conv;
pub mod expand;
pub mod fft;
pub mod flip;
mod float;
pub mod gather_scatter;
pub mod grid_sample;
mod int;
pub mod interpolate;
pub mod mask;
pub mod matmul;
mod module;
pub mod pool;
mod qtensor;
pub mod reduce;
pub mod repeat_dim;
pub mod slice;
pub mod sort;
#[cfg(feature = "sparse")]
pub mod sparse;
mod transaction;
pub mod unary;
pub mod unfold;
