/// Module with convolution operations.
pub mod conv;

/// Module with linear operations.
pub mod linear;

pub mod embedding;

/// Module with attention operations.
pub mod attention;

/// Module with CTC loss operations.
pub mod ctc;

/// Module with unfold operations.
pub mod unfold;

/// Module with pooling operations.
pub mod pool;
/// Native-dimensional interpolation using the backend's original spatial filters.
pub mod interpolation;

/// Module for grid_sample operations
pub mod grid_sample;

mod base;

pub use base::*;
