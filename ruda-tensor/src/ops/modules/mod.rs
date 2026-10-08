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

/// Saved-statistics normalization and complete first-order gradients.
pub mod normalization;
/// Actual channel-group normalization and selected first-order derivatives.
pub mod group_normalization;

/// Saved-working-output softmax and log-softmax training on the current backend.
pub mod softmax;

/// Working-storage activation forward and first-order training operations.
pub mod activation_training;
/// Shared/channel-wise PReLU forward and independently selected training derivatives.
pub mod prelu_training;

/// Module for grid_sample operations
pub mod grid_sample;

mod base;

pub use base::*;
