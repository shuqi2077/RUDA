//! # Common `ZSpace` Utilities for `Ruda`
//!
//! This is a new/experimental module.
//!
//! The goal will be to unify:
//! - shape/stride construction/validation.
//! - stride map view transformations.
//! - common Shape types.
//! - common Shape/Size/Reshape utility traits.
//!
//! The intention is to publish this as a stand-alone `zspace` module,
//! with no direct tie to `ruda`; once it is more polished.

pub mod dtype;
pub use dtype::*;

pub mod errors;
pub mod indexing;
pub mod striding;
pub use striding::reshape::*;

pub mod slice;
pub use slice::*;

pub mod quantization;
pub use quantization::*;

pub(crate) const INLINE_DIMS: usize = 5;

pub mod metadata;
pub use metadata::Metadata;
pub mod shape;
mod strides;

/// Reexport to avoid annoying rust-analyzer bug where it imports the module instead of the macro
pub use shape::*;
pub use strides::*;

/// Reexport for use in macros
pub use smallvec::{SmallVec, smallvec};

pub use crate::{s, shape, strides};

#[cfg(feature = "tensor-elements")]
pub mod element;
#[cfg(feature = "tensor-elements")]
pub mod distribution;

#[cfg(feature = "tensor-host-data")]
pub mod data;

pub mod primitive;
pub use primitive::{QTensorPrimitive, TensorMetadata};

#[cfg(feature = "tensor-host-data")]
pub mod execution;

pub mod spatial;

#[cfg(feature = "tensor-device-settings")]
pub mod device_settings;

#[cfg(feature = "tensor-host-data")]
pub mod transaction;

#[cfg(feature = "tensor-host-storage")]
pub mod host;
