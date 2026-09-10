#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides the core types that define how Ruda tensor data is represented, stored, and interpreted.

#[macro_use]
extern crate derive_new;

extern crate alloc;

use ruda_core::tensor::data;
pub use data::*;

pub use ruda_core::tensor::distribution;
pub use distribution::*;
pub use ruda_core::tensor::element;
pub use ruda_core::make_element;
pub use element::*;

pub mod device;
pub mod primitive;
pub mod ops;
#[cfg(feature = "distributed")]
pub mod distributed;

/// [`Backend`] trait and required types.
pub mod backend;
pub use backend::*;

/// Backend tensor primitives and operations.
pub mod tensor;

/// Tensor operation graphs and backend resource handles.
#[cfg(feature = "graph")]
pub mod graph;

// Re-exported types
pub use ruda_core::reader::*; // Useful so that backends don't have to add `ruda_core` as a dependency.
pub use ruda_core::bytes::{AllocationProperty, Bytes};
pub use ruda_core::device_handle::DeviceHandle;
pub use ruda_core::stream_id::StreamId;
pub use ruda_core::tensor::{BoolDType, BoolStore, DType, FloatDType, IntDType};
pub use half::{bf16, f16};

/// Shape definition.
pub mod shape {
    pub use ruda_core::tensor::shape::*;
}
pub use shape::*;

/// Slice utilities.
pub mod slice {
    pub use ruda_core::tensor::{s, slice::*};
}
pub use slice::*;

/// Indexing utilities.
pub mod indexing {
    pub use ruda_core::tensor::indexing::*;
}
pub use indexing::*;

/// Quantization data representation.
pub mod quantization {
    pub use crate::tensor::quantization::*;
    pub use ruda_core::tensor::quantization::{
        BlockSize, QuantLevel, QuantMode, QuantParam, QuantPropagation, QuantScheme, QuantStore,
        QuantValue, QuantizedBytes,
    };
}

/// Convenience macro to link to the `ruda-tensor` API docs.
///
/// Usage:
/// ```rust,ignore
/// # use ruda_tensor::doc_tensor;
/// doc_tensor!();        // Links to `Tensor` struct
/// doc_tensor!("zeros"); // Links to `Tensor::zeros` method
/// ```
#[macro_export]
macro_rules! doc_tensor {
    () => {
        "[`Tensor`](crate::api::Tensor)"
    };

    ($method:literal) => {
        concat!(
            "[`Tensor::",
            $method,
            "`](crate::api::Tensor::",
            $method,
            ")"
        )
    };
}

/// Public tensor API built on the shared backend contracts and primitives.
#[cfg(feature = "api")]
pub mod api;
