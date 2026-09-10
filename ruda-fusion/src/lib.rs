#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Ruda Fusion
//!
//! This library is a part of the Ruda project. It is a standalone crate that
//! can be used to perform automatic operation fusion on backends that support it.

#[macro_use]
extern crate derive_new;

/// Client module exposing types to communicate with the fusion server.
pub use runtime::client;
/// Stream module exposing all tensor operations that can be optimized.
pub mod stream;

/// Search module for stream optimizations.
pub(crate) mod search;

mod backend;
mod op;
mod ops;
pub(crate) mod runtime;
mod tensor;

/// Test-only introspection into fusion runtime behavior — see
/// [`inspect::FusionInspector`].
#[cfg(feature = "test-util")]
pub mod inspect;

pub use op::UnfusedOp;
pub(crate) use runtime::server::*;

pub use backend::*;
pub use ops::NoOp;
pub use tensor::*;

/// Device fusion code generation, launch and domain optimizations.
#[cfg(feature = "device")]
pub mod device;
