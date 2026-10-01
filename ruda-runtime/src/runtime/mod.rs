#![warn(missing_docs)]

//! `Ruda` runtime crate that helps creating high performance async runtimes.

/// Various identifier types used in `Ruda`.
pub mod id;

/// Kernel related traits.
pub mod kernel;

/// Stream related utilities.
pub mod stream;

/// Compute client module.
pub mod client;

/// Autotune module
pub mod tune;

/// Memory management module.
pub mod memory_management;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

/// `Ruda` config module.
pub mod config;

pub use ruda_core::benchmark;

/// Logging utilities to be used by a compute server.
pub mod logging;

/// TMA-related runtime types
pub mod tma;

/// Compiler trait and related types
pub mod compiler;
/// Runtime trait and related types
pub mod backend;
/// Simple system profiling using timestamps.
pub mod timestamp_profiler;

/// Validation utils for shared properties
pub mod validation;

/// Allocators moddule.
pub mod allocator;

pub use crate::{local_tuner, storage_id_type};
