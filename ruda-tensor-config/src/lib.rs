#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

extern crate alloc;

/// Autodiff config module.
pub mod autodiff;
/// Fusion config module.
pub mod fusion;

mod base;
mod logger;

pub use base::*;
pub use ruda_core::config::RuntimeConfig;
pub use ruda_core::config::logger::{LogCrateLevel, LogLevel, LoggerConfig, LoggerSinks};
pub use logger::*;
