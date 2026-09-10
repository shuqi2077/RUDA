use crate::backtrace::BackTrace;
use alloc::string::String;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An error that can happen when syncing a device.
#[derive(Error, Serialize, Deserialize)]
pub enum ExecutionError {
    /// A generic error happened during execution.
    ///
    /// The backtrace and context information should be included in the reason string.
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    WithContext {
        /// The reason of the error.
        reason: String,
    },
    /// A generic error happened during execution thrown in the Ruda project.
    ///
    /// The full context isn't captured by the string alone.
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    Generic {
        /// The reason of the error.
        reason: String,
        /// The backtrace.
        #[serde(skip)]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for ExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}
