use alloc::string::String;
use bytemuck::checked::CheckedCastError;
use thiserror::Error;

/// The things that can go wrong when manipulating tensor data.
#[derive(Debug, Error)]
pub enum DataError {
    /// Failed to cast the values to a specified element type.
    #[error("Failed to cast values to the specified element type.\nError:\n  {0}")]
    CastError(CheckedCastError),
    /// Invalid target element type.
    #[error("{0}")]
    TypeMismatch(String),
}
