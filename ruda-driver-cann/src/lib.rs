//! Dynamically loaded AscendCL interfaces; no SDK is needed at build time.
//! The caller owns initialization, context selection, synchronization and release.
//! This is not a Ruda tensor backend or an implementation of accelerator kernels.
#![deny(unsafe_op_in_unsafe_fn)]

mod api;
pub mod sys;
pub mod tensor;

pub use api::{CannApi, CannLibrary};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CannError {
    Library {
        path: String,
        message: String,
    },
    Symbol {
        name: String,
        message: String,
    },
    Status {
        operation: &'static str,
        code: sys::AclError,
    },
    InvalidDevice(u32),
    InvalidTensor(String),
    NullHandle(&'static str),
    Completion {
        operation: &'static str,
        launch_code: i32,
        sync_code: i32,
    },
}

impl Display for CannError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Library { path, message } => {
                write!(f, "cannot load CANN library {path}: {message}")
            }
            Self::Symbol { name, message } => {
                write!(f, "cannot load CANN symbol {name}: {message}")
            }
            Self::Status { operation, code } => {
                write!(f, "{operation} failed with ACL status {code}")
            }
            Self::InvalidDevice(id) => {
                write!(f, "CANN device ordinal {id} exceeds the ACL int32 range")
            }
            Self::InvalidTensor(message) => f.write_str(message),
            Self::NullHandle(operation) => write!(f, "{operation} returned a null handle"),
            Self::Completion {
                operation,
                launch_code,
                sync_code,
            } => write!(
                f,
                "{operation}: launch status {launch_code}, synchronization status {sync_code}"
            ),
        }
    }
}
impl std::error::Error for CannError {}

pub fn check_status(operation: &'static str, code: sys::AclError) -> Result<(), CannError> {
    if code == sys::ACL_SUCCESS {
        Ok(())
    } else {
        Err(CannError::Status { operation, code })
    }
}

/// An ordinal, not proof that a device is present or supports a particular dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CannDevice(i32);

impl CannDevice {
    pub fn new(ordinal: u32) -> Result<Self, CannError> {
        i32::try_from(ordinal)
            .map(Self)
            .map_err(|_| CannError::InvalidDevice(ordinal))
    }
    pub const fn ordinal(self) -> u32 {
        self.0 as u32
    }
    pub const fn acl_id(self) -> i32 {
        self.0
    }
}

#[cfg(test)]
mod tests;
