use crate::in_process::error::InProcessError;
use crate::rank::{NetworkError, TopologyError, error::RankError, work::WorkError};
use ruda_tensor::{DType, ExecutionError};
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum TensorDeviceError {
    UnsupportedDType(DType),
    DTypeMismatch { expected: DType, actual: DType },
    DeviceMismatch,
    InvalidBuffer(&'static str),
    InvalidOperation(&'static str),
    Data(String),
    Poisoned,
    Execution(ExecutionError),
    Rank(RankError),
    Network(NetworkError),
    Topology(TopologyError),
    Work(WorkError),
    InProcess(InProcessError),
}

impl fmt::Display for TensorDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDType(dtype) => write!(f, "collective backend does not support {dtype:?}"),
            Self::DTypeMismatch { expected, actual } => write!(f, "collective dtype mismatch: expected {expected:?}, got {actual:?}"),
            Self::DeviceMismatch => f.write_str("collective buffer belongs to another device"),
            Self::InvalidBuffer(message) | Self::InvalidOperation(message) => f.write_str(message),
            Self::Data(message) => write!(f, "collective tensor data: {message}"),
            Self::Poisoned => f.write_str("collective tensor buffer lock is poisoned"),
            Self::Execution(error) => fmt::Display::fmt(error, f),
            Self::Rank(error) => fmt::Display::fmt(error, f),
            Self::Network(error) => fmt::Display::fmt(error, f),
            Self::Topology(error) => fmt::Display::fmt(error, f),
            Self::Work(error) => fmt::Display::fmt(error, f),
            Self::InProcess(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl Error for TensorDeviceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            Self::Rank(error) => Some(error),
            Self::Network(error) => Some(error),
            Self::Topology(error) => Some(error),
            Self::Work(error) => Some(error),
            Self::InProcess(error) => Some(error),
            _ => None,
        }
    }
}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for TensorDeviceError {
            fn from(error: $source) -> Self { Self::$variant(error) }
        }
    };
}
from_error!(ExecutionError, Execution);
from_error!(RankError, Rank);
from_error!(NetworkError, Network);
from_error!(TopologyError, Topology);
from_error!(WorkError, Work);
from_error!(InProcessError, InProcess);
