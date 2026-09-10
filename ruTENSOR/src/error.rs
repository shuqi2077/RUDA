use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// Errors detected before a device operation is submitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidDescriptor(String),
    InvalidExpression(String),
    InvalidOperation(String),
    IncompatibleExtent { mode: i32, left: usize, right: usize },
    UnsupportedDType(String),
    Overflow,
    InputCount { expected: usize, actual: usize },
    ScalarCount { expected: usize, actual: usize },
    TensorMismatch { input: usize },
    OutputMismatch,
    DeviceMismatch,
    SharedOutput,
    BufferTooSmall,
    UnsupportedDevice(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(s) => write!(f, "invalid tensor descriptor: {s}"),
            Self::InvalidExpression(s) => write!(f, "invalid einsum expression: {s}"),
            Self::InvalidOperation(s) => write!(f, "invalid tensor operation: {s}"),
            Self::IncompatibleExtent { mode, left, right } =>
                write!(f, "mode {mode} has incompatible extents {left} and {right}"),
            Self::UnsupportedDType(s) => write!(f, "unsupported data type: {s}"),
            Self::Overflow => f.write_str("tensor size or address exceeds usize"),
            Self::InputCount { expected, actual } => write!(f, "expected {expected} inputs, got {actual}"),
            Self::ScalarCount { expected, actual } => write!(f, "expected {expected} coefficients, got {actual}"),
            Self::TensorMismatch { input } => write!(f, "input {input} does not match its descriptor"),
            Self::OutputMismatch => f.write_str("output does not match its descriptor"),
            Self::DeviceMismatch => f.write_str("all tensors must be on the same device"),
            Self::SharedOutput => f.write_str("execute_into requires an exclusively owned output buffer"),
            Self::BufferTooSmall => f.write_str("tensor layout exceeds its backing buffer"),
            Self::UnsupportedDevice(s) => write!(f, "unsupported device configuration: {s}"),
        }
    }
}

impl std::error::Error for Error {}
