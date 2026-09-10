pub(crate) mod stats;

pub(crate) mod check;

mod autodiff;
mod base;
mod bool;
mod cartesian_grid;
#[path = "cast.rs"]
mod tensor_cast;
mod float;
mod fmod;
mod int;
mod numeric;
mod options;
mod orderable;
mod pad;
pub use pad::IntoPadding;
mod take;
mod transaction;
pub mod sparse;

mod trunc;

pub use autodiff::*;
pub use base::*;
pub use cartesian_grid::cartesian_grid;
pub use tensor_cast::*;
pub use crate::element::cast;
pub use float::{DEFAULT_ATOL, DEFAULT_RTOL};
pub use numeric::*;
pub use options::*;
pub use transaction::*;

pub use crate::tensor::IndexingUpdateOp;

// Re-exported types
pub use crate::{
    BoolDType, BoolStore, DType, DataError, FloatDType, IntDType, TensorData, TensorMetadata,
    TensorPrimitive, Tolerance,
    distribution::*,
    element::*,
    indexing::*,
    ops::TransactionPrimitive,
    shape::*,
    slice::*,
    tensor::{Bool, Float, Int, TensorKind},
};

/// The activation module.
pub mod activation;

/// The backend module.
pub mod backend {
    pub use crate::backend::*;
}

/// The container module.
pub mod container {
    pub use crate::tensor::TensorContainer;
}

/// The grid module.
pub mod grid;

/// The linalg module.
pub mod linalg;

/// The loss module.
pub mod loss;

/// The neural network module.
pub mod module;

/// The signal processing module.
pub mod signal;

/// Operations on tensors module.
pub mod ops {
    pub use crate::backend::ops::*;
    pub use crate::tensor::{
        BoolElem, BoolTensor, Device, FloatElem, FloatTensor, IntElem, IntTensor, QuantizedTensor,
    };
}

/// Tensor quantization module.
pub mod quantization;

#[cfg(feature = "api-std")]
pub use report::*;

#[cfg(feature = "api-std")]
mod report;

pub use ops::Device; // Re-export device so that it's available from `ruda_tensor::api::Device`.

pub(crate) use check::macros::check;

pub use crate::{
    AllocationProperty, Bytes, DeviceSettings, StreamId, bf16, f16, get_device_settings, read_sync,
    set_default_dtypes, try_read_sync,
};
