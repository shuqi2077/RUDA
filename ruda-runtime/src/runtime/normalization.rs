use super::server::Handle;
use ruda_core::tensor::{DType, Shape, Strides};

/// An owned device buffer binding and its logical tensor layout.
#[derive(Clone, Debug)]
pub struct TensorBuffer {
    /// Retains the allocation for the duration of native execution.
    pub handle: Handle,
    /// Logical dimensions.
    pub shape: Shape,
    /// Element strides, including non-contiguous layouts.
    pub strides: Strides,
    /// Stored element type; native implementations must validate it.
    pub dtype: DType,
}
