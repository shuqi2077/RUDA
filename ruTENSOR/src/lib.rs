//! Tensor linear algebra on Ruda device tensors.
//!
//! Use [`einsum`] for index expressions or construct an [`OperationDescriptor`]
//! and reuse a [`Plan`] with different buffers and scalar coefficients.

mod descriptor;
mod einsum;
mod error;
mod kernel;
mod operation;
mod plan;
mod tensor;

pub use descriptor::{Mode, OperandDescriptor, TensorDescriptor};
pub use einsum::{EinsumPlan, einsum, einsum_with_options};
pub use error::{Error, Result};
pub use operation::{BinaryOp, ComputeType, OperationDescriptor, ReductionOp, UnaryOp};
pub use plan::Plan;
pub use tensor::{contract, elementwise_binary, elementwise_trinary, permute, reduce};
pub use ruda_core::tensor::DType;
pub use ruda_kernel::tensor::RudaTensor;
