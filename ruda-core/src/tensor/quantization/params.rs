use super::QuantLevel;
use crate::tensor::{DType, Shape, metadata::Metadata};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
/// The precision of accumulating elements.
pub enum QuantAcc {
    /// Full precision.
    #[default]
    F32,
    /// Half precision.
    F16,
    /// bfloat16 precision.
    BF16,
}

/// Specify if the output of an operation is quantized using the scheme of the input
/// or returned unquantized.
#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum QuantPropagation {
    /// The output is quantized using the scheme of the input.
    Propagate,
    /// The output is not quantized.
    #[default]
    Inhibit,
}

/// The quantization tensor data parameters.
#[derive(Clone, Debug)]
pub struct QParams<S> {
    /// The scaling factor.
    pub scales: S,
}

/// A quantization parameter tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QParamTensor {
    /// Start of the tensor in the buffer
    pub offset_start: usize,
    /// Offset of tensor end from the end of the buffer
    pub offset_end: usize,
    /// Metadata of the tensor
    pub metadata: Metadata,
    /// Data type of the tensor
    pub dtype: DType,
}

/// Calculate the shape of the quantization parameters for a given tensor and level
pub fn params_shape(data_shape: &Shape, level: QuantLevel) -> Shape {
    match level {
        QuantLevel::Tensor => Shape::new([1]),
        QuantLevel::Block(block_size) => {
            let mut params_shape = data_shape.clone();
            let block_size = block_size.to_dim_vec(data_shape.num_dims());

            for (shape, block_size) in params_shape.iter_mut().zip(block_size) {
                *shape = (*shape).div_ceil(block_size as usize);
            }

            params_shape
        }
    }
}
