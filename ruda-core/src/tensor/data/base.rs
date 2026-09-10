use crate::bytes::Bytes;
use crate::tensor::{DType, Shape};
use serde::{Deserialize, Serialize};

/// Data structure for tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorData {
    /// The values of the tensor (as bytes).
    pub bytes: Bytes,

    /// The shape of the tensor.
    #[serde(with = "shape_inner")]
    pub shape: Shape,

    /// The data type of the tensor.
    pub dtype: DType,
}

// For backward compatibility with shape `Vec<usize>`
mod shape_inner {
    use crate::tensor::SmallVec;

    use super::*;

    pub fn serialize<S: serde::Serializer>(
        shape: &Shape,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        shape.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Shape, D::Error> {
        let dims = SmallVec::<[usize; _]>::deserialize(deserializer)?;
        Ok(Shape::new_raw(dims))
    }
}

impl TensorData {
    /// Returns the rank (the number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements of the tensor data.
    pub fn num_elements(&self) -> usize {
        Self::numel(&self.shape)
    }

    pub(super) fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

}
