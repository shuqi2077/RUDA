use crate::{Backend, get_device_settings};
use ruda_core::tensor::{DType, Shape};

#[derive(Debug, Clone)]
/// A primitive tensor representation.
pub enum TensorPrimitive<B: Backend> {
    /// Float tensor primitive.
    Float(B::FloatTensorPrimitive),
    /// Quantized float tensor primitive.
    QFloat(B::QuantizedTensorPrimitive),
}

impl<B: Backend> TensorPrimitive<B> {
    /// Returns the full tensor representation.
    pub fn tensor(self) -> B::FloatTensorPrimitive {
        match self {
            Self::QFloat(tensor) => {
                let dtype = get_device_settings::<B>(&B::q_device(&tensor)).float_dtype;
                B::dequantize(tensor, dtype)
            }
            Self::Float(tensor) => tensor,
        }
    }

    /// Returns a mutable reference to the full tensor representation.
    pub fn get_mut_ref(&mut self) -> &mut B::FloatTensorPrimitive {
        match self {
            Self::QFloat(_tensor) => todo!(),
            Self::Float(tensor) => tensor,
        }
    }
}

impl<B: Backend> TensorMetadata for TensorPrimitive<B> {
    fn dtype(&self) -> DType {
        match self {
            TensorPrimitive::Float(tensor) => tensor.dtype(),
            TensorPrimitive::QFloat(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            TensorPrimitive::Float(tensor) => tensor.shape(),
            TensorPrimitive::QFloat(tensor) => tensor.shape(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            TensorPrimitive::Float(tensor) => tensor.rank(),
            TensorPrimitive::QFloat(tensor) => tensor.rank(),
        }
    }
}

pub use ruda_core::tensor::primitive::{QTensorPrimitive, TensorMetadata};
