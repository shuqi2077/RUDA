use super::{DType, Shape};
use super::quantization::{QuantAcc, QuantPropagation, QuantScheme};

/// Tensor metadata trait for tensor primitive.
pub trait TensorMetadata: Clone + Send + Sync + core::fmt::Debug {
    /// The dtype of the tensor.
    fn dtype(&self) -> DType;
    /// The shape of the tensor.
    fn shape(&self) -> Shape;

    /// The number of dimensions of the tensor.
    fn rank(&self) -> usize {
        self.shape().num_dims()
    }
}

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization settings for the given tensor.
    fn scheme(&self) -> &QuantScheme;
    /// The precision used for the accumulation in various kernels.
    fn acc_precision(&self) -> QuantAcc {
        QuantAcc::F32
    }
    /// How quantization is propagated during computation.
    fn propagation(&self) -> QuantPropagation {
        QuantPropagation::Inhibit
    }

    /// Returns the default tensor quantization scheme.
    fn default_scheme() -> QuantScheme {
        QuantScheme::default()
    }
}
