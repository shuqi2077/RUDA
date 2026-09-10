use crate::api::{Tensor, TensorPrimitive, backend::Backend};
use crate::tensor::quantization;
use crate::{Shape, TensorMetadata};

// We re-export those types.
pub use crate::{QTensorPrimitive, quantization::*};

/// The tensor quantization parameters.
pub type QuantizationParameters<B> = QParams<Tensor<B, 1>>;

/// The observed input calibration range.
#[derive(Clone, Debug)]
pub struct CalibrationRange<B: Backend> {
    /// Minimum observed value(s).
    pub min: Tensor<B, 1>,
    /// Maximum observed value(s).
    pub max: Tensor<B, 1>,
}

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend, const D: usize>(
    scheme: &QuantScheme,
    tensor: &Tensor<B, D>,
    calibration: &Calibration,
) -> CalibrationRange<B> {
    let (min, max) = match &tensor.primitive {
        TensorPrimitive::Float(tensor) => {
            quantization::compute_range::<B>(scheme, tensor.clone(), calibration)
        }
        TensorPrimitive::QFloat(_) => unreachable!(),
    };

    let min_shape = Shape::new([min.shape().num_elements()]);
    let max_shape = Shape::new([max.shape().num_elements()]);
    CalibrationRange {
        min: Tensor::from_primitive(TensorPrimitive::Float(B::float_reshape(min, min_shape))),
        max: Tensor::from_primitive(TensorPrimitive::Float(B::float_reshape(max, max_shape))),
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params<B: Backend>(
    scheme: &QuantScheme,
    range: CalibrationRange<B>,
) -> QuantizationParameters<B> {
    match (range.min.primitive, range.max.primitive) {
        (TensorPrimitive::Float(min), TensorPrimitive::Float(max)) => {
            let qparams = quantization::compute_q_params::<B>(scheme, min, max);
            QuantizationParameters {
                scales: Tensor::from_primitive(TensorPrimitive::Float(qparams.scales)),
            }
        }
        _ => unreachable!(),
    }
}
