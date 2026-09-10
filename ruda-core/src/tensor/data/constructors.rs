use super::TensorData;
use crate::bytes::Bytes;
use crate::tensor::{DType, QuantScheme, QuantizedBytes, Shape};
use crate::tensor::element::Element;
use alloc::vec::Vec;

impl TensorData {
    /// Creates a new tensor data structure.
    pub fn new<E: Element, S: Into<Shape>>(value: Vec<E>, shape: S) -> Self {
        // Ensure shape is valid
        let shape = shape.into();
        Self::check_data_len(&value, &shape);

        Self {
            bytes: Bytes::from_elems(value),
            shape,
            dtype: E::dtype(),
        }
    }

    /// Creates a new quantized tensor data structure.
    pub fn quantized<E: Element, S: Into<Shape>>(
        value: Vec<E>,
        shape: S,
        scheme: QuantScheme,
        qparams: &[f32],
    ) -> Self {
        let shape = shape.into();
        Self::check_data_len(&value, &shape);

        let q_bytes = QuantizedBytes::new(value, scheme, qparams);

        Self {
            bytes: q_bytes.bytes,
            shape,
            dtype: DType::QFloat(q_bytes.scheme),
        }
    }

    /// Creates a new tensor data structure from raw bytes.
    pub fn from_bytes<S: Into<Shape>>(bytes: Bytes, shape: S, dtype: DType) -> Self {
        Self {
            bytes,
            shape: shape.into(),
            dtype,
        }
    }

    /// Creates a new tensor data structure from raw bytes stored in a vector.
    ///
    /// Prefer [`TensorData::new`] or [`TensorData::quantized`] over this method unless you are
    /// certain that the bytes representation is valid.
    pub fn from_bytes_vec<S: Into<Shape>>(bytes: Vec<u8>, shape: S, dtype: DType) -> Self {
        Self {
            bytes: Bytes::from_bytes_vec(bytes),
            shape: shape.into(),
            dtype,
        }
    }

    // Check that the input vector contains a correct number of elements
    fn check_data_len<E: Element>(data: &[E], shape: &Shape) {
        let expected_data_len = Self::numel(shape);
        let num_data = data.len();
        assert_eq!(
            expected_data_len, num_data,
            "Shape {shape:?} is invalid for input of size {num_data:?}",
        );
    }

}
