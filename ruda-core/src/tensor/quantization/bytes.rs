use super::{QPARAM_ALIGN, QParams, QuantLevel, QuantScheme, QuantStore, QuantValue};
use super::packing::unpack_q_to_i8s;
use crate::bytes::Bytes;
use crate::tensor::Shape;
use alloc::vec::Vec;
use core::any::TypeId;

/// Quantized data bytes representation.
///
/// # Notes
/// 1) The quantized values are packed into 32-bit unsigned integers. For example, int8
///    quantized values pack 4 grouped values into a single `u32`. When unpacking these values,
///    we make sure to retrieve only the meaningful values (and ignore the alignment padding).
/// 2) Quantization parameters are appended to the tensor data.
///    As such, the last bytes always correspond to the scale parameter.
///    If the quantization scheme includes an offset (zero-point) parameter, it is next to last.
pub struct QuantizedBytes {
    /// The quantized values and quantization parameters represented as bytes.
    pub bytes: Bytes,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The number of quantized elements.
    pub num_elements: usize,
}

impl QuantizedBytes {
    /// Creates a new quantized bytes representation.
    pub fn new<E: bytemuck::CheckedBitPattern + bytemuck::NoUninit>(
        value: Vec<E>,
        scheme: QuantScheme,
        scales: &[f32],
    ) -> Self {
        let num_elements = value.len();
        // Only used for 8-bit quantization data comparison in tests
        if TypeId::of::<E>() != TypeId::of::<i8>() {
            panic!("Invalid quantized type");
        }

        // Re-interpret `Vec<E>` as `Vec<i8>` with `Vec::from_raw_parts`
        let i8s: Vec<i8> = bytemuck::allocation::cast_vec(value);
        let mut bytes = Bytes::from_elems(i8s);

        match scheme.level {
            QuantLevel::Tensor => {
                let scale_bytes = bytemuck::bytes_of(&scales[0]);
                bytes.extend_from_byte_slice_aligned(scale_bytes, QPARAM_ALIGN);
            }
            QuantLevel::Block(_block_size) => {
                let mut scale_bytes = Vec::with_capacity(size_of_val(scales));
                for scale in scales {
                    scale_bytes.extend_from_slice(bytemuck::bytes_of(scale));
                }
                bytes.extend_from_byte_slice_aligned(scale_bytes.as_slice(), QPARAM_ALIGN);
            }
        }

        Self {
            bytes,
            scheme,
            num_elements,
        }
    }

    /// Returns the int8 quantized values with the quantization parameters.
    pub fn into_vec_i8(self) -> (Vec<i8>, QParams<Vec<f32>>) {
        let num_params = match self.scheme.level {
            QuantLevel::Tensor => 1,
            QuantLevel::Block(block_size) => self.num_elements / block_size.num_elements(),
        };
        self.into_vec_i8_with_params(num_params)
    }

    /// Returns the int8 values and parameters using the tensor's multidimensional block layout.
    pub fn into_vec_i8_with_shape(self, shape: &Shape) -> (Vec<i8>, QParams<Vec<f32>>) {
        assert_eq!(shape.num_elements(), self.num_elements, "Quantized shape mismatch");
        let num_params = super::params_shape(shape, self.scheme.level).num_elements();
        self.into_vec_i8_with_params(num_params)
    }

    fn into_vec_i8_with_params(self, num_params: usize) -> (Vec<i8>, QParams<Vec<f32>>) {
        let (values, (qparams, num_params)) = self.split_values_off(num_params);

        // Quantization parameters are added at the end of the tensor data.
        // As such, the last bytes always correspond to the scale parameter(s).
        // For example, per-block quantization can have multiple parameters for a single tensor:
        // [scale, scale, scale, ...]
        let scale_size = core::mem::size_of::<f32>(); // scale is stored as f32
        let qparams_bytes: &[u8] = bytemuck::cast_slice(&qparams);
        let total_bytes = qparams_bytes.len();

        let scales_size = scale_size * num_params;

        let scales = bytemuck::cast_slice(&qparams_bytes[total_bytes - scales_size..]).to_vec();

        (values, QParams { scales })
    }

    fn split_i8_values(self, num_params: usize) -> (Vec<i8>, Vec<u32>) {
        let mut values = read_bytes_to_i8(self.bytes);

        let scale_size = num_params * size_of::<f32>();
        let values_end = values.len() - scale_size;

        let qparams = values.split_off(values_end);

        let qparams = qparams
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([
                chunk[0] as u8, chunk[1] as u8, chunk[2] as u8, chunk[3] as u8,
            ]))
            .collect();
        if matches!(self.scheme.store, QuantStore::PackedU32(_)) {
            values.truncate(self.num_elements);
        }
        (values, qparams)
    }

    /// Splits the quantized values of the tensor from the quantization parameters.
    ///
    /// Returns the values in i8 and a newly allocated vector containing the quantization parameters.
    fn split_values_off(self, num_params: usize) -> (Vec<i8>, (Vec<u32>, usize)) {
        if let QuantStore::PackedU32(packed_dim) = self.scheme.store {
            assert_eq!(
                packed_dim, 0,
                "Packing must be on innermost dimension for splitting off values"
            );
        }

        let (values, qparams) = match self.scheme.store {
            QuantStore::Native => self.split_i8_values(num_params),
            QuantStore::PackedU32(_) => match self.scheme.value {
                QuantValue::Q8F | QuantValue::Q8S => self.split_i8_values(num_params),
                QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                    assert!(self.bytes.len().is_multiple_of(4), "Invalid packed byte count");
                    let mut values = self.bytes
                        .chunks_exact(4)
                        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect::<Vec<_>>();
                    let scale_size = num_params; // size of f32 same as u32
                    let values_end = values.len() - scale_size;

                    let qparams = values.split_off(values_end);
                    // Sub-byte values are unpacked as i8s for value equality tests
                    let values = unpack_q_to_i8s(&values, self.num_elements, &self.scheme.value);
                    (values, qparams)
                }
                QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
                    unimplemented!("Not yet supported")
                }
            },
            QuantStore::PackedNative(_) => unimplemented!("Not yet supported"),
        };

        (values, (qparams, num_params))
    }
}

fn read_bytes_to_i8(bytes: Bytes) -> Vec<i8> {
    match bytes.try_into_vec::<i8>() {
        Ok(val) => val,
        Err(bytes) => bytemuck::allocation::cast_vec(bytes.to_vec()),
    }
}
