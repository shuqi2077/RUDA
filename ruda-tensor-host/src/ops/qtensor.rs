//! Quantized tensor operations for the Host backend.

use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use ruda_tensor::{
    DType, ExecutionError, FloatDType, TensorData, TensorMetadata,
    ops::QTensorOps,
    quantization::{
        QuantLevel, QuantScheme, QuantStore, QuantizationParametersPrimitive,
    },
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use ruda_tensor::{Bytes, Shape, Slice, bf16, f16};

use crate::{Host, HostTensor, Layout};

impl QTensorOps<Host> for Host {
    fn q_from_data(data: TensorData, _device: &Device<Host>) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_from_data(data)
    }

    fn quantize_dynamic(tensor: FloatTensor<Host>, scheme: &QuantScheme) -> QuantizedTensor<Host> {
        ruprim_host::quantization::quantize_dynamic(tensor, scheme)
    }

    fn quantize(
        tensor: FloatTensor<Host>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Host>,
    ) -> QuantizedTensor<Host> {
        ruprim_host::quantization::quantize(tensor, scheme, ruda_core::tensor::quantization::QParams { scales: qparams.scales })
    }

    fn dequantize(tensor: QuantizedTensor<Host>, dtype: FloatDType) -> FloatTensor<Host> {
        ruprim_host::quantization::dequantize(tensor, dtype)
    }

    fn q_device(_tensor: &QuantizedTensor<Host>) -> Device<Host> {
        Default::default()
    }

    fn q_to_device(tensor: QuantizedTensor<Host>, _device: &Device<Host>) -> QuantizedTensor<Host> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Host>, shape: Shape) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Host>) -> Result<TensorData, ExecutionError> {
        ruprim_host::quantization::q_into_data(tensor).await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Host>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Host>, axes: &[usize]) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Host>, axes: &[usize]) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_flip(tensor, axes)
    }

    fn q_expand(tensor: QuantizedTensor<Host>, shape: Shape) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_expand(tensor, shape)
    }

    fn q_select(
        tensor: QuantizedTensor<Host>,
        dim: usize,
        indices: IntTensor<Host>,
    ) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Host>, slices: &[Slice]) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_slice(tensor, slices)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Host>,
        indices: IntTensor<Host>,
    ) -> QuantizedTensor<Host> {
        ruprim_host::quantization::q_gather(dim, tensor, indices)
    }
}

// Tests kept here exercise flex-specific behavior: quantization scheme
// roundtrips, per-block / dynamic quantization, block-quantized layout
// ops (transpose / select / flip dequantize), and f16/f64 dequantize
// dtype paths. Plain layout-preservation / select / slice / argmax /
// argmin / gather tests are covered generically in
// crates/ruda-backend-tests/tests/tensor/float/quantization/ops/extended/
// so they run on every backend.
#[cfg(test)]
mod tests {
    use super::*;
    use ruda_tensor::{TensorMetadata, quantization::QuantValue};

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Create a float tensor
        let values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [2, 3]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        // Compute scale: symmetric, so scale = 2 * max(|min|, |max|) / (b - a)
        // max_abs = 5.0, range = 127 - (-127) = 254
        // scale = 2 * 5.0 / 254 = 0.03937008
        let scale: f32 = 2.0 * 5.0 / 254.0;
        let scales_tensor = HostTensor::from_data(TensorData::new(vec![scale], [1]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        // Quantize
        let qtensor = Host::quantize(tensor, &scheme, qparams);
        assert_eq!(qtensor.tensor().shape().to_vec(), vec![2, 3]);
        assert_eq!(qtensor.tensor().dtype(), DType::I8);

        // Check quantized values
        let q_vals: &[i8] = qtensor.tensor().storage();
        // 0 / 0.03937 = 0, 1 / 0.03937 = 25.4 -> 25, etc.
        assert_eq!(q_vals[0], 0);
        assert_eq!(q_vals[1], 25);
        assert_eq!(q_vals[5], 127);

        // Dequantize
        let result = Host::dequantize(qtensor, FloatDType::F32);
        assert_eq!(result.shape().to_vec(), vec![2, 3]);
        assert_eq!(result.dtype(), DType::F32);

        let result_vals: &[f32] = result.storage();
        // Values should be approximately equal (quantization introduces small errors)
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_quantize_dequantize_negative_values() {
        let values = vec![-3.0f32, -1.5, 0.0, 1.5, 3.0];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [5]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let scale: f32 = 2.0 * 3.0 / 254.0;
        let scales_tensor = HostTensor::from_data(TensorData::new(vec![scale], [1]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        let qtensor = Host::quantize(tensor, &scheme, qparams);
        let result = Host::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();

        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_q_from_data_into_data_roundtrip() {
        // Create quantized TensorData the standard way
        let values = vec![0i8, 25, 51, 76, 102, 127];
        let scale = 0.03937008f32;
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let data = TensorData::quantized(values.clone(), [2, 3], scheme, &[scale]);

        // Load into HostQTensor
        let qtensor = Host::q_from_data(data, &Default::default());
        assert_eq!(qtensor.tensor().shape().to_vec(), vec![2, 3]);
        assert_eq!(qtensor.scales(), vec![scale]);

        // Dequantize and check values
        let float_tensor = Host::dequantize(qtensor, FloatDType::F32);
        let result: &[f32] = float_tensor.storage();
        assert!((result[0]).abs() < 0.01); // 0 * scale ~ 0
        assert!((result[5] - 5.0).abs() < 0.05); // 127 * scale ~ 5.0
    }

    #[test]
    fn test_quantize_zero_tensor() {
        let values = vec![0.0f32; 4];
        let tensor = HostTensor::from_data(TensorData::new(values, [4]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        // Scale of 0 should be handled gracefully
        let scales_tensor = HostTensor::from_data(TensorData::new(vec![0.0f32], [1]));
        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        let qtensor = Host::quantize(tensor, &scheme, qparams);
        let q_vals: &[i8] = qtensor.tensor().storage();
        assert_eq!(q_vals, &[0, 0, 0, 0]);
    }

    #[test]
    fn test_quantize_dynamic_roundtrip() {
        let values = vec![-3.0f32, -1.5, 0.0, 1.5, 3.0, 4.5];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [2, 3]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);
        assert_eq!(qtensor.tensor().shape().to_vec(), vec![2, 3]);
        assert_eq!(qtensor.scales().len(), 1);

        // Scale should be 2 * 4.5 / 254
        let expected_scale: f32 = 2.0 * 4.5 / 254.0;
        assert!(
            (qtensor.scales()[0] - expected_scale).abs() < 1e-6,
            "scale={}, expected={}",
            qtensor.scales()[0],
            expected_scale
        );

        let result = Host::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_per_block_quantize_dequantize() {
        use ruda_tensor::quantization::BlockSize;

        let values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [8]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Block(block_size))
            .with_store(QuantStore::Native);

        // Block 1: [0, 1, 2, 3] -> max_abs=3, scale = 6/254
        // Block 2: [4, 5, 6, 7] -> max_abs=7, scale = 14/254
        let scale_1: f32 = 2.0 * 3.0 / 254.0;
        let scale_2: f32 = 2.0 * 7.0 / 254.0;
        let scales_tensor = HostTensor::from_data(TensorData::new(vec![scale_1, scale_2], [2]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        let qtensor = Host::quantize(tensor, &scheme, qparams);
        assert_eq!(qtensor.scales().len(), 2);

        let result = Host::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();

        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_quantize_dynamic_block() {
        use ruda_tensor::quantization::BlockSize;

        let values = vec![-2.0f32, -1.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [8]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Block(block_size))
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);
        assert_eq!(qtensor.scales().len(), 2);

        // Block 1: [-2, -1, 0, 1] -> alpha=2, scale = 4/254
        // Block 2: [4, 5, 6, 7] -> alpha=7, scale = 14/254
        let expected_scale_1: f32 = 2.0 * 2.0 / 254.0;
        let expected_scale_2: f32 = 2.0 * 7.0 / 254.0;
        assert!((qtensor.scales()[0] - expected_scale_1).abs() < 1e-6);
        assert!((qtensor.scales()[1] - expected_scale_2).abs() < 1e-6);

        let result = Host::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_quantize_dynamic_q8f() {
        // Q8F uses asymmetric range [-128, 127]
        let values = vec![-5.0f32, -2.5, 0.0, 2.5, 5.0, 7.5];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [6]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8F)
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);

        // Q8F range: [-128, 127], so range = 255
        // alpha = 7.5, scale = 2 * 7.5 / 255
        let expected_scale: f32 = 2.0 * 7.5 / 255.0;
        assert!(
            (qtensor.scales()[0] - expected_scale).abs() < 1e-6,
            "scale={}, expected={}",
            qtensor.scales()[0],
            expected_scale
        );

        let result = Host::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_block_quantized_transpose_dequantize() {
        use ruda_tensor::quantization::BlockSize;

        // 2x4 tensor, 2 blocks of 4
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = HostTensor::from_data(TensorData::new(values, [2, 4]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Block(block_size))
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);

        // Transpose to [4, 2], then dequantize
        let transposed = Host::q_swap_dims(qtensor, 0, 1);
        assert_eq!(transposed.tensor().shape().to_vec(), vec![4, 2]);

        let result = Host::dequantize(transposed, FloatDType::F32);
        let result_vals: &[f32] = result.storage();

        // Original [[1,2,3,4],[5,6,7,8]] transposed to [[1,5],[2,6],[3,7],[4,8]]
        let expected = [1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!(
                (exp - deq).abs() < 0.15,
                "expected={exp}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_block_quantized_select() {
        use ruda_tensor::quantization::BlockSize;

        // 2x4 tensor, 2 blocks of 4
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let tensor = HostTensor::from_data(TensorData::new(values, [2, 4]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Block(block_size))
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);

        // Select row 1 -> [10, 20, 30, 40]
        let indices = HostTensor::from_data(TensorData::new(vec![1i64], [1]));
        let selected = Host::q_select(qtensor, 0, indices);
        assert_eq!(selected.tensor().shape().to_vec(), vec![1, 4]);

        let result = Host::dequantize(selected, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        let expected = [10.0f32, 20.0, 30.0, 40.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!((exp - deq).abs() < 0.5, "expected={exp}, dequantized={deq}");
        }
    }

    #[test]
    fn test_block_quantized_flip_dequantize() {
        use ruda_tensor::quantization::BlockSize;

        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = HostTensor::from_data(TensorData::new(values, [2, 4]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Block(block_size))
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);

        // Flip along axis 0: [[5,6,7,8],[1,2,3,4]]
        let flipped = Host::q_flip(qtensor, &[0]);
        assert_eq!(flipped.tensor().shape().to_vec(), vec![2, 4]);

        let result = Host::dequantize(flipped, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        let expected = [5.0f32, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!(
                (exp - deq).abs() < 0.15,
                "expected={exp}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_quantize_dynamic_f64_tensor() {
        use ruda_tensor::quantization::QuantValue;

        let values = vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = HostTensor::new(
            Bytes::from_elems(values),
            Layout::contiguous([6].into()),
            DType::F64,
        );
        assert_eq!(tensor.dtype(), DType::F64);

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);
        assert_eq!(qtensor.tensor().dtype(), DType::I8);

        // Dequantize and verify round-trip accuracy
        let result = Host::dequantize(qtensor, FloatDType::F32);
        let result_vals: &[f32] = result.storage();
        let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        for (exp, deq) in expected.iter().zip(result_vals.iter()) {
            assert!(
                (exp - deq).abs() < 0.15,
                "expected={exp}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_dequantize_f64() {
        let values = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [4]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);
        let result = Host::dequantize(qtensor, FloatDType::F64);
        assert_eq!(result.dtype(), DType::F64);
        let result_vals: &[f64] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!(
                (*orig as f64 - deq).abs() < 0.05,
                "orig={orig}, dequantized={deq}"
            );
        }
    }

    #[test]
    fn test_dequantize_f16() {
        let values = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = HostTensor::from_data(TensorData::new(values.clone(), [4]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let qtensor = Host::quantize_dynamic(tensor, &scheme);
        let result = Host::dequantize(qtensor, FloatDType::F16);
        assert_eq!(result.dtype(), DType::F16);
        let result_vals: &[f16] = result.storage();
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!(
                (*orig - f32::from(*deq)).abs() < 0.05,
                "orig={orig}, dequantized={deq}"
            );
        }
    }
}
