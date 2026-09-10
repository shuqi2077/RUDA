use super::*;
use ruda_core::tensor::quantization::params_shape;

pub fn quantize_dynamic(tensor: HostTensor, scheme: &QuantScheme) -> HostQTensor {
    let shape = tensor.shape();
    let tensor = tensor.to_contiguous();
    let float_data = float_storage_as_f32(&tensor);
    let (a, b) = scheme.value.range();
    let range = b - a;

    let (quantized, scales) = match scheme.level {
        QuantLevel::Tensor => {
            // Pass 1: find alpha = max(|min|, |max|)
            let mut alpha: f32 = 0.0;
            for &x in &*float_data {
                let abs = x.abs();
                if abs > alpha {
                    alpha = abs;
                }
            }
            let scale = validated_scale(2.0 * alpha / range);
            let inv_scale = 1.0 / scale;

            // Pass 2: quantize
            let quantized = float_data
                .iter()
                .map(|&x| (x * inv_scale).round().clamp(a, b) as i8)
                .collect::<Vec<i8>>();

            (quantized, alloc::vec![scale])
        }
        QuantLevel::Block(block_size) => {
            let block_dims = block_size.to_dim_vec(shape.rank());
            let params_shape = params_shape(&shape, scheme.level);
            let mut alphas = alloc::vec![0.0f32; params_shape.num_elements()];
            for (index, &x) in float_data.iter().enumerate() {
                let block = block_param_index(index, &shape, &block_dims, &params_shape);
                let abs = x.abs();
                if abs > alphas[block] {
                    alphas[block] = abs;
                }
            }
            let scales = alphas.into_iter()
                .map(|alpha| validated_scale(2.0 * alpha / range))
                .collect::<Vec<_>>();
            let inv_scales = scales.iter().map(|scale| 1.0 / scale).collect::<Vec<_>>();
            let quantized = float_data.iter().enumerate()
                .map(|(index, &x)| {
                    let block = block_param_index(index, &shape, &block_dims, &params_shape);
                    (x * inv_scales[block]).round().clamp(a, b) as i8
                })
                .collect();
            (quantized, scales)
        }
    };

    let bytes = Bytes::from_elems(quantized);
    let layout = Layout::contiguous(shape);
    let qt = HostTensor::new(bytes, layout, DType::I8);

    HostQTensor::new(qt, scheme.with_store(QuantStore::Native), scales)
}

pub fn quantize(
    tensor: HostTensor,
    scheme: &QuantScheme,
    qparams: QParams<HostTensor>,
) -> HostQTensor {
    let shape = tensor.shape();
    let tensor = tensor.to_contiguous();
    let float_data = float_storage_as_f32(&tensor);

    // Extract and validate scales from the qparams tensor. The scales tensor
    // shares its dtype with the float element type, which can be any of
    // f32/f64/f16/bf16, so we normalise via float_storage_as_f32 instead of
    // assuming f32 storage.
    let scales_tensor = qparams.scales.to_contiguous();
    let scales_data = float_storage_as_f32(&scales_tensor);
    let scales: Vec<f32> = scales_data.iter().copied().map(validated_scale).collect();
    assert_eq!(
        scales.len(), params_shape(&shape, scheme.level).num_elements(),
        "quantized scale count must match the parameter shape"
    );

    let (a, b) = scheme.value.range();

    let quantized = match scheme.level {
        QuantLevel::Tensor => {
            let inv_scale = 1.0 / scales[0];
            float_data
                .iter()
                .map(|&x| (x * inv_scale).round().clamp(a, b) as i8)
                .collect::<Vec<i8>>()
        }
        QuantLevel::Block(block_size) => {
            let block_dims = block_size.to_dim_vec(shape.rank());
            let params_shape = params_shape(&shape, scheme.level);
            let inv_scales = scales.iter().map(|scale| 1.0 / scale).collect::<Vec<_>>();
            float_data.iter().enumerate()
                .map(|(index, &x)| {
                    let block = block_param_index(index, &shape, &block_dims, &params_shape);
                    (x * inv_scales[block]).round().clamp(a, b) as i8
                })
                .collect::<Vec<_>>()
        }
    };

    let bytes = Bytes::from_elems(quantized);
    let layout = Layout::contiguous(shape);
    let qt = HostTensor::new(bytes, layout, DType::I8);

    HostQTensor::new(qt, scheme.with_store(QuantStore::Native), scales)
}

pub fn dequantize(tensor: HostQTensor, dtype: FloatDType) -> HostTensor {
    let shape = tensor.tensor.shape();
    let qt = tensor.tensor.to_contiguous();
    let q_data: &[i8] = qt.storage();

    let dequantized = match tensor.scheme.level {
        QuantLevel::Tensor => {
            let scale = tensor.scales[0];
            q_data
                .iter()
                .map(|&x_q| scale * x_q as f32)
                .collect::<Vec<f32>>()
        }
        QuantLevel::Block(block_size) => {
            let block_dims = block_size.to_dim_vec(shape.rank());
            let params_shape = params_shape(&shape, tensor.scheme.level);
            q_data
                .iter().enumerate()
                .map(|(index, &x_q)| {
                    let block = block_param_index(index, &shape, &block_dims, &params_shape);
                    tensor.scales[block] * x_q as f32
                })
                .collect::<Vec<f32>>()
        }
    };

    let layout = Layout::contiguous(shape);
    match dtype {
        FloatDType::F32 | FloatDType::Flex32 => {
            HostTensor::new(Bytes::from_elems(dequantized), layout, DType::F32)
        }
        FloatDType::F64 => {
            let data: Vec<f64> = dequantized.iter().map(|&v| v as f64).collect();
            HostTensor::new(Bytes::from_elems(data), layout, DType::F64)
        }
        FloatDType::F16 => {
            let data: Vec<f16> = dequantized.iter().map(|&v| f16::from_f32(v)).collect();
            HostTensor::new(Bytes::from_elems(data), layout, DType::F16)
        }
        FloatDType::BF16 => {
            let data: Vec<bf16> = dequantized.iter().map(|&v| bf16::from_f32(v)).collect();
            HostTensor::new(Bytes::from_elems(data), layout, DType::BF16)
        }
    }
}

fn block_param_index(mut index: usize, shape: &Shape, block_dims: &[u8], params_shape: &Shape) -> usize {
    let mut parameter = 0;
    let mut stride = 1;
    for axis in (0..shape.rank()).rev() {
        let coordinate = index % shape[axis];
        index /= shape[axis];
        parameter += coordinate / block_dims[axis] as usize * stride;
        stride *= params_shape[axis];
    }
    parameter
}

/// Ensure scale is finite and nonzero to avoid division by zero or NaN propagation.
fn validated_scale(scale: f32) -> f32 {
    if scale.is_normal() {
        scale
    } else {
        f32::MIN_POSITIVE
    }
}

