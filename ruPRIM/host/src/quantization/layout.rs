use super::*;

pub fn q_reshape(tensor: HostQTensor, shape: Shape) -> HostQTensor {
    block_safe_layout_op(tensor, |t| t.reshape(shape))
}

pub fn q_swap_dims(
    tensor: HostQTensor,
    dim1: usize,
    dim2: usize,
) -> HostQTensor {
    block_safe_layout_op(tensor, |t| t.transpose(dim1, dim2))
}

pub fn q_permute(tensor: HostQTensor, axes: &[usize]) -> HostQTensor {
    block_safe_layout_op(tensor, |t| t.permute(axes))
}

pub fn q_flip(tensor: HostQTensor, axes: &[usize]) -> HostQTensor {
    block_safe_layout_op(tensor, |t| crate::flip::flip(t, axes))
}

pub fn q_expand(tensor: HostQTensor, shape: Shape) -> HostQTensor {
    block_safe_layout_op(tensor, |t| crate::expand::expand(t, shape))
}

pub fn q_select(
    tensor: HostQTensor,
    dim: usize,
    indices: HostTensor,
) -> HostQTensor {
    match tensor.scheme.level {
        QuantLevel::Tensor => HostQTensor::new(
            crate::gather_scatter::select::<i8>(tensor.tensor, dim, indices),
            tensor.scheme,
            tensor.scales,
        ),
        QuantLevel::Block(_) => {
            let scheme = tensor.scheme;
            let float_tensor = crate::quantization::dequantize(tensor, FloatDType::F32);
            let result = crate::gather_scatter::select::<f32>(float_tensor, dim, indices);
            crate::quantization::quantize_dynamic(result, &scheme)
        }
    }
}

pub fn q_slice(tensor: HostQTensor, slices: &[Slice]) -> HostQTensor {
    block_safe_layout_op(tensor, |t| crate::slice::slice(t, slices))
}

pub fn q_argmax(
    tensor: HostQTensor,
    dim: usize,
    out_dtype: ruda_core::tensor::IntDType,
) -> HostTensor {
    let tensor = crate::quantization::dequantize(tensor, FloatDType::F32);
    let result = crate::reduce::argmax(tensor, dim);
    if result.dtype() != DType::from(out_dtype) {
        crate::cast::int_cast(result, out_dtype)
    } else {
        result
    }
}

pub fn q_argmin(
    tensor: HostQTensor,
    dim: usize,
    out_dtype: ruda_core::tensor::IntDType,
) -> HostTensor {
    let tensor = crate::quantization::dequantize(tensor, FloatDType::F32);
    let result = crate::reduce::argmin(tensor, dim);
    if result.dtype() != DType::from(out_dtype) {
        crate::cast::int_cast(result, out_dtype)
    } else {
        result
    }
}

pub fn q_gather(
    dim: usize,
    tensor: HostQTensor,
    indices: HostTensor,
) -> HostQTensor {
    match tensor.scheme.level {
        QuantLevel::Tensor => HostQTensor::new(
            crate::gather_scatter::gather::<i8>(tensor.tensor, dim, indices),
            tensor.scheme,
            tensor.scales,
        ),
        QuantLevel::Block(_) => {
            let scheme = tensor.scheme;
            let float_tensor = crate::quantization::dequantize(tensor, FloatDType::F32);
            let result = crate::gather_scatter::gather::<f32>(float_tensor, dim, indices);
            crate::quantization::quantize_dynamic(result, &scheme)
        }
    }
}

/// Apply a layout operation to a quantized tensor.
/// For block-quantized tensors, dequantizes and requantizes to preserve
/// correct scale-to-block mapping.
fn block_safe_layout_op(
    qtensor: HostQTensor,
    op: impl FnOnce(HostTensor) -> HostTensor,
) -> HostQTensor {
    match qtensor.scheme.level {
        QuantLevel::Tensor => HostQTensor::new(op(qtensor.tensor), qtensor.scheme, qtensor.scales),
        QuantLevel::Block(_) => {
            let scheme = qtensor.scheme;
            let float_tensor = crate::quantization::dequantize(qtensor, FloatDType::F32);
            let result = op(float_tensor);
            crate::quantization::quantize_dynamic(result, &scheme)
        }
    }
}

