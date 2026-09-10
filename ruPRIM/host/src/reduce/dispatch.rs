use ruda_core::{bytes::Bytes, tensor::{DType, Shape, host::{HostTensor, Layout}}};
use crate::unary;

pub fn float_max_dim_with_indices(
    tensor: HostTensor,
    dim: usize,
    indices_dtype: ruda_core::tensor::IntDType,
) -> (HostTensor, HostTensor) {
    let (values, indices) = crate::reduce::max_dim_with_indices(tensor, dim);
    if indices.dtype() != DType::from(indices_dtype) {
        (values, crate::cast::int_cast(indices, indices_dtype))
    } else {
        (values, indices)
    }
}

pub fn float_min_dim_with_indices(
    tensor: HostTensor,
    dim: usize,
    indices_dtype: ruda_core::tensor::IntDType,
) -> (HostTensor, HostTensor) {
    let (values, indices) = crate::reduce::min_dim_with_indices(tensor, dim);
    if indices.dtype() != DType::from(indices_dtype) {
        (values, crate::cast::int_cast(indices, indices_dtype))
    } else {
        (values, indices)
    }
}

pub fn float_argmax(
    tensor: HostTensor,
    dim: usize,
    out_dtype: ruda_core::tensor::IntDType,
) -> HostTensor {
    let result = crate::reduce::argmax(tensor, dim);
    if result.dtype() != DType::from(out_dtype) {
        crate::cast::int_cast(result, out_dtype)
    } else {
        result
    }
}

pub fn float_argmin(
    tensor: HostTensor,
    dim: usize,
    out_dtype: ruda_core::tensor::IntDType,
) -> HostTensor {
    let result = crate::reduce::argmin(tensor, dim);
    if result.dtype() != DType::from(out_dtype) {
        crate::cast::int_cast(result, out_dtype)
    } else {
        result
    }
}

pub fn float_max_abs(tensor: HostTensor) -> HostTensor {
    let abs = unary::abs(tensor);
    crate::reduce::max(abs)
}

pub fn float_max_abs_dim(tensor: HostTensor, dim: usize) -> HostTensor {
    let abs = unary::abs(tensor);
    crate::reduce::max_dim(abs, dim)
}

pub fn int_mean(tensor: HostTensor) -> HostTensor {
    let n = tensor.layout().num_elements();
    assert!(n > 0, "int_mean: cannot take mean of empty tensor");
    let dtype = tensor.dtype();
    let sum_result = crate::reduce::sum(tensor);
    // Compute in i64 to avoid truncation of n for small int types
    macro_rules! compute_mean {
        ($ty:ty) => {{
            let data: &[$ty] = sum_result.storage();
            let mean_val = (data[0] as i64 / n as i64) as $ty;
            HostTensor::new(
                Bytes::from_elems(alloc::vec![mean_val]),
                Layout::contiguous(Shape::from(alloc::vec![1])),
                dtype,
            )
        }};
    }
    match dtype {
        DType::I64 => compute_mean!(i64),
        DType::I32 => compute_mean!(i32),
        DType::I16 => compute_mean!(i16),
        DType::I8 => compute_mean!(i8),
        other => panic!("int_mean: unsupported dtype {:?}", other),
    }
}

pub fn int_max_abs(tensor: HostTensor) -> HostTensor {
    let abs = crate::unary::int_abs(tensor);
    crate::reduce::max(abs)
}

pub fn int_max_abs_dim(tensor: HostTensor, dim: usize) -> HostTensor {
    let abs = crate::unary::int_abs(tensor);
    crate::reduce::max_dim(abs, dim)
}

