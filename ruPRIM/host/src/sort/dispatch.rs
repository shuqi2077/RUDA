use ruda_core::tensor::{DType, host::HostTensor};

pub fn float_argtopk(
    tensor: HostTensor,
    dim: usize,
    k: usize,
    out_dtype: ruda_core::tensor::IntDType,
) -> HostTensor {
    let indices = super::argtopk(tensor, dim, k);
    if indices.dtype() != DType::from(out_dtype) {
        crate::cast::int_cast(indices, out_dtype)
    } else {
        indices
    }
}

pub fn int_argtopk(tensor: HostTensor, dim: usize, k: usize) -> HostTensor {
    let dtype = tensor.dtype();
    let indices = super::argtopk(tensor, dim, k);
    if indices.dtype() != dtype {
        crate::cast::int_cast(indices, dtype.into())
    } else {
        indices
    }
}

pub fn float_sort_with_indices(
    tensor: HostTensor,
    dim: usize,
    descending: bool,
    indices_dtype: ruda_core::tensor::IntDType,
) -> (HostTensor, HostTensor) {
    let (values, indices) = crate::sort::sort_with_indices(tensor, dim, descending);
    let indices = if indices.dtype() != DType::from(indices_dtype) {
        crate::cast::int_cast(indices, indices_dtype)
    } else {
        indices
    };
    (values, indices)
}

pub fn float_argsort(
    tensor: HostTensor,
    dim: usize,
    descending: bool,
    out_dtype: ruda_core::tensor::IntDType,
) -> HostTensor {
    let indices = crate::sort::argsort(tensor, dim, descending);
    if indices.dtype() != DType::from(out_dtype) {
        crate::cast::int_cast(indices, out_dtype)
    } else {
        indices
    }
}

