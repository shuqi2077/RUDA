use super::*;

// ============================================================================
// any / all operations
// ============================================================================

/// Check if any element is non-zero (float tensors).
pub fn any_float(tensor: HostTensor, out_dtype: BoolDType) -> HostTensor {
    let has_any = match tensor.dtype() {
        DType::F32 => iter_elements::<f32>(&tensor).any(|x| x != 0.0),
        DType::F64 => iter_elements::<f64>(&tensor).any(|x| x != 0.0),
        DType::F16 => iter_elements::<f16>(&tensor).any(|x: f16| x.to_f32() != 0.0),
        DType::BF16 => iter_elements::<bf16>(&tensor).any(|x: bf16| x.to_f32() != 0.0),
        _ => panic!("any_float: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(has_any, out_dtype)
}

/// Check if any element along a dimension is non-zero (float tensors).
pub fn any_float_dim(tensor: HostTensor, dim: usize, out_dtype: BoolDType) -> HostTensor {
    reduce_bool_dim(&tensor, dim, false, |a, b| a || b, out_dtype)
}

/// Check if all elements are non-zero (float tensors).
pub fn all_float(tensor: HostTensor, out_dtype: BoolDType) -> HostTensor {
    let all = match tensor.dtype() {
        DType::F32 => iter_elements::<f32>(&tensor).all(|x| x != 0.0),
        DType::F64 => iter_elements::<f64>(&tensor).all(|x| x != 0.0),
        DType::F16 => iter_elements::<f16>(&tensor).all(|x: f16| x.to_f32() != 0.0),
        DType::BF16 => iter_elements::<bf16>(&tensor).all(|x: bf16| x.to_f32() != 0.0),
        _ => panic!("all_float: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(all, out_dtype)
}

/// Check if all elements along a dimension are non-zero (float tensors).
pub fn all_float_dim(tensor: HostTensor, dim: usize, out_dtype: BoolDType) -> HostTensor {
    reduce_bool_dim(&tensor, dim, true, |a, b| a && b, out_dtype)
}

/// Check if any element is non-zero (int tensors).
pub fn any_int(tensor: HostTensor, out_dtype: BoolDType) -> HostTensor {
    let has_any = match tensor.dtype() {
        DType::I64 => iter_elements::<i64>(&tensor).any(|x| x != 0),
        DType::I32 => iter_elements::<i32>(&tensor).any(|x| x != 0),
        DType::I16 => iter_elements::<i16>(&tensor).any(|x| x != 0),
        DType::I8 => iter_elements::<i8>(&tensor).any(|x| x != 0),
        DType::U64 => iter_elements::<u64>(&tensor).any(|x| x != 0),
        DType::U32 => iter_elements::<u32>(&tensor).any(|x| x != 0),
        DType::U16 => iter_elements::<u16>(&tensor).any(|x| x != 0),
        DType::U8 => iter_elements::<u8>(&tensor).any(|x| x != 0),
        _ => panic!("any_int: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(has_any, out_dtype)
}

/// Check if any element along a dimension is non-zero (int tensors).
pub fn any_int_dim(tensor: HostTensor, dim: usize, out_dtype: BoolDType) -> HostTensor {
    reduce_bool_dim_int(&tensor, dim, false, |a, b| a || b, out_dtype)
}

/// Check if all elements are non-zero (int tensors).
pub fn all_int(tensor: HostTensor, out_dtype: BoolDType) -> HostTensor {
    let all = match tensor.dtype() {
        DType::I64 => iter_elements::<i64>(&tensor).all(|x| x != 0),
        DType::I32 => iter_elements::<i32>(&tensor).all(|x| x != 0),
        DType::I16 => iter_elements::<i16>(&tensor).all(|x| x != 0),
        DType::I8 => iter_elements::<i8>(&tensor).all(|x| x != 0),
        DType::U64 => iter_elements::<u64>(&tensor).all(|x| x != 0),
        DType::U32 => iter_elements::<u32>(&tensor).all(|x| x != 0),
        DType::U16 => iter_elements::<u16>(&tensor).all(|x| x != 0),
        DType::U8 => iter_elements::<u8>(&tensor).all(|x| x != 0),
        _ => panic!("all_int: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(all, out_dtype)
}

/// Check if all elements along a dimension are non-zero (int tensors).
pub fn all_int_dim(tensor: HostTensor, dim: usize, out_dtype: BoolDType) -> HostTensor {
    reduce_bool_dim_int(&tensor, dim, true, |a, b| a && b, out_dtype)
}

/// Check if any bool element is true.
pub fn any_bool(tensor: HostTensor, out_dtype: BoolDType) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let data: &[u8] = tensor.bytes();
    bool_scalar(data.iter().any(|&x| x != 0), out_dtype)
}

/// Check if any bool element along a dimension is true.
pub fn any_bool_dim(tensor: HostTensor, dim: usize, out_dtype: BoolDType) -> HostTensor {
    reduce_bool_dim_raw(&tensor, dim, false, |a, b| a || b, out_dtype)
}

/// Check if all bool elements are true.
pub fn all_bool(tensor: HostTensor, out_dtype: BoolDType) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let data: &[u8] = tensor.bytes();
    bool_scalar(data.iter().all(|&x| x != 0), out_dtype)
}

/// Check if all bool elements along a dimension are true.
pub fn all_bool_dim(tensor: HostTensor, dim: usize, out_dtype: BoolDType) -> HostTensor {
    reduce_bool_dim_raw(&tensor, dim, true, |a, b| a && b, out_dtype)
}

