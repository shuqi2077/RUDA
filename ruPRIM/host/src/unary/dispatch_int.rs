use ruda_core::tensor::{DType, host::HostTensor, element::Scalar};
use num_traits::ToPrimitive;
use crate::binary::{int_scalar_op, scalar_op_typed};

pub fn int_neg(tensor: HostTensor) -> HostTensor {
    int_scalar_op(tensor, 0i64, |a, _| a.wrapping_neg())
}

pub fn int_clamp(tensor: HostTensor, min: Scalar, max: Scalar) -> HostTensor {
    if tensor.dtype() == DType::U64 {
        let min_val = min.to_u64().unwrap();
        let max_val = max.to_u64().unwrap();
        return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.clamp(min_val, max_val));
    }
    let min_val = min.to_i64().unwrap();
    let max_val = max.to_i64().unwrap();
    int_scalar_op(tensor, 0i64, move |x, _| x.clamp(min_val, max_val))
}

pub fn int_clamp_min(tensor: HostTensor, min: Scalar) -> HostTensor {
    if tensor.dtype() == DType::U64 {
        let min_val = min.to_u64().unwrap();
        return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.max(min_val));
    }
    let min_val = min.to_i64().unwrap();
    int_scalar_op(tensor, 0i64, move |x, _| x.max(min_val))
}

pub fn int_clamp_max(tensor: HostTensor, max: Scalar) -> HostTensor {
    if tensor.dtype() == DType::U64 {
        let max_val = max.to_u64().unwrap();
        return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.min(max_val));
    }
    let max_val = max.to_i64().unwrap();
    int_scalar_op(tensor, 0i64, move |x, _| x.min(max_val))
}

pub fn int_sign(tensor: HostTensor) -> HostTensor {
    if tensor.dtype() == DType::U64 {
        return scalar_op_typed(tensor, 0u64, |x: u64, _| if x > 0 { 1 } else { 0 });
    }
    int_scalar_op(tensor, 0i64, |x, _| {
        if x > 0 {
            1
        } else if x < 0 {
            -1
        } else {
            0
        }
    })
}

