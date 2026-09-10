use ruda_core::tensor::{DType, host::HostTensor, element::Scalar, TensorMetadata};
use num_traits::ToPrimitive;
use super::{binary_op_typed, int_binary_op, int_scalar_op, scalar_op_typed};

pub fn int_add(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a + b)
}

pub fn int_add_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| {
            a.wrapping_add(b)
        });
    }
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a + b)
}

pub fn int_sub(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a - b)
}

pub fn int_sub_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| {
            a.wrapping_sub(b)
        });
    }
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a - b)
}

pub fn int_mul(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a * b)
}

pub fn int_mul_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| {
            a.wrapping_mul(b)
        });
    }
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a * b)
}

pub fn int_div(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    // U64 values > i64::MAX produce wrong results through i64 cast
    if lhs.dtype() == DType::U64 {
        let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);
        return binary_op_typed(lhs, &rhs, |a: u64, b: u64| a / b);
    }
    int_binary_op(lhs, rhs, |a, b| a / b)
}

pub fn int_div_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a / b);
    }
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a / b)
}

pub fn int_remainder(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    // U64 values > i64::MAX produce wrong results through i64 cast
    if lhs.dtype() == DType::U64 {
        let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);
        return binary_op_typed(lhs, &rhs, |a: u64, b: u64| a % b);
    }
    // Python/PyTorch-style remainder: result has same sign as divisor
    int_binary_op(lhs, rhs, |a, b| ((a % b) + b) % b)
}

pub fn int_remainder_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a % b);
    }
    // Python/PyTorch-style remainder: result has same sign as divisor
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| ((a % b) + b) % b)
}

pub fn bitwise_and(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a & b)
}

pub fn bitwise_and_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a & b);
    }
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a & b)
}

pub fn bitwise_or(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a | b)
}

pub fn bitwise_or_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a | b);
    }
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a | b)
}

pub fn bitwise_xor(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a ^ b)
}

pub fn bitwise_xor_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a ^ b);
    }
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a ^ b)
}

pub fn bitwise_not(tensor: HostTensor) -> HostTensor {
    // Use scalar op with dummy value, only applying NOT to lhs
    int_scalar_op(tensor, 0, |a, _| !a)
}

pub fn bitwise_left_shift(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a.wrapping_shl(b as u32))
}

pub fn bitwise_left_shift_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a.wrapping_shl(b as u32))
}

pub fn bitwise_right_shift(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a.wrapping_shr(b as u32))
}

pub fn bitwise_right_shift_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a.wrapping_shr(b as u32))
}

pub fn int_powi(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    int_binary_op(lhs, rhs, |a, b| a.wrapping_pow(b as u32))
}

pub fn int_powi_scalar(lhs: HostTensor, rhs: ruda_core::tensor::element::Scalar) -> HostTensor {
    use num_traits::ToPrimitive;
    match rhs.to_i64().unwrap() {
        0 => crate::fill::int_ones(lhs.shape(), lhs.dtype().into()),
        1 => lhs,
        2 => crate::binary::dispatch_int::int_mul(lhs.clone(), lhs),
        _ => crate::binary::dispatch_int::int_powi_scalar_impl(lhs, rhs),
    }
}

pub fn int_powi_scalar_impl(lhs: HostTensor, rhs: ruda_core::tensor::element::Scalar) -> HostTensor {
    use num_traits::ToPrimitive;
    let exp = rhs.to_i64().unwrap() as u32;
    if lhs.dtype() == DType::U64 {
        return scalar_op_typed(lhs, exp as u64, move |x: u64, _| x.wrapping_pow(exp));
    }
    int_scalar_op(lhs, exp as i64, move |x, _| x.wrapping_pow(exp))
}

