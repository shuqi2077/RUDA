use ruda_core::tensor::{host::HostTensor, element::Scalar, TensorMetadata};
use num_traits::ToPrimitive;
use super::{BinaryOp, binary_op, scalar_op};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

pub fn float_add(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    binary_op(lhs, rhs, |a, b| a + b, |a, b| a + b, Some(BinaryOp::Add))
}

pub fn float_add_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    let rhs_val = rhs.to_f64().unwrap();
    scalar_op(lhs, rhs_val, |a, b| a + b, |a, b| a + b)
}

pub fn float_sub(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    binary_op(lhs, rhs, |a, b| a - b, |a, b| a - b, Some(BinaryOp::Sub))
}

pub fn float_sub_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    let rhs_val = rhs.to_f64().unwrap();
    scalar_op(lhs, rhs_val, |a, b| a - b, |a, b| a - b)
}

pub fn float_mul(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    binary_op(lhs, rhs, |a, b| a * b, |a, b| a * b, Some(BinaryOp::Mul))
}

pub fn float_mul_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    let rhs_val = rhs.to_f64().unwrap();
    scalar_op(lhs, rhs_val, |a, b| a * b, |a, b| a * b)
}

pub fn float_div(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    binary_op(lhs, rhs, |a, b| a / b, |a, b| a / b, Some(BinaryOp::Div))
}

pub fn float_div_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    let rhs_val = rhs.to_f64().unwrap();
    scalar_op(lhs, rhs_val, |a, b| a / b, |a, b| a / b)
}

pub fn float_remainder(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    // Python/PyTorch-style remainder: result has same sign as divisor
    binary_op(
        lhs,
        rhs,
        |a, b| ((a % b) + b) % b,
        |a, b| ((a % b) + b) % b,
        None,
    )
}

pub fn float_remainder_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    let rhs_val = rhs.to_f64().unwrap();
    // Python/PyTorch-style remainder: result has same sign as divisor
    scalar_op(
        lhs,
        rhs_val,
        |a, b| ((a % b) + b) % b,
        |a, b| ((a % b) + b) % b,
    )
}

pub fn float_powf(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    binary_op(lhs, rhs, |a: f32, b| a.powf(b), |a: f64, b| a.powf(b), None)
}

pub fn float_powf_scalar_impl(tensor: HostTensor, value: Scalar) -> HostTensor {
    let exp = value.to_f64().unwrap();
    scalar_op(tensor, exp, |a: f32, b| a.powf(b), |a: f64, b| a.powf(b))
}

pub fn float_atan2(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    binary_op(
        lhs,
        rhs,
        |a: f32, b| a.atan2(b),
        |a: f64, b| a.atan2(b),
        None,
    )
}

pub fn float_powi(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    super::integer_power::tensor(lhs, rhs)
}

pub fn float_powi_scalar(lhs: HostTensor, rhs: Scalar) -> HostTensor {
    if let Scalar::UInt(exponent) = rhs
        && exponent > i64::MAX as u64
    {
        return super::integer_power::scalar(lhs, false, exponent);
    }
    match rhs.to_i64().unwrap() {
        0 => crate::fill::float_ones(lhs.shape(), lhs.dtype().into()),
        1 => lhs,
        2 => crate::binary::dispatch_float::float_mul(lhs.clone(), lhs),
        -1 => crate::unary::recip(lhs),
        -2 => crate::unary::recip(crate::binary::dispatch_float::float_mul(lhs.clone(), lhs)),
        exponent => super::integer_power::scalar(lhs, exponent < 0, exponent.unsigned_abs()),
    }
}

pub fn float_powf_scalar(tensor: HostTensor, value: Scalar) -> HostTensor {
    if let Some(exp) = value.try_as_integer() {
        crate::binary::dispatch_float::float_powi_scalar(tensor, exp)
    } else {
        crate::binary::dispatch_float::float_powf_scalar_impl(tensor, value)
    }
}

