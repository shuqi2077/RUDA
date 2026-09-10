use ruda_kernel::dsl::{Runtime, prelude::InputScalar};
use ruda_kernel::tensor::RudaTensor;
use super::binary::numeric::{AddOp, DivOp, MulOp, PowOp, RemainderOp, SubOp, launch_binop, launch_scalar_binop};
use super::binary::int::{BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, launch_binop_int, launch_scalar_binop_int};

/// Add two tensors
pub fn add<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop::<R, AddOp>(lhs, rhs)
}

/// Add a tensor and a scalar
pub fn add_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop::<R, AddOp>(lhs, rhs)
}

/// Subtract two tensors
pub fn sub<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop::<R, SubOp>(lhs, rhs)
}

/// Subtract a tensor and a scalar
pub fn sub_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop::<R, SubOp>(lhs, rhs)
}

/// Multiply two tensors
pub fn mul<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop::<R, MulOp>(lhs, rhs)
}

/// Multiply a tensor and a scalar
pub fn mul_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop::<R, MulOp>(lhs, rhs)
}

/// Divide two tensors
pub fn div<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop::<R, DivOp>(lhs, rhs)
}

/// Divide a tensor by a scalar
pub fn div_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop::<R, DivOp>(lhs, rhs)
}

/// Calculate remainder of two tensors
pub fn remainder<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop::<R, RemainderOp>(lhs, rhs)
}

/// Calculate the remainder of a tensor with a scalar
pub fn remainder_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop::<R, RemainderOp>(lhs, rhs)
}

/// Calculate the power of two tensors
pub fn pow<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop::<R, PowOp>(lhs, rhs)
}

/// Bitwise and two tensors
pub fn bitwise_and<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop_int::<R, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise and with a scalar
pub fn bitwise_and_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop_int::<R, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise or two tensors
pub fn bitwise_or<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop_int::<R, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise or with a scalar
pub fn bitwise_or_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop_int::<R, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise xor two tensors
pub fn bitwise_xor<R: Runtime>(lhs: RudaTensor<R>, rhs: RudaTensor<R>) -> RudaTensor<R> {
    launch_binop_int::<R, BitwiseXorOp>(lhs, rhs)
}

/// Bitwise xor with a scalar
pub fn bitwise_xor_scalar<R: Runtime>(lhs: RudaTensor<R>, rhs: InputScalar) -> RudaTensor<R> {
    launch_scalar_binop_int::<R, BitwiseXorOp>(lhs, rhs)
}

