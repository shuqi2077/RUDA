use crate::{Backend, DType, FloatDType, TensorMetadata, tensor::FloatTensor};

/// Explicit working-storage SiLU; existing activation defaults are not changed.
pub fn silu_native<B: Backend>(input: FloatTensor<B>) -> FloatTensor<B> {
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    B::float_cast(B::silu(B::float_cast(input, compute)), storage)
}

/// Independent SiLU VJP, using FP32 for half inputs and FP64 when either operand uses FP64.
/// Returned storage follows the actual original input, without retaining rounded sigmoid values.
pub fn silu_native_backward<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
    assert_eq!(input.shape(), grad.shape(), "SiLU gradient shape differs");
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 || grad.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    if input.shape().num_elements() == 0 { return input; }
    let input = B::float_cast(input, compute);
    let grad = B::float_cast(grad, compute);
    let sigmoid = B::float_recip(B::float_add_scalar(B::float_exp(B::float_neg(input.clone())), 1f32.into()));
    let correction = B::float_add_scalar(B::float_mul(input, B::float_add_scalar(B::float_neg(sigmoid.clone()), 1f32.into())), 1f32.into());
    B::float_cast(B::float_mul(B::float_mul(grad, sigmoid), correction), storage)
}
