use super::SoftmaxOutput;
use crate::{Backend, DType, FloatDType, TensorMetadata, tensor::FloatTensor};

/// Stable normalization with FP32/FP64 arithmetic, without changing input storage.
pub fn softmax_with_stats<B: Backend>(input: FloatTensor<B>, dim: usize, logarithmic: bool) -> SoftmaxOutput<B> {
    let shape = input.shape();
    assert!(dim < shape.num_dims(), "softmax axis out of bounds");
    assert!(shape[dim] > 0, "softmax axis must be nonempty");
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    if shape.num_elements() == 0 {
        return SoftmaxOutput { working: B::float_cast(input.clone(), compute), output: input };
    }
    let input = B::float_cast(input, compute);
    let maximum = B::float_max_dim(B::float_detach(input.clone()), dim);
    let shifted = B::float_sub(input, maximum);
    let numerator = B::float_exp(shifted.clone());
    let denominator = B::float_sum_dim(numerator.clone(), dim);
    let working = if logarithmic {
        B::float_sub(shifted, B::float_log(denominator))
    } else { B::float_div(numerator, denominator) };
    SoftmaxOutput { output: B::float_cast(working.clone(), storage), working }
}

/// Exact first-order softmax/log-softmax VJP using saved working-storage values.
pub fn softmax_backward<B: Backend>(working: FloatTensor<B>, grad: FloatTensor<B>, dim: usize,
    logarithmic: bool) -> FloatTensor<B> {
    let shape = working.shape();
    assert!(dim < shape.num_dims(), "softmax backward axis out of bounds");
    assert!(shape[dim] > 0, "softmax backward axis must be nonempty");
    assert_eq!(grad.shape(), shape, "softmax gradient shape differs");
    assert!(matches!(working.dtype(), DType::F32 | DType::F64), "softmax saved output must use working storage");
    let compute = if working.dtype() == DType::F64 || grad.dtype() == DType::F64 {
        FloatDType::F64
    } else { FloatDType::F32 };
    let grad = B::float_cast(grad, compute);
    if shape.num_elements() == 0 { return grad; }
    let working = B::float_cast(working, compute);
    if logarithmic {
        let sum = B::float_sum_dim(grad.clone(), dim);
        B::float_sub(grad, B::float_mul(B::float_exp(working), sum))
    } else {
        let sum = B::float_sum_dim(B::float_mul(grad.clone(), working.clone()), dim);
        B::float_mul(working, B::float_sub(grad, sum))
    }
}
