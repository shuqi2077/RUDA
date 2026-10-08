use crate::{Backend, DType, FloatDType, TensorMetadata, tensor::FloatTensor};

/// Same-device ELU/CELU using the supplied alpha and original nonpositive exponential branch.
pub fn exponential_relu_native<B: Backend>(input: FloatTensor<B>, alpha: f64, continuous: bool) -> FloatTensor<B> {
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let input = B::float_cast(input, compute);
    let bool_dtype = crate::get_device_settings::<B>(&B::float_device(&input)).bool_dtype;
    let nonpositive = B::float_lower_equal_elem(input.clone(), 0f32.into(), bool_dtype);
    let exponent = if continuous { B::float_div_scalar(input.clone(), alpha.into()) } else { input.clone() };
    let value = B::float_mul_scalar(B::float_sub_scalar(B::float_exp(exponent), 1f32.into()), alpha.into());
    B::float_cast(B::float_mask_where(input, nonpositive, value), storage)
}

/// Independent original-primal VJP, retaining actual multiplication/division by alpha in CELU.
pub fn exponential_relu_native_backward<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>, alpha: f64,
    continuous: bool) -> FloatTensor<B> {
    assert_eq!(input.shape(), grad.shape(), "ELU/CELU gradient shape differs");
    if input.shape().num_elements() == 0 { return input; }
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 || grad.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let input = B::float_cast(input, compute);
    let grad = B::float_cast(grad, compute);
    let bool_dtype = crate::get_device_settings::<B>(&B::float_device(&input)).bool_dtype;
    let nonpositive = B::float_lower_equal_elem(input.clone(), 0f32.into(), bool_dtype);
    let exponent = if continuous { B::float_div_scalar(input, alpha.into()) } else { input };
    let value = B::float_mul(B::float_mul_scalar(grad.clone(), alpha.into()), B::float_exp(exponent));
    let value = if continuous { B::float_div_scalar(value, alpha.into()) } else { value };
    B::float_cast(B::float_mask_where(grad, nonpositive, value), storage)
}

/// Same-device working-storage LeakyReLU with the original supplied scalar slope.
pub fn leaky_relu_native<B: Backend>(input: FloatTensor<B>, negative_slope: f64) -> FloatTensor<B> {
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    B::float_cast(B::leaky_relu(B::float_cast(input, compute), negative_slope.into()), storage)
}

/// Original input's LeakyReLU VJP; half storage uses FP32 and an F64 operand retains FP64 working arithmetic.
pub fn leaky_relu_native_backward<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>, negative_slope: f64) -> FloatTensor<B> {
    assert_eq!(input.shape(), grad.shape(), "LeakyReLU gradient shape differs");
    if input.shape().num_elements() == 0 { return input; }
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 || grad.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let input = B::float_cast(input, compute);
    let grad = B::float_cast(grad, compute);
    let bool_dtype = crate::get_device_settings::<B>(&B::float_device(&input)).bool_dtype;
    let negative = B::float_lower_elem(input, 0f32.into(), bool_dtype);
    let scaled = B::float_mul_scalar(grad.clone(), negative_slope.into());
    B::float_cast(B::float_mask_where(grad, negative, scaled), storage)
}

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

const SQRT_2_OVER_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI * core::f64::consts::FRAC_1_SQRT_2;

fn gelu_tanh<B: Backend>(input: FloatTensor<B>) -> FloatTensor<B> {
    let cubic = B::float_mul_scalar(B::float_powf_scalar(input.clone(), 3f32.into()), 0.044715f64.into());
    B::float_tanh(B::float_mul_scalar(B::float_add(input, cubic), SQRT_2_OVER_PI.into()))
}

/// Explicit FP32/FP64 GELU, retaining the original selected erf or tanh mathematical mode.
pub fn gelu_native<B: Backend>(input: FloatTensor<B>, approximate: bool) -> FloatTensor<B> {
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let input = B::float_cast(input, compute);
    let output = if approximate {
        let value = gelu_tanh::<B>(input.clone());
        B::float_mul_scalar(B::float_mul(input, B::float_add_scalar(value, 1f32.into())), 0.5f32.into())
    } else { B::gelu(input) };
    B::float_cast(output, storage)
}

/// Original selected GELU mode's independent first-order VJP in working storage.
pub fn gelu_native_backward<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>, approximate: bool) -> FloatTensor<B> {
    assert_eq!(input.shape(), grad.shape(), "GELU gradient shape differs");
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 || grad.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    if input.shape().num_elements() == 0 { return input; }
    let input = B::float_cast(input, compute);
    let grad = B::float_cast(grad, compute);
    let output = if approximate {
        let value = gelu_tanh::<B>(input.clone());
        let inner_grad = B::float_add_scalar(B::float_mul_scalar(B::float_mul(input.clone(), input.clone()), (3.0 * 0.044715).into()), 1f32.into());
        let inner_grad = B::float_mul_scalar(inner_grad, SQRT_2_OVER_PI.into());
        let tanh_grad = B::float_add_scalar(B::float_neg(B::float_mul(value.clone(), value.clone())), 1f32.into());
        let correction = B::float_mul_scalar(B::float_mul(B::float_mul(input, tanh_grad), inner_grad), 0.5f32.into());
        let direct = B::float_mul_scalar(B::float_add_scalar(value, 1f32.into()), 0.5f32.into());
        B::float_mul(B::float_add(direct, correction), grad)
    } else { crate::ops::gelu_backward_exact::<B>(input, grad) };
    B::float_cast(output, storage)
}
