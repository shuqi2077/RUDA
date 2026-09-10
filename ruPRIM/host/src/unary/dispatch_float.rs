use ruda_core::tensor::{host::HostTensor, element::Scalar};
use num_traits::ToPrimitive;
use crate::unary;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

pub fn float_neg(tensor: HostTensor) -> HostTensor {
    unary::unary_op(tensor, |x: f32| -x, |x: f64| -x)
}

pub fn float_clamp(tensor: HostTensor, min: Scalar, max: Scalar) -> HostTensor {
    let min32 = min.to_f32().unwrap();
    let max32 = max.to_f32().unwrap();
    let min64 = min.to_f64().unwrap();
    let max64 = max.to_f64().unwrap();
    unary::unary_op(
        tensor,
        move |x: f32| x.clamp(min32, max32),
        move |x: f64| x.clamp(min64, max64),
    )
}

pub fn float_clamp_min(tensor: HostTensor, min: Scalar) -> HostTensor {
    let min32 = min.to_f32().unwrap();
    let min64 = min.to_f64().unwrap();
    unary::unary_op(
        tensor,
        move |x: f32| x.max(min32),
        move |x: f64| x.max(min64),
    )
}

pub fn float_clamp_max(tensor: HostTensor, max: Scalar) -> HostTensor {
    let max32 = max.to_f32().unwrap();
    let max64 = max.to_f64().unwrap();
    unary::unary_op(
        tensor,
        move |x: f32| x.min(max32),
        move |x: f64| x.min(max64),
    )
}

pub fn float_sign(tensor: HostTensor) -> HostTensor {
    unary::unary_op(
        tensor,
        |x: f32| {
            if x.is_nan() {
                x
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
        |x: f64| {
            if x.is_nan() {
                x
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
    )
}

pub fn float_is_nan(tensor: HostTensor, out_dtype: ruda_core::tensor::BoolDType) -> HostTensor {
    unary::float_predicate(tensor, out_dtype, |x: f32| x.is_nan(), |x: f64| x.is_nan())
}

pub fn float_is_inf(tensor: HostTensor, out_dtype: ruda_core::tensor::BoolDType) -> HostTensor {
    unary::float_predicate(
        tensor,
        out_dtype,
        |x: f32| x.is_infinite(),
        |x: f64| x.is_infinite(),
    )
}

