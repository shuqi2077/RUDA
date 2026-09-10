//! Activation function operations for the Flex backend.
//!
//! Each activation is implemented as a single-pass unary operation,
//! replacing the default multi-op compositions from Ruda's trait defaults.

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::element::Scalar;
use ruda_core::tensor::{DType, TensorMetadata};
use ruda_core::bytes::Bytes;
use half::{bf16, f16};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;
use num_traits::ToPrimitive;

use ruprim_host::binary::binary_op;
use ruprim_host::unary::unary_op;
use ruda_core::tensor::host::{HostTensor, Layout};


    pub fn relu(tensor: HostTensor) -> HostTensor {
        unary_op(tensor, |x: f32| x.max(0.0), |x: f64| x.max(0.0))
    }

    pub fn relu_backward(output: HostTensor, grad: HostTensor) -> HostTensor {
        // grad * (output > 0): zero the gradient where output was zero
        binary_op(
            output,
            grad,
            |out: f32, g| if out > 0.0 { g } else { 0.0 },
            |out: f64, g| if out > 0.0 { g } else { 0.0 },
            None,
        )
    }

    pub fn leaky_relu(tensor: HostTensor, negative_slope: Scalar) -> HostTensor {
        let ns32 = negative_slope.to_f32().unwrap();
        let ns64 = negative_slope.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| if x >= 0.0 { x } else { ns32 * x },
            move |x: f64| if x >= 0.0 { x } else { ns64 * x },
        )
    }

    pub fn prelu(tensor: HostTensor, alpha: HostTensor) -> HostTensor {
        // x if x >= 0, alpha * x otherwise
        binary_op(
            tensor,
            alpha,
            |x: f32, a| if x >= 0.0 { x } else { a * x },
            |x: f64, a| if x >= 0.0 { x } else { a * x },
            None,
        )
    }

    pub fn gelu(tensor: HostTensor) -> HostTensor {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        use ruprim_host::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        unary_op(
            tensor,
            move |x: f32| 0.5 * x * (1.0 + erf_f32(x / sqrt2_f32)),
            move |x: f64| 0.5 * x * (1.0 + erf_f64(x / sqrt2_f64)),
        )
    }

    pub fn gelu_backward(x: HostTensor, grad: HostTensor) -> HostTensor {
        // d/dx[gelu(x)] = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
        use ruprim_host::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        let inv_sqrt_2pi_f32: f32 = 1.0 / (2.0 * core::f32::consts::PI).sqrt();
        let inv_sqrt_2pi_f64: f64 = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
        binary_op(
            x,
            grad,
            move |x: f32, g| {
                let cdf = 0.5 * (1.0 + erf_f32(x / sqrt2_f32));
                let pdf = inv_sqrt_2pi_f32 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
            move |x: f64, g| {
                let cdf = 0.5 * (1.0 + erf_f64(x / sqrt2_f64));
                let pdf = inv_sqrt_2pi_f64 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
            None,
        )
    }

    pub fn sigmoid(tensor: HostTensor) -> HostTensor {
        unary_op(tensor, sigmoid_f32, sigmoid_f64)
    }

    pub fn sigmoid_backward(output: HostTensor, grad: HostTensor) -> HostTensor {
        // grad * output * (1 - output)
        binary_op(
            output,
            grad,
            |s: f32, g| g * s * (1.0 - s),
            |s: f64, g| g * s * (1.0 - s),
            None,
        )
    }

    pub fn hard_sigmoid(tensor: HostTensor, alpha: Scalar, beta: Scalar) -> HostTensor {
        let alpha32 = alpha.to_f32().unwrap();
        let beta32 = beta.to_f32().unwrap();
        let alpha64 = alpha.to_f64().unwrap();
        let beta64 = beta.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| (alpha32 * x + beta32).clamp(0.0, 1.0),
            move |x: f64| (alpha64 * x + beta64).clamp(0.0, 1.0),
        )
    }

    pub fn log_sigmoid(tensor: HostTensor) -> HostTensor {
        // Numerically stable: -softplus(-x) = -log(1 + exp(-x))
        // For x >= 0: -log(1 + exp(-x))  (standard form, exp(-x) is small)
        // For x < 0: x - log(1 + exp(x))  (avoids exp of large positive)
        unary_op(
            tensor,
            |x: f32| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
            |x: f64| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
        )
    }

    pub fn log_sigmoid_backward(x: HostTensor, grad: HostTensor) -> HostTensor {
        // d/dx[log_sigmoid(x)] = sigmoid(-x) * (-1) * (-1) = 1 - sigmoid(x) = sigmoid(-x)
        // So: grad * sigmoid(-x)
        binary_op(
            x,
            grad,
            |x: f32, g| g * sigmoid_f32(-x),
            |x: f64, g| g * sigmoid_f64(-x),
            None,
        )
    }




#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

mod fused_softmax;
pub use fused_softmax::*;

mod fused_layer_norm;
pub use fused_layer_norm::*;

// Tests kept here exercise flex-specific behavior: SIMD boundaries, rayon
// chunk boundaries, non-contiguous input handling, the flex-internal
// layer_norm op (no public API yet), and dtype-specific fused softmax
// paths (f16/bf16/f64). Plain activation/softmax smoke tests have been
// migrated to ruda-backend-tests so they cover every backend. When adding
// new tests, keep them here only if they probe flex internals; otherwise
// add them to crates/ruda-backend-tests/tests/tensor/float/activation/.
#[cfg(test)]
mod tests;
