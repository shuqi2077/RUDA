use super::float::{FloatUnaryOp, FloatUnaryOpFamily, launch_unary_float};
use ruda_core::tensor::{DType, TensorMetadata};
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};

/// SiLU for F32/F16/BF16 inputs, using FP32 arithmetic with one final cast.
/// Retains the tensor's dtype and supports non-contiguous, non-overlapping views.
pub fn launch<R: Runtime>(tensor: RudaTensor<R>) -> RudaTensor<R> {
    assert!(matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16));
    if tensor.meta.num_elements() == 0 { return tensor; }
    launch_unary_float::<R, Silu, _>(tensor, |_| {
        SiluOptionsLaunch::new(include_str!("silu.rs").to_owned())
    })
}

#[derive(RudaLaunch, RudaType)]
struct SiluOptions {
    #[ruda(comptime)]
    source: String,
}

struct Silu;

/// Shared SiLU arithmetic for standalone and fused F32/F16/BF16 kernels.
#[ruda]
pub fn apply<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    let input = Vector::<f32, N>::cast_from(input);
    let denominator = Vector::new(1f32) + Vector::exp(-input);
    Vector::cast_from(input / denominator)
}

#[ruda]
impl<F: Float, N: Size> FloatUnaryOp<F, N> for Silu {
    type Options = SiluOptions;

    fn execute(input: Vector<F, N>, _options: &Self::Options) -> Vector<F, N> {
        apply(input)
    }
}

impl FloatUnaryOpFamily for Silu {
    type Options = SiluOptions;
    type Unary<F: Float, N: Size> = Self;
}
