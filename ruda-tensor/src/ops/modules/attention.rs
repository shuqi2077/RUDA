use crate::{Backend, tensor::{FloatTensor, BoolTensor, IntTensor}};
use crate::ops::AttentionModuleOptions;
use ruda_core::tensor::{Shape, IntDType, BoolDType};
use ruda_core::tensor::element::Scalar;
use ruda_core::tensor::device_settings::DeviceSettings;
use rudnn::attention::fallback_ops::AttentionFallbackOps;

struct FrameworkAttention<B>(core::marker::PhantomData<B>);

impl<B: Backend> AttentionFallbackOps for FrameworkAttention<B> {
    type FloatTensor = FloatTensor<B>;
    type BoolTensor = BoolTensor<B>;
    type IntTensor = IntTensor<B>;
    type Device = B::Device;

    fn device_settings(device: &Self::Device) -> DeviceSettings {
        crate::get_device_settings::<B>(device)
    }
    fn float_device(tensor: &Self::FloatTensor) -> Self::Device {
        B::float_device(tensor)
    }
    fn float_transpose(tensor: Self::FloatTensor) -> Self::FloatTensor {
        B::float_transpose(tensor)
    }
    fn float_matmul(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        B::float_matmul(lhs, rhs)
    }
    fn float_mul_scalar(lhs: Self::FloatTensor, rhs: Scalar) -> Self::FloatTensor {
        B::float_mul_scalar(lhs, rhs)
    }
    fn float_div_scalar(lhs: Self::FloatTensor, rhs: Scalar) -> Self::FloatTensor {
        B::float_div_scalar(lhs, rhs)
    }
    fn float_tanh(tensor: Self::FloatTensor) -> Self::FloatTensor {
        B::float_tanh(tensor)
    }
    fn float_mask_fill(tensor: Self::FloatTensor, mask: Self::BoolTensor, value: Scalar) -> Self::FloatTensor {
        B::float_mask_fill(tensor, mask, value)
    }
    fn float_add(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        B::float_add(lhs, rhs)
    }
    fn float_max_dim(tensor: Self::FloatTensor, dim: usize) -> Self::FloatTensor {
        B::float_max_dim(tensor, dim)
    }
    fn float_clamp_min(tensor: Self::FloatTensor, min: Scalar) -> Self::FloatTensor {
        B::float_clamp_min(tensor, min)
    }
    fn float_sub(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        B::float_sub(lhs, rhs)
    }
    fn float_exp(tensor: Self::FloatTensor) -> Self::FloatTensor {
        B::float_exp(tensor)
    }
    fn float_sum_dim(tensor: Self::FloatTensor, dim: usize) -> Self::FloatTensor {
        B::float_sum_dim(tensor, dim)
    }
    fn float_div(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        B::float_div(lhs, rhs)
    }
    fn int_reshape(tensor: Self::IntTensor, shape: Shape) -> Self::IntTensor {
        B::int_reshape(tensor, shape)
    }
    fn int_arange(range: core::ops::Range<i64>, device: &Self::Device, dtype: IntDType) -> Self::IntTensor {
        B::int_arange(range, device, dtype)
    }
    fn int_add_scalar(lhs: Self::IntTensor, rhs: Scalar) -> Self::IntTensor {
        B::int_add_scalar(lhs, rhs)
    }
    fn int_lower(lhs: Self::IntTensor, rhs: Self::IntTensor, out_dtype: BoolDType) -> Self::BoolTensor {
        B::int_lower(lhs, rhs, out_dtype)
    }
    fn bool_reshape(tensor: Self::BoolTensor, shape: Shape) -> Self::BoolTensor {
        B::bool_reshape(tensor, shape)
    }
    fn bool_expand(tensor: Self::BoolTensor, shape: Shape) -> Self::BoolTensor {
        B::bool_expand(tensor, shape)
    }
}

/// Computes softmax(QKᵗ * scale) · V using separate kernels.
/// Serves as a fallback when FlashAttention is not used.
pub fn attention_fallback<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
    attn_bias: Option<FloatTensor<B>>,
    options: AttentionModuleOptions,
) -> FloatTensor<B> {
    rudnn::attention::fallback::attention_fallback::<FrameworkAttention<B>>(query, key, value, mask, attn_bias, options)
}
