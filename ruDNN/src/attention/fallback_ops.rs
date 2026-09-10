use ruda_core::tensor::{Shape, IntDType, BoolDType, TensorMetadata};
use ruda_core::tensor::element::Scalar;
use ruda_core::tensor::device_settings::DeviceSettings;

pub trait AttentionFallbackOps {
    type FloatTensor: TensorMetadata;
    type BoolTensor;
    type IntTensor;
    type Device;

    fn device_settings(device: &Self::Device) -> DeviceSettings;
    fn float_device(tensor: &Self::FloatTensor) -> Self::Device;
    fn float_transpose(tensor: Self::FloatTensor) -> Self::FloatTensor;
    fn float_matmul(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor;
    fn float_mul_scalar(lhs: Self::FloatTensor, rhs: Scalar) -> Self::FloatTensor;
    fn float_div_scalar(lhs: Self::FloatTensor, rhs: Scalar) -> Self::FloatTensor;
    fn float_tanh(tensor: Self::FloatTensor) -> Self::FloatTensor;
    fn float_mask_fill(tensor: Self::FloatTensor, mask: Self::BoolTensor, value: Scalar) -> Self::FloatTensor;
    fn float_add(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor;
    fn float_max_dim(tensor: Self::FloatTensor, dim: usize) -> Self::FloatTensor;
    fn float_clamp_min(tensor: Self::FloatTensor, min: Scalar) -> Self::FloatTensor;
    fn float_sub(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor;
    fn float_exp(tensor: Self::FloatTensor) -> Self::FloatTensor;
    fn float_sum_dim(tensor: Self::FloatTensor, dim: usize) -> Self::FloatTensor;
    fn float_div(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor;
    fn int_reshape(tensor: Self::IntTensor, shape: Shape) -> Self::IntTensor;
    fn int_arange(range: core::ops::Range<i64>, device: &Self::Device, dtype: IntDType) -> Self::IntTensor;
    fn int_add_scalar(lhs: Self::IntTensor, rhs: Scalar) -> Self::IntTensor;
    fn int_lower(lhs: Self::IntTensor, rhs: Self::IntTensor, out_dtype: BoolDType) -> Self::BoolTensor;
    fn bool_reshape(tensor: Self::BoolTensor, shape: Shape) -> Self::BoolTensor;
    fn bool_expand(tensor: Self::BoolTensor, shape: Shape) -> Self::BoolTensor;
}
