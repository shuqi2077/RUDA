use super::super::fallback_ops::AttentionFallbackOps;
use core::marker::PhantomData;
use ruda_kernel::dsl::{Runtime, prelude::InputScalar};
use ruda_kernel::tensor::{RudaTensor, capability::supports_dtype, transfer::from_data};
use ruda_core::tensor::{DType, FloatDType, IntDType, BoolDType, Shape, TensorMetadata};
use ruda_core::tensor::data::TensorData;
use ruda_core::tensor::element::{Scalar, ElementConversion};
use ruda_core::tensor::device_settings::{DeviceSettings, DeviceSettingsRegistry, select_bool_dtype};
use rublas::tensor_matmul::{matmul, MatmulStrategy};
use ruprim::elementwise::arithmetic as numeric;
use ruprim::elementwise::unary::float::unary_basic;
use ruprim::elementwise::unary::float::unary_basic::BasicFloatUnaryKind;
use ruprim::elementwise::mask::mask_fill_auto;
use ruprim::elementwise::comparison::{lower, lower_elem};
use ruprim::reduce::tensor as reduce;
use ruprim::reduce::components::instructions::ReduceOperationConfig;

pub(super) struct RudaAttentionOps<R>(PhantomData<R>);

impl<R: Runtime> AttentionFallbackOps for RudaAttentionOps<R> {
    type FloatTensor = RudaTensor<R>;
    type BoolTensor = RudaTensor<R>;
    type IntTensor = RudaTensor<R>;
    type Device = R::Device;
    fn float_device(tensor: &Self::FloatTensor) -> Self::Device {
        tensor.device.clone()
    }

    fn float_matmul(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        let dtype = lhs.dtype;
        matmul(lhs, rhs, None, MatmulStrategy::default(), dtype).unwrap()
    }

    fn float_mul_scalar(lhs: Self::FloatTensor, rhs: Scalar) -> Self::FloatTensor {
        let dtype = lhs.dtype;
        numeric::mul_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_div_scalar(lhs: Self::FloatTensor, rhs: Scalar) -> Self::FloatTensor {
        let dtype = lhs.dtype;
        numeric::div_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_tanh(tensor: Self::FloatTensor) -> Self::FloatTensor {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Tanh)
    }

    fn float_mask_fill(
        tensor: Self::FloatTensor,
        mask: Self::BoolTensor,
        value: Scalar,
    ) -> Self::FloatTensor {
        let dtype = tensor.dtype;
        let bool_dtype = mask.dtype;
        mask_fill_auto(tensor, mask, InputScalar::new(value, dtype), bool_dtype)
    }

    fn float_add(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        numeric::add(lhs, rhs)
    }

    fn float_max_dim(tensor: Self::FloatTensor, dim: usize) -> Self::FloatTensor {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::Max,
        )
        .unwrap()
    }

    fn float_sub(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        numeric::sub(lhs, rhs)
    }

    fn float_exp(tensor: Self::FloatTensor) -> Self::FloatTensor {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Exp)
    }

    fn float_sum_dim(tensor: Self::FloatTensor, dim: usize) -> Self::FloatTensor {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::Sum,
        )
        .unwrap()
    }

    fn float_div(lhs: Self::FloatTensor, rhs: Self::FloatTensor) -> Self::FloatTensor {
        numeric::div(lhs, rhs)
    }

    fn int_reshape(tensor: Self::IntTensor, shape: Shape) -> Self::IntTensor {
        ruda_kernel::tensor::reshape::reshape(tensor, shape)
    }

    fn int_add_scalar(lhs: Self::IntTensor, rhs: Scalar) -> Self::IntTensor {
        let dtype = lhs.dtype;
        numeric::add_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn int_lower(
        lhs: Self::IntTensor,
        rhs: Self::IntTensor,
        out_dtype: BoolDType,
    ) -> Self::BoolTensor {
        lower(lhs, rhs, out_dtype.into())
    }

    fn device_settings(device: &Self::Device) -> DeviceSettings {
        DeviceSettingsRegistry::get_or_insert(device, || DeviceSettings::new(
            FloatDType::F32, IntDType::I32,
            select_bool_dtype(DType::U8.into(), |dtype| supports_dtype::<R>(device, dtype)),
        ))
    }

    fn float_transpose(tensor: Self::FloatTensor) -> Self::FloatTensor {
        let ndims = tensor.shape().num_dims();
        ruda_kernel::tensor::permutation::swap_dims(tensor, ndims - 2, ndims - 1)
    }

    fn float_clamp_min(tensor: Self::FloatTensor, min: Scalar) -> Self::FloatTensor {
        let dtype = Self::device_settings(&Self::float_device(&tensor)).bool_dtype;
        let mask = lower_elem(tensor.clone(), InputScalar::new(min, tensor.dtype), dtype.into());
        Self::float_mask_fill(tensor, mask, min)
    }

    fn int_arange(range: core::ops::Range<i64>, device: &Self::Device, dtype: IntDType) -> Self::IntTensor {
        let value = range.step_by(1).map(|i| i.elem()).collect::<Vec<i32>>();
        let shape = Shape::new([value.len()]);
        let data = TensorData::new(value, shape).convert_dtype(dtype.into());
        match data.dtype {
            DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8 => from_data(data, device),
            _ => unimplemented!("Unsupported dtype for `int_from_data`"),
        }
    }

    fn bool_reshape(tensor: Self::BoolTensor, shape: Shape) -> Self::BoolTensor {
        ruda_kernel::tensor::reshape::reshape(tensor, shape)
    }

    fn bool_expand(tensor: Self::BoolTensor, shape: Shape) -> Self::BoolTensor {
        ruda_kernel::tensor::view::expand(tensor, shape)
    }
}
