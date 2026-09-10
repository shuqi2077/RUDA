use ruda_tensor::{
    DType, ExecutionError, QTensorPrimitive, Shape, Slice, TensorData, TensorMetadata,
    TensorPrimitive,
    ops::{FloatTensorOps, QTensorOps},
    quantization::{
        QuantLevel, QuantPropagation, QuantScheme, QuantValue,
        QuantizationParametersPrimitive,
    },
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use ruda_core::tensor::FloatDType;
use ruda_core::{ir::features::Plane as PlaneFeature, quant::scheme::QuantStore};

use crate::{DeviceBackend, DeviceRuntime, FloatElement, IntElement, element::BoolElement, RudaTensor};
use rublas::tensor_matmul::MatmulStrategy;

use super::{permute, swap_dims};

fn maybe_dequantize_native_fp4<R: DeviceRuntime>(
    tensor: RudaTensor<R>,
    dtype: DType,
    scalar_matmul: bool,
) -> RudaTensor<R> {
    let DType::QFloat(scheme) = tensor.dtype else {
        return tensor;
    };
    let QuantStore::PackedNative(packed_dim) = scheme.store else {
        return tensor;
    };
    if scheme.value != QuantValue::E2M1 {
        return tensor;
    }

    let packed_axis = tensor.rank() - packed_dim - 1;
    if scalar_matmul || !tensor.shape()[packed_axis].is_multiple_of(scheme.num_quants()) {
        ruda_kernel::tensor::dequantize::dequantize(tensor, dtype)
    } else {
        tensor
    }
}

pub use ruda_kernel::tensor::allocation::{empty_qtensor, empty_qtensor_optimized};

impl<R, F, I, BT> QTensorOps<Self> for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        ruda_kernel::tensor::transfer::q_from_data(data, device)
    }

    // TODO: quantize_dynamic (we can compute min-max on the fly and scale, especially when not per-tensor)

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        ruda_kernel::tensor::quantize::quantize(tensor, scheme, qparams.scales)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        ruda_kernel::tensor::dequantize::dequantize(tensor, dtype.into())
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        super::to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        let scheme = *tensor.scheme();
        match ruda_kernel::tensor::reshape::try_q_reshape(tensor, shape) {
            Ok(tensor) => tensor,
            Err((tensor, shape)) => {
                let tensor = Self::dequantize(tensor, FloatDType::F32);
                let output = Self::float_reshape(tensor, shape);
                Self::quantize_dynamic(output, &scheme)
            }
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        ruda_kernel::tensor::transfer::q_into_data(tensor).await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let scheme = *tensor.scheme();
        match scheme.level {
            QuantLevel::Tensor => ruprim::indexing::quantized_flip(tensor, axes),
            QuantLevel::Block(_) => {
                let tensor = Self::dequantize(tensor, FloatDType::F32);
                let output = Self::float_flip(tensor, axes);
                Self::quantize_dynamic(output, &scheme)
            }
        }
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        let scheme = *tensor.scheme();
        match scheme.level {
            QuantLevel::Tensor => ruprim::indexing::quantized_gather(dim, tensor, indices),
            QuantLevel::Block(_) => {
                let dtype = ruda_tensor::get_device_settings::<Self>(&tensor.device).float_dtype;
                let tensor = Self::dequantize(tensor, dtype);
                let output = Self::float_gather(dim, tensor, indices);
                Self::quantize_dynamic(output, &scheme)
            }
        }
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        let scheme = *tensor.scheme();
        match scheme.level {
            QuantLevel::Tensor => ruprim::indexing::quantized_select(tensor, dim, indices),
            QuantLevel::Block(_) => {
                let tensor = Self::dequantize(tensor, FloatDType::F32);
                let output = Self::float_select(tensor, dim, indices);
                Self::quantize_dynamic(output, &scheme)
            }
        }
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        let scheme = *tensor.scheme();
        match scheme.level {
            QuantLevel::Tensor => ruprim::indexing::quantized_slice(tensor, slices),
            QuantLevel::Block(_) => {
                let tensor = Self::dequantize(tensor, FloatDType::F32);
                let output = Self::float_slice(tensor, slices);
                Self::quantize_dynamic(output, &scheme)
            }
        }
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        super::expand(tensor, shape)
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let (propagation, scheme) = match (&lhs, &rhs) {
            (TensorPrimitive::QFloat(lhs), _) => (lhs.propagation(), *lhs.scheme()),
            (_, TensorPrimitive::QFloat(rhs)) => (rhs.propagation(), *rhs.scheme()),
            _ => unreachable!(),
        };

        // Inherit precision for mixed inputs, default to `FloatElem` for fully quantized.
        let out_dtype = match (&lhs, &rhs) {
            (TensorPrimitive::Float(lhs), _) => lhs.dtype,
            (_, TensorPrimitive::Float(rhs)) => rhs.dtype,
            _ => F::dtype(),
        };

        let (_lhs_dtype, lhs) = match lhs {
            TensorPrimitive::Float(lhs) => (lhs.dtype, lhs),
            TensorPrimitive::QFloat(lhs) => (out_dtype, lhs),
        };
        let (_rhs_dtype, rhs) = match rhs {
            TensorPrimitive::Float(rhs) => (rhs.dtype, rhs),
            TensorPrimitive::QFloat(rhs) => (out_dtype, rhs),
        };
        let has_plane_ops = lhs
            .client
            .properties()
            .features
            .plane
            .contains(PlaneFeature::Ops);
        let lhs = maybe_dequantize_native_fp4(lhs, out_dtype, !has_plane_ops);
        let rhs = maybe_dequantize_native_fp4(rhs, out_dtype, !has_plane_ops);

        let strategy = if has_plane_ops {
            MatmulStrategy::default()
        } else {
            MatmulStrategy::Naive
        };
        let out = rublas::tensor_matmul::matmul(lhs, rhs, None, strategy, out_dtype).unwrap();

        match propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}
