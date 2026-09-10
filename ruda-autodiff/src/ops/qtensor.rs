use ruda_tensor::{
    Backend, ExecutionError, TensorData,
    ops::QTensorOps,
    tensor::{
        Device, FloatTensor, IntTensor, QuantizedTensor,
        quantization::QuantizationParametersPrimitive,
    },
};
use ruda_core::tensor::{FloatDType, IntDType, QuantScheme, Shape};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor};

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        B::q_from_data(data, device)
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        assert!(
            !tensor.is_tracked() && !qparams.scales.is_tracked(),
            "quantization of tracked tensors requires a quantization-aware gradient rule, which is not implemented"
        );
        B::quantize(tensor.primitive, scheme, QuantizationParametersPrimitive {
            scales: qparams.scales.primitive,
        })
    }

    fn quantize_dynamic(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
    ) -> QuantizedTensor<Self> {
        assert!(
            !tensor.is_tracked(),
            "dynamic quantization of tracked tensors requires a quantization-aware gradient rule, which is not implemented"
        );
        B::quantize_dynamic(tensor.primitive, scheme)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        AutodiffTensor::new(B::dequantize(tensor, dtype))
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        B::q_device(tensor)
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        device: &Device<Self>,
    ) -> QuantizedTensor<Self> {
        B::q_to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::q_into_data(tensor).await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        B::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        B::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        B::q_flip(tensor, axes)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        B::q_gather(dim, tensor, indices)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        B::q_select(tensor, dim, indices)
    }

    fn q_slice(
        tensor: QuantizedTensor<Self>,
        slices: &[ruda_core::tensor::Slice],
    ) -> QuantizedTensor<Self> {
        B::q_slice(tensor, slices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmax(tensor, dim, out_dtype)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmin(tensor, dim, out_dtype)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_expand(tensor, shape)
    }
}
