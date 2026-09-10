use ruda_core::tensor::{QTensorPrimitive, TensorMetadata};
use crate::dsl::{Runtime, server::MemoryLayoutStrategy};
use super::{RudaTensor, allocation::empty_qtensor};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: Runtime>(tensor: RudaTensor<R>) -> RudaTensor<R> {
    if tensor.qparams.is_some() {
        let (values, scales) = tensor.quantized_handles().unwrap();
        if values.is_contiguous() && scales.is_contiguous() {
            return tensor;
        }
        return into_contiguous_quantized(tensor, MemoryLayoutStrategy::Contiguous);
    }

    if tensor.is_contiguous() {
        return tensor;
    }

    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);

    let output = crate::library::tensor::into_contiguous(&client, tensor.binding(), dtype.into());

    RudaTensor::new(
        client.clone(),
        output.handle,
        *output.metadata,
        device,
        dtype,
    )
}

/// Make a jit tensor contiguous with an aligned last stride. Tensor is considered already contiguous
/// if runtime can read it as is. This is equivalent in practice.
#[cfg_attr(
    feature = "device-tensor-tracing",
    tracing::instrument(level = "trace", skip(tensor))
)]
pub fn into_contiguous_aligned<R: Runtime>(tensor: RudaTensor<R>) -> RudaTensor<R> {
    if tensor.qparams.is_some() {
        let (values, scales) = tensor.quantized_handles().unwrap();
        if R::can_read_tensor(values.meta.shape(), values.meta.strides())
            && R::can_read_tensor(scales.meta.shape(), scales.meta.strides())
        {
            return tensor;
        }
        return into_contiguous_quantized(tensor, MemoryLayoutStrategy::Optimized);
    }

    if R::can_read_tensor(tensor.meta.shape(), tensor.meta.strides()) {
        return tensor;
    }

    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);

    let output =
        crate::library::tensor::into_contiguous_pitched(&client, tensor.binding(), dtype.into());

    RudaTensor::new(
        client.clone(),
        output.handle,
        *output.metadata,
        device,
        dtype,
    )
}

#[cfg_attr(
    feature = "device-tensor-tracing",
    tracing::instrument(level = "trace", skip(tensor))
)]
fn into_contiguous_quantized<R: Runtime>(
    tensor: RudaTensor<R>,
    strategy: MemoryLayoutStrategy,
) -> RudaTensor<R> {
    let output = empty_qtensor(tensor.shape(), *tensor.scheme(), &tensor.device, strategy);
    let (values, scales) = tensor.quantized_handles().unwrap();
    let (out_values, out_scales) = output.quantized_handles().unwrap();

    let (client, dtype_scales, dtype_value) = (scales.client.clone(), scales.dtype, values.dtype);

    crate::library::tensor::copy_into(
        &client,
        values.binding(),
        out_values.binding(),
        dtype_value.into(),
    );

    crate::library::tensor::copy_into(
        &client,
        scales.binding(),
        out_scales.binding(),
        dtype_scales.into(),
    );

    output
}
