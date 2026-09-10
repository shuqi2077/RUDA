use super::RudaTensor;
use crate::dsl::{Runtime, server::CopyDescriptor};
use ruda_core::tensor::{data::TensorData, execution::ExecutionError};

pub async fn into_data<R: Runtime>(
    tensor: RudaTensor<R>,
) -> Result<TensorData, ExecutionError> {
    let tensor = super::contiguous::into_contiguous_aligned(tensor);

    let elem_size = tensor.elem_size();
    let shape = tensor.meta.shape().clone();
    let strides = tensor.meta.strides().clone();
    let binding = CopyDescriptor::new(tensor.handle.binding(), shape, strides, elem_size);
    let bytes = tensor
        .client
        .read_one_tensor_async(binding)
        .await
        .map_err(|err| ExecutionError::WithContext {
            reason: format!("{err}"),
        })?;

    Ok(TensorData::from_bytes(
        bytes,
        tensor.meta.shape.clone(),
        tensor.dtype,
    ))
}

/// Read data from a `RudaTensor` synchronously
#[allow(unused, reason = "useful for debugging kernels")]
pub fn into_data_sync<R: Runtime>(tensor: RudaTensor<R>) -> TensorData {
    ruda_core::future::block_on(into_data(tensor)).unwrap()
}

