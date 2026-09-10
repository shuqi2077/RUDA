use crate::dsl::Runtime;
use super::RudaTensor;
use super::allocation::new_qtensor_optimized;
use super::readback::into_data;
use ruda_core::tensor::{DType, Shape, Metadata, TensorMetadata, QuantScheme, QuantLevel, QuantMode, QuantValue};
use ruda_core::tensor::{data::TensorData, execution::ExecutionError};

pub fn from_data<R: Runtime>(data: TensorData, device: &R::Device) -> RudaTensor<R> {
    let client = R::client(device);
    let alloc = client.create_tensor(data.bytes, data.shape.clone(), data.dtype.size());
    let shape: Shape = (&data.shape).into();
    RudaTensor::new(
        client,
        alloc.memory,
        Metadata::new(shape, alloc.strides),
        device.clone(),
        data.dtype,
    )
}

#[cfg_attr(
    feature = "device-tensor-tracing",
    tracing::instrument(level = "trace", skip(tensor, device))
)]
pub fn to_device<R: Runtime>(
    tensor: RudaTensor<R>,
    device: &R::Device,
) -> RudaTensor<R>
where
    R::Device: PartialEq,
{
    if &tensor.device == device {
        return tensor;
    }

    let mut tensor = super::contiguous::into_contiguous_aligned(tensor);
    let client = R::client(device);
    tensor.to_client(client, device.clone())
}

pub fn q_from_data<R: Runtime>(data: TensorData, device: &R::Device) -> RudaTensor<R> {
    match data.dtype {
        DType::QFloat(scheme) => match scheme {
            QuantScheme {
                level: QuantLevel::Tensor | QuantLevel::Block(_),
                mode: QuantMode::Symmetric,
                value:
                    QuantValue::Q8F
                    | QuantValue::Q8S
                    | QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::Q2F
                    | QuantValue::Q2S
                    | QuantValue::E4M3
                    | QuantValue::E5M2
                    | QuantValue::E2M1,
                ..
            } => {
                // TensorData quantized representation is the same, with multiple quantized values
                // packed into u32 and quantization parameters appended to the bytes
                new_qtensor_optimized(data.bytes, data.shape.clone(), scheme, device)
            }
        },
        _ => panic!(
            "Invalid dtype (expected DType::QFloat, got {:?})",
            data.dtype
        ),
    }
}

pub async fn q_into_data<R: Runtime>(tensor: RudaTensor<R>) -> Result<TensorData, ExecutionError> {
    if tensor.qparams.is_none() {
        return into_data(tensor).await;
    }

    let (shape, dtype) = (tensor.shape(), tensor.dtype);
    let (values, params) = tensor.quantized_handles().unwrap();

    let mut data_values = into_data(values).await?;
    let data_params = into_data(params).await?;

    data_values.bytes.extend_from_byte_slice(&data_params.bytes);

    Ok(TensorData {
        bytes: data_values.bytes,
        shape,
        dtype,
    })
}
