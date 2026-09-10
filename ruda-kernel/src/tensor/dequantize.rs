use super::RudaTensor;
use crate::dsl::Runtime;
use super::allocation::empty_device_dtype;
use ruda_core::tensor::{DType, TensorMetadata};

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R>(tensor: RudaTensor<R>, dtype: DType) -> RudaTensor<R>
where
    R: Runtime,
{
    let scheme = match tensor.dtype {
        DType::QFloat(scheme) => scheme,
        _ => return tensor,
    };

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        dtype,
    );
    let (values, params) = tensor.quantized_handles().unwrap();

    crate::quantization::dequantize::launch_ref(
        &output.client,
        values.binding(),
        output.clone().binding(),
        params.binding(),
        &scheme,
        dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
