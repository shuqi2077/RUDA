use crate::dsl::Runtime;
use super::{allocation::empty_qtensor_optimized, RudaTensor};
use ruda_core::tensor::{TensorMetadata, QuantScheme};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R>(
    tensor: RudaTensor<R>,
    scheme: &QuantScheme,
    scale: RudaTensor<R>,
) -> RudaTensor<R>
where
    R: Runtime,
{
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let dtype = tensor.dtype;
    let scale_dtype = scale.dtype;

    crate::quantization::quantize::launch_ref_with_scale_dtype(
        &output.client,
        tensor.binding(),
        out_values.binding(),
        scale.binding(),
        out_params.binding(),
        scheme,
        dtype.into(),
        scale_dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
