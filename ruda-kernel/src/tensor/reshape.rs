use crate::dsl::Runtime;
use super::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{Shape, Metadata, ReshapeAction, reshape_action};

/// Reshape a jit tensor to a new shape
pub fn reshape<R: Runtime>(mut tensor: RudaTensor<R>, shape: Shape) -> RudaTensor<R> {
    let analysis = reshape_action(tensor.meta.shape(), tensor.meta.strides(), &shape);

    match analysis {
        ReshapeAction::UpdateStrides { strides } => {
            *tensor.meta = Metadata::new(shape, strides);
            return tensor;
        }
        ReshapeAction::NoChange => return tensor,
        ReshapeAction::Recompute => (),
    }

    let out = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape,
        tensor.dtype,
    );

    crate::library::tensor::copy_into(
        &out.client,
        tensor.binding(),
        out.clone().binding(),
        out.dtype.into(),
    );

    out
}

/// Reshape a quantized tensor without recalibrating its values.
pub fn q_reshape<R: Runtime>(tensor: RudaTensor<R>, shape: Shape) -> RudaTensor<R> {
    try_q_reshape(tensor, shape).unwrap_or_else(|_| {
        panic!("No scale-preserving reshape path for this block layout")
    })
}

/// Returns the original tensor and target shape when no scale-preserving path is selected.
pub fn try_q_reshape<R: Runtime>(
    tensor: RudaTensor<R>,
    shape: Shape,
) -> Result<RudaTensor<R>, (RudaTensor<R>, Shape)> {
    super::reshape_quantized::try_reshape(tensor, shape)
}
