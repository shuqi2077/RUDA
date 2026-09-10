use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, allocation::empty_device_dtype}};
use ruda_core::tensor::{DType, Shape, TensorMetadata};

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: Runtime>(
    shape: Shape,
    device: &R::Device,
    lower_bound: f32,
    upper_bound: f32,
    dtype: DType,
) -> RudaTensor<R> {
    let client = R::client(device);
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);

    crate::random_uniform(
        &client,
        lower_bound,
        upper_bound,
        output.clone().binding(),
        dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}

/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_uniform<R: Runtime>(
    tensor: &RudaTensor<R>,
    lower_bound: f32,
    upper_bound: f32,
    dtype: DType,
) -> RudaTensor<R> {
    random_uniform(
        tensor.shape(),
        &tensor.device,
        lower_bound,
        upper_bound,
        dtype,
    )
}
