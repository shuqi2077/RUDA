use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, allocation::empty_device_dtype}};
use ruda_core::tensor::{DType, Shape};

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: Runtime>(
    shape: Shape,
    device: &R::Device,
    probability: f32,
    dtype: DType,
) -> RudaTensor<R> {
    let client = R::client(device);
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);

    crate::random_bernoulli(&client, probability, output.clone().binding(), dtype.into())
        .expect("Kernel to never fail");

    output
}
