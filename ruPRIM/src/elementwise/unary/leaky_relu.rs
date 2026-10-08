use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::{dsl::{calculate_ruda_count_elemwise, prelude::*},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous}};
use ruda_kernel::dsl as kernel_dsl;

fn storage<R: Runtime>(tensor: &RudaTensor<R>) {
    assert!(tensor.qparams.is_none() && matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16),
        "native LeakyReLU requires unquantized F32/F16/BF16");
    assert!(tensor.meta.num_elements() <= u32::MAX as usize, "native LeakyReLU index range exceeded");
}

/// FP32 LeakyReLU using the supplied scalar slope, without allocating an artificial parameter tensor.
pub fn launch<R: Runtime>(input: RudaTensor<R>, negative_slope: f32) -> RudaTensor<R> {
    storage(&input);
    if input.meta.num_elements() == 0 { return input; }
    let input = into_contiguous(input);
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let dim = RudaDim::new(input.client.properties(), input.meta.num_elements());
    let count = calculate_ruda_count_elemwise(&input.client, input.meta.num_elements(), dim);
    forward::launch(&input.client, count, dim, input.clone().into_array_arg(), output.clone().into_array_arg(),
        negative_slope, include_str!("leaky_relu.rs").to_owned(), input.dtype.into());
    output
}

/// Independent input VJP from the original primal, preserving the original nonnegative zero branch.
pub fn launch_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>, negative_slope: f32) -> RudaTensor<R> {
    storage(&input);
    storage(&grad);
    assert_eq!(input.meta.shape(), grad.meta.shape(), "LeakyReLU gradient shape differs");
    assert_eq!(input.device.to_id(), grad.device.to_id(), "LeakyReLU gradient device differs");
    assert!(input.client.same_execution_queue(&grad.client), "LeakyReLU gradient queue differs");
    if input.meta.num_elements() == 0 { return input; }
    let input = into_contiguous(input);
    let grad = into_contiguous(grad);
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let dim = RudaDim::new(input.client.properties(), input.meta.num_elements());
    let count = calculate_ruda_count_elemwise(&input.client, input.meta.num_elements(), dim);
    backward::launch(&input.client, count, dim, input.clone().into_array_arg(), grad.clone().into_array_arg(), output.clone().into_array_arg(),
        negative_slope, include_str!("leaky_relu.rs").to_owned(), [input.dtype.into(), grad.dtype.into()]);
    output
}

#[ruda(launch)]
fn forward<F: Float>(input: &Array<F>, output: &mut Array<F>, negative_slope: f32,
    #[comptime] _source: String, #[define(F)] _storage: StorageType) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let value = f32::cast_from(input[i]);
    output[i] = F::cast_from(if value < 0f32 { value * negative_slope } else { value });
}

#[ruda(launch)]
fn backward<F: Float, G: Float>(input: &Array<F>, grad: &Array<G>, output: &mut Array<F>, negative_slope: f32,
    #[comptime] _source: String, #[define(F, G)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let value = f32::cast_from(grad[i]);
    output[i] = F::cast_from(if f32::cast_from(input[i]) < 0f32 { value * negative_slope } else { value });
}
