use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::{dsl::{calculate_ruda_count_elemwise, prelude::*},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous}};
use ruda_kernel::dsl as kernel_dsl;

fn binding<R: Runtime>(input: &RudaTensor<R>, value: &RudaTensor<R>) {
    assert!(value.qparams.is_none() && matches!(value.dtype, DType::F32 | DType::F16 | DType::BF16)
        && value.meta.num_elements() <= u32::MAX as usize, "native ELU/CELU storage or index range unsupported");
    assert_eq!(input.device.to_id(), value.device.to_id(), "ELU/CELU device differs");
    assert!(input.client.same_execution_queue(&value.client), "ELU/CELU execution queue differs");
}

/// Native ELU or CELU with the original alpha and the original `x <= 0` exponential branch.
pub fn launch<R: Runtime>(input: RudaTensor<R>, alpha: f32, continuous: bool) -> RudaTensor<R> {
    binding(&input, &input);
    if input.meta.num_elements() == 0 { return input; }
    let input = into_contiguous(input);
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let dim = RudaDim::new(input.client.properties(), input.meta.num_elements());
    let count = calculate_ruda_count_elemwise(&input.client, input.meta.num_elements(), dim);
    forward::launch(&input.client, count, dim, input.clone().into_array_arg(), output.clone().into_array_arg(), alpha, continuous,
        include_str!("exponential_relu.rs").to_owned(), input.dtype.into());
    output
}

/// Original-primal ELU/CELU input VJP in FP32, with one final original-input storage cast.
pub fn launch_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>, alpha: f32, continuous: bool) -> RudaTensor<R> {
    binding(&input, &input);
    binding(&input, &grad);
    assert_eq!(input.meta.shape(), grad.meta.shape(), "ELU/CELU gradient shape differs");
    if input.meta.num_elements() == 0 { return input; }
    let input = into_contiguous(input);
    let grad = into_contiguous(grad);
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let dim = RudaDim::new(input.client.properties(), input.meta.num_elements());
    let count = calculate_ruda_count_elemwise(&input.client, input.meta.num_elements(), dim);
    backward::launch(&input.client, count, dim, input.clone().into_array_arg(), grad.clone().into_array_arg(), output.clone().into_array_arg(),
        alpha, continuous, include_str!("exponential_relu.rs").to_owned(), [input.dtype.into(), grad.dtype.into()]);
    output
}

#[ruda(launch)]
fn forward<F: Float>(input: &Array<F>, output: &mut Array<F>, alpha: f32, #[comptime] continuous: bool,
    #[comptime] _source: String, #[define(F)] _storage: StorageType) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let value = f32::cast_from(input[i]);
    let mut result = value;
    if value <= 0f32 {
        let mut exponent = value;
        if continuous { exponent /= alpha; }
        result = (exponent.exp() - 1f32) * alpha;
    }
    output[i] = F::cast_from(result);
}

#[ruda(launch)]
fn backward<F: Float, G: Float>(input: &Array<F>, grad: &Array<G>, output: &mut Array<F>, alpha: f32,
    #[comptime] continuous: bool, #[comptime] _source: String, #[define(F, G)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let value = f32::cast_from(input[i]);
    let dy = f32::cast_from(grad[i]);
    let mut result = dy;
    if value <= 0f32 {
        let mut exponent = value;
        if continuous { exponent /= alpha; }
        result = (dy * alpha) * exponent.exp();
        if continuous { result /= alpha; }
    }
    output[i] = F::cast_from(result);
}
