// SPDX-License-Identifier: Apache-2.0
use ruda_core::{future::block_on, tensor::{DType, Shape, data::TensorData}};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, transfer::from_data, readback::into_data_sync}};
use ruda_optim::fused_adamw::{AdamWState, reference::ReferenceState};

pub type Tensor = RudaTensor<CudaRuntime>;

pub fn tensor(values: &[f32], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let data = TensorData::new(values.to_vec(), shape).convert_dtype(dtype);
    from_data(data, &CudaDevice::default())
}
pub fn floats(value: &Tensor) -> Vec<f32> {
    into_data_sync(value.clone()).convert::<f32>().to_vec::<f32>().unwrap()
}
pub fn rounded(values: &[f32], dtype: DType) -> Vec<f32> {
    TensorData::new(values.to_vec(), [values.len()]).convert_dtype(dtype).convert::<f32>().to_vec::<f32>().unwrap()
}
pub fn sync() {
    block_on(CudaRuntime::client(&CudaDevice::default()).sync()).expect("CUDA execution/synchronization failed");
}
pub fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * b.abs().max(1.0), "index {i}: {a} != {b}, tolerance {tolerance}");
    }
}
pub fn close_moments(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2e-8 + 5e-5 * b.abs();
        assert!(a.is_finite() && b.is_finite() && (a-b).abs() <= tolerance,
            "moment index {i}: {a} != {b}, tolerance {tolerance}");
    }
}
pub fn check_state(actual: &AdamWState<CudaRuntime>, expected: &ReferenceState) {
    assert_eq!(actual.step(), expected.step);
    close_moments(&floats(actual.first_moment()), &expected.first);
    close_moments(&floats(actual.second_moment()), &expected.second);
    match (actual.max_second_moment(), &expected.maximum) {
        (Some(a), Some(b)) => close_moments(&floats(a), b),
        (None, None) => (),
        _ => panic!("AMSGrad state presence differs"),
    }
}
