pub use ruda_kernel::template::*;

use ruda_tensor_device::{DeviceRuntime, TensorElement, RudaTensor};

/// Create a vector containing the dimension, strides and shape of tensors.
///
/// # Example
///
/// With two tensors (lhs, rhs)
///
/// | Indexes                  | Value       |
/// |:------------------------:|:-----------:|
/// |           0..1           | D           |
/// |           1..D + 1       | lhs strides |
/// |     (D + 1)..(2 * D + 1) | rhs strides |
/// | (2 * D + 1)..(3 * D + 1) | lhs shape   |
/// | (3 * D + 1)..(4 * D + 1) | rhs shape   |
pub fn build_info<R: DeviceRuntime, E: TensorElement>(tensors: &[&RudaTensor<R>]) -> Vec<u32> {
    ruda_kernel::tensor::info::build_info::<R, E>(tensors)
}
