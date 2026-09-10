use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, allocation::empty_device_dtype}};
use ruda_core::tensor::{DType, calculate_matmul_output};

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output<R: Runtime>(
    lhs: &RudaTensor<R>,
    rhs: &RudaTensor<R>,
    dtype: DType,
) -> RudaTensor<R> {
    empty_device_dtype(
        lhs.client.clone(),
        lhs.device.clone(),
        calculate_matmul_output(lhs.meta.shape(), rhs.meta.shape()).unwrap(),
        dtype,
    )
}
