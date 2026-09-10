//! Native ACLNN tensor operations on CANN-owned tensors.
pub use ruda_driver_cann::CannError;
pub use ruda_driver_cann::tensor::{CannSession, CannTensor, DType, ScalarValue};

pub fn add(
    a: &CannTensor,
    b: &CannTensor,
    alpha: ScalarValue,
    output_dtype: DType,
) -> Result<CannTensor, CannError> {
    a.session().add(a, b, alpha, output_dtype)
}

pub fn mul(a: &CannTensor, b: &CannTensor, output_dtype: DType) -> Result<CannTensor, CannError> {
    a.session().mul(a, b, output_dtype)
}

pub fn permute(input: &CannTensor, dims: &[i64]) -> Result<CannTensor, CannError> {
    input.session().permute(input, dims)
}

pub fn sum(
    input: &CannTensor,
    dims: &[i64],
    keep_dims: bool,
    output_dtype: DType,
) -> Result<CannTensor, CannError> {
    input.session().sum(input, dims, keep_dims, output_dtype)
}
