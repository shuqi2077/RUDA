//! Native ACLNN neural-network operations on CANN-owned tensors.
pub use ruda_driver_cann::CannError;
pub use ruda_driver_cann::tensor::{CannSession, CannTensor, DType};

pub fn softmax(input: &CannTensor, dim: i64) -> Result<CannTensor, CannError> {
    input.session().softmax(input, dim)
}

pub fn silu(input: &CannTensor) -> Result<CannTensor, CannError> {
    input.session().silu(input)
}

pub fn rms_norm(
    input: &CannTensor,
    gamma: &CannTensor,
    epsilon: f64,
) -> Result<(CannTensor, CannTensor), CannError> {
    input.session().rms_norm(input, gamma, epsilon)
}
