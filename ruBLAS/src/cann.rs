//! Native ACLNN matrix multiplication on CANN-owned tensors.
pub use ruda_driver_cann::CannError;
pub use ruda_driver_cann::tensor::{CannSession, CannTensor, DType};

pub fn matmul(
    a: &CannTensor,
    b: &CannTensor,
    output_dtype: DType,
    cube_math_type: i8,
) -> Result<CannTensor, CannError> {
    a.session().matmul(a, b, output_dtype, cube_math_type)
}

/// Explicit Rust-authored Ascend 950DT kernel path; the ACLNN `matmul` above is unchanged.
/// Kernels are authored in ruda-ascend-kernels; CANN intrinsic-source lowering
/// still requires Bisheng. This is not a generic Ruda Runtime implementation.
pub use ruda_driver_cann::tensor::deepgemm::{
    DeepGemm, GemmKind, GemmSpec, GroupEnds, Transpose, KernelStats, DeviceGroupEnds,
};
