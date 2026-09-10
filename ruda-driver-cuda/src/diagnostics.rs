//! Fallible read-only preflight. No context creation or device-memory allocation.
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDriverProbeError(pub String);
impl Display for CudaDriverProbeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for CudaDriverProbeError {}

/// Return cuDriverGetVersion's 1000*major + 10*minor encoding. Catching a
/// loader panic here does NOT make panic recovery during GPU execution safe.
/// A process configured with panic=abort still aborts on a loader panic.
pub fn query_driver_api_version() -> Result<u32, CudaDriverProbeError> {
    let result = std::panic::catch_unwind(|| {
        let mut version = 0i32;
        // SAFETY: cuDriverGetVersion writes one int to a valid, uniquely
        // borrowed stack location. It neither launches work nor retains it.
        let status = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaDriverProbeError(format!("cuDriverGetVersion failed: {status:?}")));
        }
        u32::try_from(version).ok().filter(|&v| v > 0)
            .ok_or_else(|| CudaDriverProbeError("CUDA driver reported no valid API version".into()))
    });
    result.unwrap_or_else(|_| Err(CudaDriverProbeError("CUDA driver library or entry point could not be loaded".into())))
}
