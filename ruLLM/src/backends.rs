//! Optional adapters. A custom accelerator still needs a real Ruda runtime,
//! compiler and tensor kernels; a BackendKind::Custom label cannot supply them.

/// AscendCL runtime and explicit ACLNN tensor operations; not a generic DeviceBackend.
#[cfg(feature = "cann")]
pub use ruda_driver_cann as cann;

#[cfg(feature = "cann")]
pub use ruda_driver_cann::{CannApi, CannDevice, CannError, CannLibrary};

#[cfg(feature = "cann")]
impl From<CannDevice> for crate::runtime::DeviceKey {
    fn from(device: CannDevice) -> Self {
        Self { backend: crate::runtime::BackendKind::Cann, ordinal: device.ordinal() }
    }
}

#[cfg(feature = "nvidia")]
pub use ruda_driver_cuda::{CudaRuntime, CudaDevice as NvidiaDevice};

/// Explicit unfused adapter, unaffected by other crates enabling cuda-fusion.
#[cfg(feature = "nvidia")]
pub type Nvidia<F = f32, I = i32, B = u8> = ruda_tensor_device::DeviceBackend<CudaRuntime, F, I, B>;

#[cfg(feature = "amd")]
pub use ruda_driver_hip::{AmdDevice, HipRuntime};

/// Unfused HIP tensor backend for the same generic/packed Llama APIs.
/// Existing HIP platform/build prerequisites still apply.
#[cfg(feature = "amd")]
pub type Amd<F = f32, I = i32, B = u8> = ruda_tensor_device::DeviceBackend<HipRuntime, F, I, B>;

/// Read the driver-supported CUDA API version without launching inference.
/// This is not the NVIDIA driver release number and does not prove PTX support.
#[cfg(feature = "nvidia")]
pub fn cuda_driver_api_version() -> Result<crate::runtime::CudaApiVersion, crate::runtime::RuntimeError> {
    let encoded = ruda_driver_cuda::query_driver_api_version().map_err(|error| {
        crate::runtime::RuntimeError::new(crate::runtime::RuntimeErrorKind::DriverIncompatible, error.to_string())
    })?;
    crate::runtime::CudaApiVersion::from_encoded(encoded)
}

#[cfg(feature = "nvidia")]
pub fn validate_cuda_driver(requirements: &crate::runtime::CudaRequirements) -> Result<crate::runtime::CudaApiVersion, crate::runtime::RuntimeError> {
    let version = cuda_driver_api_version()?;
    requirements.check(version)?;
    Ok(version)
}
