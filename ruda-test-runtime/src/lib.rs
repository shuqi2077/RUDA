#[cfg(test_runtime_default)]
pub type TestRuntime = ruda_driver_wgpu::WgpuRuntime;

#[cfg(test_runtime_wgpu)]
pub type TestRuntime = ruda_driver_wgpu::WgpuRuntime;

#[cfg(test_runtime_cpu)]
pub type TestRuntime = ruda_driver_cpu::CpuRuntime;

#[cfg(test_runtime_cuda)]
pub type TestRuntime = ruda_driver_cuda::CudaRuntime;

#[cfg(test_runtime_hip)]
pub type TestRuntime = ruda_driver_hip::HipRuntime;
