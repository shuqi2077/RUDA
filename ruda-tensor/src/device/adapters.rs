#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::device::DeviceOps;
    use ruda_driver_wgpu::WgpuDevice;

    impl DeviceOps for WgpuDevice {}
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::device::DeviceOps;
    use ruda_driver_cuda::CudaDevice;

    impl DeviceOps for CudaDevice {}
}

#[cfg(feature = "cpu")]
mod cpu {
    use crate::device::DeviceOps;
    use ruda_driver_cpu::CpuDevice;

    impl DeviceOps for CpuDevice {}
}

#[cfg(feature = "hip")]
mod hip {
    use crate::device::DeviceOps;
    use ruda_driver_hip::AmdDevice;

    impl DeviceOps for AmdDevice {}
}
