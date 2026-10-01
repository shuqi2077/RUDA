#[cfg(feature = "runtime")]
#[test]
fn runtime_types_are_reexports() {
    use core::any::TypeId;
    assert_eq!(
        TypeId::of::<ruda::runtime::id::KernelId>(),
        TypeId::of::<ruda_runtime::runtime::id::KernelId>(),
    );
    ruda::storage_id_type!(FacadeStorageId);
    assert_ne!(FacadeStorageId::new(), FacadeStorageId::new());
}

#[cfg(feature = "kernel")]
fn accepts_runtime<R: ruda::runtime::backend::Runtime>() {
    let _: fn(&R::Device) -> ruda::runtime::client::ComputeClient<R> = R::client;
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_uses_shared_runtime() {
    accepts_runtime::<ruda::cuda::CudaRuntime>();
    let _: ruda::cuda::CudaDevice = Default::default();
}

#[cfg(feature = "hip")]
#[test]
fn hip_uses_shared_runtime() {
    accepts_runtime::<ruda::hip::HipRuntime>();
    let _: ruda::hip::AmdDevice = Default::default();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_uses_shared_runtime() {
    accepts_runtime::<ruda::wgpu::WgpuRuntime>();
    let _: ruda::wgpu::WgpuDevice = Default::default();
}
