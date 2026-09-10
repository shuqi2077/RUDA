use super::*;

/// A complete setup used to run wgpu.
///
/// These can either be created with [`init_setup`] or [`init_setup_async`].
#[derive(Clone, Debug)]
pub struct WgpuSetup {
    /// The underlying wgpu instance.
    pub instance: wgpu::Instance,
    /// The selected 'adapter'. This corresponds to a physical device.
    pub adapter: wgpu::Adapter,
    /// The wgpu device Ruda will use. Nb: There can only be one device per adapter.
    pub device: wgpu::Device,
    /// The queue Ruda commands will be submitted to.
    pub queue: wgpu::Queue,
    /// The backend used by the setup.
    pub backend: wgpu::Backend,
}

/// Create a [`WgpuDevice`] on an existing [`WgpuSetup`].
/// Useful when you want to share a device between `Ruda` and other wgpu-dependent libraries.
///
/// # Note
///
/// Please **do not** to call on the same [`setup`](WgpuSetup) more than once.
///
/// This function generates a new, globally unique ID for the device every time it is called,
/// even if called on the same device multiple times.
pub fn init_device(setup: WgpuSetup, options: RuntimeOptions) -> WgpuDevice {
    use core::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    let device_id = COUNTER.fetch_add(1, Ordering::Relaxed);
    if device_id == u32::MAX {
        core::panic!("Memory ID overflowed");
    }

    let device_id = WgpuDevice::Existing(device_id);
    let server = create_server(setup, options);
    let _ = ComputeClient::<WgpuRuntime>::init(&device_id, server);
    device_id
}

/// Like [`init_setup_async`], but synchronous.
/// On wasm, it is necessary to use [`init_setup_async`] instead.
pub fn init_setup<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) -> WgpuSetup {
    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            let _ = (device, options);
            panic!("Creating a wgpu setup synchronously is unsupported on wasm. Use init_async instead");
        } else {
            future::block_on(init_setup_async::<G>(device, options))
        }
    }
}

/// Initialize a client on the given device with the given options.
/// This function is useful to configure the runtime options
/// or to pick a different graphics API.
pub async fn init_setup_async<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuSetup {
    let setup = create_setup_for_device(device, G::backend()).await;
    let return_setup = setup.clone();
    let server = create_server(setup, options);
    let _ = ComputeClient::<WgpuRuntime>::init(device, server);
    return_setup
}

/// Select the wgpu device and queue based on the provided [device](WgpuDevice) and
/// [backend](wgpu::Backend).
pub(crate) async fn create_setup_for_device(
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> WgpuSetup {
    let (instance, adapter) = super::adapters::request_adapter(device, backend).await;
    let (device, queue) = backend::request_device(&adapter).await;

    log::info!(
        "Created wgpu compute server on device {:?} => {:?}",
        device,
        adapter.get_info()
    );

    WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend,
    }
}
