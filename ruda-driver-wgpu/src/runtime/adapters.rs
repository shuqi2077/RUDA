use super::*;

pub(super) async fn request_adapter(
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> (wgpu::Instance, wgpu::Adapter) {
    #[cfg(not(feature = "vulkan-validate"))]
    let instance_flags = {
        let debug = ServerLogger::default();
        match (debug.profile_level(), debug.compilation_activated()) {
            (Some(ProfileLevel::Full), _) => InstanceFlags::advanced_debugging(),
            (_, true) => InstanceFlags::debugging(),
            (_, false) => InstanceFlags::default(),
        }
    };
    #[cfg(feature = "vulkan-validate")]
    let instance_flags = InstanceFlags::advanced_debugging();
    log::debug!("{instance_flags:?}");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        flags: instance_flags,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    #[allow(deprecated)]
    let override_device = if matches!(
        device,
        WgpuDevice::DefaultDevice | WgpuDevice::BestAvailable
    ) {
        get_device_override()
    } else {
        None
    };

    let device = override_device.unwrap_or_else(|| device.clone());

    let adapter = match device {
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::DiscreteGpu(num) => {
            select_from_adapter_list(
                num,
                "No Discrete GPU device found",
                &instance,
                &device,
                backend,
            )
            .await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::IntegratedGpu(num) => {
            select_from_adapter_list(
                num,
                "No Integrated GPU device found",
                &instance,
                &device,
                backend,
            )
            .await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::VirtualGpu(num) => {
            select_from_adapter_list(
                num,
                "No Virtual GPU device found",
                &instance,
                &device,
                backend,
            )
            .await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::Cpu => {
            select_from_adapter_list(0, "No CPU device found", &instance, &device, backend).await
        }
        #[cfg(target_family = "wasm")]
        WgpuDevice::IntegratedGpu(_) => {
            request_adapter_with_preference(&instance, wgpu::PowerPreference::LowPower).await
        }
        WgpuDevice::Existing(_) => {
            unreachable!("Cannot select an adapter for an existing device.")
        }
        _ => {
            request_adapter_with_preference(&instance, wgpu::PowerPreference::HighPerformance).await
        }
    };

    log::info!("Using adapter {:?}", adapter.get_info());

    (instance, adapter)
}

async fn request_adapter_with_preference(
    instance: &wgpu::Instance,
    power_preference: wgpu::PowerPreference,
) -> wgpu::Adapter {
    instance
        .request_adapter(&RequestAdapterOptions {
            power_preference,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("No possible adapter available for backend. Falling back to first available.")
}

#[cfg(not(target_family = "wasm"))]
async fn select_from_adapter_list(
    num: usize,
    error: &str,
    instance: &wgpu::Instance,
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> wgpu::Adapter {
    let mut adapters_other = Vec::new();
    let mut adapters = Vec::new();

    instance
        .enumerate_adapters(backend.into())
        .await
        .into_iter()
        .for_each(|adapter| {
            let device_type = adapter.get_info().device_type;

            if let wgpu::DeviceType::Other = device_type {
                adapters_other.push(adapter);
                return;
            }

            let is_same_type = match device {
                WgpuDevice::DiscreteGpu(_) => device_type == wgpu::DeviceType::DiscreteGpu,
                WgpuDevice::IntegratedGpu(_) => device_type == wgpu::DeviceType::IntegratedGpu,
                WgpuDevice::VirtualGpu(_) => device_type == wgpu::DeviceType::VirtualGpu,
                WgpuDevice::Cpu => device_type == wgpu::DeviceType::Cpu,
                #[allow(deprecated)]
                WgpuDevice::DefaultDevice | WgpuDevice::BestAvailable => true,
                WgpuDevice::Existing(_) => {
                    unreachable!("Cannot select an adapter for an existing device.")
                }
            };

            if is_same_type {
                adapters.push(adapter);
            }
        });

    if adapters.len() <= num {
        if adapters_other.len() <= num {
            panic!(
                "{}, adapters {:?}, other adapters {:?}",
                error,
                adapters
                    .into_iter()
                    .map(|adapter| adapter.get_info())
                    .collect::<Vec<_>>(),
                adapters_other
                    .into_iter()
                    .map(|adapter| adapter.get_info())
                    .collect::<Vec<_>>(),
            );
        }

        return adapters_other.remove(num);
    }

    adapters.remove(num)
}

fn get_device_override() -> Option<WgpuDevice> {
    // If BestAvailable, check if we should instead construct as
    // if a specific device was specified.
    std::env::var("RUDA_WGPU_DEFAULT_DEVICE")
        .ok()
        .and_then(|var| {
            let override_device = if let Some(inner) = var.strip_prefix("DiscreteGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDevice::DiscreteGpu)
            } else if let Some(inner) = var.strip_prefix("IntegratedGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDevice::IntegratedGpu)
            } else if let Some(inner) = var.strip_prefix("VirtualGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDevice::VirtualGpu)
            } else if var == "Cpu" {
                Some(WgpuDevice::Cpu)
            } else {
                None
            };

            if override_device.is_none() {
                log::warn!("Unknown RUDA_WGPU_DEVICE override {var}");
            }
            override_device
        })
}
