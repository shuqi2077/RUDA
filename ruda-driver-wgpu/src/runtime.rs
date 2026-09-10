use crate::{
    AutoCompiler, AutoGraphicsApi, GraphicsApi, WgpuDevice, backend, execution::WgpuServer,
    contiguous_strides,
};
use ruda_core::device::{Device, DeviceService};
use ruda_core::{future, profile::TimingMethod};
use ruda_kernel::dsl::device::{DeviceId, ServerUtilitiesHandle};
use ruda_kernel::dsl::server::ServerUtilities;
use ruda_kernel::dsl::zspace::{Shape, Strides};
use ruda_kernel::dsl::{Runtime, ir::TargetProperties};
use ruda_core::ir::{DeviceProperties, HardwareProperties, MemoryDeviceProperties};
use ruda::runtime::allocator::ContiguousMemoryLayoutPolicy;
#[cfg(not(feature = "vulkan-validate"))]
use ruda::runtime::logging::ProfileLevel;
pub use ruda::runtime::memory_management::MemoryConfiguration;
use ruda::runtime::{client::ComputeClient, logging::ServerLogger};
use wgpu::{InstanceFlags, RequestAdapterOptions};

/// Runtime that uses the [wgpu] crate with the wgsl compiler. This is used in the Wgpu backend.
/// For advanced configuration, use [`init_setup`] to pass in runtime options or to select a
/// specific graphics API.
#[derive(Debug, Clone)]
pub struct WgpuRuntime;

impl Runtime for WgpuRuntime {
    type Compiler = AutoCompiler;
    type Server = WgpuServer;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(client: &ComputeClient<Self>) -> &'static str {
        match client.info() {
            wgpu::Backend::Vulkan => {
                #[cfg(feature = "spirv")]
                return "wgpu<spirv>";

                #[cfg(not(feature = "spirv"))]
                return "wgpu<wgsl>";
            }
            wgpu::Backend::Metal => {
                #[cfg(feature = "msl")]
                return "wgpu<msl>";

                #[cfg(not(feature = "msl"))]
                return "wgpu<wgsl>";
            }
            _ => "wgpu<wgsl>",
        }
    }

    fn max_ruda_count() -> (u32, u32, u32) {
        let max_dim = u16::MAX as u32;
        (max_dim, max_dim, max_dim)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        if shape.is_empty() {
            return true;
        }

        for (&expected, &stride) in contiguous_strides(shape).iter().zip(strides.iter()) {
            if expected != stride {
                return false;
            }
        }

        true
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            // Values are irrelevant, since no wgsl backends currently support manual mma
            mma: Default::default(),
        }
    }

    fn enumerate_devices(type_id: u16, info: &wgpu::Backend) -> Vec<DeviceId> {
        #[cfg(target_family = "wasm")]
        {
            let _ = type_id;
            let _ = info;
            // WebGPU only supports a single device currently.
            vec![DeviceId::new(0, 0)]
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });

            let adapters = enumerate_all_adapters(instance, *info);
            adapters
                .into_iter()
                .filter(|adapter| {
                    // Default doesn't filter device types.
                    if type_id == 4 {
                        return true;
                    }

                    let device_type = adapter.get_info().device_type;

                    let adapter_type_id = match device_type {
                        wgpu::DeviceType::Other => 4,
                        wgpu::DeviceType::IntegratedGpu => 1,
                        wgpu::DeviceType::DiscreteGpu => 0,
                        wgpu::DeviceType::VirtualGpu => 2,
                        wgpu::DeviceType::Cpu => 3,
                    };

                    adapter_type_id == type_id
                })
                .enumerate()
                .map(|(index, adapter)| match adapter.get_info().device_type {
                    wgpu::DeviceType::DiscreteGpu => DeviceId::new(0, index as u16),
                    wgpu::DeviceType::IntegratedGpu => DeviceId::new(1, index as u16),
                    wgpu::DeviceType::VirtualGpu => DeviceId::new(2, index as u16),
                    wgpu::DeviceType::Cpu => DeviceId::new(3, 0),
                    wgpu::DeviceType::Other => DeviceId::new(4, 0),
                })
                .collect()
        }
    }

    fn enumerate_all_devices(info: &wgpu::Backend) -> Vec<DeviceId> {
        #[cfg(target_family = "wasm")]
        {
            let _ = info;
            // WebGPU only supports a single device currently.
            vec![DeviceId::new(0, 0)]
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapters = enumerate_all_adapters(instance, *info);
            adapters
                .into_iter()
                .enumerate()
                .map(|(index, adapter)| match adapter.get_info().device_type {
                    wgpu::DeviceType::DiscreteGpu => DeviceId::new(0, index as u16),
                    wgpu::DeviceType::IntegratedGpu => DeviceId::new(1, index as u16),
                    wgpu::DeviceType::VirtualGpu => DeviceId::new(2, index as u16),
                    wgpu::DeviceType::Cpu => DeviceId::new(3, 0),
                    wgpu::DeviceType::Other => DeviceId::new(4, 0),
                })
                .collect()
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn enumerate_all_adapters(instance: wgpu::Instance, backend: wgpu::Backend) -> Vec<wgpu::Adapter> {
    // `enumerate_adapters` is now async & available on WebGPU
    ruda_core::future::block_on(instance.enumerate_adapters(backend.into()))
}

/// The values that control how a WGPU Runtime will perform its calculations.
pub struct RuntimeOptions {
    /// Control the amount of compute tasks to be aggregated into a single GPU command.
    pub tasks_max: usize,
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        #[cfg(test)]
        const DEFAULT_MAX_TASKS: usize = 32;
        #[cfg(not(test))]
        const DEFAULT_MAX_TASKS: usize = 32;

        let tasks_max = match std::env::var("RUDA_WGPU_MAX_TASKS") {
            Ok(value) => value
                .parse::<usize>()
                .expect("RUDA_WGPU_MAX_TASKS should be a positive integer."),
            Err(_) => DEFAULT_MAX_TASKS,
        };

        Self {
            tasks_max,
            memory_config: MemoryConfiguration::default(),
        }
    }
}

mod adapters;
mod device_service;
mod setup;

pub use setup::{WgpuSetup, init_device, init_setup, init_setup_async};
pub(crate) use setup::create_setup_for_device;
pub(crate) use device_service::create_server;
