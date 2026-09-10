use crate::{
    compilation::{MlirCompiler, register_supported_types},
    execution::server::CpuServer,
    device::CpuDevice,
};
use ruda_core::{device::DeviceService, profile::TimingMethod};
use ruda_kernel::dsl::{
    MemoryConfiguration, Runtime,
    client::ComputeClient,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        DeviceProperties, HardwareProperties, MemoryDeviceProperties, TargetProperties, VectorSize,
        features::Features,
    },
    server::ServerUtilities,
    zspace::{Shape, Strides},
};
use ruda::runtime::{allocator::ContiguousMemoryLayoutPolicy, logging::ServerLogger};
use ruda_kernel::library::tensor::is_contiguous;
use std::sync::Arc;
use sysinfo::System;

#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct CpuRuntime;

pub type CpuCompiler = MlirCompiler;

mod device_service;

impl Runtime for CpuRuntime {
    type Compiler = CpuCompiler;
    type Server = CpuServer;
    type Device = CpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "cpu"
    }

    fn max_ruda_count() -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        is_contiguous(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            // Values are irrelevant, since no wgsl backends currently support manual mma
            mma: Default::default(),
        }
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as ruda_kernel::dsl::server::ComputeServer>::Info,
    ) -> Vec<DeviceId> {
        vec![DeviceId {
            type_id: 0,
            index_id: 0,
        }]
    }
}
