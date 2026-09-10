use super::DummyServer;
use crate::dummy::KernelTask;
use ruda_core::device::{Device, DeviceService};
use ruda_core::ir::MemoryDeviceProperties;
use ruda_core::ir::StorageType;
use ruda::runtime::server::ComputeServer;
use ruda::runtime::{
    client::ComputeClient,
    compiler::{CompilationError, Compiler},
    logging::ServerLogger,
    memory_management::{MemoryConfiguration, MemoryManagement, MemoryManagementOptions},
    backend::Runtime,
    server::ExecutionMode,
    storage::BytesStorage,
};
use ruda_core::tensor::Shape;
use ruda_core::tensor::Strides;
use std::sync::Arc;

/// The dummy device.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct DummyDevice;

impl Device for DummyDevice {
    fn from_id(_device_id: ruda_core::device::DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> ruda_core::device::DeviceId {
        ruda_core::device::DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }
}

pub type DummyClient = ComputeClient<DummyRuntime>;

impl DeviceService for DummyServer {
    fn init(_device_id: ruda_core::device::DeviceId) -> Self {
        init_server()
    }

    fn utilities(&self) -> Arc<dyn std::any::Any + Send + Sync> {
        ComputeServer::utilities(self) as Arc<dyn std::any::Any + Send + Sync>
    }
}

fn init_server() -> DummyServer {
    let storage = BytesStorage::default();
    let mem_properties = MemoryDeviceProperties {
        max_page_size: 1024 * 1024 * 512,
        alignment: 32,
    };

    let memory_management = MemoryManagement::from_configuration(
        storage,
        &mem_properties,
        MemoryConfiguration::default(),
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new("Main CPU Memory"),
    );
    DummyServer::new(memory_management, mem_properties)
}

pub fn test_client(device: &DummyDevice) -> DummyClient {
    ComputeClient::load(device)
}

#[derive(Debug, Clone)]
pub struct DummyCompiler;

impl Compiler for DummyCompiler {
    type Representation = KernelTask;

    type CompilationOptions = ();

    fn compile(
        &mut self,
        _kernel: ruda::runtime::kernel::KernelDefinition,
        _compilation_options: &Self::CompilationOptions,
        _mode: ExecutionMode,
        _addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        unimplemented!()
    }

    fn elem_size(&self, _elem: ruda_core::ir::ElemType) -> usize {
        unimplemented!()
    }

    fn extension(&self) -> &'static str {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
pub struct DummyRuntime;

impl Runtime for DummyRuntime {
    type Compiler = DummyCompiler;

    type Server = DummyServer;

    type Device = DummyDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        unimplemented!()
    }

    fn max_ruda_count() -> (u32, u32, u32) {
        unimplemented!()
    }

    fn can_read_tensor(_shape: &Shape, _strides: &Strides) -> bool {
        unimplemented!()
    }

    fn target_properties() -> ruda_core::ir::TargetProperties {
        unimplemented!()
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as ComputeServer>::Info,
    ) -> Vec<ruda_core::device::DeviceId> {
        vec![ruda_core::device::DeviceId {
            type_id: 0,
            index_id: 0,
        }]
    }
}
