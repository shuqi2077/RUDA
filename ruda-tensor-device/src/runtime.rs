use ruda::runtime::{backend::Runtime, compiler::RudaTask};
use ruda_tensor::DeviceOps;

/// Device runtime implementing the tensor backend device contract.
pub trait DeviceRuntime: Runtime<Device = Self::RudaDevice, Server = Self::RudaServer> {
    /// The device that should also implement [ruda_tensor::backend::DeviceOps].
    type RudaDevice: ruda_tensor::DeviceOps;
    /// The server accepting compiled Kernel tasks.
    type RudaServer: ruda::runtime::server::ComputeServer<Kernel = Box<dyn RudaTask<Self::Compiler>>>;
}


impl<R: Runtime> DeviceRuntime for R
where
    R::Device: DeviceOps,
{
    type RudaDevice = R::Device;
    type RudaServer = R::Server;
}
