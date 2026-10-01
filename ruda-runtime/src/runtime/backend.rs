use alloc::boxed::Box;
use alloc::vec::Vec;
use ruda_core::device::{Device, DeviceId};
use ruda_core::ir::TargetProperties;
use ruda_core::tensor::{Shape, Strides};

use crate::runtime::{
    client::ComputeClient,
    compiler::{Compiler, RudaTask},
    server::ComputeServer,
};

/// Runtime for the `Ruda`.
pub trait Runtime: Sized + Send + Sync + 'static + core::fmt::Debug + Clone {
    /// The compiler used to compile the inner representation into tokens.
    type Compiler: Compiler;
    /// The compute server used to run kernels and perform autotuning.
    type Server: ComputeServer<Kernel = Box<dyn RudaTask<Self::Compiler>>>;
    /// The device used to retrieve the compute client.
    type Device: Device;

    /// Retrieve the compute client from the runtime device.
    fn client(device: &Self::Device) -> ComputeClient<Self>;

    /// The runtime name on the given device.
    fn name(client: &ComputeClient<Self>) -> &'static str;

    /// Stable loaded driver/runtime identity for persistent autotuning. Unknown backends return
    /// None and use session-only caches; device ordinal or API major version alone is insufficient.
    /// Adding a default preserves existing custom Runtime implementations.
    fn autotune_driver_fingerprint(_client: &ComputeClient<Self>) -> Option<alloc::string::String> {
        None
    }

    /// Return true if global input array lengths should be added to kernel info.
    fn require_array_lengths() -> bool {
        false
    }

    /// Returns the maximum ruda count on each dimension that can be launched.
    fn max_ruda_count() -> (u32, u32, u32);

    /// Whether a tensor with `shape` and `strides` can be read as is. If the result is false, the
    /// tensor should be made contiguous before reading.
    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool;

    /// Returns the properties of the target hardware architecture.
    fn target_properties() -> TargetProperties;

    /// Returns all devices available under the provided type id.
    fn enumerate_devices(
        type_id: u16,
        info: &<Self::Server as ComputeServer>::Info,
    ) -> Vec<DeviceId>;
    /// Returns all devices that can be handled by the runtime.
    fn enumerate_all_devices(info: &<Self::Server as ComputeServer>::Info) -> Vec<DeviceId> {
        Self::enumerate_devices(0, info)
    }
}
