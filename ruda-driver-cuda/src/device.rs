use ruda_core::device::{Device, DeviceId};

// It is not clear if CUDA has a limit on the number of bindings it can hold at
// any given time, but it's highly unlikely that it's more than this. We can
// also assume that we'll never have more than this many bindings in flight,
// so it's 'safe' to store only this many bindings.
/// Upper bound on in-flight kernel bindings retained by the CUDA adapter.
pub const CUDA_MAX_BINDINGS: u32 = 1024;

/// Explicit CUDA device selection. Default selects visible ordinal zero;
/// collective rank IDs are independent of this device ordinal.
#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct CudaDevice {
    /// Ordinal in the process's visible CUDA device list, not a global rank.
    /// Construct each replica and its communicator on the same selected device.
    pub index: usize,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cuda({})", self.index)
    }
}

impl Device for CudaDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index as u16,
        }
    }
}
