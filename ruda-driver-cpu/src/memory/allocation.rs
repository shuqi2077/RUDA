use ruda_core::bytes::{AllocationController, AllocationProperty};
use ruda_kernel::dsl::server::IoError;
use ruda::runtime::{
    memory_management::{ManagedMemoryBinding, MemoryManagement},
    storage::{BytesReadGuard, BytesStorage},
};

// Multiple host readbacks may overlap. They share read leases, not independent
// mutable slices into the same allocation. A mutable Bytes access copies first.
enum ReadbackBacking {
    Shared(BytesReadGuard),
    Owned(Vec<u8>),
}

pub struct CpuAllocController {
    backing: ReadbackBacking,
    // Keep the memory-pool region reserved as well as the allocation itself.
    _binding: ManagedMemoryBinding,
}

impl AllocationController for CpuAllocController {
    fn alloc_align(&self) -> usize { align_of::<u8>() }

    fn property(&self) -> AllocationProperty { AllocationProperty::Other }

    /// SAFETY: The caller must write only initialized bytes, as required by
    /// AllocationController. Mutating a readback must not mutate a shared view.
    unsafe fn memory_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        if let ReadbackBacking::Shared(resource) = &self.backing {
            // The shared lease remains live until the copy has finished.
            let copy = resource.to_vec();
            self.backing = ReadbackBacking::Owned(copy);
        }
        let ReadbackBacking::Owned(bytes) = &mut self.backing else { unreachable!() };
        // SAFETY: u8 and MaybeUninit<u8> have identical layout. This buffer is
        // private to this controller, and the caller preserves initialization.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len()) }
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        let bytes: &[u8] = match &self.backing {
            ReadbackBacking::Shared(resource) => resource,
            ReadbackBacking::Owned(bytes) => bytes,
        };
        // SAFETY: The backing (including any shared lease) outlives this slice.
        // MaybeUninit<u8> has the same size and alignment as u8.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) }
    }
}

impl CpuAllocController {
    /// Call only after the producing CPU queues have completed their writes.
    pub fn init(
        binding: ruda_kernel::dsl::server::Binding,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) -> Result<Self, IoError> {
        let memory = binding.memory.clone();
        let resource = memory_management.get_resource(
            binding.memory, binding.offset_start, binding.offset_end,
        )?;
        let resource = resource.try_read().map_err(|err| IoError::Unknown {
            description: err.to_string(),
            backtrace: ruda_core::backtrace::BackTrace::capture(),
        })?;
        Ok(Self { _binding: memory, backing: ReadbackBacking::Shared(resource) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda::runtime::{memory_management::ManagedMemoryHandle, storage::ComputeStorage};
    use ruda_core::bytes::Bytes;

    #[test]
    fn overlapping_readbacks_share_reads_but_mutation_copies() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(4).unwrap();
        let resource = storage.get(&handle);
        resource.write().copy_from_slice(&[1, 2, 3, 4]);
        let controller = || CpuAllocController {
            backing: ReadbackBacking::Shared(resource.read()),
            _binding: ManagedMemoryHandle::new().binding(),
        };
        // SAFETY: Both controllers own initialized four-byte read leases.
        let mut first = unsafe { Bytes::from_controller(Box::new(controller()), 4) };
        let second = unsafe { Bytes::from_controller(Box::new(controller()), 4) };
        storage.dealloc(handle.id);
        first[0] = 99;
        assert_eq!(&first[..], &[99, 2, 3, 4]);
        assert_eq!(&second[..], &[1, 2, 3, 4]);
        assert_eq!(&resource.read()[..], &[1, 2, 3, 4]);
        assert!(resource.try_write().is_err());
        drop(second);
        // The first readback owns a private copy now and no longer locks the
        // original allocation. There are no remaining shared leases.
        assert!(resource.try_write().is_ok());
    }
}
