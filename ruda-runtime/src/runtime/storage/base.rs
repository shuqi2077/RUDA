use crate::runtime::{memory_management::ManagedMemoryBinding, server::IoError, storage_id_type};
use core::fmt::Debug;

// This ID is used to map a handle to its actual data.
storage_id_type!(StorageId);

impl core::fmt::Display for StorageId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StorageId({})", self.value))
    }
}

/// Defines if data uses a full memory chunk or a slice of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageUtilization {
    /// The offset in bytes from the chunk start.
    pub offset: u64,
    /// The size of the slice in bytes.
    pub size: u64,
}

/// Contains the [storage id](StorageId) of a resource and the way it is used.
#[derive(new, Clone, Debug)]
pub struct StorageHandle {
    /// Storage id.
    pub id: StorageId,
    /// How the storage is used.
    pub utilization: StorageUtilization,
}

impl StorageHandle {
    /// Returns the size the handle is pointing to in memory.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub fn size(&self) -> u64 {
        self.utilization.size
    }

    /// Returns the offset of the handle.
    pub fn offset(&self) -> u64 {
        self.utilization.offset
    }

    /// Increase the current offset with the given value in bytes.
    pub fn offset_start(&self, offset_bytes: u64) -> Self {
        let utilization = StorageUtilization {
            offset: self.offset().checked_add(offset_bytes)
                .expect("storage offset overflow"),
            size: self.size().checked_sub(offset_bytes)
                .expect("storage prefix exceeds resource size"),
        };

        Self {
            id: self.id,
            utilization,
        }
    }

    /// Reduce the size of the memory handle..
    pub fn offset_end(&self, offset_bytes: u64) -> Self {
        let utilization = StorageUtilization {
            offset: self.offset(),
            size: self.size().checked_sub(offset_bytes)
                .expect("storage suffix exceeds resource size"),
        };

        Self {
            id: self.id,
            utilization,
        }
    }
}

/// Storage types are responsible for allocating and deallocating memory.
pub trait ComputeStorage: Send {
    /// The resource associated type determines the way data is implemented and how
    /// it can be accessed by kernels.
    type Resource: Send;

    /// The alignment memory is allocated with in this storage.
    fn alignment(&self) -> usize;

    /// Returns the underlying resource for a specified storage handle
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    /// Allocates `size` units of memory and returns a handle to it
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError>;

    /// Deallocates the memory pointed by the given storage id.
    ///
    /// These deallocations might need to be flushed with [`Self::flush`].
    fn dealloc(&mut self, id: StorageId);

    /// Flush deallocations when required.
    fn flush(&mut self);
}

/// Access to the underlying resource.
#[derive(new, Debug)]
pub struct ManagedResource<Resource: Send> {
    // This handle is here just to keep the underlying allocation alive.
    // If the underlying allocation becomes invalid, someone else might
    // allocate into this resource which could lead to bad behaviour.
    #[allow(unused)]
    binding: ManagedMemoryBinding,
    resource: Resource,
}

impl<Resource: Send> ManagedResource<Resource> {
    /// access the underlying resource.
    ///
    /// # Note
    ///
    /// The resource might be bigger than the part required.
    /// (e.g. a big buffer where the handle only refers to a slice of it).
    /// Only the part required by the handle is guaranteed to remain,
    /// other parts of this resource *will* be re-used.
    pub fn resource(&self) -> &Resource {
        &self.resource
    }
}


#[cfg(test)]
mod handle_safety_tests {
    use super::*;

    #[test]
    fn valid_handle_slicing_preserves_bounds() {
        let handle = StorageHandle::new(StorageId::new(), StorageUtilization { offset: 8, size: 16 });
        let sliced = handle.offset_start(4).offset_end(3);
        assert_eq!(sliced.offset(), 12);
        assert_eq!(sliced.size(), 9);
        assert_eq!(handle.offset_start(16).size(), 0);
    }

    #[test]
    #[should_panic(expected = "storage prefix exceeds resource size")]
    fn oversized_prefix_never_wraps_in_release_mode() {
        StorageHandle::new(StorageId::new(), StorageUtilization { offset: 0, size: 1 })
            .offset_start(2);
    }

    #[test]
    #[should_panic(expected = "storage suffix exceeds resource size")]
    fn oversized_suffix_never_wraps_in_release_mode() {
        StorageHandle::new(StorageId::new(), StorageUtilization { offset: 0, size: 1 })
            .offset_end(2);
    }

    #[test]
    #[should_panic(expected = "storage offset overflow")]
    fn offset_overflow_never_wraps_in_release_mode() {
        StorageHandle::new(StorageId::new(), StorageUtilization { offset: u64::MAX, size: 2 })
            .offset_start(1);
    }
}
