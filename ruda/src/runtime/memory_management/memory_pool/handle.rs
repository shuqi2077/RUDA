use crate::runtime::memory_management::MemoryHandle;
use alloc::sync::Arc;
use spin::Mutex;

/// Managed Memory handle
#[derive(Debug)]
pub struct ManagedMemoryHandle {
    descriptor: Arc<ManagedMemoryDescriptor>,
    // Holds only the reference counts of the handle.
    handle_count: Arc<()>,
}

/// Binding of a memory handle
#[derive(Debug)]
pub struct ManagedMemoryBinding {
    descriptor: Arc<ManagedMemoryDescriptor>,
}

impl Clone for ManagedMemoryHandle {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
            handle_count: self.handle_count.clone(),
        }
    }
}

/// Managed memory descriptor.
///
/// Host-side diagnostics can read the location while the device thread updates
/// it. Protect the whole location so all readers observe a consistent snapshot.
/// Send/Sync are derived from the fields; no unchecked Cell sharing is needed.
pub(crate) struct ManagedMemoryDescriptor {
    pub(crate) id: ManagedMemoryId,
    location: Mutex<MemoryLocation>,
}

impl core::fmt::Debug for ManagedMemoryDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ManagedMemoryDescriptor")
            .field("id", &self.id)
            .field("location", &self.location())
            .finish()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
/// Managed memory unique identifier.
pub struct ManagedMemoryId {
    pub(crate) value: usize,
}

impl PartialEq for ManagedMemoryDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ManagedMemoryDescriptor {}

#[derive(Clone, Copy, Debug)]
/// Defines where the [`ManagedMemoryId`] is located.
pub(crate) struct MemoryLocation {
    /// The memory pool index in the global memory management.
    pub pool: u8,
    /// The memory page index in a memory pool.
    pub page: u16,
    /// The memory slice index in a memory page.
    pub slice: u32,
    /// Whether the memory location is known/initialized.
    pub init: u8,
}

impl ManagedMemoryDescriptor {
    /// Update the memory location for the given [`ManagedMemoryId`].
    pub(crate) fn update_location(&self, location: MemoryLocation) {
        *self.location.lock() = location;
    }

    /// Update only the slice position for the given [`ManagedMemoryId`].
    pub(crate) fn update_slice(&self, slice: u32) {
        self.location.lock().slice = slice;
    }

    /// Update only the memory page position for the given [`ManagedMemoryId`].
    pub fn update_page(&self, page: u16) {
        self.location.lock().page = page;
    }

    /// Retrieves the current location.
    pub(crate) fn location(&self) -> MemoryLocation {
        *self.location.lock()
    }

    pub(crate) fn slice(&self) -> usize {
        self.location.lock().slice as usize
    }

    pub(crate) fn page(&self) -> usize {
        self.location.lock().page as usize
    }
}

impl MemoryLocation {
    /// Creates a new memory location.
    pub(crate) fn new(pool: u8, page: u16, slice: u32) -> Self {
        Self {
            pool,
            page,
            slice,
            init: 1,
        }
    }

    /// Creates a new uninitialized memory location.
    pub(crate) fn uninit() -> Self {
        Self {
            pool: 0,
            page: 0,
            slice: 0,
            init: 0,
        }
    }
}

impl ManagedMemoryHandle {
    /// Creates a new managed memory handle.
    pub fn new() -> Self {
        let value = Self::gen_id();

        Self {
            descriptor: Arc::new(ManagedMemoryDescriptor {
                id: ManagedMemoryId { value },
                location: Mutex::new(MemoryLocation::uninit()),
            }),
            handle_count: Arc::new(()),
        }
    }

    /// Retrieves the descriptor for the current handle.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }

    /// Return whether the current handle can be modified in-place.
    pub fn can_mut(&self) -> bool {
        Arc::strong_count(&self.handle_count) <= 2
    }

    /// Return whether the current handle is free.
    pub fn is_free(&self) -> bool {
        Arc::strong_count(&self.descriptor) <= 1
    }

    /// Returns the binding for the current handle.
    pub fn binding(self) -> ManagedMemoryBinding {
        ManagedMemoryBinding {
            descriptor: self.descriptor.clone(),
        }
    }

    fn gen_id() -> usize {
        static COUNTER: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        let value = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if value == usize::MAX {
            core::panic!("Memory ID overflowed");
        }
        value
    }
}

impl ManagedMemoryBinding {
    /// Retrieves the descriptor for the current binding.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }
}

impl Default for ManagedMemoryHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ManagedMemoryBinding {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
        }
    }
}

impl MemoryHandle<ManagedMemoryBinding> for ManagedMemoryHandle {
    fn can_mut(&self) -> bool {
        self.can_mut()
    }

    fn binding(self) -> ManagedMemoryBinding {
        self.binding()
    }
}

/// Calculates a best-effort heuristic for the alignment of row-aligned tensors.
/// Prefers contiguous alignments for unit dimensions, 16-byte minimum alignment for non-unit,
/// scaling with input size up to `buffer_align`.
pub fn optimal_align(shape: usize, elem_size: usize, buffer_align: usize) -> usize {
    if shape == 1 {
        elem_size
    } else {
        (shape * elem_size)
            .next_power_of_two()
            .clamp(16, buffer_align)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_id_mutability() {
        let handle1 = ManagedMemoryHandle::new();
        handle1.descriptor().update_slice(4);
        assert_eq!(handle1.descriptor().slice(), 4);

        let handle2 = ManagedMemoryHandle::new();
        handle2
            .clone()
            .descriptor()
            .update_location(handle1.descriptor().location());
        assert_eq!(handle2.descriptor().slice(), 4);
    }

    #[test]
    fn test_location_visible_through_shared_arc() {
        let handle = ManagedMemoryHandle::new();
        let handle2 = handle.clone();

        let location = MemoryLocation::new(1, 2, 3);
        handle.descriptor().update_location(location);

        assert_eq!(handle2.descriptor().location().pool, 1);
        assert_eq!(handle2.descriptor().location().page, 2);
        assert_eq!(handle2.descriptor().location().slice, 3);
        assert_eq!(handle2.descriptor().location().init, 1);

        handle.descriptor().update_slice(42);
        assert_eq!(handle2.descriptor().slice(), 42);
    }

    #[test]
    #[cfg(feature = "runtime-std")]
    fn concurrent_debug_reads_consistent_location_snapshots() {
        let handle = ManagedMemoryHandle::new();
        let writer = handle.clone();
        let task = std::thread::spawn(move || {
            for i in 1..=128_u32 {
                writer.descriptor().update_location(MemoryLocation::new(i as u8, i as u16, i));
                std::thread::yield_now();
            }
        });
        for _ in 0..128 {
            let _ = alloc::format!("{handle:?}");
            let location = handle.descriptor().location();
            assert_eq!(location.page as u32, location.slice);
            assert_eq!(location.pool as u32, location.slice);
            std::thread::yield_now();
        }
        task.join().unwrap();
        assert_eq!(handle.descriptor().slice(), 128);
    }

    #[test]
    #[cfg(feature = "runtime-std")]
    fn concurrent_field_updates_do_not_overwrite_each_other() {
        let handle = ManagedMemoryHandle::new();
        let first = handle.clone();
        let second = handle.clone();
        let page_writer = std::thread::spawn(move || {
            for page in 1..=128 { first.descriptor().update_page(page); }
        });
        let slice_writer = std::thread::spawn(move || {
            for slice in 1..=128 { second.descriptor().update_slice(slice); }
        });
        page_writer.join().unwrap();
        slice_writer.join().unwrap();
        assert_eq!(handle.descriptor().page(), 128);
        assert_eq!(handle.descriptor().slice(), 128);
    }

}
