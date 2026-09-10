use crate::execution::storage::cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryResource};
use ruda_core::bytes::{AllocationController, AllocationProperty};
use ruda::runtime::memory_management::ManagedMemoryBinding;

/// Controller for managing pinned (page-locked) host memory allocations.
///
/// This struct ensures that the associated memory binding remains alive until
/// explicitly deallocated, allowing the pinned memory to be reused for other memory operations.
pub struct PinnedMemoryManagedAllocController {
    resource: PinnedMemoryResource,
    /// The memory binding, kept alive until deallocation.
    _binding: ManagedMemoryBinding,
}

impl PinnedMemoryManagedAllocController {
    /// Creates a new allocation controller for pinned host memory.
    ///
    /// # Arguments
    ///
    /// * `binding` - The memory binding for the pinned memory.
    /// * `resource` - The pinned memory resource to manage.
    ///
    /// # Returns
    ///
    /// The controller and the corresponding `Allocation`.
    pub fn init(binding: ManagedMemoryBinding, resource: PinnedMemoryResource) -> Self {
        Self {
            _binding: binding,
            resource,
        }
    }

    fn slice_ptr(&self) -> *mut std::mem::MaybeUninit<u8> {
        if self.resource.size == 0 {
            std::ptr::without_provenance_mut(PINNED_MEMORY_ALIGNMENT)
        } else {
            self.resource.ptr.cast()
        }
    }
}

impl AllocationController for PinnedMemoryManagedAllocController {
    fn alloc_align(&self) -> usize {
        PINNED_MEMORY_ALIGNMENT
    }

    unsafe fn memory_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        // SAFETY:
        // - The ptr is valid while the binding is alive.
        // - The resource is allocated with the size of size.
        // - MaybeUninit<u8> has the same layout as u8.
        // - Empty resources use a non-null, allocation-aligned dangling pointer, never dereferenced.
        // - Caller has to promise to only write initialized data to this slice.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.slice_ptr(),
                self.resource.size,
            )
        }
    }

    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        // SAFETY:
        // - The ptr is valid while the binding is alive.
        // - The resource is allocated with the size of size.
        // - MaybeUninit<u8> has the same layout as u8.
        // - Empty resources use a non-null, allocation-aligned dangling pointer, never dereferenced.
        unsafe {
            std::slice::from_raw_parts(
                self.slice_ptr(),
                self.resource.size,
            )
        }
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }
}
