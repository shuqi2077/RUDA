use crate::runtime::server::IoError;

use super::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use alloc::{
    alloc::{Layout, alloc_zeroed, dealloc},
    sync::Arc,
    vec::Vec,
};
use core::{
    fmt,
    ops::{Deref, DerefMut, Range},
    ptr::NonNull,
};
use hashbrown::HashMap;
use ruda_core::backtrace::BackTrace;
use spin::Mutex;

/// The bytes storage maps IDs to reference-counted, initialized allocations.
/// Removing an ID prevents new lookups; outstanding resources/guards keep the
/// allocation alive until their last owner is dropped.
#[derive(Default)]
pub struct BytesStorage {
    memory: HashMap<StorageId, Arc<AllocatedBytes>>,
}

impl fmt::Debug for BytesStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BytesStorage")
    }
}

/// A checked range of a CPU allocation. Clones share the allocation and its
/// borrow registry, not independently mutable slices.
#[derive(Clone, Debug)]
pub struct BytesResource {
    allocation: Arc<AllocatedBytes>,
    range: Range<usize>,
}

/// Invalid storage lookup or conflicting safe access.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BytesAccessError {
    /// The ID was never allocated or has already been removed from the storage.
    UnknownStorage(StorageId),
    /// Offset/size overflow or a range outside the allocation.
    InvalidRange {
        /// Requested byte offset.
        offset: u64,
        /// Requested byte length.
        size: u64,
        /// Actual allocation length.
        allocation_size: usize,
    },
    /// Overlapping ranges may have multiple readers, but no overlapping writer.
    BorrowConflict {
        /// Requested byte range.
        requested: Range<usize>,
        /// Conflicting live borrow.
        existing: Range<usize>,
    },
}

impl fmt::Display for BytesAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownStorage(id) => write!(f, "unknown or released storage {id}"),
            Self::InvalidRange { offset, size, allocation_size } => write!(
                f, "invalid storage range: offset={offset}, size={size}, allocation={allocation_size}",
            ),
            Self::BorrowConflict { requested, existing } => write!(
                f, "storage range {requested:?} conflicts with live borrow {existing:?}",
            ),
        }
    }
}

impl core::error::Error for BytesAccessError {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BorrowRegion {
    range: Range<usize>,
    writable: bool,
}

struct AllocatedBytes {
    ptr: NonNull<u8>,
    layout: Layout,
    borrows: Mutex<Vec<BorrowRegion>>,
}

impl fmt::Debug for AllocatedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never read allocation contents while formatting a shared resource.
        f.debug_struct("AllocatedBytes")
            .field("size", &self.layout.size())
            .field("alignment", &self.layout.align())
            .finish_non_exhaustive()
    }
}

// SAFETY: The allocation never moves or resizes and is freed only on final Arc
// drop. All safe slice access is mediated by a mutex-protected range registry:
// overlapping readers are allowed, but a writer excludes all overlapping
// borrows. Raw-pointer users must separately uphold the documented FFI contract.
unsafe impl Send for AllocatedBytes {}
// SAFETY: See above. Shared ownership itself does not expose an unguarded slice.
unsafe impl Sync for AllocatedBytes {}

impl Drop for AllocatedBytes {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: This is the last allocation owner. No guard can outlive
            // it, and ptr/layout are the original alloc_zeroed allocation pair.
            unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

/// A live range reservation. Owns the allocation independently of the resource.
/// Dropping it releases the reservation, not necessarily the allocation.
#[derive(Debug)]
struct BorrowLease {
    allocation: Arc<AllocatedBytes>,
    region: BorrowRegion,
}

impl BorrowLease {
    fn acquire(resource: &BytesResource, writable: bool) -> Result<Self, BytesAccessError> {
        let region = BorrowRegion { range: resource.range.clone(), writable };
        let mut borrows = resource.allocation.borrows.lock();
        for existing in borrows.iter() {
            let overlap = !region.range.is_empty()
                && !existing.range.is_empty()
                && region.range.start < existing.range.end
                && existing.range.start < region.range.end;
            if overlap && (region.writable || existing.writable) {
                return Err(BytesAccessError::BorrowConflict {
                    requested: region.range,
                    existing: existing.range.clone(),
                });
            }
        }
        borrows.push(region.clone());
        drop(borrows);
        Ok(Self { allocation: resource.allocation.clone(), region })
    }

    fn ptr(&self) -> *mut u8 {
        // SAFETY: Only checked resources can create a lease. Empty ranges may
        // point one-past-end; zero-sized allocations use an aligned dangling ptr.
        unsafe { self.allocation.ptr.as_ptr().add(self.region.range.start) }
    }

    fn len(&self) -> usize {
        self.region.range.end - self.region.range.start
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: Memory is initialized and the lease owns the allocation.
        // The registry prevents overlapping writers for the lifetime of self.
        unsafe { core::slice::from_raw_parts(self.ptr(), self.len()) }
    }
}

impl Drop for BorrowLease {
    fn drop(&mut self) {
        let mut borrows = self.allocation.borrows.lock();
        // Identical read-only (or empty) reservations are interchangeable;
        // removing exactly one preserves the live-borrow count.
        let index = borrows.iter().position(|region| region == &self.region)
            .expect("live byte lease must have a registered range");
        borrows.swap_remove(index);
    }
}

/// Owned read guard. References obtained through Deref cannot outlive the guard.
#[derive(Debug)]
pub struct BytesReadGuard {
    lease: BorrowLease,
}

impl Deref for BytesReadGuard {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.lease.as_slice()
    }
}

impl AsRef<[u8]> for BytesReadGuard {
    fn as_ref(&self) -> &[u8] { self }
}

/// Owned exclusive range guard. Disjoint ranges can be borrowed independently,
/// even when a memory pool places multiple resources in the same allocation.
#[derive(Debug)]
pub struct BytesWriteGuard {
    lease: BorrowLease,
}

impl Deref for BytesWriteGuard {
    type Target = [u8];

    fn deref(&self) -> &[u8] { self.lease.as_slice() }
}

impl DerefMut for BytesWriteGuard {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: This guard holds the exclusive registry entry for its range;
        // &mut self also prevents simultaneously borrowing this guard twice.
        unsafe { core::slice::from_raw_parts_mut(self.lease.ptr(), self.lease.len()) }
    }
}

impl AsRef<[u8]> for BytesWriteGuard {
    fn as_ref(&self) -> &[u8] { self }
}

impl AsMut<[u8]> for BytesWriteGuard {
    fn as_mut(&mut self) -> &mut [u8] { self }
}

impl BytesResource {
    /// Returns a raw pointer and the checked range length for a kernel/FFI call.
    ///
    /// This does not acquire a safe-access guard. To dereference the pointer,
    /// the caller must keep this resource (or an allocation-owning lease) alive,
    /// enforce initialization/bounds, and synchronize kernel accesses with all
    /// host guards and other raw accesses. Safe host code should use read/write.
    pub fn get_write_ptr_and_length(&self) -> (*mut u8, usize) {
        // SAFETY: The range was checked against the allocation by try_get.
        let ptr = unsafe { self.allocation.ptr.as_ptr().add(self.range.start) };
        (ptr, self.range.end - self.range.start)
    }

    /// Try to acquire an exclusive range without waiting or spinning on readers.
    pub fn try_write(&self) -> Result<BytesWriteGuard, BytesAccessError> {
        Ok(BytesWriteGuard { lease: BorrowLease::acquire(self, true)? })
    }

    /// Acquire an exclusive range, panicking on conflicting live access.
    /// Use try_write to handle conflicts as ordinary errors.
    #[track_caller]
    pub fn write(&self) -> BytesWriteGuard {
        self.try_write().expect("conflicting byte-storage write")
    }

    /// Try to acquire a shared range without waiting for overlapping writers.
    pub fn try_read(&self) -> Result<BytesReadGuard, BytesAccessError> {
        Ok(BytesReadGuard { lease: BorrowLease::acquire(self, false)? })
    }

    /// Acquire a shared range, panicking on conflicting live access.
    /// Use try_read to handle conflicts as ordinary errors.
    #[track_caller]
    pub fn read(&self) -> BytesReadGuard {
        self.try_read().expect("conflicting byte-storage read")
    }
}

impl BytesStorage {
    /// Validate an ID and its byte range before exposing a resource.
    pub fn try_get(&self, handle: &StorageHandle) -> Result<BytesResource, BytesAccessError> {
        let allocation = self.memory.get(&handle.id)
            .ok_or(BytesAccessError::UnknownStorage(handle.id))?;
        let invalid = || BytesAccessError::InvalidRange {
            offset: handle.offset(), size: handle.size(), allocation_size: allocation.layout.size(),
        };
        let end = handle.offset().checked_add(handle.size()).ok_or_else(invalid)?;
        let start = usize::try_from(handle.offset()).map_err(|_| invalid())?;
        let end = usize::try_from(end).map_err(|_| invalid())?;
        if end > allocation.layout.size() {
            return Err(invalid());
        }
        Ok(BytesResource { allocation: allocation.clone(), range: start..end })
    }
}

impl ComputeStorage for BytesStorage {
    type Resource = BytesResource;

    fn alignment(&self) -> usize { 4 }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        self.try_get(handle).expect("invalid byte-storage handle")
    }

    #[cfg_attr(feature = "runtime-tracing", tracing::instrument(level = "trace", skip(self, size)))]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let too_big = || IoError::BufferTooBig { size, backtrace: BackTrace::capture() };
        let size_usize = usize::try_from(size).map_err(|_| too_big())?;
        // Match the alignment promised by ComputeStorage, not Layout<u8>'s 1.
        let layout = Layout::from_size_align(size_usize, self.alignment())
            .map_err(|_| too_big())?;
        let id = StorageId::new();
        let ptr = if size_usize == 0 {
            NonNull::<u32>::dangling().cast::<u8>()
        } else {
            // SAFETY: Layout is valid and non-zero. Null is handled as an error.
            NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or_else(too_big)?
        };
        self.memory.insert(id, Arc::new(AllocatedBytes {
            ptr, layout, borrows: Mutex::new(Vec::new()),
        }));
        Ok(StorageHandle { id, utilization: StorageUtilization { offset: 0, size } })
    }

    #[cfg_attr(feature = "runtime-tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        // Existing guards/resources keep the allocation alive. New lookups fail.
        self.memory.remove(&id);
    }

    fn flush(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_can_alloc_and_dealloc() {
        let mut storage = BytesStorage::default();
        let handle_1 = storage.alloc(64).unwrap();

        assert_eq!(handle_1.size(), 64);
        storage.dealloc(handle_1.id);
    }

    #[test_log::test]
    fn test_slices() {
        let mut storage = BytesStorage::default();
        let handle_1 = storage.alloc(64).unwrap();
        let handle_2 = StorageHandle::new(
            handle_1.id,
            StorageUtilization {
                offset: 24,
                size: 8,
            },
        );

        storage
            .get(&handle_1)
            .write()
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| {
                *b = i as u8;
            });

        let bytes = storage.get(&handle_2).read().to_vec();

        storage.dealloc(handle_1.id);
        assert_eq!(bytes, &[24, 25, 26, 27, 28, 29, 30, 31]);
    }

    /// Miri catches: "reading memory, but memory is uninitialized"
    #[test_log::test]
    fn test_read_after_alloc_without_write() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(16).unwrap();
        let resource = storage.get(&handle);
        assert!(resource.read().iter().all(|&b| b == 0));
        storage.dealloc(handle.id);
    }

    /// Miri catches: "creating allocation with size 0"
    #[test_log::test]
    fn test_zero_size_alloc_and_dealloc() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(0).unwrap();
        assert_eq!(handle.size(), 0);
        storage.dealloc(handle.id);
    }

    #[test_log::test]
    fn test_alloc_dealloc_realloc() {
        let mut storage = BytesStorage::default();
        let h1 = storage.alloc(32).unwrap();
        storage.get(&h1).write()[0] = 0xAA;
        storage.dealloc(h1.id);
        let h2 = storage.alloc(32).unwrap();
        storage.dealloc(h2.id);
    }

    #[test_log::test]
    fn test_multiple_non_overlapping_regions() {
        let mut storage = BytesStorage::default();
        let base = storage.alloc(64).unwrap();

        let regions: alloc::vec::Vec<_> = (0..4)
            .map(|i| {
                StorageHandle::new(
                    base.id,
                    StorageUtilization {
                        offset: i * 16,
                        size: 16,
                    },
                )
            })
            .collect();

        for (i, region) in regions.iter().enumerate() {
            storage.get(region).write().fill(i as u8);
        }
        for (i, region) in regions.iter().enumerate() {
            assert!(storage.get(region).read().iter().all(|&b| b == i as u8));
        }
        storage.dealloc(base.id);
    }

    #[test]
    fn guard_outlives_resource_and_storage() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(4).unwrap();
        let weak = Arc::downgrade(storage.memory.get(&handle.id).unwrap());
        let mut guard = storage.get(&handle).write();
        storage.dealloc(handle.id);
        assert!(matches!(storage.try_get(&handle), Err(BytesAccessError::UnknownStorage(_))));
        drop(storage);
        guard.copy_from_slice(&[10, 20, 30, 40]);
        assert_eq!(&guard[..], &[10, 20, 30, 40]);
        assert!(weak.upgrade().is_some());
        drop(guard);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn dropping_storage_frees_unborrowed_allocations() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(8).unwrap();
        let weak = Arc::downgrade(storage.memory.get(&handle.id).unwrap());
        drop(storage);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn cloned_resources_share_borrow_registry() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(8).unwrap();
        let first = storage.get(&handle);
        let second = first.clone();
        let mut writer = first.write();
        writer[0] = 42;
        assert!(second.try_read().is_err());
        assert!(second.try_write().is_err());
        drop(writer);
        assert_eq!(second.read()[0], 42);
        let a = first.read();
        let b = second.read();
        assert!(first.try_write().is_err());
        drop(a);
        assert!(first.try_write().is_err());
        drop(b);
        assert!(first.try_write().is_ok());
    }

    #[test]
    fn independently_looked_up_overlapping_ranges_conflict() {
        let mut storage = BytesStorage::default();
        let base = storage.alloc(16).unwrap();
        let left = storage.get(&StorageHandle::new(
            base.id, StorageUtilization { offset: 0, size: 8 },
        ));
        let overlap = storage.get(&StorageHandle::new(
            base.id, StorageUtilization { offset: 4, size: 8 },
        ));
        let _writer = left.write();
        assert!(overlap.try_read().is_err());
        assert!(overlap.try_write().is_err());
    }

    #[test]
    fn disjoint_pooled_ranges_can_be_borrowed_together() {
        let mut storage = BytesStorage::default();
        let base = storage.alloc(16).unwrap();
        let left = storage.get(&StorageHandle::new(
            base.id, StorageUtilization { offset: 0, size: 8 },
        ));
        let right = storage.get(&StorageHandle::new(
            base.id, StorageUtilization { offset: 8, size: 8 },
        ));
        let mut a = left.write();
        let mut b = right.write();
        a.fill(1);
        b.fill(2);
        assert_eq!(&a[..], &[1; 8]);
        assert_eq!(&b[..], &[2; 8]);
        drop((a, b));
        let bytes = storage.get(&base).read();
        assert_eq!(&bytes[..8], &[1; 8]);
        assert_eq!(&bytes[8..], &[2; 8]);
    }

    #[test]
    fn forged_ranges_are_rejected_before_pointer_arithmetic() {
        let mut storage = BytesStorage::default();
        let base = storage.alloc(8).unwrap();
        for (offset, size) in [(9, 0), (7, 2), (0, 9), (u64::MAX, 2)] {
            let bad = StorageHandle::new(base.id, StorageUtilization { offset, size });
            assert!(matches!(storage.try_get(&bad), Err(BytesAccessError::InvalidRange { .. })));
        }
        let end = StorageHandle::new(base.id, StorageUtilization { offset: 8, size: 0 });
        assert!(storage.try_get(&end).unwrap().read().is_empty());
    }

    #[test]
    fn empty_ranges_do_not_conflict_with_live_nonempty_ranges() {
        let mut storage = BytesStorage::default();
        let base = storage.alloc(8).unwrap();
        let _writer = storage.get(&base).write();
        let empty = storage.get(&StorageHandle::new(
            base.id, StorageUtilization { offset: 4, size: 0 },
        ));
        let a = empty.write();
        let b = empty.write();
        assert!(a.is_empty() && b.is_empty());
    }

    #[test]
    fn zero_allocations_have_usable_empty_guards_and_correct_alignment() {
        let mut storage = BytesStorage::default();
        for size in [0, 1, 17] {
            let handle = storage.alloc(size).unwrap();
            let resource = storage.get(&handle);
            let (ptr, len) = resource.get_write_ptr_and_length();
            assert_eq!(ptr as usize % storage.alignment(), 0);
            assert_eq!(len, size as usize);
            assert_eq!(resource.read().len(), len);
            storage.dealloc(handle.id);
        }
        assert!(storage.alloc(u64::MAX).is_err());
    }

    #[test]
    #[cfg(feature = "runtime-std")]
    fn active_write_blocks_cross_thread_access() {
        let mut storage = BytesStorage::default();
        let handle = storage.alloc(8).unwrap();
        let first = storage.get(&handle);
        let second = first.clone();
        let writer = first.write();
        std::thread::spawn(move || {
            assert!(second.try_read().is_err());
            assert!(second.try_write().is_err());
        }).join().unwrap();
        drop(writer);
        assert!(first.try_read().is_ok());
    }

}
