/// Minimum total element count for rayon fan-out on per-row or per-chunk loops.
/// Below this, task-dispatch overhead dominates the per-unit work.
pub const PARALLEL_THRESHOLD: usize = 256 * 1024;

/// Wrapper for raw mutable pointers that can be sent across rayon threads.
///
/// # Safety
///
/// The caller must ensure:
/// - The pointer remains valid for the lifetime of all uses
/// - No two threads write to the same offset
/// - No references to the underlying data exist during writes
pub struct SendMutPtr<T>(*mut T);

unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Write `val` at the given element offset.
    ///
    /// # Safety
    /// Offset must be in bounds and no other thread may write to the same offset.
    pub unsafe fn write(&self, offset: usize, val: T) {
        unsafe { self.0.add(offset).write(val) }
    }

    /// Returns the raw pointer offset by `offset` elements.
    ///
    /// # Safety
    /// Offset must be in bounds.
    pub unsafe fn ptr_add(&self, offset: usize) -> *mut T {
        unsafe { self.0.add(offset) }
    }
}

