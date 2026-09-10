use super::{RuntimeError, RuntimeErrorKind, TensorDType};
use std::sync::{Arc, Mutex};

/// Geometry of dense K/V pages. Quantized caches require a different layout;
/// accepting an I4 byte count here would hide scale/zero-point storage.
#[derive(Debug, Clone, Copy)]
pub struct KvMemoryGeometry {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
    pub tokens_per_page: usize,
    pub dtype: TensorDType,
}
impl KvMemoryGeometry {
    pub fn bytes_per_page(self) -> Result<u64, RuntimeError> {
        let dimensions = [self.layers, self.kv_heads, self.head_dimension, self.tokens_per_page];
        if dimensions.contains(&0) || !self.dtype.is_float() {
            return Err(RuntimeError::invalid("dense KV geometry requires positive dimensions and a floating dtype"));
        }
        let elements = dimensions.iter().try_fold(2u64, |n, &d| {
            let d = u64::try_from(d).map_err(|_| RuntimeError::invalid("KV dimension exceeds u64"))?;
            n.checked_mul(d).ok_or_else(|| RuntimeError::invalid("KV page size overflow"))
        })?;
        self.dtype.storage_bytes(elements)
    }
    /// `available_bytes` must exclude weights and unrelated allocations.
    /// Workspace and safety headroom are deducted before choosing page count.
    pub fn page_capacity(self, available_bytes: u64, workspace_bytes: u64, headroom_bytes: u64) -> Result<usize, RuntimeError> {
        let page = self.bytes_per_page()?;
        let usable = available_bytes.checked_sub(workspace_bytes)
            .and_then(|n| n.checked_sub(headroom_bytes))
            .ok_or_else(|| RuntimeError::new(RuntimeErrorKind::OutOfMemory, "workspace and headroom exceed available memory"))?;
        let count = usable / page;
        if count == 0 { return Err(RuntimeError::new(RuntimeErrorKind::OutOfMemory, "no complete KV page fits")); }
        usize::try_from(count.min(u32::MAX as u64))
            .map_err(|_| RuntimeError::invalid("KV page capacity exceeds usize"))
    }
}

#[derive(Debug)]
struct BudgetState { limit: u64, reserved: u64, peak: u64 }

/// Thread-safe LOGICAL budget shared by callers. It cannot stop another process
/// from allocating VRAM. A reservation is not itself a device-memory handle.
#[derive(Debug, Clone)]
pub struct MemoryBudget { state: Arc<Mutex<BudgetState>> }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudgetSnapshot { pub limit: u64, pub reserved: u64, pub peak: u64 }

#[derive(Debug)]
#[must_use = "dropping an unused reservation returns the logical budget"]
pub struct MemoryReservation { state: Arc<Mutex<BudgetState>>, bytes: u64 }
impl MemoryReservation { pub const fn bytes(&self) -> u64 { self.bytes } }
impl Drop for MemoryReservation {
    fn drop(&mut self) {
        // No user callbacks run under this lock, so poison can only come from an
        // internal bug. Recover for release without panicking during unwinding.
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.reserved = state.reserved.saturating_sub(self.bytes);
    }
}
impl MemoryBudget {
    pub fn new(capacity_bytes: u64, headroom_bytes: u64) -> Result<Self, RuntimeError> {
        let limit = capacity_bytes.checked_sub(headroom_bytes)
            .filter(|&n| n > 0).ok_or_else(|| RuntimeError::invalid("memory headroom leaves no usable capacity"))?;
        Ok(Self { state: Arc::new(Mutex::new(BudgetState { limit, reserved: 0, peak: 0 })) })
    }
    pub fn try_reserve(&self, bytes: u64) -> Result<MemoryReservation, RuntimeError> {
        if bytes == 0 { return Err(RuntimeError::invalid("memory reservation must be nonzero")); }
        let mut state = self.state.lock().map_err(|_| RuntimeError::internal("memory budget lock poisoned"))?;
        let reserved = state.reserved.checked_add(bytes).filter(|&n| n <= state.limit)
            .ok_or_else(|| RuntimeError::new(RuntimeErrorKind::OutOfMemory, "logical device budget exhausted"))?;
        state.reserved = reserved;
        state.peak = state.peak.max(reserved);
        Ok(MemoryReservation { state: self.state.clone(), bytes })
    }
    pub fn snapshot(&self) -> Result<MemoryBudgetSnapshot, RuntimeError> {
        let state = self.state.lock().map_err(|_| RuntimeError::internal("memory budget lock poisoned"))?;
        Ok(MemoryBudgetSnapshot { limit: state.limit, reserved: state.reserved, peak: state.peak })
    }
}
