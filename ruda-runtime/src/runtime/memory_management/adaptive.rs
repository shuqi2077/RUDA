use super::{
    ManagedMemoryHandle, MemoryPoolOptions, PoolType,
    memory_manage::DynamicPool,
    memory_pool::{MemoryPool, RelocationInput, SlicedPool, calculate_padding},
};
use crate::runtime::{server::IoError, storage::{ComputeStorage, StorageHandle}};
use alloc::{format, vec, vec::Vec};
use ruda_core::{backtrace::BackTrace, ir::MemoryDeviceProperties};

const MIB: u64 = 1024 * 1024;
const SMALL_SLICE: u64 = 64 * 1024;
const ARENA_START: usize = 2;
const ARENA_SLOTS: usize = 64;

pub(super) struct AdaptiveState {
    alignment: u64,
    max_page_size: u64,
    min_page_size: u64,
    small_page_size: u64,
    current: usize,
    outdated: Vec<usize>,
    stalled: Vec<RelocationInput>,
    stalled_valid: bool,
}

struct Relocation {
    allocation: ManagedMemoryHandle,
    target: ManagedMemoryHandle,
    source: StorageHandle,
    destination: StorageHandle,
    cursor: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetRoom {
    Held,
    MayAllocate,
}

impl AdaptiveState {
    pub(super) fn new(properties: &MemoryDeviceProperties) -> Self {
        let alignment = properties.alignment;
        assert!(alignment != 0, "Memory alignment must be nonzero");
        let max_page_size = properties.max_page_size / alignment * alignment;
        assert!(max_page_size != 0, "Device page limit must fit its alignment");
        let aligned = |size: u64| size.max(alignment).next_multiple_of(alignment).min(max_page_size);
        Self {
            alignment,
            max_page_size,
            min_page_size: aligned(2 * MIB),
            small_page_size: aligned(8 * MIB),
            current: ARENA_START,
            outdated: Vec::new(),
            stalled: Vec::new(),
            stalled_valid: false,
        }
    }

    pub(super) fn pool_options(&self) -> Vec<MemoryPoolOptions> {
        vec![
            MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 0 }, dealloc_period: None },
            MemoryPoolOptions { pool_type: PoolType::SlicedPages {
                page_size: self.small_page_size, max_slice_size: SMALL_SLICE.min(self.small_page_size),
            }, dealloc_period: None },
            MemoryPoolOptions { pool_type: PoolType::SlicedPages {
                page_size: self.min_page_size, max_slice_size: self.min_page_size,
            }, dealloc_period: None },
        ]
    }

    fn pool<'a>(&self, pools: &'a [DynamicPool]) -> &'a SlicedPool {
        match &pools[self.current] { DynamicPool::Sliced(pool) => pool, _ => unreachable!() }
    }

    fn metadata_pool(&self, size: u64) -> Option<usize> {
        if size == 0 { Some(0) }
        else if size <= SMALL_SLICE.min(self.small_page_size) { Some(1) }
        else { None }
    }

    pub(super) fn pending(&self, pools: &[DynamicPool], size: u64) -> bool {
        if self.metadata_pool(size).is_some() { return false; }
        let pool = self.pool(pools);
        let Some(needed) = size.checked_add(calculate_padding(size, self.alignment)) else { return false; };
        needed <= self.max_page_size &&
            (needed > pool.page_size() || (!self.outdated.is_empty() && !pool.can_reserve(size)))
    }

    pub(super) fn reserve<Storage: ComputeStorage>(
        &mut self, pools: &mut Vec<DynamicPool>, storage: &mut Storage, size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        if !storage.supports_relocation() {
            return Err(IoError::UnsupportedIoOperation { backtrace: BackTrace::capture() });
        }
        let needed = size.checked_add(calculate_padding(size, self.alignment))
            .filter(|&needed| needed <= self.max_page_size)
            .ok_or_else(|| IoError::BufferTooBig { size, backtrace: BackTrace::capture() })?;
        if let Some(index) = self.metadata_pool(size) {
            if let Some(handle) = pools[index].try_reserve(size) { return Ok(handle); }
            return pools[index].alloc(storage, size);
        }

        if needed > self.pool(pools).page_size() {
            self.cleanup_outdated(pools, storage);
            let page_size = size.saturating_add(MIB).checked_next_multiple_of(MIB)
                .and_then(|size| size.checked_next_multiple_of(self.alignment))
                .unwrap_or(self.max_page_size).clamp(self.min_page_size, self.max_page_size);
            self.relieve_pressure(pools, storage, page_size)?;
            // Empty slots can be reused: no live descriptor can still name them.
            let vacant = (ARENA_START..pools.len()).find(|&index| {
                matches!(&pools[index], DynamicPool::Sliced(pool) if pool.is_empty())
            });
            let index = match vacant {
                Some(index) => index,
                None if pools.len() < ARENA_START + ARENA_SLOTS => pools.len(),
                None => {
                    self.relocate(pools, storage, TargetRoom::MayAllocate)?;
                    (ARENA_START..pools.len()).find(|&index| {
                        matches!(&pools[index], DynamicPool::Sliced(pool) if pool.is_empty())
                    }).ok_or_else(|| IoError::Unknown {
                        description: "Adaptive page-size slots are all occupied by live allocations".into(),
                        backtrace: BackTrace::capture(),
                    })?
                }
            };
            let mut pool = DynamicPool::Sliced(SlicedPool::new(page_size, page_size, self.alignment, index as u8));
            let allocated = match pool.alloc(storage, size) {
                Ok(handle) => handle,
                Err(error) => {
                    if !self.relocate(pools, storage, TargetRoom::Held)? { return Err(error); }
                    pool.alloc(storage, size)?
                }
            };
            if index == pools.len() { pools.push(pool); } else { pools[index] = pool; }
            if index != self.current { self.outdated.push(self.current); }
            self.current = index;
            self.relocate(pools, storage, TargetRoom::Held)?;
            return Ok(allocated);
        }

        if let Some(handle) = pools[self.current].try_reserve(size) { return Ok(handle); }
        let page_size = self.pool(pools).page_size();
        if self.relieve_pressure(pools, storage, page_size)? {
            if let Some(handle) = pools[self.current].try_reserve(size) { return Ok(handle); }
        }
        // Pressure/OOM recovery only uses room already held. An OOM never allocates a second
        // page just to attempt to recover the first reservation.
        let allocated = pools[self.current].alloc(storage, size);
        let reclaimed = self.relocate(pools, storage, TargetRoom::Held)?;
        match allocated {
            Ok(handle) => Ok(handle),
            Err(error) => match pools[self.current].try_reserve(size) {
                Some(handle) => Ok(handle),
                None if reclaimed => pools[self.current].alloc(storage, size),
                None => Err(error),
            },
        }
    }

    fn cleanup_outdated<Storage: ComputeStorage>(&mut self, pools: &mut [DynamicPool], storage: &mut Storage) {
        for &index in &self.outdated { pools[index].cleanup(storage, 0, true); }
        self.outdated.retain(|&index| !matches!(&pools[index], DynamicPool::Sliced(pool) if pool.is_empty()));
    }

    pub(super) fn compact<Storage: ComputeStorage>(
        &mut self, pools: &mut [DynamicPool], storage: &mut Storage,
    ) -> Result<(), IoError> {
        self.cleanup_outdated(pools, storage);
        self.relocate(pools, storage, TargetRoom::Held)?;
        Ok(())
    }

    fn outdated_pages(&self, pools: &[DynamicPool]) -> usize {
        self.outdated.iter().map(|&index| match &pools[index] {
            DynamicPool::Sliced(pool) => pool.page_count(), _ => unreachable!(),
        }).sum()
    }

    fn relieve_pressure<Storage: ComputeStorage>(
        &mut self, pools: &mut [DynamicPool], storage: &mut Storage, page_size: u64,
    ) -> Result<bool, IoError> {
        if self.outdated.is_empty() { return Ok(false); }
        if storage.available_memory().is_some_and(|free| free.saturating_sub(page_size) < page_size) {
            if self.stalled_valid && storage.supports_relocation() && self.stalled_matches(pools) {
                return Ok(false);
            }
            self.stalled_valid = false;
            let mut inputs = core::mem::take(&mut self.stalled);
            inputs.clear();
            inputs.extend(self.relocation_inputs(pools));
            self.stalled = inputs;
            let cacheable = !self.stalled.iter().any(|input| {
                matches!(input, RelocationInput::Allocation { pin_revision: usize::MAX, .. })
            });
            let reclaimed = self.relocate(pools, storage, TargetRoom::Held)?;
            self.stalled_valid = !reclaimed && cacheable && self.stalled_matches(pools);
            return Ok(reclaimed);
        }
        Ok(false)
    }

    fn relocation_inputs<'a>(&'a self, pools: &'a [DynamicPool]) -> impl Iterator<Item = RelocationInput> + 'a {
        core::iter::once(RelocationInput::Pool(self.current))
            .chain(self.pool(pools).relocation_inputs(false))
            .chain(self.outdated.iter().flat_map(move |&index| {
                let pool = match &pools[index] { DynamicPool::Sliced(pool) => pool, _ => unreachable!() };
                core::iter::once(RelocationInput::Pool(index)).chain(pool.relocation_inputs(true))
            }))
    }

    fn stalled_matches(&self, pools: &[DynamicPool]) -> bool {
        self.stalled.iter().copied().eq(self.relocation_inputs(pools))
    }

    fn relocate<Storage: ComputeStorage>(
        &mut self, pools: &mut [DynamicPool], storage: &mut Storage, room: TargetRoom,
    ) -> Result<bool, IoError> {
        let result = self.relocate_inner(pools, storage, room);
        if room == TargetRoom::MayAllocate {
            pools[self.current].cleanup(storage, 0, true);
        }
        if !matches!(&result, Ok(false)) { self.stalled_valid = false; }
        result
    }

    fn relocate_inner<Storage: ComputeStorage>(
        &mut self, pools: &mut [DynamicPool], storage: &mut Storage, room: TargetRoom,
    ) -> Result<bool, IoError> {
        if self.outdated.is_empty() { return Ok(false); }
        if !storage.supports_relocation() {
            return Err(IoError::UnsupportedIoOperation { backtrace: BackTrace::capture() });
        }
        let pages_before = self.outdated_pages(pools);
        let mut pages = Vec::new();
        for &index in &self.outdated {
            match &pools[index] {
                DynamicPool::Sliced(pool) => pool.relocation_pages(&mut pages), _ => unreachable!(),
            }
        }
        pages.sort_by_key(|page| page.live_bytes);
        let mut moves = Vec::new();
        let mut allocation_error = None;
        for page in pages {
            let start = moves.len();
            for (allocation, source, cursor) in page.allocations {
                let target = match (pools[self.current].try_reserve(source.size()), room) {
                    (Some(target), _) => Some(target),
                    (None, TargetRoom::Held) => None,
                    (None, TargetRoom::MayAllocate) => match pools[self.current].alloc(storage, source.size()) {
                        Ok(target) => Some(target),
                        Err(error) => { allocation_error = Some(error); None }
                    },
                };
                let Some(target) = target else {
                    moves.truncate(start);
                    break;
                };
                let destination = pools[self.current].find_at(target.descriptor().location())?.storage.clone();
                moves.push(Relocation { allocation, target, source, destination, cursor });
            }
        }
        if moves.is_empty() {
            self.cleanup_outdated(pools, storage);
            let reclaimed = self.outdated_pages(pools) < pages_before;
            return match allocation_error {
                Some(error) if !reclaimed => Err(error),
                _ => Ok(reclaimed),
            };
        }
        storage.relocation_barrier()?;
        let copied = storage.relocation_copy_batch(moves.iter().map(|relocation| {
            (&relocation.source, &relocation.destination)
        }));
        // A failed enqueue can leave earlier copies active. No target may be
        // reused unless the completion wait proves those writes have finished.
        if let Err(error) = storage.relocation_complete() {
            core::mem::forget(moves);
            return Err(error);
        }
        copied?;
        for relocation in moves {
            let source_pool = relocation.allocation.descriptor().location().pool as usize;
            match &mut pools[source_pool] {
                DynamicPool::Sliced(pool) => pool.release_relocated(&relocation.allocation)?,
                _ => unreachable!(),
            }
            pools[self.current].bind(relocation.target, relocation.allocation, relocation.cursor)?;
        }
        self.cleanup_outdated(pools, storage);
        storage.flush();
        Ok(self.outdated_pages(pools) < pages_before)
    }
}
