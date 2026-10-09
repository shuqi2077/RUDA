use crate::runtime::{
    memory_management::{
        BytesFormat, ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation, MemoryUsage,
        memory_pool::{Slice, calculate_padding},
    },
    server::IoError,
    storage::{StorageHandle, StorageUtilization},
};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use ruda_core::backtrace::BackTrace;

/// A memory page is responsible to reserve [slices](Slice) of data based on a fixed [storage buffer](StorageHandle).
pub struct MemoryPage {
    storage: StorageHandle,
    slices: Vec<Slice>,
    /// This is a vector to be used temporary to store the updated slices.
    ///
    /// It avoids allocating a new vector all the time.
    slices_tmp: Vec<Slice>,
    /// Memory alignment.
    alignment: u64,
    location_base: MemoryLocation,
}

pub(crate) struct RelocationPage {
    pub allocations: Vec<(ManagedMemoryHandle, StorageHandle, u64)>,
    pub live_bytes: u64,
}

impl MemoryPage {
    /// Creates a new memory page with the given storage and memory alignment.
    pub fn new(storage: StorageHandle, alignment: u64, location_base: MemoryLocation) -> Self {
        let mut this = MemoryPage {
            storage: storage.clone(),
            slices: Vec::new(),
            slices_tmp: Vec::new(),
            alignment,
            location_base,
        };

        let slice = Slice::new(storage, 0);
        let slice_pos = this.slices.len() as u32;
        let mut location = this.location_base;
        location.slice = slice_pos;
        slice.handle.descriptor().update_location(location);
        this.slices.push(slice);

        this
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let slice = &mut self.slices[reserved.descriptor().slice()];
        new.descriptor()
            .update_location(reserved.descriptor().location());
        slice.cursor = cursor;
        slice.handle = new;

        Ok(())
    }

    /// Gets the [memory usage](MemoryUsage) of the current memory page.
    pub fn memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for slice in self.slices.iter() {
            usage.bytes_reserved += slice.effective_size();

            if !slice.handle.is_free() {
                usage.number_allocs += 1;
                usage.bytes_in_use += slice.storage.size();
                usage.bytes_padding += slice.padding;
            }
        }

        usage
    }

    /// Gets the [summary](MemoryPageSummary) of the current memory page.
    ///
    /// # Arguments
    ///
    /// - `memory_blocks`: whether the memory block details are included in the summary.
    pub fn summary(&self, memory_blocks: bool) -> MemoryPageSummary {
        let mut summary = MemoryPageSummary::default();

        for slice in self.slices.iter() {
            let is_free = slice.handle.is_free();
            if is_free {
                summary.amount_free += slice.effective_size();
                summary.num_free += 1;
            } else {
                summary.amount_full += slice.effective_size();
                summary.num_full += 1;
            }
            if memory_blocks {
                summary.blocks.push(MemoryBlock {
                    is_free,
                    size: slice.effective_size(),
                });
            }
        }
        summary.amount_total = self.storage.size();
        summary.num_total = self.slices.len();

        summary
    }

    /// Reserves a slice of the given size if there is enough place in the page.
    ///
    /// # Notes
    ///
    /// If the current memory page is fragmented, meaning multiple contiguous slices of data exist,
    /// you can call the [`Self::coalesce()`] function to merge those.
    #[cfg_attr(feature = "runtime-tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        for (index, slice) in self.slices.iter_mut().enumerate() {
            let can_use_slice =
                slice.storage.utilization.size >= effective_size && slice.handle.is_free();

            if !can_use_slice {
                continue;
            }

            let can_be_split = slice.storage.utilization.size > effective_size;
            let handle = slice.handle.clone();
            let storage_old = slice.storage.clone();

            // Updates the current storage utilization.
            slice.storage.utilization.size = size;
            slice.padding = padding;

            if can_be_split {
                let new_slice = Slice::new(storage_old.offset_start(effective_size), 0);
                self.add_new_slice(index, size, new_slice);
            }

            return Some(handle);
        }

        None
    }

    pub(crate) fn can_reserve(&self, size: u64) -> bool {
        let Some(needed) = size.checked_add(calculate_padding(size, self.alignment)) else { return false; };
        let mut free = 0u64;
        for slice in &self.slices {
            free = if slice.is_free() { free.saturating_add(slice.effective_size()) } else { 0 };
            if free >= needed { return true; }
        }
        false
    }

    /// Gets the [storage handle](SliceHandle) with the correct offset and size using the slice
    /// binding.
    ///
    /// If the handle isn't returned, it means the binding isn't present in the given page.
    pub fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        self.find_at(binding.descriptor().location())
    }

    pub(crate) fn find_at(&self, location: MemoryLocation) -> Result<&Slice, IoError> {
        let slice_index = location.slice as usize;

        self.slices
            .get(slice_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {} doesn't exist", slice_index).into(),
            })
    }

    pub fn update_page(&mut self, page: u16) {
        self.location_base.page = page;

        for slice in self.slices.iter() {
            slice.descriptor().update_page(page);
        }
    }

    pub(crate) fn relocation_page(&self) -> Option<RelocationPage> {
        if self.slices.iter().any(|slice| !slice.is_free() && slice.handle.is_pinned()) {
            return None;
        }
        let mut allocations = Vec::new();
        let mut live_bytes = 0;
        for slice in &self.slices {
            if !slice.is_free() {
                live_bytes += slice.storage.size();
                allocations.push((slice.handle.clone(), slice.storage.clone(), slice.cursor));
            }
        }
        (!allocations.is_empty()).then_some(RelocationPage { allocations, live_bytes })
    }

    pub(crate) fn release_relocated(&mut self, allocation: &ManagedMemoryHandle) -> Result<(), IoError> {
        let location = allocation.descriptor().location();
        let slice = self.slices.get_mut(location.slice as usize).ok_or_else(|| IoError::NotFound {
            reason: "Relocation source slice does not exist".into(), backtrace: BackTrace::capture(),
        })?;
        if slice.handle.descriptor() != allocation.descriptor() {
            return Err(IoError::NotFound { reason: "Relocation source identity changed".into(), backtrace: BackTrace::capture() });
        }
        let replacement = ManagedMemoryHandle::new();
        replacement.descriptor().update_location(location);
        slice.handle = replacement;
        slice.cursor = 0;
        Ok(())
    }

    /// Recompute the memory page metadata to make sure adjacent slices are merged together into a
    /// single slice.
    ///
    /// This is necessary to allow bigger slices to be reserved on the current page.
    #[cfg_attr(feature = "runtime-tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn coalesce(&mut self) {
        self.slices_tmp.clear();
        if !self
            .slices
            .windows(2)
            .any(|pair| pair[0].is_free() && pair[1].is_free())
        {
            return;
        }
        let storage = &self.storage;
        let location_base = self.location_base;
        let flush_run = |first: Slice, size: u64, count: usize, output: &mut Vec<Slice>| {
            let slice = if count > 1 {
                let mut storage = storage.clone();
                storage.utilization = StorageUtilization {
                    offset: first.storage.utilization.offset,
                    size,
                };
                Slice::new(storage, 0)
            } else {
                first
            };
            let mut location = location_base;
            location.slice = output.len() as u32;
            slice.descriptor().update_location(location);
            output.push(slice);
        };
        let mut pending = None;
        let mut size = 0;
        let mut count = 0;
        for slice in self.slices.drain(..) {
            if slice.is_free() {
                size += slice.effective_size();
                count += 1;
                if pending.is_none() {
                    pending = Some(slice);
                }
            } else {
                if let Some(first) = pending.take() {
                    flush_run(first, size, count, &mut self.slices_tmp);
                    size = 0;
                    count = 0;
                }
                slice.descriptor().update_slice(self.slices_tmp.len() as u32);
                self.slices_tmp.push(slice);
            }
        }
        if let Some(first) = pending {
            flush_run(first, size, count, &mut self.slices_tmp);
        }

        core::mem::swap(&mut self.slices, &mut self.slices_tmp);
    }

    fn add_new_slice(
        &mut self,
        index_previous: usize,
        reserved_size_previous: u64,
        new_slice: Slice,
    ) {
        self.slices[index_previous].storage.utilization.size = reserved_size_previous;
        let index_new = index_previous + 1;
        let mut location = self.location_base;
        location.slice = index_new as u32;
        new_slice.descriptor().update_location(location);
        self.slices.insert(index_new, new_slice);

        for (index, slice) in self.slices.iter().enumerate().skip(index_new + 1) {
            slice.handle.descriptor().update_slice(index as u32);
        }
    }

    #[cfg(test)]
    fn memory_job(&self) -> MemoryJob {
        let mut job = MemoryJob::default();
        let mut task = MemoryTask::default();

        for (index, slice) in self.slices.iter().enumerate() {
            if slice.handle.is_free() {
                task.size += slice.effective_size();
                task.tag_coalesce(index);
            } else {
                task = job.add(task);
            }
        }
        job.add(task);

        job
    }
}

#[derive(Debug, PartialEq, Eq)]
struct MemoryBlock {
    is_free: bool,
    size: u64,
}

#[derive(Default, PartialEq, Eq)]
pub struct MemoryPageSummary {
    blocks: Vec<MemoryBlock>,
    pub amount_free: u64,
    pub amount_full: u64,
    pub amount_total: u64,
    pub num_free: usize,
    pub num_full: usize,
    pub num_total: usize,
}

impl Display for MemoryBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.is_free {
            true => f.write_fmt(format_args!("Free ({})", BytesFormat::new(self.size))),
            false => f.write_fmt(format_args!("Reserved ({})", BytesFormat::new(self.size))),
        }
    }
}
impl Display for MemoryPageSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

impl Debug for MemoryPageSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n==== Memory Page Summary ====\n")?;
        f.write_str("[Info]\n")?;

        for (tag, num, amount) in [
            ("Free ", self.num_free, self.amount_free),
            ("Full ", self.num_full, self.amount_full),
            ("Total", self.num_total, self.amount_total),
        ] {
            f.write_fmt(format_args!(
                " - {tag}: {} slices ({})\n",
                num,
                BytesFormat::new(amount),
            ))?;
        }

        f.write_str("\n[Blocks]\n")?;
        let mut blocks = String::new();
        for (i, b) in self.blocks.iter().enumerate() {
            if i == 0 {
                blocks += "|";
            }
            blocks += format!(" {b} |").as_str();
        }
        let size = blocks.len();
        for _ in 0..size {
            f.write_str("-")?;
        }
        f.write_str("\n")?;
        f.write_str(&blocks)?;
        f.write_str("\n")?;
        for _ in 0..size {
            f.write_str("-")?;
        }

        f.write_str("\n=============================")?;
        f.write_str("\n")
    }
}

impl Display for MemoryPage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.summary(true)))
    }
}

#[cfg(test)]
#[derive(Default, Debug, PartialEq, Eq)]
struct MemoryJob {
    tasks: Vec<MemoryTask>,
}

#[cfg(test)]
#[derive(Default, Debug, PartialEq, Eq)]
/// The goal of the memory task is to gather contiguous slice indices that can be merged into a single slice.
struct MemoryTask {
    /// The first slice index to be merged.
    start_index: usize,
    /// The number of slices to be merged.
    count: usize,
    /// Which slice index is being merge right now.
    cursor: usize,
    /// The total size in bytes in the resulting merged slice.
    size: u64,
}

#[cfg(test)]
impl MemoryTask {
    /// Tells the task that the given slice index will be coalesced.
    fn tag_coalesce(&mut self, index: usize) {
        if self.count == 0 {
            self.start_index = index;
        }

        debug_assert!(
            self.start_index + self.count == index,
            "Only contiguous index can be coalesced in a single task"
        );

        self.count += 1;
    }
    /// Tells the task that the given slice index is being coalesce.
    fn on_coalesce(&mut self, index: usize) -> MemoryTaskStatus {
        let index_current = self.start_index + self.cursor;

        if index_current == index {
            self.cursor += 1;
            if self.cursor == 1 {
                return MemoryTaskStatus::StartMerging;
            }

            if self.cursor == self.count {
                return MemoryTaskStatus::Completed;
            } else {
                return MemoryTaskStatus::Merging;
            }
        }

        MemoryTaskStatus::Ignoring
    }
}

#[cfg(test)]
impl MemoryJob {
    fn add(&mut self, mut task: MemoryTask) -> MemoryTask {
        // A single index can't be merge with anything.
        if task.count < 2 {
            return MemoryTask::default();
        }

        let mut returned = MemoryTask::default();
        core::mem::swap(&mut task, &mut returned);
        self.tasks.push(returned);
        task
    }
}

#[cfg(test)]
#[derive(Debug)]
enum MemoryTaskStatus {
    Merging,
    StartMerging,
    Ignoring,
    Completed,
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison, clippy::identity_op)]
mod tests {
    use crate::runtime::storage::{StorageId, StorageUtilization};
    use alloc::vec;

    use super::*;

    const MB: u64 = 1024 * 1024;

    #[test_log::test]
    fn test_memory_page() {
        let mut page = new_memory_page(32 * MB);
        let slice = page
            .try_reserve(16 * MB)
            .expect("Enough space to allocate a new slice");

        assert_eq!(slice.is_free(), false);
        assert_eq!(slice.can_mut(), true);

        let storage = &page
            .find(&slice.binding())
            .expect("To find the correct storage")
            .storage;

        assert_eq!(
            storage.utilization,
            StorageUtilization {
                offset: 0,
                size: 16 * MB
            },
            "Utilization to be correct"
        );

        let summary = page.summary(true);

        assert_eq!(
            summary,
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: true,
                        size: 16 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 16 * MB
                    }
                ],
                amount_free: 32 * MB,
                amount_full: 0,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 0,
                num_total: 2
            },
            "Summary is correct before coalesce",
        );
        page.coalesce();
        let summary = page.summary(true);

        assert_eq!(
            summary,
            MemoryPageSummary {
                blocks: vec![MemoryBlock {
                    is_free: true,
                    size: 32 * MB
                },],
                amount_free: 32 * MB,
                amount_full: 0,
                amount_total: 32 * MB,
                num_free: 1,
                num_full: 0,
                num_total: 1
            },
            "Summary is correct after coalesce",
        );
    }

    #[test_log::test]
    fn test_memory_job() {
        let mut page = new_memory_page(32 * MB);
        let slice = page
            .try_reserve(16 * MB)
            .expect("Enough space to allocate a new slice");

        core::mem::drop(slice);
        let job = page.memory_job();

        assert_eq!(
            job,
            MemoryJob {
                tasks: vec![MemoryTask {
                    start_index: 0,
                    count: 2,
                    cursor: 0,
                    size: 32 * MB,
                }]
            }
        );
    }

    #[test_log::test]
    fn test_scenario() {
        let mut page = new_memory_page(32 * MB);

        let slice_1 = page
            .try_reserve(4 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_2 = page
            .try_reserve(15 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_3 = page
            .try_reserve(8 * MB)
            .expect("Enough space to allocate a new slice");
        let slice_4 = page
            .try_reserve(4 * MB)
            .expect("Enough space to allocate a new slice");

        assert_eq!(
            page.summary(true),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 15 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 8 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 1 * MB
                    }
                ],
                amount_free: 1 * MB,
                amount_full: 31 * MB,
                amount_total: 32 * MB,
                num_free: 1,
                num_full: 4,
                num_total: 5
            },
        );

        let slice_5 = page.try_reserve(8 * MB);
        assert!(slice_5.is_none(), "No more place");

        core::mem::drop(slice_2);
        let slice_5 = page.try_reserve(9 * MB);
        assert!(slice_5.is_some(), "Now we have more place");

        let slice_6 = page.try_reserve(9 * MB);
        assert!(slice_6.is_none(), "No more place");

        core::mem::drop(slice_3);
        let slice_6 = page.try_reserve(9 * MB);
        assert!(slice_6.is_none(), "No more place");

        page.coalesce();

        assert_eq!(
            page.summary(true),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 14 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 1 * MB
                    }
                ],
                amount_free: 15 * MB,
                amount_full: 17 * MB,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 3,
                num_total: 5
            },
        );

        assert_eq!(
            page.find(&slice_4.clone().binding())
                .unwrap()
                .storage
                .utilization,
            StorageUtilization {
                offset: 27 * MB,
                size: 4 * MB
            },
            "Utilization to be correct"
        );

        let slice_6 = page.try_reserve(9 * MB);
        assert!(slice_6.is_some(), "Now we have more place");
        core::mem::drop(slice_1);
        core::mem::drop(slice_4);

        page.coalesce();

        assert_eq!(
            page.find(&slice_6.clone().unwrap().binding())
                .unwrap()
                .storage
                .utilization,
            StorageUtilization {
                offset: 13 * MB,
                size: 9 * MB
            },
            "Utilization to be correct"
        );

        assert_eq!(
            page.summary(true),
            MemoryPageSummary {
                blocks: vec![
                    MemoryBlock {
                        is_free: true,
                        size: 4 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: false,
                        size: 9 * MB
                    },
                    MemoryBlock {
                        is_free: true,
                        size: 10 * MB
                    }
                ],
                amount_free: 14 * MB,
                amount_full: 18 * MB,
                amount_total: 32 * MB,
                num_free: 2,
                num_full: 2,
                num_total: 4
            },
        );
    }

    fn new_memory_page(size: u64) -> MemoryPage {
        let storage = StorageHandle::new(StorageId::new(), StorageUtilization { offset: 0, size });

        MemoryPage::new(storage, 4, MemoryLocation::new(0, 0, 0))
    }
}
