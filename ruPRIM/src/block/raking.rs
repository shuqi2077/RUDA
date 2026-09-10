use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::record::{RudaRecord, RudaRecordShared, RudaReference};

/// Raking-grid geometry with explicit subgroup and shared-bank dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RudaRakingLayout {
    pub shared_elements: usize,
    pub max_raking_threads: usize,
    pub segment_length: usize,
    pub raking_threads: usize,
    pub has_conflicts: bool,
    pub conflict_degree: usize,
    pub segment_padding: bool,
    pub grid_elements: usize,
    pub unguarded: bool,
}

impl RudaRakingLayout {
    pub fn new(threads: usize, subgroup_width: usize, shared_banks: usize) -> Result<Self, &'static str> {
        if threads == 0 || subgroup_width == 0 || shared_banks == 0 { return Err("raking dimensions must be positive"); }
        let max_raking_threads = threads.min(subgroup_width);
        let segment_length = threads.div_ceil(max_raking_threads);
        let raking_threads = threads.div_ceil(segment_length);
        let has_conflicts = shared_banks % segment_length == 0;
        let conflict_degree = if has_conflicts {
            max_raking_threads.checked_mul(segment_length).ok_or("raking size overflow")? / shared_banks
        } else { 1 };
        let segment_padding = segment_length % 2 == 0 && segment_length > 2;
        let stride = segment_length.checked_add(usize::from(segment_padding)).ok_or("raking size overflow")?;
        let grid_elements = raking_threads.checked_mul(stride).ok_or("raking size overflow")?;
        Ok(Self {
            shared_elements: threads, max_raking_threads, segment_length, raking_threads,
            has_conflicts, conflict_degree, segment_padding, grid_elements,
            unguarded: threads % raking_threads == 0,
        })
    }
}

#[ruda]
pub fn placement_index(thread: usize, #[comptime] segment_length: usize, #[comptime] padding: bool) -> usize {
    let mut index = thread;
    if padding { index += thread / segment_length; }
    index
}

#[ruda]
pub fn raking_index(thread: usize, #[comptime] segment_length: usize, #[comptime] padding: bool) -> usize {
    let stride = comptime![segment_length + usize::from(padding)];
    thread * stride
}

#[derive(RudaType)]
pub struct RudaRakingGrid<T: RudaRecord> {
    storage: RudaRecordShared<T>,
    #[ruda(comptime)]
    segment_length: usize,
    #[ruda(comptime)]
    padding: bool,
}

#[ruda]
impl<T: RudaRecord> RudaRakingGrid<T> {
    pub fn new(#[comptime] layout: RudaRakingLayout) -> Self {
        RudaRakingGrid::<T> {
            storage: RudaRecordShared::<T>::new(comptime![layout.grid_elements]),
            segment_length: comptime![layout.segment_length],
            padding: comptime![layout.segment_padding],
        }
    }

    pub fn placement(&self, thread: usize) -> RudaReference<T> {
        let index = placement_index(thread, self.segment_length, self.padding);
        RudaReference::<T>::new(self.storage.address(index))
    }

    /// First record in a raking thread's contiguous, aligned segment.
    pub fn segment(&self, thread: usize) -> RudaReference<T> {
        let index = raking_index(thread, self.segment_length, self.padding);
        RudaReference::<T>::new(self.storage.address(index))
    }

    pub fn item(&self, thread: usize, item: usize) -> RudaReference<T> {
        let index = raking_index(thread, self.segment_length, self.padding) + item;
        RudaReference::<T>::new(self.storage.address(index))
    }
}
