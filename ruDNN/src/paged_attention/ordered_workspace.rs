//! Reusable statistics and inverse metadata for atomic-free history backward.
use std::sync::Arc;
use super::{bounded, history_index, DevicePlan, HostPlan, PagedAttentionError, WORKSPACE_LIMIT_BYTES};
use ruda_core::{device::Device, tensor::{DType, Shape}};
use ruda_kernel::{dsl::prelude::Runtime, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype}};

/// Private buffers, one immutable schedule and one ordered execution queue.
/// Keep this across backward calls. No tensor handles escape; use mutable access
/// for submissions. The runtime must retain submitted bindings until completion.
pub struct OrderedBackwardWorkspace<R: Runtime> {
    pub(super) statistics: RudaTensor<R>,
    pub(super) index: RudaTensor<R>,
    host: Arc<HostPlan>,
    heads: usize,
    bytes: usize,
    pub(super) query_pruning: bool,
    pub(super) history_row_cache: bool,
    pub(super) compact_history: bool,
    pub(super) page_partition: Option<RudaTensor<R>>,
    pub(super) active_pages: usize,
}
impl<R: Runtime> OrderedBackwardWorkspace<R> {
    pub fn new(plan: &DevicePlan<R>, like: &RudaTensor<R>) -> Result<Self, PagedAttentionError> {
        if like.meta.num_dims()!=3 || like.meta.shape()[0]!=plan.host.queries || like.meta.shape()[1]==0
            || like.device.to_id()!=plan.metadata.device.to_id()
            || !like.client.same_execution_queue(&plan.metadata.client) {
            return Err(PagedAttentionError("ordered workspace query shape/device/queue mismatch"));
        }
        // Parse before index upload or GPU allocation. Never silently accept a
        // misspelled opt-in value, and never inspect env on every backward.
        let history_row_cache=super::history_row_cache::from_env()?;
        let compact_history=super::history_compaction::from_env()?;
        let heads=like.meta.shape()[1];
        let elements=bounded(&[plan.host.queries,heads,3])?.max(1);
        let stat_bytes=elements.checked_mul(4).filter(|&n| n<=WORKSPACE_LIMIT_BYTES)
            .ok_or(PagedAttentionError("ordered statistics exceed workspace budget"))?;
        let words=history_index::build(&plan.host,(WORKSPACE_LIMIT_BYTES-stat_bytes)/4)?;
        let mut bytes=stat_bytes+words.len()*4;
        let partition=if compact_history {
            Some(super::history_compaction::build(&plan.host,(WORKSPACE_LIMIT_BYTES-bytes)/4)?)
        } else { None };
        let (page_partition,active_pages)=if let Some(partition)=partition {
            bytes+=partition.pages.len()*4;
            let map=RudaTensor::new_contiguous(like.client.clone(),like.device.clone(),Shape::from([partition.pages.len()]),
                like.client.create_from_slice(bytemuck::cast_slice(&partition.pages)),DType::U32);
            (Some(map),partition.active)
        } else { (None,plan.host.pages) };
        let index=RudaTensor::new_contiguous(like.client.clone(),like.device.clone(),Shape::from([words.len()]),
            like.client.create_from_slice(bytemuck::cast_slice(&words)),DType::U32);
        let statistics=empty_device_contiguous_dtype(like.client.clone(),like.device.clone(),Shape::from([elements]),DType::F32);
        Ok(Self{statistics,index,host:plan.host.clone(),heads,bytes,query_pruning:true,history_row_cache,
            compact_history,page_partition,active_pages})
    }
    /// Skip only queries proven causally invisible. Original summation order is
    /// retained. Enabled for ordered backward; disable to benchmark the same
    /// kernel and buffers without pruning. No upload/allocation is performed.
    pub fn set_query_pruning(&mut self, enabled: bool) { self.query_pruning = enabled; }
    pub fn query_pruning(&self) -> bool { self.query_pruning }
    /// Cache this physical row's K/V/position values in thread-local storage.
    /// Disabled unless explicitly selected at construction or by this setter.
    /// No extra device tensor, upload, or synchronization. May increase register
    /// pressure/spills; measure on the target GPU before making it a default.
    pub fn set_history_row_cache(&mut self, enabled: bool) { self.history_row_cache=enabled; }
    pub fn history_row_cache(&self) -> bool { self.history_row_cache }
    /// Partition active/inactive physical pages once, then cache the immutable
    /// device index. Enabling the first time uploads `physical_pages * 4` bytes;
    /// toggling thereafter does not allocate, upload, or synchronize. Disabling
    /// retains that index until workspace drop; bytes() includes retained data.
    /// Budget errors leave both the current mode and prior workspace unchanged.
    pub fn set_history_compaction(&mut self, enabled: bool) -> Result<(), PagedAttentionError> {
        if enabled && self.page_partition.is_none() {
            let partition=super::history_compaction::build(&self.host,(WORKSPACE_LIMIT_BYTES-self.bytes)/4)?;
            let map=RudaTensor::new_contiguous(self.index.client.clone(),self.index.device.clone(),
                Shape::from([partition.pages.len()]),
                self.index.client.create_from_slice(bytemuck::cast_slice(&partition.pages)),DType::U32);
            self.bytes+=partition.pages.len()*4;
            self.active_pages=partition.active;
            self.page_partition=Some(map);
        }
        self.compact_history=enabled;
        Ok(())
    }
    pub fn history_compaction(&self)->bool { self.compact_history }
    /// Schedule-level physical page counts, NOT nonzero gradient counts. Causal
    /// tails inside active pages are still written by the original kernel.
    pub fn history_compaction_pages(&self)->Option<(usize,usize)> {
        self.page_partition.as_ref().map(|_| (self.active_pages,self.host.pages-self.active_pages))
    }
    pub fn bytes(&self)->usize { self.bytes }
    pub fn statistics_bytes(&self)->usize { self.statistics.meta.num_elements()*4 }
    pub fn matches(&self,plan:&DevicePlan<R>,like:&RudaTensor<R>)->bool {
        like.meta.num_dims()==3 && like.meta.shape()[0]==self.host.queries
            && like.meta.shape()[1]==self.heads
            // Repeated calls on this immutable schedule use identity, not an
            // O(metadata size) vector comparison. Distinct equivalent plans
            // retain the v33 structural compatibility check.
            && (Arc::ptr_eq(&self.host, &plan.host) || self.host==plan.host)
            && self.index.device.to_id()==like.device.to_id()
            && self.index.client.same_execution_queue(&like.client)
    }
}
