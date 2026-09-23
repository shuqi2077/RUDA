//! Reusable FP32 partial statistics, private to one ordered execution queue.
use super::{bounded, PagedAttentionError};
use ruda_core::{device::Device, tensor::{DType, Shape}};
use ruda_kernel::{dsl::prelude::Runtime, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype}};

/// Explicit opt-in, not an unmeasured auto-tuning rule.
pub const MAX_SPLITS: usize = 32;
/// Per-workspace guard; not a global GPU memory budget.
pub const WORKSPACE_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// No Clone and no exposed tensor: mutable access serializes reuse in safe Rust.
/// The runtime retains submitted bindings until their stream-ordered work ends.
pub struct SplitWorkspace<R: Runtime> {
    pub(super) tensor: RudaTensor<R>,
    splits: usize,
    bytes: usize,
}

impl<R: Runtime> SplitWorkspace<R> {
    pub fn required_bytes(queries: usize, heads: usize, value_dim: usize, splits: usize)
        -> Result<usize, PagedAttentionError>
    {
        if !(2..=MAX_SPLITS).contains(&splits) || heads==0 || value_dim==0 || value_dim>1024 {
            return Err(PagedAttentionError("split workspace requires 2..32 splits and valid heads/value dimension"));
        }
        let elements=bounded(&[queries,heads,splits,value_dim+2])?;
        elements.checked_mul(4).filter(|&b|b<=WORKSPACE_LIMIT_BYTES)
            .ok_or(PagedAttentionError("split workspace exceeds 64 MiB limit"))
    }

    pub fn new(like: &RudaTensor<R>, queries: usize, heads: usize, value_dim: usize, splits: usize)
        -> Result<Self, PagedAttentionError>
    {
        let bytes=Self::required_bytes(queries,heads,value_dim,splits)?;
        let max=like.client.properties().hardware.max_ruda_count;
        if queries>max.0 as usize || heads>max.1 as usize || splits>max.2 as usize {
            return Err(PagedAttentionError("split workspace exceeds the device launch grid"));
        }
        let tensor=empty_device_contiguous_dtype(like.client.clone(),like.device.clone(),
            Shape::from([queries,heads,splits,value_dim+2]),DType::F32);
        Ok(Self{tensor,splits,bytes})
    }
    pub fn bytes(&self) -> usize { self.bytes }
    pub fn splits(&self) -> usize { self.splits }
    pub fn matches(&self, like: &RudaTensor<R>, queries: usize, heads: usize, value_dim: usize, splits: usize) -> bool {
        self.splits==splits && self.tensor.meta.shape()[..]==[queries,heads,splits,value_dim+2]
            && self.tensor.device.to_id()==like.device.to_id()
            && self.tensor.client.same_execution_queue(&like.client)
    }
}
