//! Shared paged MHA/GQA/MLA kernels for packed decode and chunked prefill.
//! No model names, host tensor readback, padded KV gather or per-head KV repeat.
//! Finite Q/K/V inputs are required; non-finite arithmetic propagates normally.
mod kernel;
mod plan;
mod workspace;
pub use plan::HostPlan;
pub use workspace::{SplitWorkspace, MAX_SPLITS, WORKSPACE_LIMIT_BYTES};
use ruda_core::{device::Device, tensor::{DType, Shape}};
use ruda_kernel::{dsl::prelude::*, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype}};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PagedAttentionError(pub &'static str);
impl fmt::Display for PagedAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.0) }
}
impl std::error::Error for PagedAttentionError {}

/// Uploaded immutable metadata, reusable while the schedule is unchanged.
/// Cache data are supplied separately. Creating a new plan is one small upload.
#[derive(Clone)]
pub struct DevicePlan<R: Runtime> { host: HostPlan, metadata: RudaTensor<R> }

fn check_tensor<R: Runtime>(tensor: &RudaTensor<R>, like: &RudaTensor<R>) -> Result<(), PagedAttentionError> {
    if !tensor.is_contiguous() || tensor.qparams.is_some() || tensor.dtype != like.dtype
        || tensor.device.to_id()!=like.device.to_id()
        || !tensor.client.same_execution_queue(&like.client)
        || !matches!(tensor.dtype, DType::F32|DType::F16|DType::BF16)
    { return Err(PagedAttentionError("paged operands require contiguous, same-queue, same-dtype float tensors")); }
    bounded(tensor.meta.shape())?;
    Ok(())
}
fn bounded(shape: &[usize]) -> Result<usize, PagedAttentionError> {
    if shape.iter().any(|&d| d>u32::MAX as usize) { return Err(PagedAttentionError("paged dimension exceeds U32 indexing")); }
    shape.iter().try_fold(1usize, |n,&d| n.checked_mul(d))
        .filter(|&n| n<=u32::MAX as usize)
        .ok_or(PagedAttentionError("paged tensor exceeds U32 indexing"))
}

impl<R: Runtime> DevicePlan<R> {
    pub fn upload(host: HostPlan, like: &RudaTensor<R>) -> Self {
        let metadata=RudaTensor::new_contiguous(
            like.client.clone(), like.device.clone(), Shape::from([host.words.len()]),
            like.client.create_from_slice(bytemuck::cast_slice(&host.words)), DType::U32);
        Self { host, metadata }
    }
    pub fn host(&self) -> &HostPlan { &self.host }

    /// Packed Q [queries,Hq,D], K [pages,page_size,Hkv,D], V [...,Dv].
    /// `causal` uses absolute positions from HostPlan, including prefill chunks.
    pub fn attention(&self, q: RudaTensor<R>, k: RudaTensor<R>, v: RudaTensor<R>,
        scale: f32, causal: bool) -> Result<RudaTensor<R>, PagedAttentionError>
    { self.forward(q,k,v,None,scale,causal,None) }

    /// MLA: absorbed Q [queries,H,R], position Q [queries,H,P]; normalized
    /// latent cache [pages,page_size,1,R], rotated position cache [...,1,P].
    /// Returns compressed context [queries,H,R]. Caller applies value/output
    /// projection only to these current queries, NOT to all historical tokens.
    /// `scale` must come from the original model QK dimensions, not latent rank.
    pub fn mla(&self, q: RudaTensor<R>, qp: RudaTensor<R>, latent: RudaTensor<R>,
        kp: RudaTensor<R>, scale: f32, causal: bool) -> Result<RudaTensor<R>, PagedAttentionError>
    {
        if latent.meta.num_dims()!=4 || latent.meta.shape()[2]!=1 {
            return Err(PagedAttentionError("MLA latent cache must have one shared head"));
        }
        self.forward(q,latent.clone(),latent,Some((qp,kp)),scale,causal,None)
    }
    /// Safe allocating-output wrapper for the split path. `workspace` may be
    /// reused across schedules with matching queries/heads/value dimension and
    /// the same queue. Its lifetime is independent of immutable page metadata.
    pub fn attention_with_workspace(&self, q: RudaTensor<R>, k: RudaTensor<R>, v: RudaTensor<R>,
        scale: f32, causal: bool, workspace: &mut SplitWorkspace<R>) -> Result<RudaTensor<R>, PagedAttentionError>
    { self.forward(q,k,v,None,scale,causal,Some(workspace)) }

    /// Safe split MLA wrapper; latent/position projections remain model-owned.
    pub fn mla_with_workspace(&self, q: RudaTensor<R>, qp: RudaTensor<R>, latent: RudaTensor<R>,
        kp: RudaTensor<R>, scale: f32, causal: bool, workspace: &mut SplitWorkspace<R>)
        -> Result<RudaTensor<R>, PagedAttentionError>
    {
        if latent.meta.num_dims()!=4 || latent.meta.shape()[2]!=1 {
            return Err(PagedAttentionError("MLA latent cache must have one shared head"));
        }
        self.forward(q,latent.clone(),latent,Some((qp,kp)),scale,causal,Some(workspace))
    }

    fn forward(&self, q: RudaTensor<R>, k: RudaTensor<R>, v: RudaTensor<R>,
        position: Option<(RudaTensor<R>,RudaTensor<R>)>, scale: f32, causal: bool,
        workspace: Option<&mut SplitWorkspace<R>>)
        -> Result<RudaTensor<R>, PagedAttentionError>
    {
        if q.meta.num_dims()!=3 || v.meta.num_dims()!=4 {
            return Err(PagedAttentionError("paged Q must be rank 3 and cache rank 4"));
        }
        let shape=Shape::from([q.meta.shape()[0],q.meta.shape()[1],v.meta.shape()[3]]);
        bounded(&shape)?;
        let out=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),shape,q.dtype);
        // Fresh output and a private workspace make the internal launch safe.
        self.execute_attention_into(&q,&k,&v,position.as_ref().map(|(a,b)|(a,b)),&out,scale,causal,workspace)?;
        Ok(out)
    }

    /// Native adapters can write into preallocated output.
    ///
    /// # Safety
    /// Caller must ensure
    /// `out` is disjoint from every input and not concurrently modified.
    /// The PyTorch adapter enforces storage-overlap checks before this call.
    pub unsafe fn attention_into(&self, q: &RudaTensor<R>, k: &RudaTensor<R>, v: &RudaTensor<R>,
        position: Option<(&RudaTensor<R>,&RudaTensor<R>)>, out: &RudaTensor<R>,
        scale: f32, causal: bool) -> Result<(), PagedAttentionError>
    {
        self.execute_attention_into(q,k,v,position,out,scale,causal,None)
    }

    /// Split the visible history over multiple planes and merge FP32 statistics.
    /// Reuse one workspace on the same execution queue; no CPU synchronization
    /// or host readback is introduced. The unsplit method remains the default.
    ///
    /// # Safety
    /// Same disjoint-output contract as `attention_into`. In-flight calls must
    /// use the same ordered queue; a workspace cannot be shared across queues.
    pub unsafe fn attention_into_split(&self, q: &RudaTensor<R>, k: &RudaTensor<R>, v: &RudaTensor<R>,
        position: Option<(&RudaTensor<R>,&RudaTensor<R>)>, out: &RudaTensor<R>,
        scale: f32, causal: bool, workspace: &mut SplitWorkspace<R>) -> Result<(), PagedAttentionError>
    {
        self.execute_attention_into(q,k,v,position,out,scale,causal,Some(workspace))
    }

    fn execute_attention_into(&self, q: &RudaTensor<R>, k: &RudaTensor<R>, v: &RudaTensor<R>,
        position: Option<(&RudaTensor<R>,&RudaTensor<R>)>, out: &RudaTensor<R>,
        scale: f32, causal: bool, workspace: Option<&mut SplitWorkspace<R>>) -> Result<(), PagedAttentionError>
    {
        for t in [q,k,v,out] { check_tensor(t,q)?; }
        if self.metadata.device.to_id()!=q.device.to_id() || !self.metadata.client.same_execution_queue(&q.client) {
            return Err(PagedAttentionError("metadata belongs to another device/queue"));
        }
        if q.meta.num_dims()!=3 || k.meta.num_dims()!=4 || v.meta.num_dims()!=4 {
            return Err(PagedAttentionError("invalid paged tensor ranks"));
        }
        let h=q.meta.shape()[1]; let d=q.meta.shape()[2];
        let kh=k.meta.shape()[2]; let dv=v.meta.shape()[3];
        if h==0 || kh==0 || h%kh!=0 || d==0 || dv==0 || d>1024 || dv>1024
            || q.meta.shape()[0]!=self.host.queries || k.meta.shape()[3]!=d
            || k.meta.shape()[..3]!=v.meta.shape()[..3]
            || k.meta.shape()[0]!=self.host.pages || k.meta.shape()[1]!=self.host.page_size
            || out.meta.shape()[..]!=[self.host.queries,h,dv]
            || !scale.is_finite() || scale<=0.0
        { return Err(PagedAttentionError("incompatible paged head/cache/output shape or scale")); }
        let (qp,kp,p)=if let Some((qp,kp))=position {
            for t in [qp,kp] { check_tensor(t,q)?; }
            if qp.meta.num_dims()!=3 || kp.meta.num_dims()!=4 || kh!=1 || dv!=d
                || qp.meta.shape()[..2]!=q.meta.shape()[..2] || kp.meta.shape()[..3]!=k.meta.shape()[..3]
                || qp.meta.shape()[2]!=kp.meta.shape()[3] || qp.meta.shape()[2]==0 || qp.meta.shape()[2]>256
            { return Err(PagedAttentionError("incompatible MLA positional dimensions")); }
            (qp,kp,qp.meta.shape()[2])
        } else { (q,k,0) };
        let lanes=q.client.properties().hardware.plane_size_max;
        let max=q.client.properties().hardware.max_ruda_count;
        if !(lanes==32 || lanes==64) || self.host.queries>max.0 as usize || h>max.1 as usize {
            return Err(PagedAttentionError("paged kernel requires a full 32/64-lane plane and legal launch grid"));
        }
        let (splits,target,output_dtype)=if let Some(workspace)=workspace {
            if !workspace.matches(q,self.host.queries,h,dv,workspace.splits()) {
                return Err(PagedAttentionError("split workspace shape/device/queue mismatch"));
            }
            if workspace.splits()>max.2 as usize {
                return Err(PagedAttentionError("split count exceeds launch grid Z limit"));
            }
            (workspace.splits(),&workspace.tensor,DType::F32)
        } else { (1,out,q.dtype) };
        if self.host.queries==0 { return Ok(()); }
        kernel::attention::launch::<R>(&q.client,
            RudaCount::Static(self.host.queries as u32,h as u32,splits as u32),RudaDim::new_1d(lanes),
            q.clone().into_array_arg(),k.clone().into_array_arg(),v.clone().into_array_arg(),
            qp.clone().into_array_arg(),kp.clone().into_array_arg(),self.metadata.clone().into_array_arg(),
            target.clone().into_array_arg(),scale,self.host.queries as u32,self.host.sequences as u32,
            self.host.table_width as u32,self.host.page_size as u32,h as u32,kh as u32,
            d,dv,p,lanes as usize,causal,position.is_some(),splits,q.dtype.into(),output_dtype.into());
        if splits>1 {
            kernel::merge::launch::<R>(&q.client,
                RudaCount::Static(self.host.queries as u32,h as u32,1),RudaDim::new_1d(lanes),
                target.clone().into_array_arg(),out.clone().into_array_arg(),h as u32,
                dv,lanes as usize,splits,q.dtype.into());
        }
        Ok(())
    }

    /// Append K/V in place when uniquely owned, otherwise make a device copy
    /// first. This preserves forked-cache isolation; it does not allocate tokens
    /// or implement page ownership/COW for shared-prefix schedulers.
    pub fn append(&self, new_k: RudaTensor<R>, new_v: RudaTensor<R>,
        k: RudaTensor<R>, v: RudaTensor<R>) -> Result<(RudaTensor<R>,RudaTensor<R>),PagedAttentionError>
    {
        self.append_with_report(new_k,new_v,k,v).map(|(k,v,_)| (k,v))
    }

    /// Same operation, with host-side allocation decisions for diagnostics.
    /// The report does not synchronize the GPU and is not a peak-memory metric.
    pub fn append_with_report(&self, new_k: RudaTensor<R>, new_v: RudaTensor<R>,
        k: RudaTensor<R>, v: RudaTensor<R>)
        -> Result<(RudaTensor<R>,RudaTensor<R>,CacheAppendReport),PagedAttentionError>
    {
        self.host.validate_writes()?;
        for t in [&new_k,&new_v,&k,&v] { check_tensor(t,&new_k)?; }
        if k.meta.num_dims()!=4 || v.meta.num_dims()!=4 || new_k.meta.num_dims()!=3 || new_v.meta.num_dims()!=3 {
            return Err(PagedAttentionError("invalid cache append ranks"));
        }
        let heads=k.meta.shape()[2]; let d=k.meta.shape()[3]; let dv=v.meta.shape()[3];
        if heads==0 || d==0 || dv==0 || k.meta.shape()[..3]!=v.meta.shape()[..3]
            || k.meta.shape()[..2]!=[self.host.pages,self.host.page_size]
            || new_k.meta.shape()[..]!=[self.host.queries,heads,d]
            || new_v.meta.shape()[..]!=[self.host.queries,heads,dv]
            || !self.metadata.client.same_execution_queue(&new_k.client)
            || self.metadata.device.to_id()!=new_k.device.to_id()
        { return Err(PagedAttentionError("incompatible cache append shapes/queue")); }
        let size=bounded(&[self.host.queries,heads,d.max(dv)])?;
        // A no-op append must not allocate/copy even when a snapshot exists.
        if size==0 { return Ok((k,v,CacheAppendReport::default())); }
        // Test BOTH handles while all caller aliases are still alive. Adding
        // our own guard clones here would make a uniquely-owned cache appear
        // shared (the pool already holds its own handle), forcing a full copy.
        // If K and V alias, both decisions are false before either is moved.
        let key_mutable=k.can_mut(); let value_mutable=v.can_mut();
        let report=CacheAppendReport {
            key_copied: !key_mutable,
            value_copied: !value_mutable,
            copied_bytes: (if key_mutable { 0 } else { k.meta.num_elements() as u64*k.elem_size() as u64 })
                + (if value_mutable { 0 } else { v.meta.num_elements() as u64*v.elem_size() as u64 }),
        };
        let k=if key_mutable { k } else { k.copy() };
        let v=if value_mutable { v } else { v.copy() };
        if size!=0 {
            let dim=RudaDim::new(new_k.client.properties(),size);
            let count=ruda_kernel::dsl::calculate_ruda_count_elemwise(&new_k.client,size,dim);
            kernel::append::launch::<R>(&new_k.client,
                count,dim,
                new_k.clone().into_array_arg(),new_v.into_array_arg(),k.clone().into_array_arg(),v.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),self.host.queries as u32,self.host.sequences as u32,
                self.host.table_width as u32,self.host.page_size as u32,heads as u32,d as u32,dv as u32,new_k.dtype.into());
        }
        Ok((k,v,report))
    }
}

/// Copy-on-write decisions made before cache-append kernel submission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheAppendReport {
    pub key_copied: bool,
    pub value_copied: bool,
    pub copied_bytes: u64,
}
