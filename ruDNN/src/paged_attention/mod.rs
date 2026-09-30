//! Shared paged MHA/GQA/MLA kernels for packed decode and chunked prefill.
//! No model names, host tensor readback, padded KV gather or per-head KV repeat.
//! Finite Q/K/V inputs are required; non-finite arithmetic propagates normally.
mod kernel;
mod plan;
mod workspace;
mod history_index;
mod ordered_workspace;
mod ordered_kernel;
mod history_row_cache;
mod history_compaction;
mod history_compaction_kernel;
pub use ordered_workspace::OrderedBackwardWorkspace;
pub use plan::HostPlan;
pub use workspace::{SplitWorkspace, MAX_SPLITS, WORKSPACE_LIMIT_BYTES};
use ruda_core::{device::Device, tensor::{DType, Shape}};
use ruda_kernel::{dsl::prelude::*, tensor::{RudaTensor, allocation::empty_device_contiguous_dtype}};
use std::{fmt, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PagedAttentionError(pub &'static str);
impl fmt::Display for PagedAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.0) }
}
impl std::error::Error for PagedAttentionError {}

/// Uploaded immutable metadata, reusable while the schedule is unchanged.
/// Cache data are supplied separately. Creating a new plan is one small upload.
#[derive(Debug)]
pub struct AttentionBackward<R: Runtime> { pub dq:RudaTensor<R>, pub dk:RudaTensor<R>, pub dv:RudaTensor<R> }
#[derive(Debug)]
pub struct MlaBackward<R: Runtime> { pub dq:RudaTensor<R>, pub dqp:RudaTensor<R>, pub dlatent:RudaTensor<R>, pub dkp:RudaTensor<R> }

#[derive(Clone)]
pub struct DevicePlan<R: Runtime> { host: Arc<HostPlan>, metadata: RudaTensor<R> }

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
        Self { host: Arc::new(host), metadata }
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

    /// First-order backward for paged MHA/GQA. Scores/probabilities are
    /// recomputed from Q/K instead of saved during forward, reducing saved
    /// activation memory at the cost of a second history scan in backward.
    pub fn attention_backward(&self, q:RudaTensor<R>, k:RudaTensor<R>, v:RudaTensor<R>,
        grad_out:RudaTensor<R>, scale:f32, causal:bool)
        ->Result<AttentionBackward<R>,PagedAttentionError>
    {
        let dq=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),q.meta.shape().clone(),q.dtype);
        let dk=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),k.meta.shape().clone(),q.dtype);
        let dv=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),v.meta.shape().clone(),q.dtype);
        unsafe { self.attention_backward_into(&q,&k,&v,&grad_out,&dq,&dk,&dv,scale,causal)?; }
        Ok(AttentionBackward{dq,dk,dv})
    }

    /// Write GQA gradients directly into caller-owned buffers. This is used by
    /// native framework adapters to avoid allocating/copying a second gradient set.
    ///
    /// # Safety
    /// Gradient outputs must not alias inputs or each other and must not be
    /// concurrently accessed on another queue.
    pub unsafe fn attention_backward_into(&self, q:&RudaTensor<R>, k:&RudaTensor<R>, v:&RudaTensor<R>,
        grad_out:&RudaTensor<R>, dq:&RudaTensor<R>, dk:&RudaTensor<R>, dv:&RudaTensor<R>,
        scale:f32, causal:bool) ->Result<(),PagedAttentionError>
    {
        self.attention_backward_selected_into(q,k,v,grad_out,Some(dq),Some(dk),Some(dv),scale,causal).map(|_| ())
    }

    /// First-order backward for absorbed MLA. `latent` participates as both K
    /// and V, so its gradient contains score and value contributions. Position
    /// cache/query gradients are returned separately. No expanded per-head
    /// historical K/V tensor is materialized.
    pub fn mla_backward(&self, q:RudaTensor<R>, qp:RudaTensor<R>, latent:RudaTensor<R>,
        kp:RudaTensor<R>, grad_out:RudaTensor<R>, scale:f32, causal:bool)
        ->Result<MlaBackward<R>,PagedAttentionError>
    {
        let dq=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),q.meta.shape().clone(),q.dtype);
        let dqp=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),qp.meta.shape().clone(),q.dtype);
        let dlatent=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),latent.meta.shape().clone(),q.dtype);
        let dkp=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),kp.meta.shape().clone(),q.dtype);
        unsafe { self.mla_backward_into(&q,&qp,&latent,&kp,&grad_out,&dq,&dqp,&dlatent,&dkp,scale,causal)?; }
        Ok(MlaBackward{dq,dqp,dlatent,dkp})
    }

    /// Direct-output MLA backward for framework adapters.
    ///
    /// # Safety
    /// Outputs must be disjoint from all inputs and from one another.
    pub unsafe fn mla_backward_into(&self, q:&RudaTensor<R>, qp:&RudaTensor<R>, latent:&RudaTensor<R>,
        kp:&RudaTensor<R>, grad_out:&RudaTensor<R>, dq:&RudaTensor<R>, dqp:&RudaTensor<R>,
        dlatent:&RudaTensor<R>, dkp:&RudaTensor<R>, scale:f32, causal:bool)
        ->Result<(),PagedAttentionError>
    {
        self.mla_backward_selected_into(q,qp,latent,kp,grad_out,Some(dq),Some(dqp),Some(dlatent),Some(dkp),scale,causal).map(|_| ())
    }

    /// GQA backward, materializing only the requested gradients. `None` means
    /// no output allocation, history workspace or writes for that branch.
    ///
    /// # Safety
    /// Requested outputs must be disjoint from all inputs and each other;
    /// all accesses must be ordered on the plan queue. History gradients use
    /// floating-point atomics and are not bitwise deterministic.
    pub unsafe fn attention_backward_selected_into(&self,
        q:&RudaTensor<R>, k:&RudaTensor<R>, v:&RudaTensor<R>, grad_out:&RudaTensor<R>,
        dq:Option<&RudaTensor<R>>, dk:Option<&RudaTensor<R>>, dv:Option<&RudaTensor<R>>,
        scale:f32, causal:bool) ->Result<BackwardReport,PagedAttentionError>
    {
        self.execute_backward_into(q,k,v,None,grad_out,dq,dk,dv,None,None,scale,causal,None)
    }

    /// Absorbed MLA backward with independently optional Q, Q-position, latent
    /// and K-position gradients. The latent derivative always contains BOTH
    /// key and value contributions when requested.
    ///
    /// # Safety
    /// Same output disjointness/queue requirements as GQA selected backward.
    pub unsafe fn mla_backward_selected_into(&self,
        q:&RudaTensor<R>, qp:&RudaTensor<R>, latent:&RudaTensor<R>, kp:&RudaTensor<R>,
        grad_out:&RudaTensor<R>, dq:Option<&RudaTensor<R>>, dqp:Option<&RudaTensor<R>>,
        dlatent:Option<&RudaTensor<R>>, dkp:Option<&RudaTensor<R>>,
        scale:f32, causal:bool) ->Result<BackwardReport,PagedAttentionError>
    {
        self.execute_backward_into(q,latent,latent,Some((qp,kp)),grad_out,
            dq,dlatent,None,dqp,dkp,scale,causal,None)
    }

    /// Atomic-free selected GQA backward. Private row statistics are reused;
    /// physical history positions have a single writer. Fixed accumulation order
    /// is not a promise of equal bits across devices or with the atomic strategy.
    /// # Safety
    /// Same disjoint-output/ordered-queue contract as selected backward. No input
    /// or output may alias workspace buffers; those handles are kept private.
    pub unsafe fn attention_backward_ordered_into(&self,
        q:&RudaTensor<R>,k:&RudaTensor<R>,v:&RudaTensor<R>,grad_out:&RudaTensor<R>,
        dq:Option<&RudaTensor<R>>,dk:Option<&RudaTensor<R>>,dv:Option<&RudaTensor<R>>,
        scale:f32,causal:bool,workspace:&mut OrderedBackwardWorkspace<R>)
        ->Result<BackwardReport,PagedAttentionError> {
        self.execute_backward_into(q,k,v,None,grad_out,dq,dk,dv,None,None,scale,causal,Some(workspace))
    }
    /// Atomic-free MLA backward; latent includes its key AND value contribution.
    /// # Safety
    /// Same disjoint-output/ordered-queue contract as attention_backward_ordered_into.
    pub unsafe fn mla_backward_ordered_into(&self,
        q:&RudaTensor<R>,qp:&RudaTensor<R>,latent:&RudaTensor<R>,kp:&RudaTensor<R>,grad_out:&RudaTensor<R>,
        dq:Option<&RudaTensor<R>>,dqp:Option<&RudaTensor<R>>,dlatent:Option<&RudaTensor<R>>,dkp:Option<&RudaTensor<R>>,
        scale:f32,causal:bool,workspace:&mut OrderedBackwardWorkspace<R>)
        ->Result<BackwardReport,PagedAttentionError> {
        self.execute_backward_into(q,latent,latent,Some((qp,kp)),grad_out,dq,dlatent,None,dqp,dkp,scale,causal,Some(workspace))
    }

    fn execute_backward_into(&self, q:&RudaTensor<R>, k:&RudaTensor<R>, v:&RudaTensor<R>,
        position:Option<(&RudaTensor<R>,&RudaTensor<R>)>, grad_out:&RudaTensor<R>,
        dq:Option<&RudaTensor<R>>, dk:Option<&RudaTensor<R>>, dv:Option<&RudaTensor<R>>,
        dqp:Option<&RudaTensor<R>>, dkp:Option<&RudaTensor<R>>, scale:f32, causal:bool, ordered:Option<&mut OrderedBackwardWorkspace<R>>)
        ->Result<BackwardReport,PagedAttentionError>
    {
        for t in [q,k,v,grad_out] { check_tensor(t,q)?; }
        if self.metadata.device.to_id()!=q.device.to_id() || !self.metadata.client.same_execution_queue(&q.client) {
            return Err(PagedAttentionError("metadata belongs to another device/queue"));
        }
        if q.meta.num_dims()!=3 || k.meta.num_dims()!=4 || v.meta.num_dims()!=4 || grad_out.meta.num_dims()!=3 {
            return Err(PagedAttentionError("invalid paged backward tensor ranks"));
        }
        let h=q.meta.shape()[1]; let d=q.meta.shape()[2]; let kh=k.meta.shape()[2]; let value_dim=v.meta.shape()[3];
        if h==0 || kh==0 || h%kh!=0 || d==0 || value_dim==0 || d>1024 || value_dim>1024
            || q.meta.shape()[0]!=self.host.queries || k.meta.shape()[3]!=d
            || k.meta.shape()[..3]!=v.meta.shape()[..3]
            || k.meta.shape()[0]!=self.host.pages || k.meta.shape()[1]!=self.host.page_size
            || grad_out.meta.shape()[..]!=[self.host.queries,h,value_dim]
            || !scale.is_finite() || scale<=0.0
        { return Err(PagedAttentionError("incompatible paged backward shape or scale")); }
        let (qp,kp,position_dim,mla)=if let Some((qp,kp))=position {
            for t in [qp,kp] { check_tensor(t,q)?; }
            if qp.meta.num_dims()!=3 || kp.meta.num_dims()!=4 || kh!=1 || value_dim!=d
                || qp.meta.shape()[..2]!=q.meta.shape()[..2] || kp.meta.shape()[..3]!=k.meta.shape()[..3]
                || qp.meta.shape()[2]!=kp.meta.shape()[3] || qp.meta.shape()[2]==0 || qp.meta.shape()[2]>256
                || dv.is_some()
            { return Err(PagedAttentionError("incompatible MLA backward positional/output dimensions")); }
            (qp,kp,qp.meta.shape()[2],true)
        } else {
            if dqp.is_some() || dkp.is_some() { return Err(PagedAttentionError("GQA has no positional gradients")); }
            (q,k,0,false)
        };
        // Validate every supplied output BEFORE any allocation, clear or launch.
        for (dst,src) in [(dq,q),(dk,k),(dv,v),(dqp,qp),(dkp,kp)] {
            if let Some(dst)=dst {
                check_tensor(dst,q)?;
                if dst.meta.shape()!=src.meta.shape() { return Err(PagedAttentionError("paged gradient shape mismatch")); }
            }
        }
        let lanes=q.client.properties().hardware.plane_size_max;
        let max=q.client.properties().hardware.max_ruda_count;
        if !(lanes==32 || lanes==64) || self.host.queries>max.0 as usize || h>max.1 as usize {
            return Err(PagedAttentionError("paged backward requires a full 32/64-lane plane and legal launch grid"));
        }
        let ordered_history=ordered.is_some() && (dk.is_some() || dv.is_some() || dkp.is_some());
        if let Some(workspace)=ordered.as_ref() {
            if !workspace.matches(self,q) { return Err(PagedAttentionError("ordered backward workspace schedule/device/queue mismatch")); }
        }
        let history_slots=bounded(&[self.host.pages,self.host.page_size])?;
        let mut launch_history_slots=history_slots;
        let mut inactive_zero_elements=0usize;
        let mut inactive_zero_width=1usize;
        let mut compact_history=false;
        if ordered_history {
            let workspace=ordered.as_ref().unwrap();
            compact_history=workspace.compact_history && workspace.active_pages<self.host.pages;
            if compact_history {
                launch_history_slots=bounded(&[workspace.active_pages,self.host.page_size])?;
                inactive_zero_width=(if dk.is_some(){d}else{0})
                    .max(if dv.is_some(){value_dim}else{0})
                    .max(if dkp.is_some(){position_dim}else{0});
                inactive_zero_elements=bounded(&[self.host.pages-workspace.active_pages,self.host.page_size,kh,inactive_zero_width])?;
            }
        }
        if ordered_history && (launch_history_slots>max.0 as usize || kh>max.1 as usize) {
            return Err(PagedAttentionError("ordered history grid exceeds device limits"));
        }
        let mut report=BackwardReport::default();
        if [dq,dk,dv,dqp,dkp].iter().all(Option::is_none) { return Ok(report); }
        let clear = |dst:&RudaTensor<R>| ->u64 {
            let size=dst.meta.num_elements();
            if size==0 { return 0; }
            let dim=RudaDim::new(dst.client.properties(),size);
            let count=ruda_kernel::dsl::calculate_ruda_count_elemwise(&dst.client,size,dim);
            kernel::zero_storage::launch::<R>(&dst.client,count,dim,dst.clone().into_array_arg(),dst.dtype.into());
            1
        };
        // An empty query set contributes EXACTLY zero to all history gradients.
        // The old early return exposed uninitialized caller output storage.
        if self.host.queries==0 {
            for dst in [dq,dk,dv,dqp,dkp].into_iter().flatten() { report.kernel_launches+=clear(dst); }
            return Ok(report);
        }
        // For FP32 output the caller buffer is already the accumulation buffer.
        // Lower precision allocates FP32 scratch only for requested history grads.
        let mut history_workspace_bytes=0usize;
        let mut make_history = |dst:Option<&RudaTensor<R>>| ->Option<RudaTensor<R>> {
            dst.filter(|_| !ordered_history).map(|out| {
                if out.dtype==DType::F32 { out.clone() } else {
                    history_workspace_bytes+=out.meta.num_elements()*4;
                    empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),out.meta.shape().clone(),DType::F32)
                }
            })
        };
        let dk32=make_history(dk); let dv32=make_history(dv); let dkp32=make_history(dkp);
        report.history_workspace_bytes=history_workspace_bytes;
        for dst in [dk32.as_ref(),dv32.as_ref(),dkp32.as_ref()].into_iter().flatten() {
            report.kernel_launches+=clear(dst);
        }
        // The DSL requires a descriptor for every static argument. Disabled
        // output branches use small, private placeholders; never alias inputs
        // as mutable outputs. Compile-time guards must make them unreachable.
        let dummy_out=if dq.is_none() || dqp.is_none() || (ordered_history && (dk.is_none() || dv.is_none() || dkp.is_none())) {
            report.placeholder_bytes+=q.elem_size();
            Some(empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),Shape::from([1]),q.dtype))
        } else { None };
        let dummy_history=if dk32.is_none() || dv32.is_none() || dkp32.is_none() {
            report.placeholder_bytes+=4;
            Some(empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),Shape::from([1]),DType::F32))
        } else { None };
        let dq_arg=dq.or(dummy_out.as_ref()).expect("dQ descriptor");
        let dqp_arg=dqp.or(dummy_out.as_ref()).expect("dQpos descriptor");
        let dk_arg=dk32.as_ref().or(dummy_history.as_ref()).expect("dK descriptor");
        let dv_arg=dv32.as_ref().or(dummy_history.as_ref()).expect("dV descriptor");
        let dkp_arg=dkp32.as_ref().or(dummy_history.as_ref()).expect("dKpos descriptor");
        // Every legal GQA/MLA call already has at least one disabled history
        // branch. Reuse its PRIVATE descriptor when statistics are not written;
        // do not add a new allocation to the default atomic path.
        let stats=if ordered_history { &ordered.as_ref().unwrap().statistics }
            else { dummy_history.as_ref().expect("unused statistics descriptor") };
        kernel::backward::launch::<R>(&q.client,RudaCount::Static(self.host.queries as u32,h as u32,1),RudaDim::new_1d(lanes),
            q.clone().into_array_arg(),k.clone().into_array_arg(),v.clone().into_array_arg(),
            qp.clone().into_array_arg(),kp.clone().into_array_arg(),grad_out.clone().into_array_arg(),self.metadata.clone().into_array_arg(),
            dq_arg.clone().into_array_arg(),dqp_arg.clone().into_array_arg(),dk_arg.clone().into_array_arg(),dv_arg.clone().into_array_arg(),dkp_arg.clone().into_array_arg(),stats.clone().into_array_arg(),
            scale,self.host.queries as u32,self.host.sequences as u32,self.host.table_width as u32,self.host.page_size as u32,
            h as u32,kh as u32,d,value_dim,position_dim,lanes as usize,causal,mla,
            dq.is_some(),dk.is_some() && !ordered_history,dv.is_some() && !ordered_history,
            dqp.is_some(),dkp.is_some() && !ordered_history,ordered_history,
            ordered_history && (dk.is_some() || dkp.is_some()),q.dtype.into());
        report.kernel_launches+=1;
        if ordered_history {
            let workspace=ordered.as_ref().unwrap();
            report.ordered_workspace_bytes=workspace.bytes();
            report.ordered_history=true;
            let k_out=dk.or(dummy_out.as_ref()).expect("ordered dK descriptor");
            let v_out=dv.or(dummy_out.as_ref()).expect("ordered dV descriptor");
            let kp_out=dkp.or(dummy_out.as_ref()).expect("ordered dKpos descriptor");
            let (cache_key_slots,cache_value_slots,cache_position_slots)=history_row_cache::slots(
                workspace.history_row_cache,mla,dk.is_some() || dkp.is_some(),
                d,value_dim,position_dim,lanes as usize);
            // Partitioned zero and active kernels write disjoint physical pages.
            // Both use the same execution queue and retained bindings. Never
            // clear an active output after it may have received a contribution.
            let history_pages=if compact_history { workspace.page_partition.as_ref().expect("history page partition") }
                else { &workspace.index }; // compile-time-disabled read descriptor
            if inactive_zero_elements!=0 {
                let dim=RudaDim::new(q.client.properties(),inactive_zero_elements);
                let count=ruda_kernel::dsl::calculate_ruda_count_elemwise(&q.client,inactive_zero_elements,dim);
                history_compaction_kernel::zero_inactive::launch::<R>(&q.client,count,dim,
                    history_pages.clone().into_array_arg(),k_out.clone().into_array_arg(),v_out.clone().into_array_arg(),kp_out.clone().into_array_arg(),
                    workspace.active_pages as u32,self.host.page_size as u32,kh as u32,inactive_zero_elements as u32,
                    d,value_dim,position_dim,inactive_zero_width,dk.is_some(),dv.is_some(),dkp.is_some(),q.dtype.into());
                report.kernel_launches+=1;
            }
            if launch_history_slots!=0 {
            ordered_kernel::history_backward::launch::<R>(&q.client,
                RudaCount::Static(launch_history_slots as u32,kh as u32,1),RudaDim::new_1d(lanes),
                q.clone().into_array_arg(),k.clone().into_array_arg(),v.clone().into_array_arg(),
                qp.clone().into_array_arg(),kp.clone().into_array_arg(),grad_out.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),workspace.index.clone().into_array_arg(),stats.clone().into_array_arg(),history_pages.clone().into_array_arg(),
                k_out.clone().into_array_arg(),v_out.clone().into_array_arg(),kp_out.clone().into_array_arg(),
                scale,self.host.queries as u32,self.host.sequences as u32,self.host.page_size as u32,h as u32,kh as u32,
                d,value_dim,position_dim,lanes as usize,history_index::QUERY_BLOCK_ROWS,workspace.query_pruning,
                workspace.history_row_cache,compact_history,cache_key_slots,cache_value_slots,cache_position_slots,
                causal,mla,dk.is_some(),dv.is_some(),dkp.is_some(),q.dtype.into());
            report.kernel_launches+=1;
            }
        }
        for (src,dst) in [(dk32.as_ref(),dk),(dv32.as_ref(),dv),(dkp32.as_ref(),dkp)] {
            if let (Some(src),Some(dst))=(src,dst) {
                let size=dst.meta.num_elements();
                if dst.dtype==DType::F32 || size==0 { continue; }
                let dim=RudaDim::new(dst.client.properties(),size);
                let count=ruda_kernel::dsl::calculate_ruda_count_elemwise(&dst.client,size,dim);
                kernel::cast_f32::launch::<R>(&dst.client,count,dim,src.clone().into_array_arg(),dst.clone().into_array_arg(),dst.dtype.into());
                report.kernel_launches+=1;
            }
        }
        Ok(report)
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

/// Allocation/launch accounting for this backward call only, not peak VRAM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackwardReport {
    pub kernel_launches:u64,
    pub history_workspace_bytes:usize,
    pub placeholder_bytes:usize,
    pub ordered_workspace_bytes:usize,
    pub ordered_history:bool,
}

#[cfg(test)]
mod tests_query_pruning;

#[cfg(test)]
mod tests_history_row_cache;

#[cfg(test)]
mod tests_history_compaction;
