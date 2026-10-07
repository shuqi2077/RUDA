use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec,string::ToString};
use core::fmt;
use ruda_model::{module::{AutodiffModule,ModuleVisitor,Param,ParamId},record::{Record,PrecisionSettings,RecorderError},
    tensor::{Tensor,DType,FloatDType,TensorMetadata,BroadcastTensorCollective,backend::{Backend,AutodiffBackend}}};
use serde::{Serialize,Deserialize};
use super::{GradientsAccumulator,GradientsParams,GradientsParamsRecord,GradientTransformError,FullyShardedOptimizerParameter};
use super::gradient_transform::{validate_work_dtype,representable};

pub(super) type Placement=Vec<(u64,Vec<usize>,u32,u32,DType,bool)>;

mod weighted;
pub use weighted::*;
mod reshard;
mod norm;
pub use norm::*;

/// Exact continuation counters for globally summed, already reduce-scattered local gradients.
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct FullyShardedAccumulationState {
    /// Explicit F32/F64 arithmetic for native local gradient addition and normalization.
    pub dtype:DType,
    /// Explicit fixed multiplier applied to every loss SUM in this window.
    pub loss_scale:f64,
    /// Exact sum of actual global supervised counts, not a sum of per-rank batch means.
    pub global_count:u64,
    /// Actual issued forward/backward windows, including zero-supervision windows.
    pub microbatches:u64,
}
impl<B:Backend> Record<B> for FullyShardedAccumulationState {
    type Item<S:PrecisionSettings>=Self;
    fn into_item<S:PrecisionSettings>(self) -> Self {self}
    fn from_item<S:PrecisionSettings>(item:Self,_device:&B::Device) -> Self {item}
}

/// Logical ownership and exact normalization continuation, without a second copy of pending gradients.
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct FullyShardedAccumulationContract {
    state:FullyShardedAccumulationState,
    placement:Placement,
}
impl<B:Backend> Record<B> for FullyShardedAccumulationContract {
    type Item<S:PrecisionSettings>=Self;
    fn into_item<S:PrecisionSettings>(self) -> Self {self}
    fn from_item<S:PrecisionSettings>(item:Self,_device:&B::Device) -> Self {item}
}
impl FullyShardedAccumulationContract {
    /// Exact saved scale/count/precision and actual issued-window count.
    pub fn state(&self) -> &FullyShardedAccumulationState {&self.state}
    /// Match the actual local module after native model-state restoration, without reading values.
    pub fn validate_for<B:AutodiffBackend,M:AutodiffModule<B>>(&self,module:&M) -> Result<(),FullyShardedAccumulationError> {
        work_dtype(&self.state)?;inspect::<B,M>(module,&self.placement,true)?;
        if self.state.microbatches==0 && self.state.global_count!=0 {return Err(FullyShardedAccumulationError::State);}Ok(())
    }
}

/// Pending native local gradients together with their exact normalization and logical shard ownership.
#[derive(Clone,Debug)]
pub struct FullyShardedGradientsRecord {
    version:u32,
    gradients:GradientsParamsRecord,
    state:FullyShardedAccumulationState,
    placement:Placement,
}
impl<B:Backend> Record<B> for FullyShardedGradientsRecord {
    type Item<S:PrecisionSettings>=(u32,GradientsParamsRecord,FullyShardedAccumulationState,Placement);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,<GradientsParamsRecord as Record<B>>::into_item::<S>(self.gradients),self.state,self.placement)
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,gradients:<GradientsParamsRecord as Record<B>>::from_item::<S>(item.1,device),state:item.2,placement:item.3}
    }
}

/// Actual argument, continuation or local-shard geometry error; no optimizer update is made here.
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum FullyShardedAccumulationError {
    /// Existing module-gradient/scalar precision contract failed.
    Gradient(GradientTransformError),
    /// Actual parameter IDs, rank/world, logical axes or local vector storage differ.
    Placement(&'static str),
    /// Loss must be an actual differentiable floating scalar on the caller's native backend.
    Loss,
    /// Exact global-count or microbatch-count addition overflowed.
    CounterOverflow,
    /// No actual backward window has been issued.
    EmptyWindow,
    /// Saved precision/options/counters or gradient state are inconsistent.
    State,
}
impl From<GradientTransformError> for FullyShardedAccumulationError {
    fn from(value:GradientTransformError) -> Self {Self::Gradient(value)}
}
impl fmt::Display for FullyShardedAccumulationError {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Gradient(value)=>fmt::Display::fmt(value,f),Self::Placement(value)=>write!(f,"FSDP gradient placement: {value}"),
            Self::Loss=>f.write_str("FSDP backward needs the actual differentiable scalar loss SUM"),
            Self::CounterOverflow=>f.write_str("exact FSDP accumulation counter overflow"),Self::EmptyWindow=>f.write_str("no actual FSDP backward window"),
            Self::State=>f.write_str("inconsistent FSDP pending-gradient checkpoint")}
    }
}
impl core::error::Error for FullyShardedAccumulationError {}

/// Completed original local shard gradients and the precise window that produced them.
pub struct FullyShardedAccumulatedGradients {
    /// Actual locally owned derivatives only; globally unused leaves remain absent.
    pub gradients:GradientsParams,
    /// Original scale, integer effective count and actual issued-window count.
    pub state:FullyShardedAccumulationState,
}

/// Resumable accumulation of actual global SUM derivatives into original native local FSDP leaves.
/// Every rank must run the scope-completed loss backward, even if its local supervised count is zero.
/// Gather backward already sums/reduce-scatters; this never performs a second gradient all-reduce.
/// Optimizer/scheduler advancement, clipping, RNG/data continuation and skip decisions remain explicit.
pub struct FullyShardedGradientsAccumulator<M> {
    accumulator:GradientsAccumulator<M>,
    state:FullyShardedAccumulationState,
    placement:Placement,
}
impl<M> FullyShardedGradientsAccumulator<M> {
    /// Same original ownership/counters, without copying or resetting pending native gradients.
    pub fn continuation(&self) -> FullyShardedAccumulationContract {FullyShardedAccumulationContract {state:self.state.clone(),placement:self.placement.clone()}}
    pub(crate) fn inner(&self) -> &GradientsAccumulator<M> {&self.accumulator}
    /// Reattach exact original FSDP continuation to gradients restored by a combined native training record.
    pub fn from_accumulator<B:AutodiffBackend>(module:&M,accumulator:GradientsAccumulator<M>,continuation:FullyShardedAccumulationContract)
        -> Result<Self,FullyShardedAccumulationError> where M:AutodiffModule<B> {
        continuation.validate_for::<B,M>(module)?;accumulator.pending().validate_for::<B,M>(module)?;
        if continuation.state.microbatches==0 && !accumulator.pending().is_empty() {return Err(FullyShardedAccumulationError::State);}
        for id in accumulator.pending().container.ids() {
            let spec=continuation.placement.iter().find(|entry|entry.0==id.val()).ok_or(FullyShardedAccumulationError::State)?;
            let value=accumulator.pending().container.get::<B::InnerBackend>(id).ok_or(FullyShardedAccumulationError::State)?;
            if !spec.5 || value.dtype()!=continuation.state.dtype {return Err(FullyShardedAccumulationError::State);}
        }
        Ok(Self {accumulator,state:continuation.state,placement:continuation.placement})
    }
    /// Bind the actual module and explicit original logical placements before issuing an accumulation window.
    /// Existing optimizer parameter bindings are reused only for identity/shape/topology, not Muon role selection.
    pub fn new<B,C>(module:&M,parameters:&[FullyShardedOptimizerParameter<C>],dtype:FloatDType,loss_scale:f64)
        -> Result<Self,FullyShardedAccumulationError>
        where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
        validate_work_dtype(dtype)?;
        let state=FullyShardedAccumulationState {dtype:dtype.into(),loss_scale,global_count:0,microbatches:0};
        work_dtype(&state)?;
        let mut placement=Vec::with_capacity(parameters.len());let mut ids=BTreeSet::new();
        for binding in parameters {
            if !ids.insert(binding.parameter) {return Err(FullyShardedAccumulationError::Placement("duplicate canonical parameter binding"));}
            placement.push((binding.parameter.val(),binding.logical_shape.clone(),binding.communicator.rank(),binding.communicator.world_size(),DType::F32,false));
        }
        placement.sort_by_key(|entry|entry.0);
        placement=inspect::<B,M>(module,&placement,false)?;
        Ok(Self {accumulator:GradientsAccumulator::new(),state,placement})
    }
    /// Inspect actual counters without draining pending local gradients.
    pub fn state(&self) -> &FullyShardedAccumulationState {&self.state}
    /// Borrow only actual pending local derivatives; no global tensors are materialized.
    pub fn pending(&self) -> &GradientsParams {self.accumulator.pending()}
    /// Actual number of canonical local leaves, including explicitly frozen leaves.
    pub fn parameter_count(&self) -> usize {self.placement.len()}

    fn next(&self,global_count:u64) -> Result<(u64,u64),FullyShardedAccumulationError> {
        Ok((self.state.global_count.checked_add(global_count).ok_or(FullyShardedAccumulationError::CounterOverflow)?,
            self.state.microbatches.checked_add(1).ok_or(FullyShardedAccumulationError::CounterOverflow)?))
    }
    /// Add already SUM/reduce-scattered derivatives and their exact caller-coordinated global count.
    /// Inputs are gradients of loss_scale * local SUM, not normalized local/rank means.
    pub fn accumulate_sum<B:AutodiffBackend>(&mut self,module:&M,gradients:&GradientsParams,global_count:u64)
        -> Result<(),FullyShardedAccumulationError> where M:AutodiffModule<B> {
        let (total,microbatches)=self.next(global_count)?;
        inspect::<B,M>(module,&self.placement,true)?;
        let dtype=work_dtype(&self.state)?;
        let mut gradients=gradients.cast_for::<B,M>(module,dtype)?;
        for id in gradients.container.ids().into_iter().copied().collect::<Vec<_>>() {
            let spec=self.placement.iter().find(|entry|entry.0==id.val()).ok_or(FullyShardedAccumulationError::State)?;
            if !spec.5 {return Err(FullyShardedAccumulationError::Placement("frozen parameter has a supplied derivative"));}
            let elements=elements(&spec.1)?;let slots=elements.div_ceil(spec.3 as usize);
            let real=elements.saturating_sub(spec.2 as usize*slots).min(slots);
            if real<slots {
                let value=gradients.remove::<B::InnerBackend,1>(id).ok_or(FullyShardedAccumulationError::State)?;
                let zeros=Tensor::zeros([slots-real],(&value.device(),value.dtype()));
                gradients.register(id,value.slice_assign([real..slots],zeros));
            }
        }
        self.accumulator.accumulate_with_dtype::<B>(module,gradients,dtype);
        self.state.global_count=total;self.state.microbatches=microbatches;Ok(())
    }
    /// Backpropagate the actual unnormalized scope-completed loss on EVERY participating rank.
    /// Apply only the explicitly configured fixed scale; zero local/global counts do not suppress collectives.
    pub fn backward_sum<B:AutodiffBackend>(&mut self,module:&M,loss_sum:Tensor<B,1>,global_count:u64)
        -> Result<(),FullyShardedAccumulationError> where M:AutodiffModule<B> {
        self.next(global_count)?;inspect::<B,M>(module,&self.placement,true)?;
        if loss_sum.dims()!=[1] || !loss_sum.is_require_grad() || !loss_sum.dtype().is_float() || matches!(loss_sum.dtype(),DType::QFloat(_)) {
            return Err(FullyShardedAccumulationError::Loss);
        }
        let dtype=work_dtype(&self.state)?;
        let loss=loss_sum.cast(dtype).mul_scalar(self.state.loss_scale);
        let gradients=GradientsParams::from_grads(loss.backward(),module);
        self.accumulate_sum::<B>(module,&gradients,global_count)
    }
    /// Drain original scaled global SUM derivatives; no implicit normalization or optimizer update occurs.
    pub fn finish_sums(&mut self) -> FullyShardedAccumulatedGradients {
        let result=FullyShardedAccumulatedGradients {gradients:self.accumulator.grads(),state:self.state.clone()};
        self.state.global_count=0;self.state.microbatches=0;result
    }
    /// Normalize actual local shards once by fixed scale and whole-window GLOBAL integer count.
    /// A zero count uses the original clamp-to-one causal-loss convention, without a skip policy.
    pub fn finish_mean<B:AutodiffBackend>(&mut self,module:&M) -> Result<FullyShardedAccumulatedGradients,FullyShardedAccumulationError>
        where M:AutodiffModule<B> {
        if self.state.microbatches==0 {return Err(FullyShardedAccumulationError::EmptyWindow);}
        inspect::<B,M>(module,&self.placement,true)?;let dtype=work_dtype(&self.state)?;
        let gradients=self.accumulator.pending().unscaled_for::<B,M>(module,self.state.loss_scale,dtype)?
            .unscaled_for::<B,M>(module,self.state.global_count.max(1) as f64,dtype)?;
        let state=self.state.clone();self.accumulator.grads();self.state.global_count=0;self.state.microbatches=0;
        Ok(FullyShardedAccumulatedGradients {gradients,state})
    }
    /// Explicit native-storage optimizer handoff after whole-window work-precision normalization.
    /// Work gradients are narrowed only here, at the caller's request; FP32-master paths use finish_mean.
    pub fn finish_native_mean<B:AutodiffBackend>(&mut self,module:&M) -> Result<FullyShardedAccumulatedGradients,FullyShardedAccumulationError>
        where M:AutodiffModule<B> {
        if self.state.microbatches==0 {return Err(FullyShardedAccumulationError::EmptyWindow);}
        inspect::<B,M>(module,&self.placement,true)?;let dtype=work_dtype(&self.state)?;
        let gradients=self.accumulator.pending().unscaled_for::<B,M>(module,self.state.loss_scale,dtype)?
            .unscaled_for::<B,M>(module,self.state.global_count.max(1) as f64,dtype)?.cast_to_parameter_storage::<B,M>(module)?;
        let mut result=self.finish_sums();result.gradients=gradients;Ok(result)
    }
    /// Save actual pending local buffers and exact placement/options/counters together, without clearing them.
    pub fn try_to_record<B:AutodiffBackend>(&self) -> Result<FullyShardedGradientsRecord,RecorderError> where M:AutodiffModule<B> {
        Ok(FullyShardedGradientsRecord {version:1,gradients:self.accumulator.try_to_record::<B>()?,state:self.state.clone(),placement:self.placement.clone()})
    }
    /// Native async readback variant for the same complete pending-gradient continuation.
    pub async fn to_record_async<B:AutodiffBackend>(&self) -> Result<FullyShardedGradientsRecord,RecorderError> where M:AutodiffModule<B> {
        Ok(FullyShardedGradientsRecord {version:1,gradients:self.accumulator.to_record_async::<B>().await?,state:self.state.clone(),placement:self.placement.clone()})
    }
    /// Restore the same actual ownership/precision/scale and canonical model IDs; rejected records leave state intact.
    pub fn load_record<B:AutodiffBackend>(&mut self,module:&M,record:FullyShardedGradientsRecord,device:&B::Device) -> Result<(),RecorderError>
        where M:AutodiffModule<B> {
        let failed=||RecorderError::Unknown(FullyShardedAccumulationError::State.to_string());
        if record.version!=1 || record.placement!=self.placement || record.state.dtype!=self.state.dtype || record.state.loss_scale!=self.state.loss_scale
            || (record.state.microbatches==0 && record.state.global_count!=0) {return Err(failed());}
        let dtype=work_dtype(&record.state).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        inspect::<B,M>(module,&self.placement,true).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        let gradients=GradientsParams::from_record::<B::InnerBackend>(record.gradients,device)?;
        gradients.validate_for::<B,M>(module).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        if record.state.microbatches==0 && !gradients.is_empty() {return Err(failed());}
        for id in gradients.container.ids() {
            let spec=self.placement.iter().find(|entry|entry.0==id.val()).ok_or_else(failed)?;
            let primitive=gradients.container.get::<B::InnerBackend>(id).ok_or_else(failed)?;
            if !spec.5 || primitive.dtype()!=record.state.dtype {return Err(failed());}
        }
        let mut accumulator=GradientsAccumulator::new();accumulator.accumulate_with_dtype::<B>(module,gradients,dtype);
        self.accumulator=accumulator;self.state=record.state;Ok(())
    }
}

fn work_dtype(state:&FullyShardedAccumulationState) -> Result<FloatDType,FullyShardedAccumulationError> {
    let dtype=match state.dtype {DType::F32=>FloatDType::F32,DType::F64=>FloatDType::F64,_=>return Err(FullyShardedAccumulationError::State)};
    if !representable(state.loss_scale,dtype) || state.loss_scale<=0.0 || (dtype==FloatDType::F32 && state.loss_scale as f32==0.0) {
        return Err(GradientTransformError::InvalidScalar.into());
    }Ok(dtype)
}
fn elements(shape:&[usize]) -> Result<usize,FullyShardedAccumulationError> {
    if shape.is_empty() || shape.contains(&0) {return Err(FullyShardedAccumulationError::Placement("positive original logical axes required"));}
    shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis)).ok_or(FullyShardedAccumulationError::Placement("original parameter size overflows"))
}
pub(super) fn inspect<B:AutodiffBackend,M:AutodiffModule<B>>(module:&M,placement:&Placement,match_storage:bool) -> Result<Placement,FullyShardedAccumulationError> {
    let mut visitor=Inspect {declared:placement.iter().map(|entry|(entry.0,entry)).collect(),found:BTreeMap::new(),match_storage,error:None};
    module.visit(&mut visitor);
    if let Some(error)=visitor.error {return Err(error);}
    if visitor.found.len()!=placement.len() {return Err(FullyShardedAccumulationError::Placement("module/binding membership differs"));}
    Ok(visitor.found.into_values().collect())
}
struct Inspect<'a> {
    declared:BTreeMap<u64,&'a (u64,Vec<usize>,u32,u32,DType,bool)>,
    found:BTreeMap<u64,(u64,Vec<usize>,u32,u32,DType,bool)>,
    match_storage:bool,
    error:Option<FullyShardedAccumulationError>,
}
impl<B:AutodiffBackend> ModuleVisitor<B> for Inspect<'_> {
    fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {
        if self.error.is_some() {return;}
        let Some(spec)=self.declared.get(&param.id.val()).copied() else {self.error=Some(FullyShardedAccumulationError::Placement("unbound actual model parameter"));return;};
        let count=match elements(&spec.1) {Ok(count)=>count,Err(error)=>{self.error=Some(error);return;}};
        if spec.3==0 || spec.2>=spec.3 || D!=1 {self.error=Some(FullyShardedAccumulationError::Placement("actual flat leaf and valid rank/world required"));return;}
        let value=param.val();let slots=count.div_ceil(spec.3 as usize);
        if slots.checked_mul(spec.3 as usize).is_none() || value.shape().num_elements()!=slots
            || !matches!(value.dtype(),DType::F32|DType::F16|DType::BF16) {
            self.error=Some(FullyShardedAccumulationError::Placement("native local length/storage differs"));return;
        }
        let found=(spec.0,spec.1.clone(),spec.2,spec.3,value.dtype(),value.is_require_grad());
        if (self.match_storage && found!=*spec) || self.found.get(&spec.0).is_some_and(|previous|previous!=&found) {
            self.error=Some(FullyShardedAccumulationError::Placement("original tied leaf metadata changed"));return;
        }
        self.found.insert(spec.0,found);
    }
}
