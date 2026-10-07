use super::*;
use crate::ElementwiseShardOptimizer;

/// Original coordinate optimizer whose native rank-tagged histories can become corresponding flat local histories.
/// The conversion uses each original concrete state type, not a newly initialized replacement moment buffer.
pub trait FlatShardElementwiseOptimizer<B:Backend>:ElementwiseShardOptimizer<B> {
    /// Partition one original full native history, retaining every scalar clock and optional-state branch.
    fn partition_native_history(record:AdaptorRecordV1<Self,B>,shard:&FlatOptimizerTensorShard) -> Result<Self::State<1>,OptimizerShardError>
        where Self:Sized;
}
impl<B:Backend,O:ElementwiseShardOptimizer<B>> FlatShardElementwiseOptimizer<B> for O
    where O::State<0>:FlatOptimizerCheckpointState<B,0,FlatState=O::State<1>>,
        O::State<1>:FlatOptimizerCheckpointState<B,1,FlatState=O::State<1>>,
        O::State<2>:FlatOptimizerCheckpointState<B,2,FlatState=O::State<1>>,
        O::State<3>:FlatOptimizerCheckpointState<B,3,FlatState=O::State<1>>,
        O::State<4>:FlatOptimizerCheckpointState<B,4,FlatState=O::State<1>>,
        O::State<5>:FlatOptimizerCheckpointState<B,5,FlatState=O::State<1>>,
        O::State<6>:FlatOptimizerCheckpointState<B,6,FlatState=O::State<1>>,
        O::State<7>:FlatOptimizerCheckpointState<B,7,FlatState=O::State<1>>,
        O::State<8>:FlatOptimizerCheckpointState<B,8,FlatState=O::State<1>> {
    fn partition_native_history(record:AdaptorRecordV1<Self,B>,shard:&FlatOptimizerTensorShard) -> Result<Self::State<1>,OptimizerShardError> {
        match record {
            AdaptorRecordV1::Rank0(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank1(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank2(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank3(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank4(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank5(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank6(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank7(state)=>state.into_flat_shard(shard),
            AdaptorRecordV1::Rank8(state)=>state.into_flat_shard(shard),
        }
    }
}

/// Original non-tensor clocks and optional-state structure, independent of native buffer placement.
/// Equality compares actual saved continuation, not hashes, replayed steps or inferred optimizer age.
pub trait OptimizerCheckpointScalars {
    /// Exact native scalar/optional-state representation for this concrete original state type.
    type Scalars:PartialEq;
    /// Read actual scalar counters/options without materializing or modifying native buffers.
    fn checkpoint_scalars(&self) -> Self::Scalars;
}

/// Native full-geometry history that can become the corresponding original flat local history type.
pub trait FlatOptimizerCheckpointState<B:Backend,const D:usize>:OptimizerCheckpointBuffers<B,D> {
    /// Same numerical algorithm/scalar metadata with native one-dimensional local buffer storage.
    type FlatState:OptimizerCheckpointBuffers<B,1>;
    /// Partition actual original moments/history/master values, retaining genuine optional absence.
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError>;
}

/// Original logical axes and exact equal padded data-shard ownership for native optimizer history.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct FlatOptimizerTensorShard {
    /// Actual original parameter geometry; [] is a rank-zero scalar, not an invented [1] logical axis.
    pub global_shape:Vec<usize>,
    /// Actual original data group owner.
    pub rank:u32,
    /// Actual number of equal padded local vectors.
    pub world_size:u32,
}
impl FlatOptimizerTensorShard {
    /// Declare the actual source geometry/ownership before importing a completed native optimization history.
    pub fn new(global_shape:Vec<usize>,rank:u32,world_size:u32) -> Result<Self,OptimizerShardError> {
        let value=Self {global_shape,rank,world_size};value.geometry()?;Ok(value)
    }
    /// Original logical element count and physical local slot count, with checked topology arithmetic.
    pub fn geometry(&self) -> Result<(usize,usize),OptimizerShardError> {
        if self.global_shape.contains(&0) || self.world_size==0 || self.rank>=self.world_size {
            return Err(OptimizerShardError::Placement("positive original FSDP axes and valid data rank/world required"));
        }
        let count=self.global_shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis)).ok_or(OptimizerShardError::Placement("original flat parameter size overflows"))?;
        let slots=count.div_ceil(self.world_size as usize);
        slots.checked_mul(self.world_size as usize).ok_or(OptimizerShardError::Placement("padded flat parameter size overflows"))?;
        Ok((count,slots))
    }
    /// Exact actual nonpadding global coordinate interval owned by this rank, including an empty real interval.
    pub fn interval(&self) -> Result<Range<usize>,OptimizerShardError> {
        let (count,slots)=self.geometry()?;let start=(self.rank as usize*slots).min(count);
        Ok(start..start.saturating_add(slots).min(count))
    }
    /// Slice original native full buffer into this owner's vector, retaining dtype/device and zero rank padding.
    pub fn partition<B:Backend,const D:usize>(&self,value:Tensor<B,D>) -> Result<Tensor<B,1>,OptimizerShardError> {
        let (count,slots)=self.geometry()?;
        if value.dims().as_slice()!=self.global_shape.as_slice() {return Err(OptimizerShardError::Shape("original flat optimizer buffer geometry"));}
        if !value.dtype().is_float() || matches!(value.dtype(),DType::QFloat(_)) {return Err(OptimizerShardError::DType("native full optimizer buffer required"));}
        if self.world_size==1 {return Ok(value.reshape([count]));}
        let start=self.rank as usize*slots;let real=count.saturating_sub(start).min(slots);
        let mut local=Tensor::zeros([slots],(&value.device(),value.dtype()));
        if real>0 {local=local.slice_assign([0..real],value.reshape([count]).slice([start..start+real]));}Ok(local)
    }
    /// Validate every actual source buffer before constructing any local history; no values are read.
    pub fn validate_state<B:Backend,const D:usize,S:OptimizerCheckpointBuffers<B,D>>(&self,state:&S) -> Result<(),OptimizerShardError> {
        self.geometry()?;
        if D!=self.global_shape.len() {return Err(OptimizerShardError::Shape("source optimizer history rank differs from original parameter"));}
        let mut error=None;let mut device=None;
        state.visit_checkpoint_buffers(&mut |value| {
            if value.dims().as_slice()!=self.global_shape.as_slice() {error=Some(OptimizerShardError::Shape("original native history dimensions"));}
            if !value.dtype().is_float() || matches!(value.dtype(),DType::QFloat(_)) {error=Some(OptimizerShardError::DType("original native history precision"));}
            if device.as_ref().is_some_and(|previous|*previous!=value.device()) {error=Some(OptimizerShardError::Device("one native parameter history spans different devices"));}
            device=Some(value.device());
        });
        if let Some(error)=error {return Err(error);}Ok(())
    }
}

impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for AdaptiveMomentumState<B,D> {
    type FlatState=AdaptiveMomentumState<B,1>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
        shard.validate_state(&self)?;
        Ok(AdaptiveMomentumState {time:self.time,moment_1:shard.partition(self.moment_1)?,moment_2:shard.partition(self.moment_2)?,
            max_moment_2:self.max_moment_2.map(|value|shard.partition(value)).transpose()?})
    }
}
impl<B:Backend,const D:usize> OptimizerCheckpointScalars for AdaptiveMomentumState<B,D> {
    type Scalars=(usize,bool);
    fn checkpoint_scalars(&self) -> Self::Scalars {(self.time,self.max_moment_2.is_some())}
}
impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for AdaptiveNesterovMomentumState<B,D> {
    type FlatState=AdaptiveNesterovMomentumState<B,1>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
        shard.validate_state(&self)?;
        Ok(AdaptiveNesterovMomentumState {time:self.time,exp_avg:shard.partition(self.exp_avg)?,exp_avg_sq:shard.partition(self.exp_avg_sq)?,
            exp_avg_diff:shard.partition(self.exp_avg_diff)?,neg_pre_grad:shard.partition(self.neg_pre_grad)?})
    }
}
impl<B:Backend,const D:usize> OptimizerCheckpointScalars for AdaptiveNesterovMomentumState<B,D> {
    type Scalars=usize;
    fn checkpoint_scalars(&self) -> usize {self.time}
}
macro_rules! adaptive_flat {
    ($(($state:ident,$momentum:ident)),+) => {$(
        impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for $state<B,D> {
            type FlatState=$state<B,1>;
            fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
                Ok($state {momentum:self.momentum.into_flat_shard(shard)?})
            }
        }
        impl<B:Backend,const D:usize> OptimizerCheckpointScalars for $state<B,D> {
            type Scalars=<$momentum<B,D> as OptimizerCheckpointScalars>::Scalars;
            fn checkpoint_scalars(&self) -> Self::Scalars {self.momentum.checkpoint_scalars()}
        }
    )+};
}
adaptive_flat!((AdamState,AdaptiveMomentumState),(AdamWState,AdaptiveMomentumState),(AdanState,AdaptiveNesterovMomentumState));

impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for SgdState<B,D> {
    type FlatState=SgdState<B,1>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
        shard.validate_state(&self)?;Ok(SgdState {momentum:self.momentum.map(|value|value.into_flat_shard(shard)).transpose()?})
    }
}
impl<B:Backend,const D:usize> OptimizerCheckpointScalars for SgdState<B,D> {
    type Scalars=bool;
    fn checkpoint_scalars(&self) -> bool {self.momentum.is_some()}
}
impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for SquareAvgState<B,D> {
    type FlatState=SquareAvgState<B,1>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {Ok(SquareAvgState {square_avg:shard.partition(self.square_avg)?})}
}
impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for CenteredState<B,D> {
    type FlatState=CenteredState<B,1>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
        shard.validate_state(&self)?;Ok(CenteredState {avg:shard.partition(self.avg)?,grad_avg:self.grad_avg.map(|value|shard.partition(value)).transpose()?})
    }
}
impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for RmsPropState<B,D> {
    type FlatState=RmsPropState<B,1>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
        shard.validate_state(&self)?;Ok(RmsPropState {square_avg:self.square_avg.into_flat_shard(shard)?,centered:self.centered.into_flat_shard(shard)?,
            momentum:self.momentum.map(|value|value.into_flat_shard(shard)).transpose()?})
    }
}
impl<B:Backend,const D:usize> OptimizerCheckpointScalars for RmsPropState<B,D> {
    type Scalars=(bool,bool);
    fn checkpoint_scalars(&self) -> Self::Scalars {(self.centered.grad_avg.is_some(),self.momentum.is_some())}
}
impl<B:Backend,const D:usize,S:FlatOptimizerCheckpointState<B,D>> FlatOptimizerCheckpointState<B,D> for Fp32MasterState<B,D,S> {
    type FlatState=Fp32MasterState<B,1,S::FlatState>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
        shard.validate_state(&self)?;
        let mut fp32=true;self.visit_checkpoint_buffers(&mut |value|fp32&=value.dtype()==DType::F32);
        if !fp32 {return Err(OptimizerShardError::DType("authoritative native master and inner history must retain FP32"));}
        Ok(Fp32MasterState {master:shard.partition(self.master)?,inner:self.inner.map(|state|state.into_flat_shard(shard)).transpose()?})
    }
}
impl<B:Backend,const D:usize,S:Record<B>+Clone+OptimizerCheckpointScalars> OptimizerCheckpointScalars for Fp32MasterState<B,D,S> {
    type Scalars=Option<S::Scalars>;
    fn checkpoint_scalars(&self) -> Self::Scalars {self.inner.as_ref().map(|state|state.checkpoint_scalars())}
}
