use ruda_model::{record::{Record,PrecisionSettings},tensor::{Tensor,DType,BroadcastTensorCollective,backend::Backend}};
use serde::{Serialize,Deserialize};
use super::{SimpleOptimizer,ElementwiseShardOptimizer,OptimizerCheckpointBuffers,OptimizerCheckpointScalars,
    FlatOptimizerCheckpointState,FlatOptimizerTensorShard,OptimizerShardError,FullyShardedElementwiseError,FullyShardedOptimizerParameter};
use crate::LearningRate;
mod native_records;

/// An explicit choice between two ORIGINAL native optimizer implementations.
/// Different parameter groups can use different algorithms/master wrappers while
/// retaining one canonical history map and the existing grouped FSDP/ZeRO APIs.
/// Nest choices for more algorithms; no name/shape-based selection, default role,
/// update formula, precision policy or automatic history conversion is introduced.
#[derive(Clone,Debug)]
pub enum OptimizerChoice<L,R> {
    Left(L),
    Right(R),
}

/// Actual original concrete history, tagged with its explicit algorithm branch.
/// Records preserve the tag and use the original state's recorder implementation.
/// An absent history stays absent; changing branch never silently reinitializes it.
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
pub enum OptimizerChoiceState<L,R> {
    Left(L),
    Right(R),
}

impl<B:Backend,L:Record<B>,R:Record<B>> Record<B> for OptimizerChoiceState<L,R> {
    type Item<P:PrecisionSettings>=OptimizerChoiceState<L::Item<P>,R::Item<P>>;
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        match self {Self::Left(state)=>OptimizerChoiceState::Left(state.into_item::<P>()),
            Self::Right(state)=>OptimizerChoiceState::Right(state.into_item::<P>())}
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        match item {OptimizerChoiceState::Left(state)=>Self::Left(L::from_item::<P>(state,device)),
            OptimizerChoiceState::Right(state)=>Self::Right(R::from_item::<P>(state,device))}
    }
}

impl<B:Backend,L:SimpleOptimizer<B>,R:SimpleOptimizer<B>> SimpleOptimizer<B> for OptimizerChoice<L,R> {
    type State<const D:usize>=OptimizerChoiceState<L::State<D>,R::State<D>>;
    fn step<const D:usize>(&self,lr:LearningRate,tensor:Tensor<B,D>,gradient:Tensor<B,D>,state:Option<Self::State<D>>)
        -> (Tensor<B,D>,Option<Self::State<D>>) {
        match self {
            Self::Left(optimizer)=>{
                let state=match state {None=>None,Some(OptimizerChoiceState::Left(state))=>Some(state),
                    Some(OptimizerChoiceState::Right(_))=>panic!("saved original optimizer branch differs")};
                let (value,state)=optimizer.step(lr,tensor,gradient,state);
                (value,state.map(OptimizerChoiceState::Left))
            },
            Self::Right(optimizer)=>{
                let state=match state {None=>None,Some(OptimizerChoiceState::Right(state))=>Some(state),
                    Some(OptimizerChoiceState::Left(_))=>panic!("saved original optimizer branch differs")};
                let (value,state)=optimizer.step(lr,tensor,gradient,state);
                (value,state.map(OptimizerChoiceState::Right))
            },
        }
    }
    fn to_device<const D:usize>(state:Self::State<D>,device:&B::Device) -> Self::State<D> {
        match state {OptimizerChoiceState::Left(state)=>OptimizerChoiceState::Left(L::to_device(state,device)),
            OptimizerChoiceState::Right(state)=>OptimizerChoiceState::Right(R::to_device(state,device))}
    }
}

impl<B:Backend,L:ElementwiseShardOptimizer<B>,R:ElementwiseShardOptimizer<B>> ElementwiseShardOptimizer<B> for OptimizerChoice<L,R> {
    fn validate_element_sharding(&self) -> Result<(),&'static str> {
        match self {Self::Left(optimizer)=>optimizer.validate_element_sharding(),Self::Right(optimizer)=>optimizer.validate_element_sharding()}
    }
    fn shard_gradient_dtype(&self,storage:DType) -> DType {
        match self {Self::Left(optimizer)=>optimizer.shard_gradient_dtype(storage),Self::Right(optimizer)=>optimizer.shard_gradient_dtype(storage)}
    }
    fn validate_fully_sharded_execution(&self) -> Result<(),&'static str> {
        match self {Self::Left(optimizer)=>optimizer.validate_fully_sharded_execution(),Self::Right(optimizer)=>optimizer.validate_fully_sharded_execution()}
    }
    fn validate_fully_sharded_history(&self,state:&Self::State<1>) -> Result<(),&'static str> {
        match (self,state) {
            (Self::Left(optimizer),OptimizerChoiceState::Left(state))=>optimizer.validate_fully_sharded_history(state),
            (Self::Right(optimizer),OptimizerChoiceState::Right(state))=>optimizer.validate_fully_sharded_history(state),
            _=>Err("saved original optimizer branch differs"),
        }
    }
    fn step_fully_sharded<C:BroadcastTensorCollective<B>>(&self,lr:LearningRate,tensor:Tensor<B,1>,gradient:Tensor<B,1>,
        state:Option<Self::State<1>>,binding:&FullyShardedOptimizerParameter<C>)
        -> Result<(Tensor<B,1>,Option<Self::State<1>>),FullyShardedElementwiseError<C::Error>> {
        match self {
            Self::Left(optimizer)=>{
                let state=match state {None=>None,Some(OptimizerChoiceState::Left(state))=>Some(state),
                    Some(OptimizerChoiceState::Right(_))=>return Err(FullyShardedElementwiseError::State("saved original optimizer branch differs"))};
                let (value,state)=optimizer.step_fully_sharded(lr,tensor,gradient,state,binding)?;
                Ok((value,state.map(OptimizerChoiceState::Left)))
            },
            Self::Right(optimizer)=>{
                let state=match state {None=>None,Some(OptimizerChoiceState::Right(state))=>Some(state),
                    Some(OptimizerChoiceState::Left(_))=>return Err(FullyShardedElementwiseError::State("saved original optimizer branch differs"))};
                let (value,state)=optimizer.step_fully_sharded(lr,tensor,gradient,state,binding)?;
                Ok((value,state.map(OptimizerChoiceState::Right)))
            },
        }
    }
}

impl<B:Backend,const D:usize,L:OptimizerCheckpointBuffers<B,D>,R:OptimizerCheckpointBuffers<B,D>>
    OptimizerCheckpointBuffers<B,D> for OptimizerChoiceState<L,R> {
    fn visit_checkpoint_buffers<F:FnMut(&Tensor<B,D>)>(&self,visit:&mut F) {
        match self {Self::Left(state)=>state.visit_checkpoint_buffers(visit),Self::Right(state)=>state.visit_checkpoint_buffers(visit)}
    }
    fn map_checkpoint_buffers<F:FnMut(Tensor<B,D>)->Tensor<B,D>>(self,map:&mut F) -> Self {
        match self {Self::Left(state)=>Self::Left(state.map_checkpoint_buffers(map)),Self::Right(state)=>Self::Right(state.map_checkpoint_buffers(map))}
    }
}

impl<L:OptimizerCheckpointScalars,R:OptimizerCheckpointScalars> OptimizerCheckpointScalars for OptimizerChoiceState<L,R> {
    type Scalars=OptimizerChoiceState<L::Scalars,R::Scalars>;
    fn checkpoint_scalars(&self) -> Self::Scalars {
        match self {Self::Left(state)=>OptimizerChoiceState::Left(state.checkpoint_scalars()),
            Self::Right(state)=>OptimizerChoiceState::Right(state.checkpoint_scalars())}
    }
}

impl<B:Backend,const D:usize,L:FlatOptimizerCheckpointState<B,D>,R:FlatOptimizerCheckpointState<B,D>>
    FlatOptimizerCheckpointState<B,D> for OptimizerChoiceState<L,R> {
    type FlatState=OptimizerChoiceState<L::FlatState,R::FlatState>;
    fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
        match self {Self::Left(state)=>state.into_flat_shard(shard).map(OptimizerChoiceState::Left),
            Self::Right(state)=>state.into_flat_shard(shard).map(OptimizerChoiceState::Right)}
    }
}
