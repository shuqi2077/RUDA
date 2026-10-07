use alloc::vec::Vec;
use serde::{Serialize,Deserialize};
use ruda_model::{record::{Record,PrecisionSettings},tensor::{Tensor,TensorPrimitive,DType,Bool,BroadcastTensorCollective,backend::Backend}};
use super::{Muon,MuonState,MuonError,MuonShardedError,MomentumState,LearningRate};

pub(crate) fn repartition_flat_buffer<B:Backend>(sources:&[Tensor<B,1>],elements:usize,rank:u32,world:u32) -> Result<Tensor<B,1>,MuonError> {
    let first=sources.first().ok_or(MuonError::InvalidConfig("complete original flat buffer rank set is empty"))?;
    if elements==0 || world==0 || rank>=world {return Err(MuonError::InvalidConfig("invalid original flat buffer geometry/topology"));}
    let old=elements.div_ceil(sources.len());let size=elements.div_ceil(world as usize);
    old.checked_mul(sources.len()).ok_or(MuonError::InvalidConfig("original flat buffer padded size overflows"))?;
    size.checked_mul(world as usize).ok_or(MuonError::InvalidConfig("destination flat buffer padded size overflows"))?;
    for source in sources {
        if source.dims()!=[old] {return Err(MuonError::ShapeMismatch("original flat buffer rank interval"));}
        if source.dtype()!=first.dtype() {return Err(MuonError::DTypeMismatch("original flat buffer rank interval"));}
        if source.device()!=first.device() {return Err(MuonError::DeviceMismatch("original flat buffer rank interval"));}
    }
    let start=rank as usize*size;let end=(start+size).min(elements);
    let mut destination=Tensor::zeros([size],(&first.device(),first.dtype()));
    for (source_rank,source) in sources.iter().enumerate() {
        let source_start=source_rank*old;let begin=start.max(source_start);let finish=end.min((source_start+old).min(elements));
        if begin<finish {destination=destination.slice_assign([begin-start..finish-start],source.clone().slice([begin-source_start..finish-source_start]));}
    }
    Ok(destination)
}

/// Original complete matrix backed by equal padded rank-ordered flat element intervals.
/// Unlike row/column TP shards, an interval may cut across any original matrix row.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct MuonFlatShardLayout {
    /// Actual original stored axes; the original Muon matrix-layout option still controls LR orientation.
    pub shape:[usize;2],
}

impl MuonFlatShardLayout {
    /// Declare the original logical matrix, never its local padded physical vector as a new matrix.
    pub fn new(shape:[usize;2]) -> Self {Self {shape}}
    /// Validate actual flat physical storage/topology and return (logical elements, local padded slots).
    pub fn geometry(&self,rank:u32,world:u32,local:usize) -> Result<(usize,usize),MuonError> {
        if world==0 || rank>=world {return Err(MuonError::InvalidConfig("invalid flat Muon rank/world"));}
        if self.shape.contains(&0) {return Err(MuonError::EmptyMatrix);}
        let elements=self.shape[0].checked_mul(self.shape[1]).ok_or(MuonError::InvalidConfig("flat Muon logical matrix size overflows"))?;
        let size=elements.div_ceil(world as usize);
        size.checked_mul(world as usize).ok_or(MuonError::InvalidConfig("flat Muon gather padding overflows"))?;
        if local!=size {return Err(MuonError::ShapeMismatch("local flat matrix shard"));}
        Ok((elements,size))
    }
    /// Original real element interval; ranks with only padding retain an empty logical interval.
    pub fn range(&self,rank:u32,world:u32) -> Result<core::ops::Range<usize>,MuonError> {
        if world==0 {return Err(MuonError::InvalidConfig("invalid flat Muon world"));}
        let elements=self.shape[0].checked_mul(self.shape[1]).ok_or(MuonError::InvalidConfig("flat Muon logical matrix size overflows"))?;
        let size=elements.div_ceil(world as usize);self.geometry(rank,world,size)?;
        let start=(rank as usize*size).min(elements);let end=(rank as usize*size+size).min(elements);Ok(start..end)
    }
    /// Slice the actual original matrix into its declared padded flat interval on the original backend.
    pub fn partition<B:Backend>(&self,value:Tensor<B,2>,rank:u32,world:u32) -> Result<Tensor<B,1>,MuonError> {
        if value.dims()!=self.shape {return Err(MuonError::ShapeMismatch("original complete matrix"));}
        if world==0 {return Err(MuonError::InvalidConfig("invalid flat Muon world"));}
        let elements=self.shape[0].checked_mul(self.shape[1]).ok_or(MuonError::InvalidConfig("flat Muon logical matrix size overflows"))?;
        let size=elements.div_ceil(world as usize);self.geometry(rank,world,size)?;let range=self.range(rank,world)?;
        let mut local=Tensor::zeros([size],(&value.device(),value.dtype()));
        if !range.is_empty() {local=local.slice_assign([0..range.len()],value.reshape([elements]).slice([range]));}
        Ok(local)
    }
    pub(crate) fn trim_padding<B:Backend>(&self,value:Tensor<B,1>,rank:u32,world:u32) -> Result<Tensor<B,1>,MuonError> {
        self.geometry(rank,world,value.dims()[0])?;let real=self.range(rank,world)?.len();
        let mask=Tensor::<B,1,Bool>::zeros(value.dims(),&value.device());
        let mask=if real<value.dims()[0] {mask.slice_assign([real..value.dims()[0]],Tensor::zeros([value.dims()[0]-real],&value.device()).bool_not())} else {mask};
        Ok(value.mask_fill(mask,0))
    }
    pub(crate) fn gather<B,C>(&self,value:Tensor<B,1>,communicator:&C) -> Result<Tensor<B,2>,MuonShardedError<C::Error>>
        where B:Backend,C:BroadcastTensorCollective<B> {
        let (elements,size)=self.geometry(communicator.rank(),communicator.world_size(),value.dims()[0]).map_err(MuonShardedError::Muon)?;
        let dtype=value.dtype();let device=value.device();
        let full=if communicator.world_size()==1 {value} else {
            let full=communicator.all_gather_float(value.into_primitive().tensor()).map_err(MuonShardedError::Collective)?;
            Tensor::<B,1>::from_primitive(TensorPrimitive::Float(full))
        };
        if full.dims()!=[size*communicator.world_size() as usize] {return Err(MuonShardedError::Muon(MuonError::ShapeMismatch("flat Muon gathered matrix")));}
        if full.dtype()!=dtype {return Err(MuonShardedError::Muon(MuonError::DTypeMismatch("flat Muon gathered matrix")));}
        if full.device()!=device {return Err(MuonShardedError::Muon(MuonError::DeviceMismatch("flat Muon gathered matrix")));}
        Ok(full.slice([0..elements]).reshape(self.shape))
    }
}

/// Exact native momentum in this rank's original flat interval, including its original logical placement.
#[derive(Clone)]
pub struct MuonFlatShardedState<B:Backend> {
    version:u32,
    rank:u32,
    world:u32,
    layout:MuonFlatShardLayout,
    local:MuonState<B,1>,
}
impl<B:Backend> MuonFlatShardedState<B> {
    /// Partition an actual loaded complete original momentum buffer, without initializing replacement history.
    pub fn from_global_state(state:MuonState<B,2>,layout:&MuonFlatShardLayout,rank:u32,world:u32) -> Result<Self,MuonError> {
        let local=layout.partition(state.momentum.velocity().clone(),rank,world)?;
        Ok(Self {version:1,rank,world,layout:layout.clone(),local:MuonState::new(MomentumState::new(local))})
    }
    /// Actual rank-local momentum; padded elements are not an independent logical parameter.
    pub fn momentum(&self) -> &Tensor<B,1> {self.local.momentum.velocity()}
    /// Original saved matrix geometry.
    pub fn layout(&self) -> &MuonFlatShardLayout {&self.layout}
    /// Actual original owner rank.
    pub fn rank(&self) -> u32 {self.rank}
    /// Actual original data-axis rank count.
    pub fn world_size(&self) -> u32 {self.world}
    /// Check original placement and real physical interval without numerical tensor reads.
    pub fn validate_placement(&self,rank:u32,world:u32,layout:&MuonFlatShardLayout) -> Result<(),MuonError> {
        if self.version!=1 || self.rank!=rank || self.world!=world || &self.layout!=layout {return Err(MuonError::IncompatibleRecord);}
        layout.geometry(rank,world,self.momentum().dims()[0])?;Ok(())
    }
    /// Move only actual local momentum, retaining the original topology/algorithm state.
    pub fn to_device(mut self,device:&B::Device) -> Self {self.local.momentum=self.local.momentum.to_device(device);self}
    /// Offline complete-rank-set conversion of original actual momentum, without a full matrix allocation.
    /// Move sources to one destination device first and repartition model/master/data continuation separately.
    pub fn repartition_from_ranks(sources:&[Self],rank:u32,world:u32) -> Result<Self,MuonError> {
        let first=sources.first().ok_or(MuonError::InvalidConfig("complete original flat Muon state set is empty"))?;
        let old_world=u32::try_from(sources.len()).map_err(|_|MuonError::InvalidConfig("flat Muon source rank count overflows"))?;
        for (source_rank,source) in sources.iter().enumerate() {source.validate_placement(source_rank as u32,old_world,&first.layout)?;}
        let elements=first.layout.shape[0].checked_mul(first.layout.shape[1]).ok_or(MuonError::InvalidConfig("flat Muon original state size overflows"))?;
        let buffers=sources.iter().map(|source|source.momentum().clone()).collect::<Vec<_>>();
        let velocity=repartition_flat_buffer(&buffers,elements,rank,world)?;
        Ok(Self {version:1,rank,world,layout:first.layout.clone(),local:MuonState::new(MomentumState::new(velocity))})
    }
}
impl<B:Backend> Record<B> for MuonFlatShardedState<B> {
    type Item<P:PrecisionSettings>=(u32,u32,u32,MuonFlatShardLayout,DType,<MuonState<B,1> as Record<B>>::Item<P>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        let dtype=self.momentum().dtype();(self.version,self.rank,self.world,self.layout,dtype,self.local.into_item::<P>())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        let mut local=MuonState::<B,1>::from_item::<P>(item.5,device);
        local.momentum=MomentumState::new(local.momentum.velocity().clone().cast(item.4));
        Self {version:item.0,rank:item.1,world:item.2,layout:item.3,local}
    }
}

impl<B:Backend> Muon<B> {
    /// Inspect the actual local slice while using the original full matrix for LR orientation and precision rules.
    pub fn validate_step_flat_sharded<C>(&self,lr:LearningRate,tensor:&Tensor<B,1>,grad:&Tensor<B,1>,
        state:Option<&MuonFlatShardedState<B>>,layout:&MuonFlatShardLayout,communicator:&C) -> Result<(),MuonShardedError<C::Error>>
        where C:BroadcastTensorCollective<B> {
        let validate=|| -> Result<(),MuonError> {
            layout.geometry(communicator.rank(),communicator.world_size(),tensor.dims()[0])?;
            if tensor.dims()!=grad.dims() {return Err(MuonError::ShapeMismatch("gradient"));}
            if tensor.dtype()!=grad.dtype() {return Err(MuonError::DTypeMismatch("gradient"));}
            if tensor.device()!=grad.device() {return Err(MuonError::DeviceMismatch("gradient"));}
            if self.stable_normalization && tensor.dtype()!=DType::F32 {return Err(MuonError::InvalidConfig("stable normalization requires FP32 tensors"));}
            if let Some(state)=state {
                state.validate_placement(communicator.rank(),communicator.world_size(),layout)?;
                if state.momentum().dtype()!=tensor.dtype() {return Err(MuonError::DTypeMismatch("momentum"));}
                if state.momentum().device()!=tensor.device() {return Err(MuonError::DeviceMismatch("momentum"));}
            }
            self.validate_effective_learning_rate(lr,&layout.shape,tensor.dtype())
        };validate().map_err(MuonShardedError::Muon)
    }
    /// Actual FSDP-gradient update: momentum remains local, original complete-matrix Newton–Schulz is reused.
    /// Gradients must already be globally normalized/reduce-scattered; this only concatenates unique
    /// element intervals, never sums them a second time. Only the transformed current matrix is transiently
    /// gathered; this is not a memory-free or communication-free matrix optimizer.
    pub fn try_step_flat_sharded<C>(&self,lr:LearningRate,tensor:Tensor<B,1>,grad:Tensor<B,1>,state:Option<MuonFlatShardedState<B>>,
        layout:&MuonFlatShardLayout,communicator:C) -> Result<(Tensor<B,1>,MuonFlatShardedState<B>),MuonShardedError<C::Error>>
        where C:BroadcastTensorCollective<B> {
        self.validate_step_flat_sharded(lr,&tensor,&grad,state.as_ref(),layout,&communicator)?;
        let grad=layout.trim_padding(grad,communicator.rank(),communicator.world_size()).map_err(MuonShardedError::Muon)?;
        let (update,momentum)=self.momentum_update(grad,state.map(|state|state.local));
        let update=self.zeropower_via_newtonschulz(layout.gather(update,&communicator)?);
        let update=layout.partition(update,communicator.rank(),communicator.world_size()).map_err(MuonShardedError::Muon)?;
        let adjusted=self.adjust_lr(lr,&layout.shape);
        let tensor=match self.weight_decay_penalty {Some(penalty)=>tensor.mul_scalar(1.0-lr*penalty as f64),None=>tensor};
        let tensor=layout.trim_padding(tensor-update.mul_scalar(adjusted),communicator.rank(),communicator.world_size()).map_err(MuonShardedError::Muon)?;
        let momentum=MomentumState::new(layout.trim_padding(momentum.velocity().clone(),communicator.rank(),communicator.world_size()).map_err(MuonShardedError::Muon)?);
        Ok((tensor,MuonFlatShardedState {version:1,rank:communicator.rank(),world:communicator.world_size(),layout:layout.clone(),local:MuonState::new(momentum)}))
    }
}
