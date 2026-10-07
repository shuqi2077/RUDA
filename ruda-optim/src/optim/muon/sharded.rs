use alloc::vec::Vec;
use core::fmt;
use serde::{Serialize,Deserialize};
use ruda_model::{record::{Record,PrecisionSettings},tensor::{Tensor,TensorPrimitive,BroadcastTensorCollective,backend::Backend}};
use super::{Muon,MuonState,MuonError,NewtonSchulzParams,LearningRate};

/// Explicit contiguous, nonoverlapping actual matrix shards in rank order.
/// Replica groups and data-parallel loss reductions must be handled separately.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct MuonMatrixShardLayout {
    /// Actual stored tensor axis: zero for row shards, one for column shards.
    pub axis: usize,
    /// Actual unpadded feature count on every participating rank, including this rank.
    pub lengths: Vec<usize>,
}

impl MuonMatrixShardLayout {
    /// Caller-declared physical matrix partitions; no weight layout or head group is guessed.
    pub fn new(axis: usize,lengths: Vec<usize>) -> Self {Self {axis,lengths}}

    /// Validate actual rank/local geometry and return the original complete matrix shape.
    pub fn global_shape(&self,rank: u32,world: u32,local: [usize;2]) -> Result<[usize;2],MuonError> {
        if self.axis > 1 || world == 0 || rank >= world || self.lengths.len() != world as usize || self.lengths.contains(&0) {
            return Err(MuonError::InvalidConfig("Muon matrix shards need a valid axis and actual positive rank lengths"));
        }
        if local[self.axis] != self.lengths[rank as usize] {return Err(MuonError::ShapeMismatch("rank-local matrix shard"));}
        let total = self.lengths.iter().try_fold(0usize,|sum,length|sum.checked_add(*length))
            .ok_or(MuonError::InvalidConfig("global Muon matrix axis overflows"))?;
        let mut global = local;global[self.axis] = total;
        if global.contains(&0) {return Err(MuonError::EmptyMatrix);}
        Ok(global)
    }

    /// Actual global position of this rank, without including gather padding slots.
    pub fn range(&self,rank: u32) -> Result<core::ops::Range<usize>,MuonError> {
        let rank = rank as usize;
        if rank >= self.lengths.len() {return Err(MuonError::InvalidConfig("Muon shard rank is outside its layout"));}
        let start = self.lengths[..rank].iter().try_fold(0usize,|sum,length|sum.checked_add(*length))
            .ok_or(MuonError::InvalidConfig("Muon shard position overflows"))?;
        let end = start.checked_add(self.lengths[rank]).ok_or(MuonError::InvalidConfig("Muon shard end overflows"))?;
        Ok(start..end)
    }
}

/// Explicit sharded Muon metadata or underlying tensor transport error.
#[derive(Debug)]
pub enum MuonShardedError<E: fmt::Debug> {
    /// Existing native Muon configuration/shape/storage/state validation failure.
    Muon(MuonError),
    /// Actual transport failure; no local-only update or host numerical fallback is applied.
    Collective(E),
}
impl<E: fmt::Debug> fmt::Display for MuonShardedError<E> {
    fn fmt(&self,f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Muon(error)=>write!(f,"{error}"),Self::Collective(error)=>write!(f,"sharded Muon collective failed: {error:?}")}
    }
}
impl<E: fmt::Debug> core::error::Error for MuonShardedError<E> {}

/// Native local momentum and exact rank/layout metadata for an explicit sharded Muon matrix.
#[derive(Clone)]
pub struct MuonShardedState<B: Backend> {
    version: u32,
    rank: u32,
    layout: MuonMatrixShardLayout,
    global_shape: [usize;2],
    local: MuonState<B,2>,
}

impl<B: Backend> MuonShardedState<B> {
    /// Actual owning rank, not a serialized root-rank replacement.
    pub fn rank(&self) -> u32 {self.rank}
    /// Original rank-ordered physical shard layout.
    pub fn layout(&self) -> &MuonMatrixShardLayout {&self.layout}
    /// Actual complete matrix shape used for orientation and learning-rate scaling.
    pub fn global_shape(&self) -> [usize;2] {self.global_shape}
    /// Validate the original record schema and actual matrix/rank placement without tensor updates.
    pub fn validate_placement(&self,rank: u32,layout: &MuonMatrixShardLayout,global_shape: [usize;2]) -> Result<(),MuonError> {
        if self.version != 1 || self.rank != rank || &self.layout != layout || self.global_shape != global_shape {Err(MuonError::IncompatibleRecord)} else {Ok(())}
    }
    /// Read actual native local momentum without gathering the complete buffer.
    pub fn momentum(&self) -> &Tensor<B,2> {self.local.momentum.velocity()}
    /// Move only this rank's actual momentum while retaining its original placement metadata.
    pub fn to_device(mut self,device: &B::Device) -> Self {self.local.momentum = self.local.momentum.to_device(device);self}
}

impl<B: Backend> Record<B> for MuonShardedState<B> {
    type Item<S: PrecisionSettings> = (u32,u32,usize,Vec<usize>,[usize;2],<MuonState<B,2> as Record<B>>::Item<S>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.rank,self.layout.axis,self.layout.lengths,self.global_shape,self.local.into_item::<S>())
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,rank:item.1,layout:MuonMatrixShardLayout::new(item.2,item.3),global_shape:item.4,
            local:MuonState::<B,2>::from_item::<S>(item.5,device)}
    }
}

fn sum<B,C,const D: usize>(value: Tensor<B,D>,communicator: &C) -> Result<Tensor<B,D>,MuonShardedError<C::Error>>
    where B: Backend,C: BroadcastTensorCollective<B> {
    let primitive = communicator.all_reduce_sum(value.into_primitive().tensor()).map_err(MuonShardedError::Collective)?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(primitive)))
}

fn maximum<B,C>(value: Tensor<B,1>,communicator: &C) -> Result<Tensor<B,1>,MuonShardedError<C::Error>>
    where B: Backend,C: BroadcastTensorCollective<B> {
    let primitive = communicator.all_gather_float(value.into_primitive().tensor()).map_err(MuonShardedError::Collective)?;
    Ok(Tensor::<B,1>::from_primitive(TensorPrimitive::Float(primitive)).max())
}

fn gather_matrix<B,C>(value: Tensor<B,2>,layout: &MuonMatrixShardLayout,communicator: &C) -> Result<Tensor<B,2>,MuonShardedError<C::Error>>
    where B: Backend,C: BroadcastTensorCollective<B> {
    let leading = value.swap_dims(0,layout.axis);let [local,other] = leading.dims();
    let slots = *layout.lengths.iter().max().expect("validated nonempty Muon rank layout");
    let total = slots.checked_mul(layout.lengths.len()).ok_or(MuonShardedError::Muon(MuonError::InvalidConfig("Muon gather padding overflows")))?;
    let padded = if local == slots {leading} else {
        let device = leading.device();let dtype = leading.dtype();
        Tensor::cat(alloc::vec![leading,Tensor::<B,2>::zeros([slots-local,other],(&device,dtype))],0)
    };
    let gathered = communicator.all_gather_float(padded.into_primitive().tensor()).map_err(MuonShardedError::Collective)?;
    let gathered = Tensor::<B,2>::from_primitive(TensorPrimitive::Float(gathered));
    assert_eq!(gathered.dims(),[total,other],"Muon transport returned incompatible matrix gather geometry");
    let parts = layout.lengths.iter().enumerate().map(|(rank,length)|gathered.clone().slice_dim(0,rank*slots..rank*slots+length)).collect();
    Ok(Tensor::cat(parts,0).swap_dims(0,layout.axis))
}

impl<B: Backend> Muon<B> {
    /// Validate actual shard/state metadata without launching an update or a collective.
    /// Returns the original full matrix shape used by the native numerical configuration.
    pub fn validate_step_sharded<C>(&self,lr: LearningRate,tensor: &Tensor<B,2>,grad: &Tensor<B,2>,state: Option<&MuonShardedState<B>>,
        layout: &MuonMatrixShardLayout,communicator: &C) -> Result<[usize;2],MuonShardedError<C::Error>>
        where C: BroadcastTensorCollective<B> {
        let shape = layout.global_shape(communicator.rank(),communicator.world_size(),tensor.dims()).map_err(MuonShardedError::Muon)?;
        if let Some(state) = state {
            state.validate_placement(communicator.rank(),layout,shape).map_err(MuonShardedError::Muon)?;
        }
        self.validate_step_shape(lr,tensor,grad,state.map(|state|&state.local),&shape).map_err(MuonShardedError::Muon)?;
        Ok(shape)
    }

    /// Explicit model-parallel full-matrix Muon, with native SGD/EMA/Nesterov and local momentum.
    /// Norms, orientation, quintic coefficients and LR scaling use the original complete matrix.
    /// Column-oriented shards reduce the full native left Gram matrix each iteration. Row-oriented
    /// shards gather the update matrix for the original native iteration, then slice the real result;
    /// this path retains a complete gradient/update matrix, not complete parameter/momentum buffers.
    /// All ranks must provide corresponding unique shards, matching global shape/dtype/configuration and collective
    /// order. Gradients must already be synchronized/unscaled as required by the caller's training regime.
    /// No shard-local orthogonalization, implicit data-parallel averaging, BF16 cast or parameter gather occurs.
    pub fn try_step_sharded<C>(&self,lr: LearningRate,tensor: Tensor<B,2>,grad: Tensor<B,2>,state: Option<MuonShardedState<B>>,
        layout: &MuonMatrixShardLayout,communicator: C) -> Result<(Tensor<B,2>,MuonShardedState<B>),MuonShardedError<C::Error>>
        where C: BroadcastTensorCollective<B> {
        let rank = communicator.rank();let world = communicator.world_size();
        let shape = self.validate_step_sharded(lr,&tensor,&grad,state.as_ref(),layout,&communicator)?;
        let (update,momentum) = self.momentum_update(grad,state.map(|state|state.local));
        let transposed = shape[0] > shape[1];let oriented_axis = if transposed {1-layout.axis} else {layout.axis};
        let update = if world == 1 {self.zeropower_via_newtonschulz(update)} else if oriented_axis == 0 {
            let global = gather_matrix(update,layout,&communicator)?;
            self.zeropower_via_newtonschulz(global).slice_dim(layout.axis,layout.range(rank).map_err(MuonShardedError::Muon)?)
        } else {
            let mut x = if transposed {update.swap_dims(0,1)} else {update};
            if self.stable_normalization {
                let scale = maximum(x.clone().abs().max(),&communicator)?.clamp_min(f32::MIN_POSITIVE);
                let scaled = x.div(scale.clone().unsqueeze());let floor = scale.recip().mul_scalar(self.epsilon);
                let norm = sum(scaled.clone().square().sum(),&communicator)?.sqrt().max_pair(floor);
                x = scaled.div(norm.unsqueeze());
            } else {
                let norm = sum(x.clone().powf_scalar(2.0).sum(),&communicator)?.sqrt().clamp_min(self.epsilon);
                x = x.div(norm.unsqueeze());
            }
            let NewtonSchulzParams {a,b,c,steps} = self.ns_params;
            for _ in 0..steps {
                let gram = sum(x.clone().matmul(x.clone().swap_dims(0,1)),&communicator)?;
                let squared = gram.clone().matmul(gram.clone());
                let polynomial = gram.mul_scalar(b).add(squared.mul_scalar(c));
                x = x.clone().mul_scalar(a).add(polynomial.matmul(x));
            }
            if transposed {x.swap_dims(0,1)} else {x}
        };
        let adjusted = self.adjust_lr(lr,&shape);
        let tensor = match self.weight_decay_penalty {Some(penalty)=>tensor.mul_scalar(1.0-lr*penalty as f64),None=>tensor};
        Ok((tensor-update.mul_scalar(adjusted),MuonShardedState {version:1,rank,layout:layout.clone(),global_shape:shape,local:MuonState::new(momentum)}))
    }
}
