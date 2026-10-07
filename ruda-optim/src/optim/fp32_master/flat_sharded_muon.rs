use super::*;
use crate::{Muon,MuonError,MuonFlatShardLayout,MuonFlatShardedState,MuonShardedError};
use ruda_model::tensor::BroadcastTensorCollective;

impl Fp32MasterOptimizer<crate::MuonAdamWConfig> {
    /// Attach explicit FP32 masters to complete flat-FSDP Muon/AdamW routing using the existing wrapper options.
    /// Loss-scale division and optional clipping are preserved; no dynamic scaling or skip policy is introduced.
    pub fn init_fully_sharded<B,M,C>(&self,module:&M,parameters:&[crate::FullyShardedOptimizerParameter<C>])
        -> Result<crate::FullyShardedMuonAdamW<M,B,C>,MuonShardedError<C::Error>>
        where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
        self.optimizer.init_fully_sharded_master(module,parameters,self.gradient_scale,self.grad_clipping.clone())
    }
}

impl<B:Backend> Fp32MasterState<B,1,MuonFlatShardedState<B>> {
    /// Partition a caller-loaded authoritative FP32 matrix/master and its actual original momentum together.
    /// Do not seed a master from already-rounded model storage when continuing a completed optimization run.
    pub fn from_global_muon_state(state:Fp32MasterState<B,2,crate::MuonState<B,2>>,layout:&MuonFlatShardLayout,rank:u32,world:u32)
        -> Result<Self,MuonError> {
        if state.master.dtype()!=DType::F32 {return Err(MuonError::DTypeMismatch("master"));}
        if let Some(inner)=&state.inner {
            if inner.momentum.velocity().dims()!=state.master.dims() {return Err(MuonError::ShapeMismatch("momentum"));}
            if inner.momentum.velocity().dtype()!=DType::F32 {return Err(MuonError::DTypeMismatch("momentum"));}
            if inner.momentum.velocity().device()!=state.master.device() {return Err(MuonError::DeviceMismatch("momentum"));}
        }
        let master=layout.partition(state.master,rank,world)?;
        let inner=state.inner.map(|state|MuonFlatShardedState::from_global_state(state,layout,rank,world)).transpose()?;
        Ok(Self {master,inner})
    }
    /// Move only the rank-local authoritative master and actual local momentum.
    pub fn to_flat_shard_device(self,device:&B::Device) -> Self {
        Self {master:self.master.to_device(device),inner:self.inner.map(|state|state.to_device(device))}
    }
}

impl<B:Backend> Fp32MasterOptimizer<Muon<B>> {
    /// Actual local storage/master/momentum and original matrix LR metadata, without implicit precision choices.
    pub fn validate_step_flat_sharded<C>(&self,lr:LearningRate,tensor:&Tensor<B,1>,grad:&Tensor<B,1>,
        state:Option<&Fp32MasterState<B,1,MuonFlatShardedState<B>>>,layout:&MuonFlatShardLayout,communicator:&C)
        -> Result<(),MuonShardedError<C::Error>> where C:BroadcastTensorCollective<B> {
        let validate=|| -> Result<(),MuonError> {
            layout.geometry(communicator.rank(),communicator.world_size(),tensor.dims()[0])?;
            if !matches!(tensor.dtype(),DType::F32|DType::F16|DType::BF16) {return Err(MuonError::InvalidConfig("FP32 flat masters require FP32/FP16/BF16 model storage"));}
            if grad.dims()!=tensor.dims() {return Err(MuonError::ShapeMismatch("gradient"));}
            if !matches!(grad.dtype(),DType::F32|DType::F16|DType::BF16) {return Err(MuonError::DTypeMismatch("gradient"));}
            if grad.device()!=tensor.device() {return Err(MuonError::DeviceMismatch("gradient"));}
            if let Some(state)=state {
                if state.master.dims()!=tensor.dims() {return Err(MuonError::ShapeMismatch("master"));}
                if state.master.dtype()!=DType::F32 {return Err(MuonError::DTypeMismatch("master"));}
                if state.master.device()!=tensor.device() {return Err(MuonError::DeviceMismatch("master"));}
                if let Some(inner)=&state.inner {
                    inner.validate_placement(communicator.rank(),communicator.world_size(),layout)?;
                    if inner.momentum().dtype()!=DType::F32 {return Err(MuonError::DTypeMismatch("momentum"));}
                    if inner.momentum().device()!=tensor.device() {return Err(MuonError::DeviceMismatch("momentum"));}
                }
            }
            self.optimizer.validate_effective_learning_rate(lr,&layout.shape,DType::F32)
        };validate().map_err(MuonShardedError::Muon)
    }
    /// Preserve the original FP32-master algorithm over the actual logical FSDP matrix, including optional clipping.
    /// Explicit loss-scale division precedes clipping. Norm clipping uses the original complete matrix and the
    /// original tensor clipper, not local-shard norms or a second data-gradient sum. Masters/momentum remain local.
    pub fn try_step_flat_sharded<C>(&self,lr:LearningRate,tensor:Tensor<B,1>,grad:Tensor<B,1>,
        state:Option<Fp32MasterState<B,1,MuonFlatShardedState<B>>>,layout:&MuonFlatShardLayout,communicator:C)
        -> Result<(Tensor<B,1>,Fp32MasterState<B,1,MuonFlatShardedState<B>>),MuonShardedError<C::Error>>
        where C:BroadcastTensorCollective<B> {
        self.validate_step_flat_sharded(lr,&tensor,&grad,state.as_ref(),layout,&communicator)?;
        let storage=tensor.dtype();
        let (master,inner)=match state {Some(state)=>(state.master,state.inner),None=>(tensor.cast(DType::F32),None)};
        let grad=grad.cast(DType::F32);let grad=if self.gradient_scale==1.0 {grad} else {grad/self.gradient_scale};
        let grad=match &self.grad_clipping {
            Some(GradientClipping::Norm(_))=>{
                let full=layout.gather(grad,&communicator)?;
                layout.partition(self.grad_clipping.as_ref().expect("selected original clipping").clip_gradient(full),communicator.rank(),communicator.world_size())
                    .map_err(MuonShardedError::Muon)?
            },
            Some(clipping)=>clipping.clip_gradient(grad),None=>grad,
        };
        let (master,inner)=self.optimizer.try_step_flat_sharded(lr,master,grad,inner,layout,communicator)?;
        Ok((master.clone().cast(storage),Fp32MasterState {master,inner:Some(inner)}))
    }
}
