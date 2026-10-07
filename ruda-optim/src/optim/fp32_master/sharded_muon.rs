use super::{Fp32MasterOptimizer,Fp32MasterState,GradientClipping,LearningRate,DType,Tensor,Backend};
use crate::{Muon,MuonError,MuonMatrixShardLayout,MuonShardedState,MuonShardedError};
use ruda_model::tensor::{BroadcastTensorCollective,TensorPrimitive};

impl<B:Backend> Fp32MasterState<B,2,crate::MuonState<B,2>> {
    /// Explicitly partition a loaded complete FP32 Muon master and its original momentum together.
    /// The model must use the same actual row/column interval and unchanged numerical configuration.
    pub fn into_muon_shard(self,layout:&MuonMatrixShardLayout,rank:u32,world:u32)
        -> Result<Fp32MasterState<B,2,MuonShardedState<B>>,MuonError> {
        if self.master.dtype() != DType::F32 {return Err(MuonError::DTypeMismatch("master"));}
        if layout.axis > 1 || rank >= world || layout.lengths.len() != world as usize {return Err(MuonError::InvalidConfig("invalid FP32 master shard placement"));}
        let global = self.master.dims();let mut local = global;local[layout.axis] = layout.lengths[rank as usize];
        if layout.global_shape(rank,world,local)? != global {return Err(MuonError::ShapeMismatch("global master"));}
        if let Some(state) = &self.inner {
            if state.momentum.velocity().dims() != global {return Err(MuonError::ShapeMismatch("global momentum"));}
            if state.momentum.velocity().dtype() != DType::F32 {return Err(MuonError::DTypeMismatch("momentum"));}
            if state.momentum.velocity().device() != self.master.device() {return Err(MuonError::DeviceMismatch("momentum"));}
        }
        let master = self.master.slice_dim(layout.axis,layout.range(rank)?);
        let inner = self.inner.map(|state|MuonShardedState::from_global_state(state,layout,rank,world)).transpose()?;
        Ok(Fp32MasterState {master,inner})
    }
}

impl Fp32MasterOptimizer<crate::MuonAdamWConfig> {
    /// Explicitly opt a generic native sharded-Muon/AdamW model into FP32 master updates.
    /// The wrapper's loss scale and optional clipping apply to both original optimizer groups.
    /// Muon norm clipping uses the complete matrix; AdamW retains its per-parameter clipping.
    /// Existing AdamW config clipping stays at its original adaptor stage before the wrapper.
    pub fn init_sharded<B,M,C>(&self,module: &M,parameters: &[crate::MuonShardedParameter<C>])
        -> Result<crate::Fp32MasterMuonShardedAdamW<M,B,C>,MuonShardedError<C::Error>>
        where B:ruda_model::tensor::backend::AutodiffBackend,M:ruda_model::module::AutodiffModule<B>,
            C:BroadcastTensorCollective<B::InnerBackend> {
        self.optimizer.init_sharded_fp32_master::<B,M,C>(module,parameters,self.gradient_scale,self.grad_clipping.clone())
    }
}

impl<B: Backend> Fp32MasterState<B,2,MuonShardedState<B>> {
    /// Move this rank's master and momentum together, retaining its declared placement.
    pub fn to_shard_device(self,device: &B::Device) -> Self {
        Self {master:self.master.to_device(device),inner:self.inner.map(|state|state.to_device(device))}
    }
}

impl<B: Backend> Fp32MasterOptimizer<Muon<B>> {
    /// Inspect native storage, FP32 state and actual rank layout without casting or communicating.
    /// Gradients may use FP32 accumulation or FP16/BF16 storage; conversion is explicit in the step.
    pub fn validate_step_sharded<C>(&self,lr: LearningRate,tensor: &Tensor<B,2>,grad: &Tensor<B,2>,
        state: Option<&Fp32MasterState<B,2,MuonShardedState<B>>>,layout: &MuonMatrixShardLayout,communicator: &C)
        -> Result<[usize;2],MuonShardedError<C::Error>>
        where C: BroadcastTensorCollective<B> {
        let validate = || -> Result<[usize;2],MuonError> {
            let shape = layout.global_shape(communicator.rank(),communicator.world_size(),tensor.dims())?;
            if !matches!(tensor.dtype(),DType::F32|DType::F16|DType::BF16) {
                return Err(MuonError::InvalidConfig("FP32 master Muon requires FP32/FP16/BF16 parameter storage"));
            }
            if tensor.dims() != grad.dims() {return Err(MuonError::ShapeMismatch("gradient"));}
            if !matches!(grad.dtype(),DType::F32|DType::F16|DType::BF16) {return Err(MuonError::DTypeMismatch("gradient"));}
            if tensor.device() != grad.device() {return Err(MuonError::DeviceMismatch("gradient"));}
            if let Some(state) = state {
                if state.master.dims() != tensor.dims() {return Err(MuonError::ShapeMismatch("master"));}
                if state.master.dtype() != DType::F32 {return Err(MuonError::DTypeMismatch("master"));}
                if state.master.device() != tensor.device() {return Err(MuonError::DeviceMismatch("master"));}
                if let Some(inner) = &state.inner {
                    inner.validate_placement(communicator.rank(),layout,shape)?;
                    if inner.momentum().dims() != tensor.dims() {return Err(MuonError::ShapeMismatch("momentum"));}
                    if inner.momentum().dtype() != DType::F32 {return Err(MuonError::DTypeMismatch("momentum"));}
                    if inner.momentum().device() != tensor.device() {return Err(MuonError::DeviceMismatch("momentum"));}
                }
            }
            self.optimizer.validate_effective_learning_rate(lr,&shape,DType::F32)?;
            Ok(shape)
        };
        validate().map_err(MuonShardedError::Muon)
    }

    /// Update an explicitly sharded matrix using FP32 master parameters and native Muon state.
    /// Loss-scale division precedes optional clipping. Norm clipping covers the original whole
    /// matrix, summing squared derivatives over unique actual shards, never clipping each rank
    /// independently. Separate replica/data-parallel reductions remain the caller's responsibility.
    /// Parameter storage remains FP32/FP16/BF16; the full-matrix Newton-Schulz method is unchanged.
    pub fn try_step_sharded<C>(&self,lr: LearningRate,tensor: Tensor<B,2>,grad: Tensor<B,2>,
        state: Option<Fp32MasterState<B,2,MuonShardedState<B>>>,layout: &MuonMatrixShardLayout,communicator: C)
        -> Result<(Tensor<B,2>,Fp32MasterState<B,2,MuonShardedState<B>>),MuonShardedError<C::Error>>
        where C: BroadcastTensorCollective<B> {
        self.validate_step_sharded(lr,&tensor,&grad,state.as_ref(),layout,&communicator)?;
        let storage_dtype = tensor.dtype();
        let (master,inner) = match state {Some(state)=>(state.master,state.inner),None=>(tensor.cast(DType::F32),None)};
        let grad = grad.cast(DType::F32);
        let grad = if self.gradient_scale == 1.0 {grad} else {grad/self.gradient_scale};
        let grad = match &self.grad_clipping {
            Some(GradientClipping::Norm(threshold)) if communicator.world_size() > 1 => {
                let sum = communicator.all_reduce_sum(grad.clone().square().sum().into_primitive().tensor())
                    .map_err(MuonShardedError::Collective)?;
                let norm = Tensor::<B,1>::from_primitive(TensorPrimitive::Float(sum)).sqrt();
                let coefficient = (*threshold/norm.add_scalar(ruda_model::tensor::FloatDType::F32.finfo().min_positive)).clamp_max(1.0);
                grad.mul(coefficient.unsqueeze())
            }
            Some(clipping)=>clipping.clip_gradient(grad),None=>grad,
        };
        let (master,inner) = self.optimizer.try_step_sharded(lr,master,grad,inner,layout,communicator)?;
        let tensor = master.clone().cast(storage_dtype);
        Ok((tensor,Fp32MasterState {master,inner:Some(inner)}))
    }
}
