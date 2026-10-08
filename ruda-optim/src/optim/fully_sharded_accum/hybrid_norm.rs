use super::*;
use ruda_model::tensor::TensorPrimitive;

/// Explicit local logical partition and its full-copy multiplicity in the norm group.
/// Tensor/expert partitions may have different IDs, shapes and data groups on each
/// rank. `replicas` counts complete copies of this logical partition across the
/// coordinator, not the ranks which collectively own one data-sharded copy.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct HybridShardedGradientNormParameter {
    pub parameter:ParamId,
    pub logical_shape:Vec<usize>,
    pub shard_rank:u32,
    pub shard_world:u32,
    pub replicas:u32,
}

impl HybridShardedGradientNormParameter {
    /// Original flat leaf ownership with an explicit, caller-known replica count.
    /// A tensor-parallel partition is a distinct logical partition; do not replace
    /// its actual shape with the unpartitioned matrix shape or count it as a copy.
    pub fn new(parameter:ParamId,logical_shape:Vec<usize>,shard_rank:u32,shard_world:u32,replicas:u32) -> Self {
        Self {parameter,logical_shape,shard_rank,shard_world,replicas}
    }

    /// Reuse an actual optimizer binding without retaining or invoking its transport.
    /// Bindings of different transport types can populate the same metadata list.
    pub fn from_shard<B,C>(binding:&FullyShardedOptimizerParameter<C>,replicas:u32) -> Self
    where B:AutodiffBackend,C:BroadcastTensorCollective<B::InnerBackend> {
        Self::new(binding.parameter,binding.logical_shape.clone(),binding.communicator.rank(),
            binding.communicator.world_size(),replicas)
    }
}

impl<B:Backend> Record<B> for HybridShardedGradientNormParameter {
    type Item<P:PrecisionSettings>=(u64,Vec<usize>,u32,u32,u32);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.parameter.val(),self.logical_shape,self.shard_rank,self.shard_world,self.replicas)
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,_device:&B::Device) -> Self {
        Self::new(ParamId::from(item.0),item.1,item.2,item.3,item.4)
    }
}

/// Native scalar statistics for one completed, explicitly declared norm reduction.
/// Retains scaled coordinates so clipping remains representable when the returned
/// global norm itself overflows the requested work dtype. This is not an automatic
/// clipping policy and contains no gradient vector or optimizer state.
#[derive(Debug)]
pub struct HybridShardedGradientNorm<B:AutodiffBackend> {
    norm:Tensor<B::InnerBackend,1>,
    safe_maximum:Tensor<B::InnerBackend,1>,
    scaled_root:Tensor<B::InnerBackend,1>,
    dtype:FloatDType,
}

impl<B:AutodiffBackend> HybridShardedGradientNorm<B> {
    /// Actual global L2 norm of the original unreplicated logical gradients.
    pub fn norm(&self) -> Tensor<B::InnerBackend,1> {self.norm.clone()}

    /// Explicit coefficient for max_norm/(norm+epsilon), capped at one.
    /// No finite-value replacement, optimizer skip or parameter update is applied.
    pub fn clipping_coefficient(&self,max_norm:f64,epsilon:f64)
        -> Result<Tensor<B::InnerBackend,1>,FullyShardedAccumulationError> {
        validate_limit(max_norm,epsilon,self.dtype)?;
        let device=self.norm.device();let dtype=DType::from(self.dtype);
        let limit=Tensor::<B::InnerBackend,1>::ones([1],(&device,dtype)).mul_scalar(max_norm)/self.safe_maximum.clone();
        let offset=Tensor::<B::InnerBackend,1>::ones([1],(&device,dtype)).mul_scalar(epsilon)/self.safe_maximum.clone();
        Ok((limit/(self.scaled_root.clone()+offset)).clamp_max(1))
    }
}

/// Read-only global norm over actual DP-sharded TP/expert-owned flat model leaves.
/// Each logical partition contributes sum(gradient^2)/replicas; the common
/// coordinator reduces only a native maximum scalar and a native sum scalar.
/// Every actual floating leaf, including frozen leaves, needs an explicit binding.
/// Tied IDs are counted once, missing gradients remain absent, and rank padding is
/// excluded. Different owners need not expose matching parameter lists or shapes.
/// The caller supplies a coordinator covering every partition and accurate copy
/// multiplicities; neither topology nor replica identity is inferred from sizes.
/// Call after the original normalization/unscaling, never on per-rank loss means.
pub fn measure_hybrid_sharded_gradient_norm<B,M,C>(module:&M,gradients:&GradientsParams,
    parameters:&[HybridShardedGradientNormParameter],communicator:C,dtype:FloatDType,device:&B::Device)
    -> Result<HybridShardedGradientNorm<B>,FullyShardedGradientNormError<C::Error>>
where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    measure::<B,M,C>(module,gradients,parameters,&communicator,dtype,device).map(|(_,statistics)|statistics)
}

/// Explicit opt-in clipping using the same original global logical norm.
/// Does not call any parameter's data-group reduction, normalize a loss, change
/// optimizer state or cast a gradient to parameter storage. Only present native
/// derivatives are replaced, in the caller's explicit F32/F64 work dtype; padding
/// becomes zero. Transport/metadata errors leave the source container unchanged.
pub fn clip_hybrid_sharded_gradient_norm<B,M,C>(module:&M,gradients:&mut GradientsParams,
    parameters:&[HybridShardedGradientNormParameter],communicator:C,max_norm:f64,epsilon:f64,
    dtype:FloatDType,device:&B::Device)
    -> Result<Tensor<B::InnerBackend,1>,FullyShardedGradientNormError<C::Error>>
where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    validate_work_dtype(dtype).map_err(|error|FullyShardedGradientNormError::Arguments(error.into()))?;
    validate_limit(max_norm,epsilon,dtype).map_err(FullyShardedGradientNormError::Arguments)?;
    let (mut proposed,statistics)=measure::<B,M,C>(module,gradients,parameters,&communicator,dtype,device)?;
    let coefficient=statistics.clipping_coefficient(max_norm,epsilon).map_err(FullyShardedGradientNormError::Arguments)?;
    for id in proposed.container.ids().into_iter().copied().collect::<Vec<_>>() {
        let value=proposed.remove::<B::InnerBackend,1>(id)
            .ok_or(FullyShardedGradientNormError::Arguments(FullyShardedAccumulationError::State))?;
        let device=value.device();proposed.register(id,value*coefficient.clone().to_device(&device));
    }
    *gradients=proposed;Ok(statistics.norm)
}

fn validate_limit(max_norm:f64,epsilon:f64,dtype:FloatDType) -> Result<(),FullyShardedAccumulationError> {
    if !representable(max_norm,dtype) || max_norm<=0.0 || !representable(epsilon,dtype) || epsilon<0.0
        || (dtype==FloatDType::F32 && max_norm as f32==0.0) {
        return Err(GradientTransformError::InvalidScalar.into());
    }
    Ok(())
}

fn measure<B,M,C>(module:&M,gradients:&GradientsParams,parameters:&[HybridShardedGradientNormParameter],
    communicator:&C,dtype:FloatDType,device:&B::Device)
    -> Result<(GradientsParams,HybridShardedGradientNorm<B>),FullyShardedGradientNormError<C::Error>>
where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    let invalid=FullyShardedGradientNormError::Arguments;
    validate_work_dtype(dtype).map_err(|error|invalid(error.into()))?;
    let world=communicator.world_size();
    if world==0 || communicator.rank()>=world {
        return Err(FullyShardedGradientNormError::Protocol("invalid explicit hybrid norm coordinator"));
    }
    let mut declared=Vec::with_capacity(parameters.len());let mut copies=BTreeMap::new();
    for parameter in parameters {
        if parameter.replicas==0 || copies.insert(parameter.parameter,parameter.replicas).is_some() {
            return Err(FullyShardedGradientNormError::Protocol("invalid replica multiplicity or duplicate local binding"));
        }
        declared.push((parameter.parameter.val(),parameter.logical_shape.clone(),parameter.shard_rank,
            parameter.shard_world,DType::F32,false));
    }
    declared.sort_by_key(|entry|entry.0);
    let placement=inspect::<B,M>(module,&declared,false).map_err(invalid)?;
    let placement:BTreeMap<_,_>=placement.into_iter().map(|entry|(entry.0,entry)).collect();
    let mut proposed=gradients.cast_for::<B,M>(module,dtype).map_err(|error|invalid(error.into()))?;
    let ids=proposed.container.ids().into_iter().copied().collect::<Vec<_>>();
    let mut local_maximum=Tensor::<B::InnerBackend,1>::zeros([1],(device,DType::from(dtype)));
    for id in &ids {
        let spec=placement.get(&id.val()).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        if !spec.5 {return Err(invalid(FullyShardedAccumulationError::Placement("frozen parameter has a supplied norm derivative")));}
        let value=proposed.get::<B::InnerBackend,1>(*id).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        let total=elements(&spec.1).map_err(invalid)?;let slots=total.div_ceil(spec.3 as usize);
        let real=total.saturating_sub(spec.2 as usize*slots).min(slots);
        if real>0 {local_maximum=local_maximum.max_pair(value.clone().slice([0..real]).abs().max().to_device(device));}
        if real<slots {
            let zeros=Tensor::zeros([slots-real],(&value.device(),value.dtype()));
            proposed.register(*id,value.slice_assign([real..slots],zeros));
        }
    }
    let maximum=if world==1 {local_maximum} else {Tensor::<B::InnerBackend,1>::from_primitive(TensorPrimitive::Float(
        communicator.all_gather_float(local_maximum.into_primitive().tensor()).map_err(FullyShardedGradientNormError::Collective)?))};
    if maximum.dims()!=[world as usize] || maximum.dtype()!=DType::from(dtype) || maximum.device()!=*device {
        return Err(FullyShardedGradientNormError::Protocol("hybrid norm maximum transport changed scalar storage/device"));
    }
    let maximum=maximum.max();let safe_maximum=maximum.clone().mask_fill(maximum.clone().equal_elem(0),1);
    let mut square_sum=Tensor::<B::InnerBackend,1>::zeros([1],(device,DType::from(dtype)));
    for id in &ids {
        let value=proposed.get::<B::InnerBackend,1>(*id).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        let gradient_device=value.device();let value=value/safe_maximum.clone().to_device(&gradient_device);
        let replicas=copies.get(id).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        square_sum=square_sum+value.square().sum().div_scalar(*replicas as f64).to_device(device);
    }
    let square_sum=if world==1 {square_sum} else {Tensor::<B::InnerBackend,1>::from_primitive(TensorPrimitive::Float(
        communicator.all_reduce_sum(square_sum.into_primitive().tensor()).map_err(FullyShardedGradientNormError::Collective)?))};
    if square_sum.dims()!=[1] || square_sum.dtype()!=DType::from(dtype) || square_sum.device()!=*device {
        return Err(FullyShardedGradientNormError::Protocol("hybrid norm sum transport changed scalar storage/device"));
    }
    let scaled_root=square_sum.sqrt();let norm=maximum*scaled_root.clone();
    Ok((proposed,HybridShardedGradientNorm {norm,safe_maximum,scaled_root,dtype}))
}
