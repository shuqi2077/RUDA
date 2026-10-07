use super::*;
use ruda_model::tensor::TensorPrimitive;

/// Actual native whole-model FSDP norm calculation/transport error.
#[derive(Debug)]
pub enum FullyShardedGradientNormError<E:fmt::Debug> {
    /// Original scalar collective failed.
    Collective(E),
    /// Local gradient/module/ownership metadata or explicit numerical arguments are invalid.
    Arguments(FullyShardedAccumulationError),
    /// Scalar collective changed native shape/precision/device or original topology.
    Protocol(&'static str),
}
impl<E:fmt::Debug> fmt::Display for FullyShardedGradientNormError<E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Collective(value)=>write!(f,"FSDP norm collective: {value:?}"),Self::Arguments(value)=>fmt::Display::fmt(value,f),
            Self::Protocol(value)=>write!(f,"FSDP norm protocol: {value}")}
    }
}
impl<E:fmt::Debug> core::error::Error for FullyShardedGradientNormError<E> {}

/// Measure/clip the actual GLOBAL L2 norm across every selected local shard of one complete model.
/// This reduces scalar norm statistics only, never already SUM/reduce-scattered gradient vectors.
/// Use the explicit common data group for these placements, after normalization/unscaling and before native casts.
/// The stable maximum-scaled sum avoids squaring large finite gradients directly; work precision is explicit.
/// Absent leaves remain absent, rank padding is excluded, and no nonfinite/optimizer skip policy is applied.
pub fn clip_fully_sharded_gradient_norm<B,M,C>(module:&M,gradients:&mut GradientsParams,parameters:&[FullyShardedOptimizerParameter<C>],
    communicator:C,max_norm:f64,epsilon:f64,dtype:FloatDType,device:&B::Device)
    -> Result<Tensor<B::InnerBackend,1>,FullyShardedGradientNormError<C::Error>>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    let invalid=|value|FullyShardedGradientNormError::Arguments(value);
    validate_work_dtype(dtype).map_err(|error|invalid(error.into()))?;
    if !representable(max_norm,dtype) || max_norm<=0.0 || !representable(epsilon,dtype) || epsilon<0.0
        || (dtype==FloatDType::F32 && max_norm as f32==0.0) {
        return Err(invalid(GradientTransformError::InvalidScalar.into()));
    }
    let world=communicator.world_size();let rank=communicator.rank();
    if world==0 || rank>=world {return Err(FullyShardedGradientNormError::Protocol("invalid explicit norm data group"));}
    let mut declared=Vec::with_capacity(parameters.len());let mut ids=BTreeSet::new();
    for binding in parameters {
        if binding.communicator.rank()!=rank || binding.communicator.world_size()!=world || !ids.insert(binding.parameter) {
            return Err(FullyShardedGradientNormError::Protocol("norm parameters do not share the declared data ownership"));
        }
        declared.push((binding.parameter.val(),binding.logical_shape.clone(),rank,world,DType::F32,false));
    }
    declared.sort_by_key(|entry|entry.0);let placement=inspect::<B,M>(module,&declared,false).map_err(invalid)?;
    let mut proposed=gradients.cast_for::<B,M>(module,dtype).map_err(|error|invalid(error.into()))?;
    let mut local_maximum=Tensor::<B::InnerBackend,1>::zeros([1],(device,DType::from(dtype)));
    for id in proposed.container.ids().into_iter().copied().collect::<Vec<_>>() {
        let spec=placement.iter().find(|entry|entry.0==id.val()).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        if !spec.5 {return Err(invalid(FullyShardedAccumulationError::Placement("frozen parameter has a supplied norm derivative")));}
        let value=proposed.get::<B::InnerBackend,1>(id).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        let total=elements(&spec.1).map_err(invalid)?;let slots=total.div_ceil(world as usize);
        let real=total.saturating_sub(rank as usize*slots).min(slots);
        if real>0 {local_maximum=local_maximum.max_pair(value.clone().slice([0..real]).abs().max().to_device(device));}
        if real<slots {
            let zeros=Tensor::zeros([slots-real],(&value.device(),value.dtype()));
            proposed.register(id,value.slice_assign([real..slots],zeros));
        }
    }
    let maximum=if world==1 {local_maximum} else {Tensor::<B::InnerBackend,1>::from_primitive(TensorPrimitive::Float(
        communicator.all_gather_float(local_maximum.into_primitive().tensor()).map_err(FullyShardedGradientNormError::Collective)?))};
    if maximum.dims()!=[world as usize] || maximum.dtype()!=DType::from(dtype) || maximum.device()!=*device {
        return Err(FullyShardedGradientNormError::Protocol("norm maximum transport changed scalar storage/device"));
    }
    let maximum=maximum.max();let safe_maximum=maximum.clone().mask_fill(maximum.clone().equal_elem(0),1);
    let mut square_sum=Tensor::<B::InnerBackend,1>::zeros([1],(device,DType::from(dtype)));
    for id in proposed.container.ids() {
        let value=proposed.get::<B::InnerBackend,1>(*id).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        let gradient_device=value.device();let value=value/safe_maximum.clone().to_device(&gradient_device);
        square_sum=square_sum+value.square().sum().to_device(device);
    }
    let square_sum=if world==1 {square_sum} else {Tensor::<B::InnerBackend,1>::from_primitive(TensorPrimitive::Float(
        communicator.all_reduce_sum(square_sum.into_primitive().tensor()).map_err(FullyShardedGradientNormError::Collective)?))};
    if square_sum.dims()!=[1] || square_sum.dtype()!=DType::from(dtype) || square_sum.device()!=*device {
        return Err(FullyShardedGradientNormError::Protocol("norm sum transport changed scalar storage/device"));
    }
    let root=square_sum.sqrt();let norm=maximum*root.clone();
    // Compute the ratio in scaled coordinates as well: an unrepresentably large native norm
    // must not turn a finite, representable clipping coefficient into an accidental zero.
    let limit=Tensor::<B::InnerBackend,1>::ones([1],(device,DType::from(dtype))).mul_scalar(max_norm)/safe_maximum.clone();
    let offset=Tensor::<B::InnerBackend,1>::ones([1],(device,DType::from(dtype))).mul_scalar(epsilon)/safe_maximum;
    let coefficient=(limit/(root+offset)).clamp_max(1);
    for id in proposed.container.ids().into_iter().copied().collect::<Vec<_>>() {
        let value=proposed.remove::<B::InnerBackend,1>(id).ok_or_else(||invalid(FullyShardedAccumulationError::State))?;
        let device=value.device();proposed.register(id,value*coefficient.clone().to_device(&device));
    }
    *gradients=proposed;Ok(norm)
}
