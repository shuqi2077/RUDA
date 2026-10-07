use super::*;
use ruda_model::tensor::backend::Backend;

pub(super) struct Snapshot<B:AutodiffBackend> {pub manifest:FlatManifest,pub values:TensorContainer<ParamId>,backend:PhantomData<B>}

pub(super) fn parameter_geometry(shape:&[usize],rank:u32,world:u32,local:usize) -> Result<(usize,usize),MuonError> {
    if shape.is_empty() || shape.contains(&0) || world==0 || rank>=world {return Err(MuonError::InvalidConfig("positive original FSDP axes and valid rank/world are required"));}
    let elements=shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis)).ok_or(MuonError::InvalidConfig("original FSDP parameter size overflows"))?;
    let slots=elements.div_ceil(world as usize);
    slots.checked_mul(world as usize).ok_or(MuonError::InvalidConfig("padded FSDP parameter size overflows"))?;
    if local!=slots {return Err(MuonError::ShapeMismatch("local FSDP parameter interval"));}Ok((elements,slots))
}

pub(super) fn trim_padding<B,C>(value:Tensor<B,1>,binding:&FullyShardedOptimizerParameter<C>) -> Result<Tensor<B,1>,MuonError>
    where B:Backend,C:BroadcastTensorCollective<B> {
    let rank=binding.communicator.rank();let world=binding.communicator.world_size();
    let (elements,slots)=parameter_geometry(&binding.logical_shape,rank,world,value.dims()[0])?;
    let start=rank as usize*slots;let real=elements.saturating_sub(start).min(slots);
    if real==slots {return Ok(value);}
    let mask=Tensor::<B,1,Bool>::zeros([slots],&value.device()).slice_assign([real..slots],Tensor::zeros([slots-real],&value.device()).bool_not());
    Ok(value.mask_fill(mask,0))
}

pub(super) fn clip_logical_gradient<B,C>(value:Tensor<B,1>,binding:&FullyShardedOptimizerParameter<C>,clipping:&GradientClipping)
    -> Result<Tensor<B,1>,MuonShardedError<C::Error>> where B:Backend,C:BroadcastTensorCollective<B> {
    if matches!(clipping,GradientClipping::Value(_)) {return Ok(clipping.clip_gradient(value));}
    let rank=binding.communicator.rank();let world=binding.communicator.world_size();
    let (elements,slots)=parameter_geometry(&binding.logical_shape,rank,world,value.dims()[0]).map_err(MuonShardedError::Muon)?;
    let dtype=value.dtype();let device=value.device();
    let full=if world==1 {value} else {
        let full=binding.communicator.all_gather_float(value.into_primitive().tensor()).map_err(MuonShardedError::Collective)?;
        Tensor::<B,1>::from_primitive(TensorPrimitive::Float(full))
    };
    if full.dims()!=[slots*world as usize] {return Err(MuonShardedError::Muon(MuonError::ShapeMismatch("FSDP complete clipping tensor")));}
    if full.dtype()!=dtype {return Err(MuonShardedError::Muon(MuonError::DTypeMismatch("FSDP complete clipping tensor")));}
    if full.device()!=device {return Err(MuonShardedError::Muon(MuonError::DeviceMismatch("FSDP complete clipping tensor")));}
    // The original native clipper's norm spans every real element, not each shard's physical vector.
    let full=clipping.clip_gradient(full.slice([0..elements]));let start=rank as usize*slots;let real=elements.saturating_sub(start).min(slots);
    let mut local=Tensor::zeros([slots],(&device,dtype));
    if real>0 {local=local.slice_assign([0..real],full.slice([start..start+real]));}Ok(local)
}

struct Inspect<'a,B:AutodiffBackend,C:BroadcastTensorCollective<B::InnerBackend>> {
    bindings:&'a [FullyShardedOptimizerParameter<C>],indices:&'a HashMap<ParamId,usize>,muon:&'a Muon<B::InnerBackend>,
    master:bool,grads:Option<&'a GradientsParams>,lr:LearningRate,entries:HashMap<ParamId,(usize,bool,String)>,values:TensorContainer<ParamId>,
    seen:usize,error:Option<MuonShardedError<C::Error>>,backend:PhantomData<B>,
}

pub(super) fn snapshot<B,M,C>(module:&M,bindings:&[FullyShardedOptimizerParameter<C>],indices:&HashMap<ParamId,usize>,muon:&Muon<B::InnerBackend>,
    master:bool,grads:Option<&GradientsParams>,lr:LearningRate) -> Result<Snapshot<B>,MuonShardedError<C::Error>>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    let mut inspect=Inspect {bindings,indices,muon,master,grads,lr,entries:HashMap::new(),values:TensorContainer::new(),seen:0,error:None,backend:PhantomData};
    module.visit(&mut inspect);
    if let Some(error)=inspect.error {return Err(error);}
    if inspect.entries.len()!=bindings.len() {return Err(MuonShardedError::Muon(MuonError::ModelChanged));}
    for binding in bindings {
        if !inspect.entries.contains_key(&binding.parameter) {return Err(MuonShardedError::Muon(MuonError::UnknownParameter(binding.parameter.val())));}
    }
    if grads.is_some_and(|grads|grads.len()!=inspect.seen) {return Err(MuonShardedError::Muon(MuonError::UnusedGradients));}
    let mut manifest=inspect.entries.into_iter().map(|(id,(size,trainable,dtype))|(id.val(),size,trainable,dtype)).collect::<FlatManifest>();
    manifest.sort_by_key(|entry|entry.0);Ok(Snapshot {manifest,values:inspect.values,backend:PhantomData})
}

impl<B:AutodiffBackend,C:BroadcastTensorCollective<B::InnerBackend>> ModuleVisitor<B> for Inspect<'_,B,C> {
    fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {
        if self.error.is_some() {return;}
        let value=param.val();let signature=(value.shape().num_elements(),param.is_require_grad(),format!("{:?}",value.dtype()));
        if let Some(previous)=self.entries.get(&param.id) {
            if previous!=&signature {self.error=Some(MuonShardedError::Muon(MuonError::ModelChanged));}
            if let Some(previous)=self.values.get::<B::InnerBackend>(&param.id) {
                let previous=Tensor::<B::InnerBackend,1>::from_primitive(previous);
                if previous.device()!=value.device() {self.error=Some(MuonShardedError::Muon(MuonError::ModelChanged));}
            }
            return;
        }
        if D!=1 {self.error=Some(MuonShardedError::Muon(MuonError::ShapeMismatch("FSDP model leaves must be flat vectors")));return;}
        #[cfg(feature="distributed")]
        if value.is_distributed() {self.error=Some(MuonShardedError::Muon(MuonError::UnsupportedDistributed));return;}
        let Some(&index)=self.indices.get(&param.id) else {self.error=Some(MuonShardedError::Muon(MuonError::UnknownParameter(param.id.val())));return;};
        let binding=&self.bindings[index];let inner=value.inner();
        if let Err(error)=parameter_geometry(&binding.logical_shape,binding.communicator.rank(),binding.communicator.world_size(),signature.0) {
            self.error=Some(MuonShardedError::Muon(error));return;
        }
        if binding.use_muon && !signature.1 {self.error=Some(MuonShardedError::Muon(MuonError::FrozenParameter(param.id.val())));return;}
        if self.master && signature.1 && !matches!(inner.dtype(),DType::F32|DType::F16|DType::BF16) {
            self.error=Some(MuonShardedError::Muon(MuonError::InvalidConfig("FP32 FSDP masters require FP32/FP16/BF16 model storage")));return;
        }
        let gradient=if signature.1 {self.grads.and_then(|grads|grads.get::<B::InnerBackend,1>(param.id))} else {None};
        if let Some(gradient)=&gradient {
            self.seen+=1;
            let dtype=if self.master {matches!(gradient.dtype(),DType::F32|DType::F16|DType::BF16)} else {gradient.dtype()==inner.dtype()};
            let failure=if gradient.dims()!=[signature.0] {Some(MuonError::ShapeMismatch("gradient"))}
                else if !dtype {Some(MuonError::DTypeMismatch("gradient"))}
                else if gradient.device()!=inner.device() {Some(MuonError::DeviceMismatch("gradient"))} else {None};
            if let Some(error)=failure {self.error=Some(MuonShardedError::Muon(error));return;}
        }
        if binding.use_muon {
            let Ok(shape)=binding.logical_shape.as_slice().try_into() else {self.error=Some(MuonShardedError::Muon(MuonError::ExpectedMatrix {rank:binding.logical_shape.len()}));return;};
            let layout=MuonFlatShardLayout::new(shape);
            let tensor=Tensor::<B::InnerBackend,1>::from_primitive(inner.clone().into_primitive());
            let validated=if self.master {Fp32MasterOptimizer::new(self.muon.clone()).validate_step_flat_sharded(self.lr,&tensor,gradient.as_ref().unwrap_or(&tensor),None,&layout,&binding.communicator)}
                else {self.muon.validate_step_flat_sharded(self.lr,&tensor,gradient.as_ref().unwrap_or(&tensor),None,&layout,&binding.communicator)};
            if let Err(error)=validated {self.error=Some(error);return;}
        }
        self.entries.insert(param.id,signature);self.values.register::<B::InnerBackend>(param.id,inner.into_primitive());
    }
}
