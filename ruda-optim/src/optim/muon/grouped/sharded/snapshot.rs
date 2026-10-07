use super::{AutodiffBackend,AutodiffModule,MuonShardedParameter,MuonShardedError,MuonError,Muon,Manifest,Tensor,ParamId,GradientsParams,LearningRate,
    BroadcastTensorCollective,HashMap,TensorContainer,format};
use alloc::{vec::Vec,string::String};
use ruda_model::{module::{ModuleVisitor,Param},tensor::DType};

pub(super) struct Snapshot<B: AutodiffBackend> {pub manifest:Manifest,pub values:TensorContainer<ParamId>,backend:core::marker::PhantomData<B>}

struct Inspect<'a,B: AutodiffBackend,C: BroadcastTensorCollective<B::InnerBackend>> {
    bindings: &'a [MuonShardedParameter<C>],indices: &'a HashMap<ParamId,usize>,muon: &'a Muon<B::InnerBackend>,
    grads: Option<&'a GradientsParams>,lr: LearningRate,entries: HashMap<ParamId,(Vec<usize>,bool,String)>,values:TensorContainer<ParamId>,
    seen: usize,error:Option<MuonShardedError<C::Error>>,master:bool,backend:core::marker::PhantomData<B>,
}

pub(super) fn snapshot<B,M,C>(module: &M,bindings: &[MuonShardedParameter<C>],indices: &HashMap<ParamId,usize>,muon: &Muon<B::InnerBackend>,
    grads: Option<&GradientsParams>,lr: LearningRate) -> Result<Snapshot<B>,MuonShardedError<C::Error>>
    where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
    snapshot_with_precision::<B,M,C>(module,bindings,indices,muon,grads,lr,false)
}

pub(super) fn snapshot_master<B,M,C>(module: &M,bindings: &[MuonShardedParameter<C>],indices: &HashMap<ParamId,usize>,muon: &Muon<B::InnerBackend>,
    grads: Option<&GradientsParams>,lr: LearningRate) -> Result<Snapshot<B>,MuonShardedError<C::Error>>
    where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
    snapshot_with_precision::<B,M,C>(module,bindings,indices,muon,grads,lr,true)
}

fn snapshot_with_precision<B,M,C>(module: &M,bindings: &[MuonShardedParameter<C>],indices: &HashMap<ParamId,usize>,muon: &Muon<B::InnerBackend>,
    grads: Option<&GradientsParams>,lr: LearningRate,master:bool) -> Result<Snapshot<B>,MuonShardedError<C::Error>>
    where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
    let mut inspect = Inspect {bindings,indices,muon,grads,lr,entries:HashMap::new(),values:TensorContainer::new(),seen:0,error:None,master,backend:core::marker::PhantomData};
    module.visit(&mut inspect);
    if let Some(error) = inspect.error {return Err(error);}
    for binding in bindings {
        if !inspect.entries.contains_key(&binding.parameter) {return Err(MuonShardedError::Muon(MuonError::UnknownParameter(binding.parameter.val())));}
    }
    if grads.is_some_and(|grads|grads.len() != inspect.seen) {return Err(MuonShardedError::Muon(MuonError::UnusedGradients));}
    let mut manifest: Manifest = inspect.entries.into_iter().map(|(id,(shape,_,dtype))|(id.val(),shape,indices.contains_key(&id),dtype)).collect();
    manifest.sort_by_key(|entry|entry.0);
    Ok(Snapshot {manifest,values:inspect.values,backend:core::marker::PhantomData})
}

impl<B: AutodiffBackend,C: BroadcastTensorCollective<B::InnerBackend>> ModuleVisitor<B> for Inspect<'_,B,C> {
    fn visit_float<const D: usize>(&mut self,param: &Param<Tensor<B,D>>) {
        if self.error.is_some() {return;}
        let tensor = param.val();let shape = tensor.shape().to_vec();let trainable = param.is_require_grad();let dtype = format!("{:?}",tensor.dtype());
        if let Some(prior) = self.entries.get(&param.id) {
            if prior != &(shape,trainable,dtype) {self.error = Some(MuonShardedError::Muon(MuonError::ModelChanged));}
            return;
        }
        self.entries.insert(param.id,(shape,trainable,dtype));
        if D > 8 {self.error = Some(MuonShardedError::Muon(MuonError::InvalidConfig("optimizer records support at most rank 8")));return;}
        #[cfg(feature="distributed")]
        if tensor.is_distributed() {self.error = Some(MuonShardedError::Muon(MuonError::UnsupportedDistributed));return;}
        let selected = self.indices.get(&param.id).copied();
        if selected.is_some() && !trainable {self.error = Some(MuonShardedError::Muon(MuonError::FrozenParameter(param.id.val())));return;}
        if self.master && trainable && !matches!(tensor.dtype(),DType::F32|DType::F16|DType::BF16) {
            self.error = Some(MuonShardedError::Muon(MuonError::InvalidConfig("FP32 master optimizers require FP32/FP16/BF16 trainable storage")));return;
        }
        let inner = tensor.inner();
        let gradient = if trainable {self.grads.and_then(|grads|grads.get::<B::InnerBackend,D>(param.id))} else {None};
        if let Some(gradient) = &gradient {
            self.seen += 1;
            let compatible_dtype = if self.master {matches!(gradient.dtype(),DType::F32|DType::F16|DType::BF16)} else {gradient.dtype() == inner.dtype()};
            let error = if gradient.shape() != inner.shape() {Some(MuonError::ShapeMismatch("gradient"))}
                else if !compatible_dtype {Some(MuonError::DTypeMismatch("gradient"))}
                else if gradient.device() != inner.device() {Some(MuonError::DeviceMismatch("gradient"))} else {None};
            if let Some(error) = error {self.error = Some(MuonShardedError::Muon(error));return;}
        }
        if let Some(index) = selected {
            if D != 2 {self.error = Some(MuonShardedError::Muon(MuonError::ExpectedMatrix {rank:D}));return;}
            let binding = &self.bindings[index];
            let tensor = Tensor::<B::InnerBackend,2>::from_primitive(inner.clone().into_primitive());
            let gradient = gradient.map(|gradient|Tensor::<B::InnerBackend,2>::from_primitive(gradient.into_primitive()));
            let validated = if self.master {
                crate::Fp32MasterOptimizer::new((*self.muon).clone()).validate_step_sharded(self.lr,&tensor,gradient.as_ref().unwrap_or(&tensor),None,&binding.layout,&binding.communicator)
            } else {self.muon.validate_step_sharded(self.lr,&tensor,gradient.as_ref().unwrap_or(&tensor),None,&binding.layout,&binding.communicator)};
            if let Err(error) = validated {
                self.error = Some(error);return;
            }
            self.values.register::<B::InnerBackend>(param.id,inner.into_primitive());
        }
    }
}
