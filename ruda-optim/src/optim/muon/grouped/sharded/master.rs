use super::{AutodiffBackend,AutodiffModule,BroadcastTensorCollective,MuonShardedParameter,MuonShardedError,MuonError,MuonAdamWConfig,
    Muon,MuonShardedState,Tensor,TensorContainer,HashMap,ParamId,Vec,String,Manifest,Placement,Updates,used,format,
    Optimizer,OptimizerAdaptor,AdamW,GradientsParams,MultiGradientsParams,LearningRate,Record,PrecisionSettings,valid_lr};
use super::snapshot::snapshot_master;
use crate::{Fp32MasterOptimizer,Fp32MasterState,grad_clipping::GradientClipping,record::{AdaptorRecord,AdaptorRecordV1}};
use ruda_model::tensor::DType;

type Masters<B: AutodiffBackend> = HashMap<ParamId,Fp32MasterState<<B as AutodiffBackend>::InnerBackend,2,MuonShardedState<<B as AutodiffBackend>::InnerBackend>>>;
type AdamMasters<B: AutodiffBackend> = HashMap<ParamId,AdaptorRecord<Fp32MasterOptimizer<AdamW>,B>>;

/// Explicit FP32 master updates for model-level sharded Muon and auxiliary native AdamW.
/// Original model storage, tied parameter IDs and missing-gradient semantics are retained.
/// No master precision, clipping, gradient scaling or non-finite policy is enabled implicitly.
#[derive(Clone)]
pub struct Fp32MasterMuonShardedAdamW<M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    muon:Fp32MasterOptimizer<Muon<B::InnerBackend>>,native_muon:Muon<B::InnerBackend>,states:Masters<B>,
    adamw:OptimizerAdaptor<Fp32MasterOptimizer<AdamW>,M,B>,bindings:Vec<MuonShardedParameter<C>>,indices:HashMap<ParamId,usize>,
    manifest:Manifest,config_key:String,adamw_lr_ratio:f64,
}

impl MuonAdamWConfig {
    pub(crate) fn init_sharded_fp32_master<B,M,C>(&self,module: &M,parameters: &[MuonShardedParameter<C>],scale:f32,clipping:Option<GradientClipping>)
        -> Result<Fp32MasterMuonShardedAdamW<M,B,C>,MuonShardedError<C::Error>>
        where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
        self.muon.validate().map_err(MuonShardedError::Muon)?;
        self.adamw.validate_hyperparameters().map_err(|error|MuonShardedError::Muon(MuonError::InvalidConfig(error)))?;
        valid_lr(self.adamw_lr_ratio).map_err(MuonShardedError::Muon)?;
        if parameters.is_empty() {return Err(MuonShardedError::Muon(MuonError::EmptyMuonGroup));}
        let mut bindings = parameters.to_vec();bindings.sort_by_key(|binding|binding.parameter);
        let mut indices = HashMap::new();
        for (index,binding) in bindings.iter().enumerate() {
            if indices.insert(binding.parameter,index).is_some() {return Err(MuonShardedError::Muon(MuonError::DuplicateParameter(binding.parameter.val())));}
        }
        let native_muon:Muon<B::InnerBackend> = self.muon.try_build().map_err(MuonShardedError::Muon)?;
        let inspected = snapshot_master::<B,M,C>(module,&bindings,&indices,&native_muon,None,0.0)?;
        let original_adamw = self.adamw.init::<B,M>();
        let original_clipping = original_adamw.grad_clipping().cloned();
        let mut muon = Fp32MasterOptimizer::new(native_muon.clone()).with_gradient_scale(scale);
        let mut adamw = Fp32MasterOptimizer::new(original_adamw.optim().clone()).with_gradient_scale(scale);
        let clip_key = match &clipping {Some(GradientClipping::Value(value))=>format!("value:{value:?}"),
            Some(GradientClipping::Norm(value))=>format!("norm:{value:?}"),None=>String::from("none")};
        if let Some(clipping) = clipping {muon = muon.with_grad_clipping(clipping.clone());adamw = adamw.with_grad_clipping(clipping);}
        let mut adamw = adamw.init();
        if let Some(clipping) = original_clipping {adamw = adamw.with_grad_clipping(clipping);}
        Ok(Fp32MasterMuonShardedAdamW {muon,native_muon,states:HashMap::new(),adamw,bindings,indices,manifest:inspected.manifest,
            config_key:format!("fp32-master-sharded-muon-adamw-v1:{self:?}:scale:{scale:?}:clip:{clip_key}"),adamw_lr_ratio:self.adamw_lr_ratio})
    }
}

impl<M,B,C> Fp32MasterMuonShardedAdamW<M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    /// Count unique explicitly selected physical matrix shards, excluding tied aliases.
    pub fn muon_parameter_count(&self) -> usize {self.bindings.len()}

    fn placement(&self) -> Placement {
        self.bindings.iter().map(|binding|(binding.parameter.val(),binding.communicator.rank(),binding.communicator.world_size(),binding.layout.clone())).collect()
    }

    /// Native mixed-precision model update with independently supplied Muon and AdamW rates.
    /// A globally unused matrix preserves its master/momentum; an absent local derivative is zero
    /// when another actual shard uses the matrix. Norm clipping therefore covers all real shards.
    /// Proposed optimizer state commits only after transport succeeds, not as a device transaction.
    pub fn try_step_with_lrs(&mut self,muon_lr:LearningRate,adamw_lr:LearningRate,module:M,mut grads:GradientsParams)
        -> Result<M,MuonShardedError<C::Error>> {
        valid_lr(muon_lr).map_err(MuonShardedError::Muon)?;valid_lr(adamw_lr).map_err(MuonShardedError::Muon)?;
        let inspected = snapshot_master::<B,M,C>(&module,&self.bindings,&self.indices,&self.native_muon,Some(&grads),muon_lr)?;
        if inspected.manifest != self.manifest {return Err(MuonShardedError::Muon(MuonError::ModelChanged));}
        let mut states = self.states.clone();
        let mut mapper = Updates::<B> {values:TensorContainer::new(),updated:TensorContainer::new(),backend:core::marker::PhantomData};
        for binding in &self.bindings {
            let id = binding.parameter;
            let tensor = Tensor::<B::InnerBackend,2>::from_primitive(inspected.values.get::<B::InnerBackend>(&id).expect("validated actual Muon parameter"));
            let gradient = grads.remove::<B::InnerBackend,2>(id);
            if !used::<B::InnerBackend,C>(gradient.is_some(),&tensor.device(),&binding.communicator)? {continue;}
            let gradient = gradient.unwrap_or_else(||Tensor::zeros(tensor.dims(),(&tensor.device(),DType::F32)));
            let state = states.remove(&id).map(|state|if state.master.device() == tensor.device() {state} else {state.to_shard_device(&tensor.device())});
            let (tensor,state) = self.muon.try_step_sharded(muon_lr,tensor,gradient,state,&binding.layout,binding.communicator.clone())?;
            mapper.values.register::<B::InnerBackend>(id,tensor.into_primitive());states.insert(id,state);
        }
        let module = module.map(&mut mapper);
        let mut adamw = self.adamw.clone();let module = adamw.step(adamw_lr,module,grads);
        self.states = states;self.adamw = adamw;
        Ok(module)
    }

    /// Restore corresponding model IDs, storage formats, configuration and physical placement.
    /// FP32 master and optimizer buffers must stay FP32; use a full-precision recorder for continuation.
    pub fn try_load_record(mut self,record:Fp32MasterMuonShardedAdamWRecord<B>) -> Result<Self,MuonError> {
        if record.version != 1 || record.config_key != self.config_key || record.manifest != self.manifest || record.placement != self.placement() {
            return Err(MuonError::IncompatibleRecord);
        }
        let known:HashMap<_,_> = self.manifest.iter().map(|entry|(ParamId::from(entry.0),entry)).collect();
        for (id,state) in &record.muon {
            let index = self.indices.get(id).ok_or(MuonError::IncompatibleRecord)?;let binding = &self.bindings[*index];
            let expected = known.get(id).ok_or(MuonError::IncompatibleRecord)?;
            let shape:[usize;2] = expected.1.as_slice().try_into().map_err(|_|MuonError::IncompatibleRecord)?;
            let global = binding.layout.global_shape(binding.communicator.rank(),binding.communicator.world_size(),shape)?;
            if state.master.dims() != shape || state.master.dtype() != DType::F32 {return Err(MuonError::IncompatibleRecord);}
            let inner = state.inner.as_ref().ok_or(MuonError::IncompatibleRecord)?;
            inner.validate_placement(binding.communicator.rank(),&binding.layout,global)?;
            if inner.momentum().dims() != shape || inner.momentum().dtype() != DType::F32 || inner.momentum().device() != state.master.device() {
                return Err(MuonError::IncompatibleRecord);
            }
        }
        for (id,state) in &record.adamw {
            if self.indices.contains_key(id) {return Err(MuonError::IncompatibleRecord);}
            let expected = known.get(id).ok_or(MuonError::IncompatibleRecord)?;
            macro_rules! check {
                ($state:expr) => {{
                    let master = &$state.master;let inner = $state.inner.as_ref().ok_or(MuonError::IncompatibleRecord)?;let m = &inner.momentum;
                    if master.shape().to_vec() != expected.1 || master.dtype() != DType::F32 || m.time == 0
                        || m.moment_1.shape().to_vec() != expected.1 || m.moment_2.shape().to_vec() != expected.1
                        || m.moment_1.dtype() != DType::F32 || m.moment_2.dtype() != DType::F32
                        || m.moment_1.device() != master.device() || m.moment_2.device() != master.device()
                        || m.max_moment_2.as_ref().is_some_and(|value|value.shape().to_vec() != expected.1 || value.dtype() != DType::F32 || value.device() != master.device()) {
                        return Err(MuonError::IncompatibleRecord);
                    }
                }};
            }
            match state {AdaptorRecord::V1(state)=>match state {
                AdaptorRecordV1::Rank0(value)=>check!(value),AdaptorRecordV1::Rank1(value)=>check!(value),AdaptorRecordV1::Rank2(value)=>check!(value),
                AdaptorRecordV1::Rank3(value)=>check!(value),AdaptorRecordV1::Rank4(value)=>check!(value),AdaptorRecordV1::Rank5(value)=>check!(value),
                AdaptorRecordV1::Rank6(value)=>check!(value),AdaptorRecordV1::Rank7(value)=>check!(value),AdaptorRecordV1::Rank8(value)=>check!(value),
            }}
        }
        drop(known);
        self.states = record.muon;self.adamw = self.adamw.load_record(record.adamw);
        Ok(self)
    }
}

/// Paired rank-local FP32 masters and optimizer buffers with original native routing metadata.
/// Save the model record alongside it; transports and credentials are not serialized.
#[derive(Clone)]
pub struct Fp32MasterMuonShardedAdamWRecord<B:AutodiffBackend> {
    version:u32,config_key:String,manifest:Manifest,placement:Placement,muon:Masters<B>,adamw:AdamMasters<B>,
}
impl<B:AutodiffBackend> Fp32MasterMuonShardedAdamWRecord<B> {
    /// Actual saved selected matrix masters and momentum states.
    pub fn muon_state_count(&self) -> usize {self.muon.len()}
    /// Actual saved auxiliary AdamW masters and moments, excluding unused/frozen parameters.
    pub fn adamw_state_count(&self) -> usize {self.adamw.len()}
}
impl<B:AutodiffBackend> Record<B> for Fp32MasterMuonShardedAdamWRecord<B> {
    type Item<S:PrecisionSettings> = (u32,String,Manifest,Placement,<Masters<B> as Record<B::InnerBackend>>::Item<S>,<AdamMasters<B> as Record<B>>::Item<S>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.config_key,self.manifest,self.placement,<Masters<B> as Record<B::InnerBackend>>::into_item::<S>(self.muon),
            <AdamMasters<B> as Record<B>>::into_item::<S>(self.adamw))
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,config_key:item.1,manifest:item.2,placement:item.3,muon:<Masters<B> as Record<B::InnerBackend>>::from_item::<S>(item.4,device),
            adamw:<AdamMasters<B> as Record<B>>::from_item::<S>(item.5,device)}
    }
}
impl<M,B,C> Optimizer<M,B> for Fp32MasterMuonShardedAdamW<M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    type Record = Fp32MasterMuonShardedAdamWRecord<B>;
    fn step(&mut self,lr:LearningRate,module:M,grads:GradientsParams) -> M {
        self.try_step_with_lrs(lr,lr*self.adamw_lr_ratio,module,grads).unwrap_or_else(|error|panic!("{error}"))
    }
    fn step_multi(&mut self,_lr:LearningRate,_module:M,_grads:MultiGradientsParams) -> M {
        panic!("explicit FP32 master Muon matrix shards use step on each actual rank, not implicit step_multi tensor flattening")
    }
    fn to_record(&self) -> Self::Record {
        Fp32MasterMuonShardedAdamWRecord {version:1,config_key:self.config_key.clone(),manifest:self.manifest.clone(),placement:self.placement(),
            muon:self.states.clone(),adamw:self.adamw.to_record()}
    }
    fn load_record(self,record:Self::Record) -> Self {self.try_load_record(record).unwrap_or_else(|error|panic!("{error}"))}
}
