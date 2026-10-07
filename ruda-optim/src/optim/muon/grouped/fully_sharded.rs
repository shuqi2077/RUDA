use super::*;
use super::sharded::{used,Updates};
use crate::{SimpleOptimizer,AdamWState,Fp32MasterOptimizer,Fp32MasterState,grad_clipping::GradientClipping,
    MuonFlatShardLayout,MuonFlatShardedState,MuonShardedError};
use ruda_model::tensor::{DType,TensorPrimitive,BroadcastTensorCollective,Bool,container::TensorContainer};

mod snapshot;
use snapshot::{snapshot,parameter_geometry,trim_padding,clip_logical_gradient};
mod state;
pub use state::{FullyShardedMuonAdamWRecord,FullyShardedMuonAdamWState};

type States<B:AutodiffBackend> = HashMap<ParamId,FullyShardedMuonAdamWState<<B as AutodiffBackend>::InnerBackend>>;
type FlatManifest=Vec<(u64,usize,bool,String)>;
type Placement=Vec<(u64,Vec<usize>,u32,u32,bool)>;

/// Actual original logical parameter identity and flat data-shard ownership, with an explicit optimizer role.
/// Bind every native floating parameter, including omitted/frozen roles, without classifying shapes or names.
#[derive(Clone,Debug)]
pub struct FullyShardedOptimizerParameter<C> {
    /// Actual canonical model parameter ID, not a per-role replacement identity.
    pub parameter:ParamId,
    /// Original logical axes, excluding all local rank padding.
    pub logical_shape:Vec<usize>,
    /// True only for explicitly selected original hidden matrices; other roles use original AdamW.
    pub use_muon:bool,
    /// Actual data group for this parameter; unrelated DP/TP groups are not inferred.
    pub communicator:C,
}
impl<C> FullyShardedOptimizerParameter<C> {
    /// Declare an actual loaded parameter's logical shape, transport and exact optimizer role.
    pub fn new(parameter:ParamId,logical_shape:Vec<usize>,use_muon:bool,communicator:C) -> Self {
        Self {parameter,logical_shape,use_muon,communicator}
    }
}

#[derive(Clone)]
struct MasterSettings {scale:f32,clipping:Option<GradientClipping>}

/// Original Muon on explicit full logical matrices and original AdamW on all other actual local shards.
/// Parameter IDs/roles/topology are fixed at initialization; selected matrices are not optimized as fragments.
/// FSDP gradients must already be globally normalized and reduce-scattered. No second gradient sum, automatic
/// mixed precision, nonfinite skip policy, loss scaling or hidden optimizer role selection is enabled.
#[derive(Clone)]
pub struct FullyShardedMuonAdamW<M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    muon:Muon<B::InnerBackend>,
    adamw:AdamW,
    adamw_clipping:Option<GradientClipping>,
    master:Option<MasterSettings>,
    bindings:Vec<FullyShardedOptimizerParameter<C>>,
    indices:HashMap<ParamId,usize>,
    manifest:FlatManifest,
    config_key:String,
    states:States<B>,
    adamw_lr_ratio:f64,
    model:PhantomData<M>,
}

impl MuonAdamWConfig {
    /// Attach the complete original mixed optimizer to actual flat FSDP model leaves.
    /// Native parameter/gradient precision and all original Muon/AdamW numerical settings remain unchanged.
    pub fn init_fully_sharded<B,M,C>(&self,module:&M,parameters:&[FullyShardedOptimizerParameter<C>])
        -> Result<FullyShardedMuonAdamW<M,B,C>,MuonShardedError<C::Error>>
        where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
        self.init_fully_sharded_inner(module,parameters,None)
    }
    pub(crate) fn init_fully_sharded_master<B,M,C>(&self,module:&M,parameters:&[FullyShardedOptimizerParameter<C>],
        scale:f32,clipping:Option<GradientClipping>) -> Result<FullyShardedMuonAdamW<M,B,C>,MuonShardedError<C::Error>>
        where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
        self.init_fully_sharded_inner(module,parameters,Some(MasterSettings {scale,clipping}))
    }
    fn init_fully_sharded_inner<B,M,C>(&self,module:&M,parameters:&[FullyShardedOptimizerParameter<C>],master:Option<MasterSettings>)
        -> Result<FullyShardedMuonAdamW<M,B,C>,MuonShardedError<C::Error>>
        where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
        self.muon.validate().map_err(MuonShardedError::Muon)?;
        self.adamw.validate_hyperparameters().map_err(|error|MuonShardedError::Muon(MuonError::InvalidConfig(error)))?;
        valid_lr(self.adamw_lr_ratio).map_err(MuonShardedError::Muon)?;
        if !parameters.iter().any(|binding|binding.use_muon) {return Err(MuonShardedError::Muon(MuonError::EmptyMuonGroup));}
        let mut bindings=parameters.to_vec();bindings.sort_by_key(|binding|binding.parameter);
        let mut indices=HashMap::new();
        for (index,binding) in bindings.iter().enumerate() {
            if indices.insert(binding.parameter,index).is_some() {return Err(MuonShardedError::Muon(MuonError::DuplicateParameter(binding.parameter.val())));}
        }
        let muon=self.muon.try_build().map_err(MuonShardedError::Muon)?;
        let inspected=snapshot::<B,M,C>(module,&bindings,&indices,&muon,master.is_some(),None,0.0)?;
        let adamw=self.adamw.init::<B,M>();
        let precision_key=match &master {
            None=>String::from("native"),Some(settings)=>{
                let clip=match &settings.clipping {None=>String::from("none"),Some(GradientClipping::Value(value))=>format!("value:{value:?}"),
                    Some(GradientClipping::Norm(value))=>format!("norm:{value:?}")};
                format!("fp32:scale:{:?}:clip:{clip}",settings.scale)
            },
        };
        Ok(FullyShardedMuonAdamW {muon,adamw:adamw.optim().clone(),adamw_clipping:adamw.grad_clipping().cloned(),master,
            bindings,indices,manifest:inspected.manifest,config_key:format!("flat-fsdp-muon-adamw-v1:{self:?}:{precision_key}"),
            states:HashMap::new(),adamw_lr_ratio:self.adamw_lr_ratio,model:PhantomData})
    }
}

impl<M,B,C> FullyShardedMuonAdamW<M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    /// Number of real explicitly selected matrices, excluding tied roles and unused padding.
    pub fn muon_parameter_count(&self) -> usize {self.bindings.iter().filter(|binding|binding.use_muon).count()}
    /// Number of actual parameter identities with allocated native momentum/master state.
    pub fn state_parameter_count(&self) -> usize {self.states.len()}
    fn placement(&self) -> Placement {
        self.bindings.iter().map(|binding|(binding.parameter.val(),binding.logical_shape.clone(),binding.communicator.rank(),
            binding.communicator.world_size(),binding.use_muon)).collect()
    }
    /// Update the complete native FSDP model once per canonical ID with independently declared optimizer rates.
    /// Presence votes distinguish globally unused parameters from an absent local derivative. Only the latter
    /// contributes zeros to an otherwise used parameter; globally unused momentum and decay remain unchanged.
    /// Proposed optimizer state commits after transport succeeds, not as an asynchronous device transaction.
    pub fn try_step_with_lrs(&mut self,muon_lr:LearningRate,adamw_lr:LearningRate,module:M,mut grads:GradientsParams)
        -> Result<M,MuonShardedError<C::Error>> {
        valid_lr(muon_lr).map_err(MuonShardedError::Muon)?;valid_lr(adamw_lr).map_err(MuonShardedError::Muon)?;
        let inspected=snapshot::<B,M,C>(&module,&self.bindings,&self.indices,&self.muon,self.master.is_some(),Some(&grads),muon_lr)?;
        if inspected.manifest!=self.manifest {return Err(MuonShardedError::Muon(MuonError::ModelChanged));}
        let mut states=self.states.clone();
        let mut mapper=Updates::<B> {values:TensorContainer::new(),updated:TensorContainer::new(),backend:PhantomData};
        for binding in &self.bindings {
            let id=binding.parameter;
            let spec=self.manifest.iter().find(|entry|entry.0==id.val()).expect("validated actual FSDP parameter");
            if !spec.2 {continue;}
            let tensor=Tensor::<B::InnerBackend,1>::from_primitive(inspected.values.get::<B::InnerBackend>(&id).expect("validated actual FSDP local leaf"));
            let gradient=grads.remove::<B::InnerBackend,1>(id);
            if !used::<B::InnerBackend,C>(gradient.is_some(),&tensor.device(),&binding.communicator)? {continue;}
            let gradient=gradient.unwrap_or_else(||Tensor::zeros(tensor.dims(),(&tensor.device(),if self.master.is_some() {DType::F32} else {tensor.dtype()})));
            let gradient=trim_padding(gradient,binding).map_err(MuonShardedError::Muon)?;
            let state=states.remove(&id).map(|state|state.to_device(&tensor.device()));
            if let Some(state)=&state {state.validate(binding,tensor.dims()[0],tensor.dtype(),self.master.is_some(),self.adamw.uses_amsgrad()).map_err(MuonShardedError::Muon)?;}
            let (tensor,state)=if binding.use_muon {
                let shape=binding.logical_shape.as_slice().try_into().map_err(|_|MuonShardedError::Muon(MuonError::ExpectedMatrix {rank:binding.logical_shape.len()}))?;
                let layout=MuonFlatShardLayout::new(shape);
                if let Some(settings)=&self.master {
                    let mut optimizer=Fp32MasterOptimizer::new(self.muon.clone()).with_gradient_scale(settings.scale);
                    if let Some(clipping)=&settings.clipping {optimizer=optimizer.with_grad_clipping(clipping.clone());}
                    let (value,state)=optimizer.try_step_flat_sharded(muon_lr,tensor,gradient,state.and_then(|state|state.master_muon),&layout,binding.communicator.clone())?;
                    (value,FullyShardedMuonAdamWState::master_muon(state))
                } else {
                    let (value,state)=self.muon.try_step_flat_sharded(muon_lr,tensor,gradient,state.and_then(|state|state.muon),&layout,binding.communicator.clone())?;
                    (value,FullyShardedMuonAdamWState::native_muon(state))
                }
            } else {
                let gradient=match &self.adamw_clipping {Some(clipping)=>clip_logical_gradient(gradient,binding,clipping)?,None=>gradient};
                if let Some(settings)=&self.master {
                    let gradient=gradient.cast(DType::F32);let gradient=if settings.scale==1.0 {gradient} else {gradient/settings.scale};
                    let gradient=match &settings.clipping {Some(clipping)=>clip_logical_gradient(gradient,binding,clipping)?,None=>gradient};
                    let gradient=trim_padding(gradient,binding).map_err(MuonShardedError::Muon)?;
                    let optimizer=Fp32MasterOptimizer::new(self.adamw.clone());
                    let (value,state)=optimizer.step(adamw_lr,tensor,gradient,state.and_then(|state|state.master_adamw));
                    let mut state=state.expect("original FP32 AdamW state");
                    state.master=trim_padding(state.master,binding).map_err(MuonShardedError::Muon)?;
                    (value,FullyShardedMuonAdamWState::master_adamw(state))
                } else {
                    let storage=tensor.dtype();let gradient=trim_padding(gradient,binding).map_err(MuonShardedError::Muon)?;
                    let (value,state)=self.adamw.step(adamw_lr,tensor,gradient,state.and_then(|state|state.adamw));
                    (value,FullyShardedMuonAdamWState::native_adamw(state.expect("original AdamW state"),storage))
                }
            };
            let tensor=trim_padding(tensor,binding).map_err(MuonShardedError::Muon)?;
            mapper.values.register::<B::InnerBackend>(id,tensor.into_primitive());states.insert(id,state);
        }
        let module=module.map(&mut mapper);self.states=states;Ok(module)
    }
}

impl<M,B,C> Optimizer<M,B> for FullyShardedMuonAdamW<M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    type Record=FullyShardedMuonAdamWRecord<B>;
    fn step(&mut self,lr:LearningRate,module:M,grads:GradientsParams) -> M {
        self.try_step_with_lrs(lr,lr*self.adamw_lr_ratio,module,grads).unwrap_or_else(|error|panic!("{error}"))
    }
    fn step_multi(&mut self,_lr:LearningRate,_module:M,_grads:MultiGradientsParams) -> M {
        panic!("actual FSDP owners use step on each real rank, not implicit mixed-matrix flattening")
    }
    fn to_record(&self) -> Self::Record {
        FullyShardedMuonAdamWRecord {version:1,config_key:self.config_key.clone(),manifest:self.manifest.clone(),placement:self.placement(),states:self.states.clone()}
    }
    fn load_record(self,record:Self::Record) -> Self {self.try_load_record(record).unwrap_or_else(|error|panic!("{error}"))}
}
