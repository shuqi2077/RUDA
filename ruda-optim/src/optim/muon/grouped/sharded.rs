use alloc::{format,string::String,vec::Vec};
use hashbrown::HashMap;
use ruda_model::{module::{AutodiffModule,ModuleMapper,Param,ParamId},record::{Record,PrecisionSettings},
    tensor::{Tensor,TensorPrimitive,DType,ElementConversion,BroadcastTensorCollective,backend::{Backend,AutodiffBackend},container::TensorContainer}};
use super::{MuonAdamWConfig,Muon,MuonError,Manifest,AdamRecords,valid_lr,validate_adam_records,AdamW,OptimizerAdaptor,
    GradientsParams,LearningRate,MultiGradientsParams,Optimizer};
use super::super::{MuonMatrixShardLayout,MuonShardedState,MuonShardedError};

mod snapshot;
use snapshot::snapshot;
mod state;
pub use state::MuonShardedAdamWRecord;
mod master;
pub use master::{Fp32MasterMuonShardedAdamW,Fp32MasterMuonShardedAdamWRecord};

type ShardRecords<B: AutodiffBackend> = HashMap<ParamId,MuonShardedState<<B as AutodiffBackend>::InnerBackend>>;
type Placement = Vec<(u64,u32,u32,MuonMatrixShardLayout)>;

/// Exact matrix identity, physical rank-ordered shard layout and its actual transport.
#[derive(Clone,Debug)]
pub struct MuonShardedParameter<C> {
    /// Actual native model parameter ID; matrix rank alone never selects a role.
    pub parameter: ParamId,
    /// Corresponding unique row/column shards, excluding separate KV/data replicas.
    pub layout: MuonMatrixShardLayout,
    /// Explicit communicator for this actual matrix, not a guessed model-wide group.
    pub communicator: C,
}
impl<C> MuonShardedParameter<C> {
    /// Bind an actual loaded matrix to the supplied placement and transport.
    pub fn new(parameter: ParamId,layout: MuonMatrixShardLayout,communicator: C) -> Self {Self {parameter,layout,communicator}}
}

/// Full-matrix sharded Muon on explicit matrix roles, with existing AdamW on all other parameters.
/// Collective order is sorted by original parameter ID. Each participating rank must agree on
/// IDs/layout/configuration and corresponding gradients; replicas/data-parallel reductions stay explicit.
#[derive(Clone)]
pub struct MuonShardedAdamW<M,B,C>
    where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
    muon: Muon<B::InnerBackend>,
    states: ShardRecords<B>,
    adamw: OptimizerAdaptor<AdamW,M,B>,
    bindings: Vec<MuonShardedParameter<C>>,
    indices: HashMap<ParamId,usize>,
    manifest: Manifest,
    config_key: String,
    adamw_lr_ratio: f64,
}

impl MuonAdamWConfig {
    /// Use the original Muon/AdamW settings on explicitly declared native local matrix shards.
    /// Embeddings, heads, bias and norms are never automatically classified as Muon matrices.
    pub fn init_sharded<B,M,C>(&self,module: &M,parameters: &[MuonShardedParameter<C>]) -> Result<MuonShardedAdamW<M,B,C>,MuonShardedError<C::Error>>
        where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
        self.muon.validate().map_err(MuonShardedError::Muon)?;
        self.adamw.validate_hyperparameters().map_err(|error|MuonShardedError::Muon(MuonError::InvalidConfig(error)))?;
        valid_lr(self.adamw_lr_ratio).map_err(MuonShardedError::Muon)?;
        if parameters.is_empty() {return Err(MuonShardedError::Muon(MuonError::EmptyMuonGroup));}
        let mut bindings = parameters.to_vec();bindings.sort_by_key(|binding|binding.parameter);
        let mut indices = HashMap::new();
        for (index,binding) in bindings.iter().enumerate() {
            if indices.insert(binding.parameter,index).is_some() {return Err(MuonShardedError::Muon(MuonError::DuplicateParameter(binding.parameter.val())));}
        }
        let muon: Muon<B::InnerBackend> = self.muon.try_build().map_err(MuonShardedError::Muon)?;
        let inspected = snapshot::<B,M,C>(module,&bindings,&indices,&muon,None,0.0)?;
        Ok(MuonShardedAdamW {muon,states:HashMap::new(),adamw:self.adamw.init(),bindings,indices,manifest:inspected.manifest,
            config_key:format!("muon-sharded-adamw-v1:{self:?}"),adamw_lr_ratio:self.adamw_lr_ratio})
    }
}

fn used<B,C>(present: bool,device: &B::Device,communicator: &C) -> Result<bool,MuonShardedError<C::Error>>
    where B: Backend,C: BroadcastTensorCollective<B> {
    if communicator.world_size() == 1 {return Ok(present);}
    let vote = Tensor::<B,1>::from_floats([if present {1.0f32} else {0.0}],device).cast(DType::F32);
    let gathered = communicator.all_gather_float(vote.into_primitive().tensor()).map_err(MuonShardedError::Collective)?;
    Ok(Tensor::<B,1>::from_primitive(TensorPrimitive::Float(gathered)).max().into_scalar().elem::<f32>() > 0.0)
}

impl<M,B,C> MuonShardedAdamW<M,B,C>
    where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
    /// Number of actual explicitly selected matrices, without counting tied aliases twice.
    pub fn muon_parameter_count(&self) -> usize {self.bindings.len()}

    fn placement(&self) -> Placement {
        self.bindings.iter().map(|binding|(binding.parameter.val(),binding.communicator.rank(),binding.communicator.world_size(),binding.layout.clone())).collect()
    }

    /// Update a generic native model with independent rates, preserving IDs/mappers/trainable flags.
    /// One gathered presence scalar per selected matrix distinguishes globally unused matrices
    /// from used matrices with a zero local derivative. Globally missing gradients skip momentum
    /// and decay; a missing local gradient contributes zero when another shard uses the matrix.
    /// Tiny presence votes require a host scalar read, not a host numerical optimizer fallback.
    /// Transport failures do not commit proposed optimizer states. This is not a device transaction:
    /// kernels can execute asynchronously, and completed model/optimizer records remain recovery points.
    pub fn try_step_with_lrs(&mut self,muon_lr: LearningRate,adamw_lr: LearningRate,module: M,mut grads: GradientsParams)
        -> Result<M,MuonShardedError<C::Error>> {
        valid_lr(muon_lr).map_err(MuonShardedError::Muon)?;valid_lr(adamw_lr).map_err(MuonShardedError::Muon)?;
        let inspected = snapshot::<B,M,C>(&module,&self.bindings,&self.indices,&self.muon,Some(&grads),muon_lr)?;
        if inspected.manifest != self.manifest {return Err(MuonShardedError::Muon(MuonError::ModelChanged));}
        let mut states = self.states.clone();let updates = TensorContainer::new();
        let mut mapper = Updates::<B> {values:updates,updated:TensorContainer::new(),backend:core::marker::PhantomData};
        for binding in &self.bindings {
            let id = binding.parameter;
            let tensor = Tensor::<B::InnerBackend,2>::from_primitive(inspected.values.get::<B::InnerBackend>(&id).expect("validated actual Muon parameter"));
            let gradient = grads.remove::<B::InnerBackend,2>(id);
            if !used::<B::InnerBackend,C>(gradient.is_some(),&tensor.device(),&binding.communicator)? {continue;}
            let gradient = gradient.unwrap_or_else(||Tensor::zeros(tensor.dims(),(&tensor.device(),tensor.dtype())));
            let state = states.remove(&id).map(|state|if state.momentum().device() == tensor.device() {state} else {state.to_device(&tensor.device())});
            let (tensor,state) = self.muon.try_step_sharded(muon_lr,tensor,gradient,state,&binding.layout,binding.communicator.clone())?;
            mapper.values.register::<B::InnerBackend>(id,tensor.into_primitive());states.insert(id,state);
        }
        let module = module.map(&mut mapper);
        let mut adamw = self.adamw.clone();let module = adamw.step(adamw_lr,module,grads);
        self.states = states;self.adamw = adamw;
        Ok(module)
    }
}

struct Updates<B: AutodiffBackend> {values:TensorContainer<ParamId>,updated:TensorContainer<ParamId>,backend:core::marker::PhantomData<B>}
impl<B: AutodiffBackend> ModuleMapper<B> for Updates<B> {
    fn map_float<const D: usize>(&mut self,param: Param<Tensor<B,D>>) -> Param<Tensor<B,D>> {
        let id = param.id;
        if let Some(updated) = self.updated.get::<B>(&id) {return param.map(|_|Tensor::from_primitive(updated));}
        if let Some(updated) = self.values.get::<B::InnerBackend>(&id) {
            param.map(|value| {
                let tensor = Tensor::<B,D>::from_inner(Tensor::from_primitive(updated)).set_require_grad(value.is_require_grad());
                self.updated.register::<B>(id,tensor.clone().into_primitive());tensor
            })
        } else {param}
    }
}

impl<M,B,C> Optimizer<M,B> for MuonShardedAdamW<M,B,C>
    where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
    type Record = MuonShardedAdamWRecord<B>;
    fn step(&mut self,lr: LearningRate,module: M,grads: GradientsParams) -> M {
        self.try_step_with_lrs(lr,lr*self.adamw_lr_ratio,module,grads).unwrap_or_else(|error|panic!("{error}"))
    }
    fn step_multi(&mut self,_lr: LearningRate,_module: M,_grads: MultiGradientsParams) -> M {
        panic!("explicit Muon matrix shards use step on each actual rank, not implicit step_multi tensor flattening")
    }
    fn to_record(&self) -> Self::Record {
        MuonShardedAdamWRecord {version:1,config_key:self.config_key.clone(),manifest:self.manifest.clone(),placement:self.placement(),
            muon:self.states.clone(),adamw:self.adamw.to_record()}
    }
    fn load_record(self,record: Self::Record) -> Self {self.try_load_record(record).unwrap_or_else(|error|panic!("{error}"))}
}
