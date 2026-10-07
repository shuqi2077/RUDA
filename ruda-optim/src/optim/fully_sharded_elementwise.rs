use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use core::{fmt,marker::PhantomData};
use hashbrown::HashMap;
use ruda_model::{module::{AutodiffModule,ModuleVisitor,ModuleMapper,Param,ParamId},record::{Record,PrecisionSettings},
    tensor::{Tensor,TensorPrimitive,DType,ElementConversion,BroadcastTensorCollective,container::TensorContainer,backend::{Backend,AutodiffBackend}}};
use crate::{LearningRate,grad_clipping::GradientClipping};
use super::{ElementwiseShardOptimizer,OptimizerCheckpointBuffers,FullyShardedOptimizerParameter,FullyShardedAccumulationError,
    GradientsParams,MultiGradientsParams,Optimizer,fully_sharded_accum::{Placement,inspect}};

mod continuation;

/// Actual native transport, local-parameter/state geometry or source-optimizer argument failure.
#[derive(Debug)]
pub enum FullyShardedElementwiseError<E:fmt::Debug> {
    /// Original native collective failed.
    Collective(E),
    /// Actual source module, logical ownership or local gradient metadata differs.
    Arguments(FullyShardedAccumulationError),
    /// Original optimizer contains a tensor-wide operation or invalid explicit learning rate.
    Configuration(&'static str),
    /// Actual state/transport storage or dimensions do not match original ownership.
    State(&'static str),
}
impl<E:fmt::Debug> fmt::Display for FullyShardedElementwiseError<E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Collective(value)=>write!(f,"native FSDP optimizer collective: {value:?}"),Self::Arguments(value)=>fmt::Display::fmt(value,f),
            Self::Configuration(value)=>write!(f,"native FSDP optimizer configuration: {value}"),Self::State(value)=>write!(f,"native FSDP optimizer state: {value}")}
    }
}
impl<E:fmt::Debug> core::error::Error for FullyShardedElementwiseError<E> {}

/// Actual local elementwise histories and exact logical ownership; numerical configuration stays caller-prepared.
/// Like the original native adaptor, this record contains state, not a replacement optimizer configuration.
pub struct FullyShardedElementwiseRecord<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> {
    version:u32,
    placement:Placement,
    states:HashMap<ParamId,O::State<1>>,
}
impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> Clone for FullyShardedElementwiseRecord<B,O> {
    fn clone(&self) -> Self {Self {version:self.version,placement:self.placement.clone(),states:self.states.clone()}}
}
impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> Record<B> for FullyShardedElementwiseRecord<B,O>
    where O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    type Item<S:PrecisionSettings>=(u32,Placement,<HashMap<ParamId,O::State<1>> as Record<B::InnerBackend>>::Item<S>,Vec<(u64,Vec<DType>)>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        let dtypes=self.states.iter().map(|(id,state)|{let mut dtypes=Vec::new();state.visit_checkpoint_buffers(&mut |value|dtypes.push(value.dtype()));(id.val(),dtypes)}).collect();
        (self.version,self.placement,<HashMap<ParamId,O::State<1>> as Record<B::InnerBackend>>::into_item::<S>(self.states),dtypes)
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        let dtypes=item.3.into_iter().collect::<BTreeMap<_,_>>();
        let states=<HashMap<ParamId,O::State<1>> as Record<B::InnerBackend>>::from_item::<S>(item.2,device);
        let states=states.into_iter().map(|(id,state)| {
            let mut dtypes=dtypes.get(&id.val()).expect("actual recorded native buffer precision required").iter();
            let state=state.map_checkpoint_buffers(&mut |value|value.cast(*dtypes.next().expect("actual recorded native buffer dtype required")));
            assert!(dtypes.next().is_none(),"recorded native buffer count differs");(id,state)
        }).collect();
        Self {version:item.0,placement:item.1,states}
    }
}

/// Original coordinate-wise optimizer on actual flat parameter shards, with full logical clipping semantics.
/// Supports native Adam/AdamW/SGD/AdaGrad/RMSProp/Adan and explicitly wrapped FP32 masters.
/// Global unused parameters keep absent/unchanged state and decay; tied identities update once.
/// Gradients must already be normalized/SUM-reduce-scattered. No second gradient reduction or skip policy is added.
#[derive(Clone)]
pub struct FullyShardedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend> {
    optimizer:O,
    clipping:Option<GradientClipping>,
    bindings:Vec<FullyShardedOptimizerParameter<C>>,
    placement:Placement,
    states:HashMap<ParamId,O::State<1>>,
    model:PhantomData<(M,B)>,
}
impl<O,M,B,C> FullyShardedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
        O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    /// Initialize a fresh local state container around the actual caller-prepared original optimizer/options.
    /// Existing full-model native histories are not silently discarded or implicitly converted by this constructor.
    pub fn new(optimizer:O,module:&M,parameters:&[FullyShardedOptimizerParameter<C>],clipping:Option<GradientClipping>)
        -> Result<Self,FullyShardedElementwiseError<C::Error>> {
        optimizer.validate_element_sharding().map_err(FullyShardedElementwiseError::Configuration)?;
        let mut bindings=parameters.to_vec();bindings.sort_by_key(|binding|binding.parameter);let mut ids=BTreeSet::new();
        let mut declared=Vec::with_capacity(bindings.len());
        for binding in &bindings {
            if !ids.insert(binding.parameter) {return Err(FullyShardedElementwiseError::State("duplicate canonical parameter binding"));}
            declared.push((binding.parameter.val(),binding.logical_shape.clone(),binding.communicator.rank(),binding.communicator.world_size(),DType::F32,false));
        }
        let placement=inspect::<B,M>(module,&declared,false).map_err(FullyShardedElementwiseError::Arguments)?;
        Ok(Self {optimizer,clipping,bindings,placement,states:HashMap::new(),model:PhantomData})
    }
    /// Actual native original numerical implementation/configuration, not a separately reimplemented update.
    pub fn optimizer(&self) -> &O {&self.optimizer}
    /// Number of canonical local leaves with actual allocated native histories.
    pub fn state_parameter_count(&self) -> usize {self.states.len()}
    /// Inspect one actual native local history, without manufacturing unused states.
    pub fn state(&self,id:ParamId) -> Option<&O::State<1>> {self.states.get(&id)}
    /// Update actual local leaves once, with state committed only after original transports succeed.
    pub fn try_step(&mut self,lr:LearningRate,module:M,mut gradients:GradientsParams) -> Result<M,FullyShardedElementwiseError<C::Error>> {
        if !lr.is_finite() || lr<0.0 {return Err(FullyShardedElementwiseError::Configuration("learning rate must be finite and nonnegative"));}
        inspect::<B,M>(&module,&self.placement,true).map_err(FullyShardedElementwiseError::Arguments)?;
        gradients.validate_for::<B,M>(&module).map_err(|error|FullyShardedElementwiseError::Arguments(error.into()))?;
        let mut leaves=Leaves::<B> {values:BTreeMap::new()};module.visit(&mut leaves);
        let mut states=self.states.clone();let mut mapper=Updates::<B> {values:BTreeMap::new(),canonical:TensorContainer::new()};
        for binding in &self.bindings {
            let id=binding.parameter;let spec=self.placement.iter().find(|entry|entry.0==id.val()).expect("validated original FSDP binding");
            let value=leaves.values.remove(&id).expect("validated original FSDP leaf");
            if !spec.5 {
                if gradients.get::<B::InnerBackend,1>(id).is_some() {return Err(FullyShardedElementwiseError::State("frozen parameter has a supplied derivative"));}
                continue;
            }
            let gradient=gradients.remove::<B::InnerBackend,1>(id);let present=gradient.is_some();
            let dtype=self.optimizer.shard_gradient_dtype(value.dtype());
            let used=if binding.communicator.world_size()==1 {present} else {
                let flag=Tensor::<B::InnerBackend,1>::ones([1],(&value.device(),DType::F32)).mul_scalar(if present {1.0} else {0.0});
                let flags=Tensor::<B::InnerBackend,1>::from_primitive(TensorPrimitive::Float(binding.communicator.all_gather_float(flag.into_primitive().tensor()).map_err(FullyShardedElementwiseError::Collective)?));
                if flags.dims()!=[binding.communicator.world_size() as usize] || flags.dtype()!=DType::F32 || flags.device()!=value.device() {
                    return Err(FullyShardedElementwiseError::State("presence transport changed original shape/storage/device"));
                }
                flags.greater_elem(0).any().into_scalar().elem::<bool>()
            };
            if !used {continue;}
            let gradient=gradient.map(|gradient|gradient.cast(dtype)).unwrap_or_else(||Tensor::zeros(value.dims(),(&value.device(),dtype)));
            let gradient=trim(gradient,binding);
            let gradient=if let Some(clipping)=&self.clipping {clip(gradient,binding,clipping)?} else {gradient};
            let state=states.remove(&id).map(|state|O::to_device(state,&value.device()));
            if let Some(state)=&state {validate_state::<B::InnerBackend,O>(state,value.dims(),dtype).map_err(FullyShardedElementwiseError::State)?;}
            let (value,state)=self.optimizer.step(lr,value,gradient,state);let value=trim(value,binding);
            if let Some(state)=state {states.insert(id,state);}
            mapper.values.insert(id,value);
        }
        let module=module.map(&mut mapper);self.states=states;Ok(module)
    }
    /// Restore actual local native histories onto the unchanged caller-prepared original numerical configuration.
    pub fn try_load_record(mut self,record:FullyShardedElementwiseRecord<B,O>) -> Result<Self,FullyShardedElementwiseError<C::Error>> {
        if record.version!=1 || record.placement!=self.placement {return Err(FullyShardedElementwiseError::State("saved original ownership differs"));}
        for (id,state) in &record.states {
            let spec=self.placement.iter().find(|entry|entry.0==id.val()).ok_or(FullyShardedElementwiseError::State("unknown saved state parameter"))?;
            let elements=spec.1.iter().product::<usize>();let slots=elements.div_ceil(spec.3 as usize);
            validate_state::<B::InnerBackend,O>(state,[slots],self.optimizer.shard_gradient_dtype(spec.4)).map_err(FullyShardedElementwiseError::State)?;
        }
        self.states=record.states;Ok(self)
    }
}
impl<O,M,B,C> Optimizer<M,B> for FullyShardedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
        O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    type Record=FullyShardedElementwiseRecord<B,O>;
    fn step(&mut self,lr:LearningRate,module:M,gradients:GradientsParams) -> M {self.try_step(lr,module,gradients).unwrap_or_else(|error|panic!("{error}"))}
    fn step_multi(&mut self,_lr:LearningRate,_module:M,_gradients:MultiGradientsParams) -> M {panic!("actual FSDP owners use step on each original rank")}
    fn to_record(&self) -> Self::Record {Self::Record {version:1,placement:self.placement.clone(),states:self.states.clone()}}
    fn load_record(self,record:Self::Record) -> Self {self.try_load_record(record).unwrap_or_else(|error|panic!("{error}"))}
}

fn validate_state<B:Backend,O:ElementwiseShardOptimizer<B>>(state:&O::State<1>,shape:[usize;1],dtype:DType)
    -> Result<(),&'static str> where O::State<1>:OptimizerCheckpointBuffers<B,1> {
    let mut valid=true;state.visit_checkpoint_buffers(&mut |buffer|{valid&=buffer.dims()==shape && buffer.dtype()==dtype;});
    if !valid {return Err("original native local history geometry/work precision differs");}Ok(())
}
fn trim<B:Backend,C:BroadcastTensorCollective<B>>(value:Tensor<B,1>,binding:&FullyShardedOptimizerParameter<C>) -> Tensor<B,1> {
    let elements=binding.logical_shape.iter().product::<usize>();let slots=value.dims()[0];
    let real=elements.saturating_sub(binding.communicator.rank() as usize*slots).min(slots);
    if real==slots {value} else {let zero=Tensor::zeros([slots-real],(&value.device(),value.dtype()));value.slice_assign([real..slots],zero)}
}
fn clip<B:Backend,C:BroadcastTensorCollective<B>>(value:Tensor<B,1>,binding:&FullyShardedOptimizerParameter<C>,clipping:&GradientClipping)
    -> Result<Tensor<B,1>,FullyShardedElementwiseError<C::Error>> {
    if matches!(clipping,GradientClipping::Value(_)) {return Ok(clipping.clip_gradient(value));}
    let slots=value.dims()[0];let dtype=value.dtype();let device=value.device();let world=binding.communicator.world_size();
    let full=if world==1 {value} else {Tensor::<B,1>::from_primitive(TensorPrimitive::Float(binding.communicator.all_gather_float(value.into_primitive().tensor()).map_err(FullyShardedElementwiseError::Collective)?))};
    if full.dims()!=[slots*world as usize] || full.dtype()!=dtype || full.device()!=device {return Err(FullyShardedElementwiseError::State("logical clipping transport changed original storage/device"));}
    let elements=binding.logical_shape.iter().product::<usize>();let full=clipping.clip_gradient(full.slice([0..elements]));
    let start=binding.communicator.rank() as usize*slots;let real=elements.saturating_sub(start).min(slots);
    let mut local=Tensor::zeros([slots],(&device,dtype));if real>0 {local=local.slice_assign([0..real],full.slice([start..start+real]));}Ok(local)
}
struct Leaves<B:AutodiffBackend> {values:BTreeMap<ParamId,Tensor<B::InnerBackend,1>>}
impl<B:AutodiffBackend> ModuleVisitor<B> for Leaves<B> {
    fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {
        self.values.entry(param.id).or_insert_with(||Tensor::from_primitive(param.val().inner().into_primitive()));
    }
}
struct Updates<B:AutodiffBackend> {values:BTreeMap<ParamId,Tensor<B::InnerBackend,1>>,canonical:TensorContainer<ParamId>}
impl<B:AutodiffBackend> ModuleMapper<B> for Updates<B> {
    fn map_float<const D:usize>(&mut self,param:Param<Tensor<B,D>>) -> Param<Tensor<B,D>> {
        if let Some(value)=self.canonical.get::<B>(&param.id) {return param.map(|_|Tensor::from_primitive(value));}
        let Some(value)=self.values.remove(&param.id) else {return param;};
        let value=Tensor::<B,D>::from_inner(Tensor::from_primitive(value.into_primitive())).set_require_grad(param.is_require_grad());
        self.canonical.register::<B>(param.id,value.clone().into_primitive());param.map(|_|value)
    }
}
