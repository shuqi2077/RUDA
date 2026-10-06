//! Element-sharded optimizer states and reduce-scattered gradients (ZeRO-2).
use super::*;
use crate::{Adam, AdamW, Sgd, LearningRate, SimpleOptimizer};
use ruda_model::record::{PrecisionSettings, Record};

/// An optimizer whose update is independent for each element of a parameter.
///
/// Matrix optimizers such as Muon must not implement this trait: flattening and
/// independently updating fragments changes their mathematical algorithm.
/// Tensor-wide clipping/normalization belongs before sharding, not in this update.
pub trait ElementwiseShardOptimizer<B: Backend>: SimpleOptimizer<B> {}
impl<B: Backend> ElementwiseShardOptimizer<B> for Adam {}
impl<B: Backend> ElementwiseShardOptimizer<B> for AdamW {}
impl<B: Backend> ElementwiseShardOptimizer<B> for Sgd<B> {}

/// Collective transport with equal-size axis-zero shards, preserving dtype.
/// The selected transport's own staging/synchronization semantics still apply.
pub trait ShardedCommunicator<B: Backend>: DataParallelCommunicator<B> {
    /// Sum all rank inputs and return this rank's equal leading-axis slice.
    fn reduce_scatter_float(&self,value:B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive,TensorDeviceError>;
    /// Concatenate equally shaped rank-local tensors in rank order.
    fn all_gather_float(&self,value:B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive,TensorDeviceError>;
}

impl<B: Backend> ShardedCommunicator<B> for RankCommunicator<TensorDevice<B>> {
    fn reduce_scatter_float(&self,value:B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive,TensorDeviceError> {
        RankCommunicator::reduce_scatter_float(self,value,ReduceOperation::Sum)
    }
    fn all_gather_float(&self,value:B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive,TensorDeviceError> {
        RankCommunicator::all_gather_float(self,value)
    }
}

/// ZeRO-2 over an initialized model: full parameters, local gradients/states.
///
/// Each distinct parameter is padded and flattened into equal rank slices.
/// Reduction happens in FP32 and is normalized by the global effective token
/// count. Only a local slice reaches the optimizer. Updated parameter slices
/// are gathered back into the original tensor shape, preserving tied aliases.
pub struct Zero2<B,M,O,C=RankCommunicator<TensorDevice<<B as AutodiffBackend>::InnerBackend>>>
where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:ShardedCommunicator<B::InnerBackend> {
    session:DataParallel<B,C>,
    optimizer:O,
    states:HashMap<ParamId,O::State<1>>,
    ids:Vec<ParamId>,
    model:PhantomData<M>,
}

/// Rank-local slice states. Restore with matching local model IDs and topology.
pub struct Zero2Record<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> {
    version:u32,
    rank:u32,
    world_size:u32,
    ids:Vec<u64>,
    states:HashMap<ParamId,O::State<1>>,
}

/// A completed ZeRO-2 update, with its exact global effective weight.
pub struct Zero2Step<M> {
    /// Updated replicated model with original aliases and parameter IDs.
    pub model:M,
    /// Total contributing effective samples/tokens.
    pub global_weight:u64,
}

impl<B,M,O,C> Zero2<B,M,O,C>
where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:ShardedCommunicator<B::InnerBackend> {
    /// Attach a bare elementwise optimizer before the first update.
    /// Every rank must supply identical optimizer options; no scheduler is implicit.
    pub fn new(session:DataParallel<B,C>,model:&M,optimizer:O)->Result<Self,DataParallelError> {
        let mut schema=Schema::new(session.synchronize_buffers);
        model.visit(&mut schema);
        let mut error=schema.device_error;
        if schema.contract!=session.contract || schema.ids!=session.ids {
            error=Some("model differs from initialized ZeRO-2 replica".into());
        }
        if schema.contract.iter().any(|p|p.trainable && p.shape.contains(&0)) {
            error=Some("ZeRO-2 requires nonempty trainable parameters".into());
        }
        for failure in gather::<B::InnerBackend,C,_>(&session.communicator,&error)? {
            if let Some(failure)=failure {return Err(contract(failure));}
        }
        let mut ids=Vec::new();
        for (spec,id) in schema.contract.iter().zip(&schema.ids) {
            if spec.trainable && !ids.contains(id) {ids.push(*id);}
        }
        Ok(Self{session,optimizer,states:HashMap::new(),ids,model:PhantomData})
    }

    /// Number of local parameter-slice states allocated so far.
    pub fn state_parameter_count(&self)->usize {self.states.len()}

    /// Rank whose equal-size slices this optimizer owns.
    pub fn rank(&self)->u32 {self.session.rank()}

    /// Reduce loss-sum gradients, update local slices, gather new weights.
    /// Globally unused parameters retain no update or weight decay. A zero-weight
    /// rank participates in the same collectives with zero gradient contributions.
    pub fn step(&mut self,lr:LearningRate,model:M,gradients:GradientsParams,
                local_weight:u64,policy:MissingGradientPolicy)->Result<Zero2Step<M>,DataParallelError> {
        let rates=gather::<B::InnerBackend,C,_>(&self.session.communicator,&lr.to_bits())?;
        if !lr.is_finite() || lr<0. || rates.iter().any(|&value|value!=lr.to_bits()) {
            return Err(contract("learning rates must be finite, nonnegative and identical"));
        }
        let mut schema=Schema::new(self.session.synchronize_buffers);
        model.visit(&mut schema);
        let mut check=Check::<B>{gradients:&gradients,ids:Vec::new(),present:Vec::new(),
            error:schema.device_error,device:self.session.communicator.device(),fp32_gradients:true};
        model.visit(&mut check);
        if schema.contract!=self.session.contract || schema.ids!=self.session.ids {
            check.error=Some("model structure or IDs changed after ZeRO-2 initialization".into());
        }
        if check.present.iter().filter(|&&value|value).count()!=gradients.len() {
            check.error=Some("gradient container includes an unknown or frozen parameter".into());
        }
        if policy==MissingGradientPolicy::Error && local_weight>0 && check.present.contains(&false) {
            check.error=Some("positive-weight rank is missing a trainable gradient".into());
        }
        if model.devices().iter().any(|device|device!=self.session.communicator.device()) {
            check.error=Some("replica moved off its communicator device".into());
        }
        let windows=gather::<B::InnerBackend,C,_>(&self.session.communicator,&Window{
            weight:local_weight,policy,fp32_gradients:true,present:check.present.clone(),error:check.error})?;
        let mut global_weight=0u64;
        let mut active=vec![false;check.present.len()];
        for (rank,window) in windows.iter().enumerate() {
            if let Some(failure)=&window.error {return Err(contract(format!("rank {rank}: {failure}")));}
            if window.policy!=policy || !window.fp32_gradients || window.present.len()!=active.len() {
                return Err(contract("ZeRO-2 gradient policies differ"));
            }
            global_weight=global_weight.checked_add(window.weight).ok_or_else(||contract("global weight overflow"))?;
            if window.weight>0 {for (any,present) in active.iter_mut().zip(&window.present) {*any|=present;}}
        }
        if global_weight==0 {return Err(contract("cannot normalize a zero-weight global window"));}
        // Proposed states are separate until every collective completed. No
        // caller-visible model or optimizer state is committed on transport error.
        let mut mapper=Update::<B,O,C>{communicator:&self.session.communicator,optimizer:&self.optimizer,
            states:&self.states,proposed:HashMap::new(),gradients,updated:TensorContainer::new(),
            visited:Vec::new(),active:&active,index:0,local_weight,global_weight,learning_rate:lr,error:None};
        let model=model.map(&mut mapper);
        if let Some(failure)=mapper.error {return Err(failure.into());}
        for (id,state) in mapper.proposed {
            if let Some(state)=state {self.states.insert(id,state);} else {self.states.remove(&id);}
        }
        Ok(Zero2Step{model,global_weight})
    }

    /// Snapshot states at a completed accumulation/update boundary.
    /// Use full-precision recorder settings for FP32 optimizer state.
    pub fn to_record(&self)->Zero2Record<B,O> {
        Zero2Record{version:1,rank:self.rank(),world_size:self.session.world_size(),
            ids:self.ids.iter().map(ParamId::val).collect(),states:self.states.clone()}
    }

    /// Load matching rank-local states collectively, without changing weights.
    pub fn load_record(&mut self,record:Zero2Record<B,O>)->Result<(),DataParallelError> {
        let error=if record.version!=1 || record.rank!=self.rank() || record.world_size!=self.session.world_size()
            || record.ids!=self.ids.iter().map(ParamId::val).collect::<Vec<_>>()
            || record.states.keys().any(|id|!self.ids.contains(id)) {
            Some("ZeRO-2 record topology, IDs or state keys differ".to_string())
        } else {None};
        for failure in gather::<B::InnerBackend,C,_>(&self.session.communicator,&error)? {
            if let Some(failure)=failure {return Err(contract(failure));}
        }
        self.states=record.states.into_iter().map(|(id,state)|(id,O::to_device(state,self.session.communicator.device()))).collect();
        Ok(())
    }
}

struct Update<'a,B,O,C>
where B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>,C:ShardedCommunicator<B::InnerBackend> {
    communicator:&'a C,
    optimizer:&'a O,
    states:&'a HashMap<ParamId,O::State<1>>,
    proposed:HashMap<ParamId,Option<O::State<1>>>,
    gradients:GradientsParams,
    updated:TensorContainer<ParamId>,
    visited:Vec<ParamId>,
    active:&'a [bool],
    index:usize,
    local_weight:u64,
    global_weight:u64,
    learning_rate:LearningRate,
    error:Option<TensorDeviceError>,
}

impl<B,O,C> ModuleMapper<B> for Update<'_,B,O,C>
where B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>,C:ShardedCommunicator<B::InnerBackend> {
    fn map_float<const D:usize>(&mut self,param:Param<Tensor<B,D>>)->Param<Tensor<B,D>> {
        if self.error.is_some() || !param.is_require_grad() {return param;}
        if let Some(tensor)=self.updated.get::<B>(&param.id) {
            let (id,_,mapper)=param.consume();
            return Param::from_mapped_value(id,Tensor::from_primitive(tensor),mapper);
        }
        if self.visited.contains(&param.id) {return param;}
        self.visited.push(param.id);
        let active=self.active[self.index];
        self.index+=1;
        if !active {return param;}
        let (id,value,mapper)=param.consume();
        let dims=value.dims();
        let elements=value.shape().num_elements();
        let world=self.communicator.world_size() as usize;
        let size=elements.div_ceil(world);
        let length=size*world;
        let start=self.communicator.rank() as usize*size;
        let storage=value.dtype();
        let flat=value.clone().inner().reshape([elements]);
        let padded=Tensor::<B::InnerBackend,1>::zeros([length],&value.device()).cast(storage)
            .slice_assign([0..elements],flat);
        let gradient=if self.local_weight==0 {None} else {self.gradients.remove::<B::InnerBackend,D>(id)};
        let gradient=gradient.map(|g|g.reshape([elements]).cast(DType::F32));
        let gradients=Tensor::<B::InnerBackend,1>::zeros([length],&value.device()).cast(DType::F32);
        let gradients=if let Some(gradient)=gradient {gradients.slice_assign([0..elements],gradient)} else {gradients};
        let result=(||->Result<_,TensorDeviceError>{
            let gradient=self.communicator.reduce_scatter_float(gradients.into_primitive().tensor())?;
            let gradient=Tensor::<B::InnerBackend,1>::from_primitive(TensorPrimitive::Float(gradient))
                .div_scalar(self.global_weight as f64).cast(storage);
            let local=padded.slice([start..start+size]);
            let (updated,state)=self.optimizer.step(self.learning_rate,local,gradient,self.states.get(&id).cloned());
            let full=self.communicator.all_gather_float(updated.into_primitive().tensor())?;
            let full=Tensor::<B::InnerBackend,1>::from_primitive(TensorPrimitive::Float(full))
                .slice([0..elements]).reshape(dims);
            let full=Tensor::<B,D>::from_inner(full).require_grad();
            self.proposed.insert(id,state);
            Ok(full)
        })();
        let value=match result {Ok(updated)=>updated,Err(failure)=>{self.error=Some(failure);value}};
        self.updated.register::<B>(id,value.clone().into_primitive());
        Param::from_mapped_value(id,value,mapper)
    }
}

impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> Record<B::InnerBackend> for Zero2Record<B,O> {
    type Item<S:PrecisionSettings>=<(u32,u32,u32,Vec<u64>,HashMap<ParamId,O::State<1>>) as Record<B::InnerBackend>>::Item<S>;
    fn into_item<S:PrecisionSettings>(self)->Self::Item<S> {
        (self.version,self.rank,self.world_size,self.ids,self.states).into_item::<S>()
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&<B::InnerBackend as Backend>::Device)->Self {
        let (version,rank,world_size,ids,states)=Record::<B::InnerBackend>::from_item::<S>(item,device);
        Self{version,rank,world_size,ids,states}
    }
}
