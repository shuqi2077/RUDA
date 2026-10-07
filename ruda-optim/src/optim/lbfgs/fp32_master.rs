use super::*;
use super::sharded::ShardedReductions;
use hashbrown::HashMap;
use ruda_model::{record::PrecisionSettings,tensor::BroadcastTensorCollective};

mod partition;
pub use partition::LBFGSMasterTensorShard;

/// Actual unique trainable parameter identity, shape and incoming model storage precision.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct LBFGSMasterParameter {
    /// Original native ParamId value; tied occurrences are represented once.
    pub id:u64,
    /// Original parameter axes before native flattening.
    pub shape:Vec<usize>,
    /// Original model storage, retained during every objective evaluation and returned update.
    pub storage:DType,
}

fn parameter_length(parameters:&[LBFGSMasterParameter]) -> Result<usize,LBFGSShardError> {
    let mut seen = HashSet::new();
    parameters.iter().try_fold(0usize,|total,parameter| {
        if !seen.insert(parameter.id) {return Err(LBFGSShardError::Record);}
        if !matches!(parameter.storage,DType::F16|DType::BF16|DType::F32) {return Err(LBFGSShardError::DType);}
        let length = parameter.shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis))
            .ok_or(LBFGSShardError::Shape("master parameter shape overflows"))?;
        total.checked_add(length).ok_or(LBFGSShardError::Shape("master vector length overflows"))
    })
}

struct MasterParameters<B:AutodiffBackend> {
    parameters:Vec<LBFGSMasterParameter>,
    tensors:Vec<Tensor<B::InnerBackend,1>>,
    seen:HashMap<ParamId,usize>,
    device:Option<B::Device>,
    materialize:bool,
    error:Option<LBFGSShardError>,
}

impl<B:AutodiffBackend> ModuleVisitor<B> for MasterParameters<B> {
    fn visit_float<const D:usize>(&mut self,parameter:&Param<Tensor<B,D>>) {
        if self.error.is_some() {return;}
        let value = parameter.val();
        if !value.is_require_grad() {return;}
        if !matches!(value.dtype(),DType::F16|DType::BF16|DType::F32) {
            self.error = Some(LBFGSShardError::DType);return;
        }
        if let Some(device) = &self.device {
            if &value.device() != device {self.error = Some(LBFGSShardError::Device);return;}
        } else {self.device = Some(value.device());}
        let metadata = LBFGSMasterParameter {id:parameter.id.val(),shape:value.dims().to_vec(),storage:value.dtype()};
        if let Some(index) = self.seen.get(&parameter.id) {
            if self.parameters[*index] != metadata {self.error = Some(LBFGSShardError::Record);}
            return;
        }
        self.seen.insert(parameter.id,self.parameters.len());
        self.parameters.push(metadata);
        if self.materialize {
            let length = value.shape().num_elements();
            self.tensors.push(value.inner().cast(DType::F32).reshape([length]));
        }
    }
}

/// Authoritative FP32 parameter vector and original native L-BFGS continuation state.
/// Full-precision recorder settings retain FP32 master/history values; do not save these with
/// a recorder that rounds floating tensors. Restore the matching model record and configuration.
#[derive(Clone)]
pub struct LBFGSFp32MasterState<B:Backend> {
    version:u32,
    parameters:Vec<LBFGSMasterParameter>,
    master:Option<Tensor<B,1>>,
    optimizer:LBFGSState<B>,
    placement:Option<(u32,LBFGSShardLayout)>,
}

impl<B:Backend> LBFGSFp32MasterState<B> {
    /// Actual authoritative master, genuinely absent before the first prepared parameter vector.
    pub fn master(&self) -> Option<&Tensor<B,1>> {self.master.as_ref()}
    /// Original unique-parameter order, identity, shape and storage precision.
    pub fn parameters(&self) -> &[LBFGSMasterParameter] {&self.parameters}
    /// Actual original native history, direction, gradient and scalar counters.
    pub fn optimizer(&self) -> &LBFGSState<B> {&self.optimizer}
    /// Explicit sharded rank/layout, or None for ordinary complete-vector optimization.
    pub fn placement(&self) -> Option<(u32,&LBFGSShardLayout)> {
        self.placement.as_ref().map(|(rank,layout)|(*rank,layout))
    }
    /// Validate schema, real vector geometry and original authoritative precision.
    pub fn validate(&self) -> Result<(),LBFGSShardError> {
        if self.version != 1 {return Err(LBFGSShardError::Record);}
        let length = parameter_length(&self.parameters)?;
        if let Some(master) = &self.master {
            if length == 0 || master.dims() != [length] {return Err(LBFGSShardError::Shape("FP32 master vector"));}
            if master.dtype() != DType::F32 {return Err(LBFGSShardError::DType);}
            self.optimizer.validate_vectors(length,Some(master))?;
        } else {
            if length != 0 || self.placement.is_some() {return Err(LBFGSShardError::Record);}
            self.optimizer.validate_vectors(0,None)?;
        }
        if let Some((rank,layout)) = &self.placement {
            let world = u32::try_from(layout.lengths.len()).map_err(|_|LBFGSShardError::Layout("master rank count overflows"))?;
            layout.validate(*rank,world)?;
            if layout.lengths[*rank as usize] != length {return Err(LBFGSShardError::Shape("FP32 master shard interval"));}
        }
        Ok(())
    }
    /// Move actual local master/history tensors, retaining IDs, incoming storage and placement.
    pub fn to_device(mut self,device:&B::Device) -> Self {
        self.master = self.master.map(|master|master.to_device(device));
        self.optimizer = self.optimizer.to_device(device);self
    }
}

impl<B:Backend> Record<B> for LBFGSFp32MasterState<B> {
    type Item<P:PrecisionSettings> = (u32,Vec<LBFGSMasterParameter>,Option<(u32,Vec<usize>)>,
        <(Option<Tensor<B,1>>,LBFGSState<B>) as Record<B>>::Item<P>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.parameters,self.placement.map(|(rank,layout)|(rank,layout.lengths)),
            (self.master,self.optimizer).into_item::<P>())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        let (master,optimizer) = <(Option<Tensor<B,1>>,LBFGSState<B>) as Record<B>>::from_item::<P>(item.3,device);
        Self {version:item.0,parameters:item.1,placement:item.2.map(|(rank,lengths)|(rank,LBFGSShardLayout::new(lengths))),master,optimizer}
    }
}

/// Explicit FP32-master L-BFGS over actual FP16/BF16/FP32 model parameters, including mixed storage.
/// Original native two-loop/line-search settings remain unchanged. Objective evaluation uses
/// the original model storage, while authoritative updates and native history remain FP32.
#[derive(Clone)]
pub struct LBFGSFp32Master<B:AutodiffBackend> {
    optimizer:LBFGS<B>,
    master:Option<Tensor<B::InnerBackend,1>>,
    parameters:Vec<LBFGSMasterParameter>,
    placement:Option<(u32,LBFGSShardLayout)>,
}

impl LBFGSConfig {
    /// Opt in to actual FP32-master native L-BFGS without altering the ordinary optimizer.
    pub fn init_fp32_master<B:AutodiffBackend>(&self) -> LBFGSFp32Master<B> {
        LBFGSFp32Master {optimizer:self.init(),master:None,parameters:Vec::new(),placement:None}
    }
}

impl<B:AutodiffBackend> LBFGSFp32Master<B> {
    /// Save actual authoritative master, original native continuation and exact parameter placement.
    pub fn to_record(&self) -> LBFGSFp32MasterState<B::InnerBackend> {
        LBFGSFp32MasterState {version:1,parameters:self.parameters.clone(),master:self.master.clone(),
            optimizer:self.optimizer.to_record(),placement:self.placement.clone()}
    }
    /// Restore actual authoritative state, not rounded values reconstructed from model storage.
    pub fn load_record(mut self,record:LBFGSFp32MasterState<B::InnerBackend>) -> Result<Self,LBFGSShardError> {
        record.validate()?;self.optimizer = self.optimizer.load_record(record.optimizer);
        self.master = record.master;self.parameters = record.parameters;self.placement = record.placement;Ok(self)
    }
    /// Move actual master/native histories and retain exact incoming model-storage metadata.
    pub fn to_device(mut self,device:&B::Device) -> Self {
        self.optimizer = self.optimizer.to_device(device);self.master = self.master.map(|master|master.to_device(device));self
    }

    fn prepare<M:Module<B>>(&self,module:&M) -> Result<(Vec<LBFGSMasterParameter>,Tensor<B::InnerBackend,1>),LBFGSShardError> {
        let mut visitor = MasterParameters::<B> {parameters:Vec::new(),tensors:Vec::new(),seen:HashMap::new(),device:None,
            materialize:self.master.is_none(),error:None};
        module.visit(&mut visitor);
        if let Some(error) = visitor.error {return Err(error);}
        let length = parameter_length(&visitor.parameters)?;
        if length == 0 {return Err(LBFGSShardError::Shape("FP32 master requires a nonempty trainable vector"));}
        let master = if let Some(master) = &self.master {
            if self.parameters != visitor.parameters {return Err(LBFGSShardError::Record);}
            if master.dims() != [length] {return Err(LBFGSShardError::Shape("FP32 master parameter length"));}
            if master.dtype() != DType::F32 {return Err(LBFGSShardError::DType);}
            if Some(master.device()) != visitor.device {return Err(LBFGSShardError::Device);}
            master.clone()
        } else {Tensor::cat(visitor.tensors,0)};
        self.optimizer.state.validate_vectors(length,Some(&master))?;
        Ok((visitor.parameters,master))
    }

    /// Native complete-vector FP32 updates with original model-storage objective evaluations.
    /// Save/reload matching model and master records together; external model mutation must not
    /// silently replace the authoritative master. No automatic loss scaling or step skipping is added.
    pub fn step<M,F>(&mut self,lr:LearningRate,module:M,mut closure:F) -> Result<(M,f64),LBFGSShardError>
        where M:AutodiffModule<B>+Clone,F:FnMut(M)->(f64,GradientsParams) {
        if self.placement.is_some() {return Err(LBFGSShardError::Record);}
        let (parameters,master) = self.prepare(&module)?;
        let (model,loss,master) = self.optimizer.try_step_with_reductions(lr,module,|model|Ok(closure(model)),
            &mut LocalReductions,Some(master),true).unwrap_or_else(|error|match error {});
        self.master = master;self.parameters = parameters;Ok((model,loss))
    }

    /// FP32-master updates of disjoint actual parameter shards using original global L-BFGS reductions.
    /// Each closure supplies the identical complete objective and its exact local derivative.
    pub fn step_sharded<M,F,C>(&mut self,lr:LearningRate,module:M,mut closure:F,layout:&LBFGSShardLayout,communicator:&C)
        -> Result<(M,f64),LBFGSShardedError<C::Error>>
        where M:AutodiffModule<B>+Clone,F:FnMut(M)->(f64,GradientsParams),C:BroadcastTensorCollective<B::InnerBackend> {
        self.try_step_sharded(lr,module,|model|Ok(closure(model)),layout,communicator)
    }

    /// Fallible sharded objective variant, preserving the installed master/history if an error is returned.
    /// Explicit placement is recorded; rank changes require conversion of matching model/master records.
    pub fn try_step_sharded<M,F,C>(&mut self,lr:LearningRate,module:M,closure:F,layout:&LBFGSShardLayout,communicator:&C)
        -> Result<(M,f64),LBFGSShardedError<C::Error>>
        where M:AutodiffModule<B>+Clone,F:FnMut(M)->Result<(f64,GradientsParams),LBFGSShardedError<C::Error>>,
            C:BroadcastTensorCollective<B::InnerBackend> {
        layout.validate(communicator.rank(),communicator.world_size())?;
        let placement = Some((communicator.rank(),layout.clone()));
        if self.master.is_some() && self.placement != placement {return Err(LBFGSShardError::Record.into());}
        let (parameters,master) = self.prepare(&module)?;
        if master.dims() != [layout.lengths[communicator.rank() as usize]] {
            return Err(LBFGSShardError::Shape("FP32 master local shard length").into());
        }
        let (model,loss,master) = self.optimizer.try_step_with_reductions(lr,module,closure,
            &mut ShardedReductions {communicator},Some(master),true)?;
        self.master = master;self.parameters = parameters;self.placement = placement;Ok((model,loss))
    }
}
