use super::*;
use core::fmt;
use ruda_model::record::{Record,PrecisionSettings};

/// Exact actual parameter checkpoint or offline shard-conversion contract failure.
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum FullyShardedParameterError {
    /// Invalid source/destination rank topology or logical/physical geometry.
    Geometry(&'static str),
    /// Original parameter ID or version/topology metadata differs.
    Record,
    /// Actual original storage precision differs.
    DType,
    /// Actual original trainability differs on an autodiff backend.
    Trainability,
    /// Source shards have not been placed on the same explicitly selected destination device.
    Device,
}
impl fmt::Display for FullyShardedParameterError {
    fn fmt(&self,formatter:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(message)=>write!(formatter,"fully sharded parameter geometry: {message}"),
            Self::Record=>write!(formatter,"fully sharded parameter checkpoint identity/topology differs"),
            Self::DType=>write!(formatter,"fully sharded parameter storage precision differs"),
            Self::Trainability=>write!(formatter,"fully sharded parameter trainability differs"),
            Self::Device=>write!(formatter,"fully sharded parameter source devices differ"),
        }
    }
}
impl core::error::Error for FullyShardedParameterError {}

fn geometry(shape:&[usize],rank:usize,world:usize) -> Result<(usize,usize),FullyShardedParameterError> {
    if world==0 || rank>=world || shape.contains(&0) {
        return Err(FullyShardedParameterError::Geometry("positive logical axes and valid rank/world are required"));
    }
    let elements = shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis))
        .ok_or(FullyShardedParameterError::Geometry("logical size overflows"))?;
    let size = elements.div_ceil(world);
    size.checked_mul(world).ok_or(FullyShardedParameterError::Geometry("padded physical size overflows"))?;
    Ok((elements,size))
}

/// Native local value plus exact original logical axes, topology, storage and trainability.
/// Recreate matching optimizer/scheduler/gradient continuation separately. Full-precision record
/// settings preserve actual stored values; local tensor precision is restored explicitly on load.
#[derive(Clone)]
pub struct FullyShardedParameterRecord<B:Backend> {
    version:u32,
    shape:Vec<usize>,
    rank:usize,
    world:usize,
    storage:DType,
    trainable:bool,
    local:Param<Tensor<B,1>>,
}

impl<B:Backend> FullyShardedParameterRecord<B> {
    pub(super) fn into_exact_item(self) -> (u32,Vec<usize>,usize,usize,u64,bool,ruda_model::tensor::TensorData) {
        (self.version,self.shape,self.rank,self.world,self.local.id.val(),self.trainable,self.local.val().into_data())
    }
    pub(super) fn from_exact_item(item:(u32,Vec<usize>,usize,usize,u64,bool,ruda_model::tensor::TensorData),device:&B::Device) -> Self {
        let storage=item.6.dtype;
        let value=Tensor::<B,1>::from_data(item.6,(device,storage)).set_require_grad(item.5);
        Self {version:item.0,shape:item.1,rank:item.2,world:item.3,storage,trainable:item.5,local:Param::initialized(ParamId::from(item.4),value)}
    }
    /// Actual original logical parameter dimensions, without padding.
    pub fn logical_shape(&self) -> &[usize] {&self.shape}
    /// Actual owning rank of this saved local interval.
    pub fn rank(&self) -> usize {self.rank}
    /// Actual saved original data-group world size.
    pub fn world_size(&self) -> usize {self.world}
    /// Actual saved original source parameter ID, including canonical shared-leaf identity.
    pub fn id(&self) -> ParamId {self.local.id}
    /// Original authoritative local value, not a gathered complete tensor.
    pub fn local(&self) -> &Param<Tensor<B,1>> {&self.local}
    /// Original saved training flag, also available when inspecting a record on a native inference backend.
    pub fn is_trainable(&self) -> bool {self.trainable}
    /// Validate actual saved storage, axes and original schema without reading tensor payloads to host.
    pub fn validate(&self) -> Result<(),FullyShardedParameterError> {
        if self.version!=1 {return Err(FullyShardedParameterError::Record);}
        let (_,size) = geometry(&self.shape,self.rank,self.world)?;
        let value = self.local.val();
        if value.dims()!=[size] {return Err(FullyShardedParameterError::Geometry("saved local slice length differs"));}
        if !matches!(self.storage,DType::F32|DType::F16|DType::BF16) || value.dtype()!=self.storage {return Err(FullyShardedParameterError::DType);}
        if B::ad_enabled(&value.device()) && value.is_require_grad()!=self.trainable {return Err(FullyShardedParameterError::Trainability);}
        Ok(())
    }
    /// Restore one canonical local leaf for subsequent explicit shared model assembly.
    /// No full logical parameter or new parameter ID is created.
    pub fn into_parameter(self) -> Result<ShardedParameter<B>,FullyShardedParameterError> {
        self.validate()?;
        Ok(ShardedParameter::from_local(self.local,self.shape,self.rank,self.world))
    }
    /// Move saved local values only, retaining all original scalar/identity/topology metadata.
    pub fn to_device(mut self,device:&B::Device) -> Self {
        self.local = self.local.map(|value|value.to_device(device));self
    }
    /// Offline rank conversion retaining original saved trainability even on a non-AD conversion backend.
    /// Actual source values must already share the selected destination device; optimizer state remains separate.
    pub fn repartition_from_ranks(sources:&[Self],rank:usize,world:usize) -> Result<Self,FullyShardedParameterError> {
        let first=sources.first().ok_or(FullyShardedParameterError::Geometry("complete saved parameter rank set is empty"))?;
        for saved in sources {
            saved.validate()?;
            if saved.trainable!=first.trainable {return Err(FullyShardedParameterError::Trainability);}
        }
        let shards=sources.iter().cloned().map(Self::into_parameter).collect::<Result<Vec<_>,_>>()?;
        let mut record=ShardedParameter::repartition_from_shards(&shards,rank,world)?.parameter_record()?;
        record.trainable=first.trainable;record.validate()?;Ok(record)
    }
}

impl<B:Backend> Record<B> for FullyShardedParameterRecord<B> {
    type Item<P:PrecisionSettings> = (u32,Vec<usize>,usize,usize,DType,bool,<Param<Tensor<B,1>> as Record<B>>::Item<P>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.shape,self.rank,self.world,self.storage,self.trainable,self.local.into_item::<P>())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        let local = <Param<Tensor<B,1>> as Record<B>>::from_item::<P>(item.6,device)
            .map(|value|value.cast(item.4).set_require_grad(item.5));
        Self {version:item.0,shape:item.1,rank:item.2,world:item.3,storage:item.4,trainable:item.5,local}
    }
}

impl<B:Backend> ShardedParameter<B> {
    /// Snapshot the actual local value together with exact original logical/topology metadata.
    pub fn parameter_record(&self) -> Result<FullyShardedParameterRecord<B>,FullyShardedParameterError> {
        let value = self.local.val();
        let record = FullyShardedParameterRecord {version:1,shape:self.logical_shape.clone(),rank:self.rank,world:self.world_size,
            storage:value.dtype(),trainable:value.is_require_grad(),local:self.local.clone()};
        record.validate()?;Ok(record)
    }
    /// Load matching saved local storage into this original canonical leaf, retaining its Param mapper.
    /// Reassemble tied roles from that restored canonical leaf; this does not mutate other module copies.
    pub fn load_parameter_record(mut self,record:FullyShardedParameterRecord<B>) -> Result<Self,FullyShardedParameterError> {
        record.validate()?;
        if record.rank!=self.rank || record.world!=self.world_size || record.shape!=self.logical_shape || record.id()!=self.local.id {
            return Err(FullyShardedParameterError::Record);
        }
        let original = self.local.val();
        if original.dtype()!=record.storage {return Err(FullyShardedParameterError::DType);}
        if B::ad_enabled(&original.device()) && original.is_require_grad()!=record.trainable {return Err(FullyShardedParameterError::Trainability);}
        let value = record.local.val().to_device(&original.device()).detach().set_require_grad(original.is_require_grad());
        self.local = self.local.map(|_|value);Ok(self)
    }

    /// Offline repartition of a complete original rank-ordered checkpoint set into a new local interval.
    /// Move sources to the chosen destination device first; only real overlaps are copied, while
    /// destination padding retains the original zero-padding convention. Parameter IDs, storage and
    /// frozen flags stay unchanged. Convert matching optimizer/gradient records before resuming;
    /// this is not an automatic live communicator migration or a complete training-state restore.
    pub fn repartition_from_shards(sources:&[Self],rank:usize,world:usize) -> Result<Self,FullyShardedParameterError> {
        let first = sources.first().ok_or(FullyShardedParameterError::Geometry("complete source shard set is empty"))?;
        let (elements,source_size) = geometry(&first.logical_shape,0,sources.len())?;
        let (_,size) = geometry(&first.logical_shape,rank,world)?;
        let original = first.local.val();
        for (source_rank,source) in sources.iter().enumerate() {
            source.parameter_record()?;
            if source.rank!=source_rank || source.world_size!=sources.len() || source.logical_shape!=first.logical_shape || source.local.id!=first.local.id {
                return Err(FullyShardedParameterError::Record);
            }
            let value = source.local.val();
            if value.dtype()!=original.dtype() {return Err(FullyShardedParameterError::DType);}
            if value.device()!=original.device() {return Err(FullyShardedParameterError::Device);}
            if value.is_require_grad()!=original.is_require_grad() {return Err(FullyShardedParameterError::Trainability);}
        }
        let start = rank*size;let end = (start+size).min(elements);
        let mut local = Tensor::<B,1>::zeros([size],(&original.device(),original.dtype()));
        for (source_rank,source) in sources.iter().enumerate() {
            let source_start = source_rank*source_size;
            let overlap_start = start.max(source_start);
            let overlap_end = end.min((source_start+source_size).min(elements));
            if overlap_start<overlap_end {
                let piece = source.local.val().slice([overlap_start-source_start..overlap_end-source_start]);
                local = local.slice_assign([overlap_start-start..overlap_end-start],piece);
            }
        }
        let local = local.detach().set_require_grad(original.is_require_grad());
        let local = first.local.clone().map(|_|local);
        Ok(Self::from_local(local,first.logical_shape.clone(),rank,world))
    }
}
