use super::*;
use ruda_model::tensor::TensorData;
mod delta;
pub use delta::FullyShardedStorageDeltaRecord;

fn packed_geometry(shape:&[usize],rank:usize,world:usize) -> Result<(usize,usize),FullyShardedParameterError> {
    if world==0 || rank>=world || shape.contains(&0) {return Err(FullyShardedParameterError::Geometry("invalid packed axes/rank/world"));}
    let elements=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).ok_or(FullyShardedParameterError::Geometry("packed logical size overflows"))?;
    let size=elements.div_ceil(world);
    size.checked_mul(world).ok_or(FullyShardedParameterError::Geometry("packed padded size overflows"))?;
    Ok((elements,size))
}

/// Exact immutable local integer parameter record with original logical axes,
/// rank, world and source ID. Serialization preserves actual integer bytes,
/// independently of PrecisionSettings::IntElem and the backend default IntElem.
#[derive(Clone)]
pub struct FullyShardedPackedParameterRecord<B:Backend> {
    version:u32,
    parameter:ShardedPackedParameter<B>,
}

impl<B:Backend> FullyShardedPackedParameterRecord<B> {
    /// Actual immutable source ID, including explicitly tied roles.
    pub fn id(&self) -> ParamId {self.parameter.local.id}
    /// Original complete integer axes without padding.
    pub fn logical_shape(&self) -> &[usize] {&self.parameter.logical_shape}
    /// Saved original data rank.
    pub fn rank(&self) -> usize {self.parameter.rank}
    /// Saved original data world size.
    pub fn world_size(&self) -> usize {self.parameter.world_size}
    /// Actual saved local words, without a complete gathered base.
    pub fn local(&self) -> &Param<Tensor<B,1,Int>> {&self.parameter.local}
    /// Validate exact saved metadata without decoding integer words to host values.
    pub fn validate(&self) -> Result<(),FullyShardedParameterError> {
        if self.version!=1 {return Err(FullyShardedParameterError::Record);}
        let (_,size)=packed_geometry(self.logical_shape(),self.rank(),self.world_size())?;
        if self.local().val().dims()!=[size] {return Err(FullyShardedParameterError::Geometry("saved packed slice length differs"));}
        if !matches!(self.local().val().dtype(),DType::U8|DType::I32|DType::I64) {return Err(FullyShardedParameterError::DType);}
        Ok(())
    }
    /// Restore the actual saved local slice with its original source identity.
    pub fn into_parameter(self) -> Result<ShardedPackedParameter<B>,FullyShardedParameterError> {
        self.validate()?;Ok(self.parameter)
    }
    /// Move only local packed words to the explicit destination device.
    pub fn to_device(mut self,device:&B::Device) -> Self {
        self.parameter.local=self.parameter.local.map(|value|value.to_device(device));self
    }
    /// Offline complete-rank-set repartition, copying real word overlaps exactly.
    pub fn repartition_from_ranks(sources:&[Self],rank:usize,world:usize) -> Result<Self,FullyShardedParameterError> {
        let first=sources.first().ok_or(FullyShardedParameterError::Geometry("complete packed source rank set is empty"))?;
        let (elements,source_size)=packed_geometry(first.logical_shape(),0,sources.len())?;
        let (_,size)=packed_geometry(first.logical_shape(),rank,world)?;
        let original=first.local().val();
        for (source_rank,source) in sources.iter().enumerate() {
            source.validate()?;
            if source.rank()!=source_rank || source.world_size()!=sources.len() || source.logical_shape()!=first.logical_shape() || source.id()!=first.id() {
                return Err(FullyShardedParameterError::Record);
            }
            if source.local().val().dtype()!=original.dtype() {return Err(FullyShardedParameterError::DType);}
            if source.local().val().device()!=original.device() {return Err(FullyShardedParameterError::Device);}
        }
        let start=rank*size;let end=(start+size).min(elements);
        let mut local=Tensor::<B,1,Int>::zeros([size],(&original.device(),original.dtype()));
        for (source_rank,source) in sources.iter().enumerate() {
            let source_start=source_rank*source_size;
            let overlap_start=start.max(source_start);
            let overlap_end=end.min((source_start+source_size).min(elements));
            if overlap_start<overlap_end {
                local=local.slice_assign([overlap_start-start..overlap_end-start],
                    source.local().val().slice([overlap_start-source_start..overlap_end-source_start]));
            }
        }
        let local=first.local().clone().map(|_|local);
        ShardedPackedParameter::from_local(local,first.logical_shape().to_vec(),rank,world).parameter_record()
    }
}

impl<B:Backend> Record<B> for FullyShardedPackedParameterRecord<B> {
    type Item<P:PrecisionSettings>=(u32,Vec<usize>,usize,usize,u64,TensorData);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.parameter.logical_shape,self.parameter.rank,self.parameter.world_size,
            self.parameter.local.id.val(),self.parameter.local.val().into_data())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        let dtype=item.5.dtype;
        let local=Param::initialized(ParamId::from(item.4),Tensor::<B,1,Int>::from_data(item.5,(device,dtype)));
        Self {version:item.0,parameter:ShardedPackedParameter::from_local(local,item.1,item.2,item.3)}
    }
}

impl<B:Backend> ShardedPackedParameter<B> {
    /// Snapshot actual local words and original logical/topology metadata.
    pub fn parameter_record(&self) -> Result<FullyShardedPackedParameterRecord<B>,FullyShardedParameterError> {
        let record=FullyShardedPackedParameterRecord {version:1,parameter:self.clone()};record.validate()?;Ok(record)
    }
    /// Restore matching local words, preserving the destination parameter mapper.
    pub fn load_parameter_record(mut self,record:FullyShardedPackedParameterRecord<B>) -> Result<Self,FullyShardedParameterError> {
        record.validate()?;
        if record.id()!=self.local.id || record.logical_shape()!=self.logical_shape || record.rank()!=self.rank || record.world_size()!=self.world_size {
            return Err(FullyShardedParameterError::Record);
        }
        if record.local().val().dtype()!=self.local.val().dtype() {return Err(FullyShardedParameterError::DType);}
        let value=record.local().val().to_device(&self.local.val().device());
        self.local=self.local.map(|_|value);Ok(self)
    }
}

fn packed_schema<B:Backend,M:FullyShardedModule<B>>(module:&M) -> Result<BTreeMap<ParamId,ShardedPackedParameter<B>>,FullyShardedParameterError> {
    let mut shards=BTreeMap::<ParamId,ShardedPackedParameter<B>>::new();let mut error=None;
    module.visit_packed_shards(&mut |parameter| {
        if error.is_some() {return;}
        if let Err(reason)=parameter.parameter_record() {error=Some(reason);return;}
        if let Some(previous)=shards.get(&parameter.local.id) {
            if previous.logical_shape!=parameter.logical_shape || previous.rank!=parameter.rank || previous.world_size!=parameter.world_size {
                error=Some(FullyShardedParameterError::Record);
            } else if previous.local.val().dtype()!=parameter.local.val().dtype() {error=Some(FullyShardedParameterError::DType);
            } else if previous.local.val().device()!=parameter.local.val().device() {error=Some(FullyShardedParameterError::Device);}
        } else {shards.insert(parameter.local.id,parameter.clone());}
    });
    if let Some(error)=error {return Err(error);}
    struct Check<'a,B:Backend> {shards:&'a BTreeMap<ParamId,ShardedPackedParameter<B>>,seen:BTreeSet<ParamId>,error:Option<FullyShardedParameterError>}
    impl<B:Backend> ModuleVisitor<B> for Check<'_,B> {
        fn visit_int<const D:usize>(&mut self,parameter:&Param<Tensor<B,D,Int>>) {
            let Some(shard)=self.shards.get(&parameter.id) else {self.error=Some(FullyShardedParameterError::Record);return;};
            if D!=1 || parameter.val().shape().num_elements()!=shard.local.val().dims()[0] {
                self.error=Some(FullyShardedParameterError::Geometry("module contains nonlocal integer storage"));
            }
            if parameter.val().dtype()!=shard.local.val().dtype() {self.error=Some(FullyShardedParameterError::DType);}
            if parameter.val().device()!=shard.local.val().device() {self.error=Some(FullyShardedParameterError::Device);}
            self.seen.insert(parameter.id);
        }
    }
    let mut check=Check {shards:&shards,seen:BTreeSet::new(),error:None};module.visit(&mut check);
    if let Some(error)=check.error {return Err(error);}
    if check.seen.len()!=shards.len() {return Err(FullyShardedParameterError::Record);}
    Ok(shards)
}

/// Complete canonical local floating AND packed-integer model storage checkpoint.
/// Native tensor bytes retain their original dtype irrespective of record precision
/// settings; this preserves AWQ coefficients and mixed FP32-adapter/half-base storage.
/// Optimizer, scheduler, pending gradients, data and RNG use separate continuation records.
#[derive(Clone)]
pub struct FullyShardedStorageRecord<B:Backend> {
    version:u32,
    floating:FullyShardedModuleParameterRecord<B>,
    packed:Vec<FullyShardedPackedParameterRecord<B>>,
}

impl<B:Backend> FullyShardedStorageRecord<B> {
    /// Capture every actual canonical local value exactly once across tied roles.
    pub fn capture<M:FullyShardedModule<B>>(module:&M) -> Result<Self,FullyShardedParameterError> {
        let floating=FullyShardedModuleParameterRecord {version:1,parameters:schema(module)?.into_values()
            .map(|value|value.parameter_record()).collect::<Result<Vec<_>,_>>()?};
        let packed=packed_schema(module)?.into_values().map(|value|value.parameter_record()).collect::<Result<Vec<_>,_>>()?;
        let record=Self {version:1,floating,packed};record.validate()?;Ok(record)
    }
    /// Actual original canonical floating value records, including frozen scales/bias.
    pub fn floating(&self) -> &[FullyShardedParameterRecord<B>] {self.floating.parameters()}
    /// Actual original canonical immutable packed value records.
    pub fn packed(&self) -> &[FullyShardedPackedParameterRecord<B>] {&self.packed}
    /// Validate saved local intervals and reject duplicate or cross-kind identities.
    pub fn validate(&self) -> Result<(),FullyShardedParameterError> {
        if self.version!=1 {return Err(FullyShardedParameterError::Record);}
        self.floating.validate()?;
        let mut ids=self.floating.parameters().iter().map(|value|value.id()).collect::<BTreeSet<_>>();
        for value in &self.packed {value.validate()?;if !ids.insert(value.id()) {return Err(FullyShardedParameterError::Record);}}
        Ok(())
    }
    /// Match the complete actual prepared float and integer logical/topology schema.
    pub fn validate_for<M:FullyShardedModule<B>>(&self,module:&M) -> Result<(),FullyShardedParameterError> {
        self.validate()?;self.floating.validate_float_for(module)?;
        let targets=packed_schema(module)?;
        if targets.len()!=self.packed.len() {return Err(FullyShardedParameterError::Record);}
        self.validate_selected_for(module)
    }
    fn validate_selected_for<M:FullyShardedModule<B>>(&self,module:&M) -> Result<(),FullyShardedParameterError> {
        self.validate()?;
        let floating=schema(module)?;let targets=packed_schema(module)?;
        for saved in self.floating.parameters() {
            let target=floating.get(&saved.id()).ok_or(FullyShardedParameterError::Record)?;
            if target.logical_shape!=saved.logical_shape() || target.rank!=saved.rank() || target.world_size!=saved.world_size() {return Err(FullyShardedParameterError::Record);}
            if target.local.val().dtype()!=saved.local().val().dtype() {return Err(FullyShardedParameterError::DType);}
            if B::ad_enabled(&target.local.val().device()) && target.local.val().is_require_grad()!=saved.is_trainable() {return Err(FullyShardedParameterError::Trainability);}
        }
        for saved in &self.packed {
            let target=targets.get(&saved.id()).ok_or(FullyShardedParameterError::Record)?;
            if target.logical_shape!=saved.logical_shape() || target.rank!=saved.rank() || target.world_size!=saved.world_size() {
                return Err(FullyShardedParameterError::Record);
            }
            if target.local.val().dtype()!=saved.local().val().dtype() {return Err(FullyShardedParameterError::DType);}
        }
        Ok(())
    }
    /// Restore one authoritative resumed floating leaf and exact integer snapshot
    /// per ID, retaining every destination Param mapper and prepared module option.
    pub fn restore_into<M:FullyShardedModule<B>>(self,module:M) -> Result<M,FullyShardedParameterError> {
        self.validate_for(&module)?;
        self.restore_selected_into(module)
    }
    fn restore_selected_into<M:FullyShardedModule<B>>(self,module:M) -> Result<M,FullyShardedParameterError> {
        self.validate_selected_for(&module)?;
        let targets=schema(&module)?;let packed_targets=packed_schema(&module)?;
        let mut floats=BTreeMap::new();let mut integers=BTreeMap::new();
        for saved in self.floating.parameters {
            let original=targets.get(&saved.id()).ok_or(FullyShardedParameterError::Record)?.local.val();
            floats.insert(saved.id(),saved.local().val().to_device(&original.device()).detach().set_require_grad(original.is_require_grad()));
        }
        for saved in self.packed {
            let original=packed_targets.get(&saved.id()).ok_or(FullyShardedParameterError::Record)?.local.val();
            integers.insert(saved.id(),saved.local().val().to_device(&original.device()));
        }
        struct Restore<B:Backend> {values:BTreeMap<ParamId,Tensor<B,1,Int>>,seen:BTreeSet<ParamId>}
        impl<B:Backend> ModuleMapper<B> for Restore<B> {
            fn map_int<const D:usize>(&mut self,parameter:Param<Tensor<B,D,Int>>) -> Param<Tensor<B,D,Int>> {
                let Some(value)=self.values.get(&parameter.id).cloned() else {return parameter;};
                self.seen.insert(parameter.id);parameter.map(|_|Tensor::<B,D,Int>::from_primitive(value.into_primitive()))
            }
        }
        let module=restore_values(module,floats)?;
        let mut restore=Restore {values:integers,seen:BTreeSet::new()};let module=module.map(&mut restore);
        if restore.values.len()!=restore.seen.len() {return Err(FullyShardedParameterError::Record);}
        Ok(module)
    }
    /// Offline complete-rank-set conversion of real floating and packed words.
    /// Local overlaps are copied; no complete dense or integer base is gathered.
    pub fn repartition_from_ranks(sources:&[Self],rank:usize,world:usize) -> Result<Self,FullyShardedParameterError> {
        let first=sources.first().ok_or(FullyShardedParameterError::Geometry("complete storage rank set is empty"))?;
        let mut ranks=Vec::with_capacity(sources.len());
        for source in sources {
            source.validate()?;
            if source.packed.len()!=first.packed.len() {return Err(FullyShardedParameterError::Record);}
            ranks.push(source.packed.iter().map(|value|(value.id(),value)).collect::<BTreeMap<_,_>>());
        }
        let floats=sources.iter().map(|source|source.floating.clone()).collect::<Vec<_>>();
        let floating=FullyShardedModuleParameterRecord::repartition_from_ranks(&floats,rank,world)?;
        let mut packed=Vec::with_capacity(first.packed.len());
        for value in &first.packed {
            let values=ranks.iter().map(|source|source.get(&value.id()).ok_or(FullyShardedParameterError::Record)
                .map(|saved|(*saved).clone())).collect::<Result<Vec<_>,_>>()?;
            packed.push(FullyShardedPackedParameterRecord::repartition_from_ranks(&values,rank,world)?);
        }
        let record=Self {version:1,floating,packed};record.validate()?;Ok(record)
    }
}

impl<B:Backend> Record<B> for FullyShardedStorageRecord<B> {
    type Item<P:PrecisionSettings>=(u32,
        Vec<(u32,Vec<usize>,usize,usize,u64,bool,TensorData)>,
        Vec<<FullyShardedPackedParameterRecord<B> as Record<B>>::Item<P>>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.floating.parameters.into_iter().map(|value|value.into_exact_item()).collect(),
            self.packed.into_iter().map(|value|value.into_item::<P>()).collect())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        Self {version:item.0,floating:FullyShardedModuleParameterRecord {version:1,parameters:item.1.into_iter()
            .map(|value|FullyShardedParameterRecord::from_exact_item(value,device)).collect()},
            packed:item.2.into_iter().map(|value|FullyShardedPackedParameterRecord::from_item::<P>(value,device)).collect()}
    }
}
