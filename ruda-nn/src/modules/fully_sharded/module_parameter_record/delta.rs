use super::*;
use alloc::string::String;

// Original identity, logical axes, rank, world, storage and trainability, without any base tensor payload.
type Signature=(u64,Vec<usize>,usize,usize,DType,bool);

/// Actual caller-selected updated local values, separate from an explicitly identified original base.
/// This can carry A/B, updated norms or any selected real parameters; it never guesses an adapter role.
/// Omitted base values, model options and complete training continuation remain caller-owned.
#[derive(Clone)]
pub struct FullyShardedModuleDeltaRecord<B:Backend> {
    version:u32,
    base_id:String,
    signatures:Vec<Signature>,
    updates:FullyShardedModuleParameterRecord<B>,
}

fn signatures<B:Backend>(shards:&BTreeMap<ParamId,ShardedParameter<B>>) -> Vec<Signature> {
    shards.iter().map(|(id,parameter)|(id.val(),parameter.logical_shape.clone(),parameter.rank,parameter.world_size,
        parameter.local.val().dtype(),parameter.local.val().is_require_grad())).collect()
}

impl<B:Backend> FullyShardedModuleDeltaRecord<B> {
    /// Save only explicitly selected real IDs, preserving canonical ties and every selected leaf's native metadata.
    /// The supplied base identifier names the required original values; this is not a tensor-content hash.
    pub fn capture<M:FullyShardedModule<B>>(module:&M,base_id:&str,updated_ids:&[ParamId]) -> Result<Self,FullyShardedParameterError> {
        require_float_only(module)?;
        if base_id.is_empty() {return Err(FullyShardedParameterError::Record);}
        let shards=schema(module)?;let ids=updated_ids.iter().copied().collect::<BTreeSet<_>>();
        if ids.len()!=updated_ids.len() || ids.iter().any(|id|!shards.contains_key(id)) {return Err(FullyShardedParameterError::Record);}
        let metadata=signatures(&shards);
        let parameters=shards.into_iter().filter(|(id,_)|ids.contains(id)).map(|(_,parameter)|parameter.parameter_record()).collect::<Result<Vec<_>,_>>()?;
        Ok(Self {version:1,base_id:base_id.into(),signatures:metadata,updates:FullyShardedModuleParameterRecord {version:1,parameters}})
    }
    /// Caller-declared identity of the required original base values.
    pub fn base_id(&self) -> &str {&self.base_id}
    /// Actual selected local value records, with no omitted base tensor payload.
    pub fn updates(&self) -> &[FullyShardedParameterRecord<B>] {self.updates.parameters()}
    /// Validate original signatures and the exact selected subset without host numerical tensor reads.
    pub fn validate(&self) -> Result<(),FullyShardedParameterError> {
        if self.version!=1 || self.base_id.is_empty() {return Err(FullyShardedParameterError::Record);}
        self.updates.validate()?;
        let mut previous=None;let mut metadata=BTreeMap::new();
        for signature in &self.signatures {
            if previous.is_some_and(|id|id>=signature.0) {return Err(FullyShardedParameterError::Record);}
            previous=Some(signature.0);
            if signature.3==0 || signature.2>=signature.3 || signature.1.is_empty() || signature.1.contains(&0) {
                return Err(FullyShardedParameterError::Geometry("delta original shard signature is invalid"));
            }
            let elements=signature.1.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis))
                .ok_or(FullyShardedParameterError::Geometry("delta original logical size overflows"))?;
            elements.div_ceil(signature.3).checked_mul(signature.3)
                .ok_or(FullyShardedParameterError::Geometry("delta original padded size overflows"))?;
            if !matches!(signature.4,DType::F32|DType::F16|DType::BF16) {return Err(FullyShardedParameterError::DType);}
            metadata.insert(signature.0,signature);
        }
        for update in self.updates.parameters() {
            let signature=metadata.get(&update.id().val()).ok_or(FullyShardedParameterError::Record)?;
            if update.logical_shape()!=signature.1.as_slice() || update.rank()!=signature.2 || update.world_size()!=signature.3 {return Err(FullyShardedParameterError::Record);}
            if update.local().val().dtype()!=signature.4 {return Err(FullyShardedParameterError::DType);}
            if update.is_trainable()!=signature.5 {return Err(FullyShardedParameterError::Trainability);}
        }
        Ok(())
    }
    /// Match the required caller-identified base and all prepared native parameter roles/flags.
    /// The caller must actually load that base's original values; this does not authenticate their contents.
    pub fn validate_for<M:FullyShardedModule<B>>(&self,module:&M,base_id:&str) -> Result<(),FullyShardedParameterError> {
        require_float_only(module)?;
        self.validate()?;
        let targets=schema(module)?;
        if self.base_id!=base_id || self.signatures.len()!=targets.len() {return Err(FullyShardedParameterError::Record);}
        for signature in &self.signatures {
            let target=targets.get(&ParamId::from(signature.0)).ok_or(FullyShardedParameterError::Record)?;
            if target.logical_shape!=signature.1 || target.rank!=signature.2 || target.world_size!=signature.3 {return Err(FullyShardedParameterError::Record);}
            if target.local.val().dtype()!=signature.4 {return Err(FullyShardedParameterError::DType);}
            if B::ad_enabled(&target.local.val().device()) && target.local.val().is_require_grad()!=signature.5 {return Err(FullyShardedParameterError::Trainability);}
        }
        Ok(())
    }
    /// Update only saved real IDs, preserving omitted original values, local aliases and destination Param mappers.
    /// The actual selected resumed leaves are created once per ID, never once per repeated module role.
    pub fn restore_into<M:FullyShardedModule<B>>(self,module:M,base_id:&str) -> Result<M,FullyShardedParameterError> {
        self.validate_for(&module,base_id)?;let targets=schema(&module)?;let mut values=BTreeMap::new();
        for saved in self.updates.parameters {
            let target=targets.get(&saved.id()).ok_or(FullyShardedParameterError::Record)?.local.val();
            let value=saved.local().val().to_device(&target.device()).detach().set_require_grad(target.is_require_grad());
            values.insert(saved.id(),value);
        }
        restore_values(module,values)
    }
    /// Offline conversion of selected actual updates from a complete rank-ordered source delta set.
    /// Repartition/load the matching original base and training-state records separately before applying it.
    pub fn repartition_from_ranks(sources:&[Self],rank:usize,world:usize) -> Result<Self,FullyShardedParameterError> {
        if world==0 || rank>=world {return Err(FullyShardedParameterError::Geometry("valid delta destination rank/world is required"));}
        let first=sources.first().ok_or(FullyShardedParameterError::Geometry("complete delta rank set is empty"))?;
        let selected=first.updates.parameters.iter().map(|record|record.id()).collect::<BTreeSet<_>>();
        for (source_rank,source) in sources.iter().enumerate() {
            source.validate()?;
            if source.base_id!=first.base_id || source.signatures.len()!=first.signatures.len()
                || source.updates.parameters.iter().map(|record|record.id()).collect::<BTreeSet<_>>()!=selected {return Err(FullyShardedParameterError::Record);}
            for (signature,original) in source.signatures.iter().zip(&first.signatures) {
                if signature.0!=original.0 || signature.1!=original.1 || signature.2!=source_rank || signature.3!=sources.len()
                    || signature.4!=original.4 || signature.5!=original.5 {return Err(FullyShardedParameterError::Record);}
            }
        }
        let records=sources.iter().map(|source|source.updates.clone()).collect::<Vec<_>>();
        let updates=FullyShardedModuleParameterRecord::repartition_from_ranks(&records,rank,world)?;
        let signatures=first.signatures.iter().map(|signature|(signature.0,signature.1.clone(),rank,world,signature.4,signature.5)).collect();
        let record=Self {version:1,base_id:first.base_id.clone(),signatures,updates};record.validate()?;Ok(record)
    }
}

impl<B:Backend> Record<B> for FullyShardedModuleDeltaRecord<B> {
    type Item<P:PrecisionSettings>=(u32,String,Vec<Signature>,<FullyShardedModuleParameterRecord<B> as Record<B>>::Item<P>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {(self.version,self.base_id,self.signatures,self.updates.into_item::<P>())}
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        Self {version:item.0,base_id:item.1,signatures:item.2,updates:FullyShardedModuleParameterRecord::<B>::from_item::<P>(item.3,device)}
    }
}
