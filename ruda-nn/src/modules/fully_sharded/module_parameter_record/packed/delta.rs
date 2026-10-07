use super::*;
use alloc::string::String;

type Signature=(u64,Vec<usize>,usize,usize,DType,bool);

fn signatures<B:Backend>(record:&FullyShardedStorageRecord<B>) -> Vec<Signature> {
    let mut values=record.floating().iter().map(|value|(value.id().val(),value.logical_shape().to_vec(),value.rank(),value.world_size(),
        value.local().val().dtype(),value.is_trainable())).chain(record.packed().iter().map(|value|
        (value.id().val(),value.logical_shape().to_vec(),value.rank(),value.world_size(),value.local().val().dtype(),false))).collect::<Vec<_>>();
    values.sort_by_key(|value|value.0);values
}

/// Actual caller-selected local updates over a caller-identified original packed/dense base.
/// Omitted original values must be loaded by the caller. Signatures preserve every
/// canonical float AND integer role; exact saved bytes do not narrow adapter/base storage.
#[derive(Clone)]
pub struct FullyShardedStorageDeltaRecord<B:Backend> {
    version:u32,
    base_id:String,
    signatures:Vec<Signature>,
    updates:FullyShardedStorageRecord<B>,
}
impl<B:Backend> FullyShardedStorageDeltaRecord<B> {
    /// Capture only explicitly selected actual IDs (for example A/B and trainable norms).
    /// No adapter suffix convention, automatic base freezing or weight content hash is inferred.
    pub fn capture<M:FullyShardedModule<B>>(module:&M,base_id:&str,updated_ids:&[ParamId]) -> Result<Self,FullyShardedParameterError> {
        if base_id.is_empty() {return Err(FullyShardedParameterError::Record);}
        let mut updates=FullyShardedStorageRecord::capture(module)?;let metadata=signatures(&updates);
        let ids=updated_ids.iter().copied().collect::<BTreeSet<_>>();
        let all=metadata.iter().map(|value|ParamId::from(value.0)).collect::<BTreeSet<_>>();
        if ids.len()!=updated_ids.len() || !ids.is_subset(&all) {return Err(FullyShardedParameterError::Record);}
        updates.floating.parameters.retain(|value|ids.contains(&value.id()));updates.packed.retain(|value|ids.contains(&value.id()));
        let record=Self {version:1,base_id:base_id.into(),signatures:metadata,updates};record.validate()?;Ok(record)
    }
    /// Original caller-declared required base identity, not an automatically computed hash.
    pub fn base_id(&self) -> &str {&self.base_id}
    /// Only actually selected exact local value records; omitted base payload is absent.
    pub fn updates(&self) -> &FullyShardedStorageRecord<B> {&self.updates}
    /// Validate exact original complete signatures and every selected saved local interval.
    pub fn validate(&self) -> Result<(),FullyShardedParameterError> {
        if self.version!=1 || self.base_id.is_empty() {return Err(FullyShardedParameterError::Record);}
        self.updates.validate()?;let mut previous=None;let mut metadata=BTreeMap::new();
        for value in &self.signatures {
            if previous.is_some_and(|id|id>=value.0) {return Err(FullyShardedParameterError::Record);}
            previous=Some(value.0);packed_geometry(&value.1,value.2,value.3)?;
            match value.4 {DType::F16|DType::BF16|DType::F32=>{},DType::U8|DType::I32|DType::I64 if !value.5=>{},_=>return Err(FullyShardedParameterError::DType)}
            metadata.insert(value.0,value);
        }
        for actual in signatures(&self.updates) {
            if metadata.get(&actual.0).is_none_or(|saved|**saved!=actual) {return Err(FullyShardedParameterError::Record);}
        }
        Ok(())
    }
    /// Match all original actual packed/floating roles and the declared required base.
    /// The caller remains responsible for loading that base's original tensor contents.
    pub fn validate_for<M:FullyShardedModule<B>>(&self,module:&M,base_id:&str) -> Result<(),FullyShardedParameterError> {
        self.validate()?;
        if self.base_id!=base_id {return Err(FullyShardedParameterError::Record);}
        let native=FullyShardedStorageRecord::capture(module)?;let actual=signatures(&native);
        if actual.len()!=self.signatures.len() {return Err(FullyShardedParameterError::Record);}
        let floats=schema(module)?;
        for (saved,target) in self.signatures.iter().zip(&actual) {
            if saved.0!=target.0 || saved.1!=target.1 || saved.2!=target.2 || saved.3!=target.3 || saved.4!=target.4 {return Err(FullyShardedParameterError::Record);}
            if let Some(parameter)=floats.get(&ParamId::from(saved.0)) {
                if B::ad_enabled(&parameter.local.val().device()) && saved.5!=target.5 {return Err(FullyShardedParameterError::Trainability);}
            } else if saved.5 {return Err(FullyShardedParameterError::Trainability);}
        }
        self.updates.validate_selected_for(module)
    }
    /// Restore exactly selected real updates, retaining omitted base values and prepared options.
    /// Resumed floating leaves are canonical per ID across all actual tied projection roles.
    pub fn restore_into<M:FullyShardedModule<B>>(self,module:M,base_id:&str) -> Result<M,FullyShardedParameterError> {
        self.validate_for(&module,base_id)?;self.updates.restore_selected_into(module)
    }
    /// Convert actual selected local updates and complete original schema to a new rank topology.
    /// Repartition/load the matching original base and optimizer/pending-state records separately.
    pub fn repartition_from_ranks(sources:&[Self],rank:usize,world:usize) -> Result<Self,FullyShardedParameterError> {
        if world==0 || rank>=world {return Err(FullyShardedParameterError::Geometry("invalid delta destination rank/world"));}
        let first=sources.first().ok_or(FullyShardedParameterError::Geometry("complete storage delta rank set is empty"))?;
        let selected=signatures(&first.updates).into_iter().map(|value|value.0).collect::<BTreeSet<_>>();
        for (source_rank,source) in sources.iter().enumerate() {
            source.validate()?;
            if source.base_id!=first.base_id || source.signatures.len()!=first.signatures.len()
                || signatures(&source.updates).into_iter().map(|value|value.0).collect::<BTreeSet<_>>()!=selected {return Err(FullyShardedParameterError::Record);}
            for (actual,original) in source.signatures.iter().zip(&first.signatures) {
                if actual.0!=original.0 || actual.1!=original.1 || actual.2!=source_rank || actual.3!=sources.len()
                    || actual.4!=original.4 || actual.5!=original.5 {return Err(FullyShardedParameterError::Record);}
            }
        }
        let updates=sources.iter().map(|source|source.updates.clone()).collect::<Vec<_>>();
        let updates=FullyShardedStorageRecord::repartition_from_ranks(&updates,rank,world)?;
        let metadata=first.signatures.iter().map(|value|(value.0,value.1.clone(),rank,world,value.4,value.5)).collect();
        let record=Self {version:1,base_id:first.base_id.clone(),signatures:metadata,updates};record.validate()?;Ok(record)
    }
}
impl<B:Backend> Record<B> for FullyShardedStorageDeltaRecord<B> {
    type Item<P:PrecisionSettings>=(u32,String,Vec<Signature>,<FullyShardedStorageRecord<B> as Record<B>>::Item<P>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {(self.version,self.base_id,self.signatures,self.updates.into_item::<P>())}
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        Self {version:item.0,base_id:item.1,signatures:item.2,updates:FullyShardedStorageRecord::from_item::<P>(item.3,device)}
    }
}
