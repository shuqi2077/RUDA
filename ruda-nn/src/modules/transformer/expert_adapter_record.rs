use alloc::{collections::{BTreeMap,BTreeSet},format,string::String,vec::Vec};
use ruda_model::{module::{Module,ModuleVisitor,Param},record::{Record,PrecisionSettings,Recorder,RecorderError},
    serde::{Serialize,Deserialize},tensor::{Tensor,DType,backend::Backend}};
use crate::{ExpertAdapterTarget,ExpertAdapterProjections,ExpertAdapterMapper,ExpertAdapterRecordBase,
    ExpertLoRAAdapterSchema,PackedExpertLoRA,ExpertAdapterProjectionRef};
use super::{TransformerProjectionShape,Nf4MoeTransformerModel,Nf4MoeTransformerLayer};

type ParameterKey=(u64,bool);
fn key<B:Backend>(parameter:&Param<Tensor<B,3>>) -> ParameterKey {(parameter.id.val(),parameter.val().is_require_grad())}
fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid native expert model adapter record: {reason}"))}

/// Actual loaded zero-based layer and original expert matrix role.
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub struct ExpertAdapterPath {
    /// Original model layer index, without model-family inferred names.
    pub layer:usize,
    /// Actual original gate/up/down projection.
    pub role:ExpertAdapterTarget,
}
/// Exact source role and its canonical A/B identities in the model adapter file.
#[derive(Clone,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub struct ExpertAdapterProjectionEntry {
    /// Actual original layer/role path.
    pub path:ExpertAdapterPath,
    /// Original source representation and actual adapter configuration.
    pub schema:ExpertLoRAAdapterSchema,
    /// Actual A, then B identities and their independent training flags.
    pub parameters:[ParameterKey;2],
}
struct ParameterEntry<B:Backend> {
    key:ParameterKey,
    shape:[usize;3],
    dtype:DType,
    record:<Param<Tensor<B,3>> as Module<B>>::Record,
}
impl<B:Backend> ParameterEntry<B> {
    fn capture(parameter:&Param<Tensor<B,3>>) -> Self {
        let value=parameter.val();let key=key(parameter);let shape=value.dims();let dtype=value.dtype();
        let record=parameter.clone().map(|value| {
            let trainable=value.is_require_grad();value.detach().set_require_grad(trainable)
        }).into_record().map(|value|value.detach());
        Self {key,shape,dtype,record}
    }
}
impl<B:Backend> Record<B> for ParameterEntry<B> {
    type Item<S:PrecisionSettings> = (ParameterKey,[usize;3],DType,<<Param<Tensor<B,3>> as Module<B>>::Record as Record<B>>::Item<S>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {(self.key,self.shape,self.dtype,self.record.into_item::<S>())}
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {key:item.0,shape:item.1,dtype:item.2,record:<<Param<Tensor<B,3>> as Module<B>>::Record as Record<B>>::from_item::<S>(item.3,device)}
    }
}

/// Whole-model expert A/B-only delta, storing one tensor per original shared ID and training flag.
/// Excludes base words/scales/cubes, attention/router/shared/head adapters, optimizer and transport state.
pub struct ExpertTransformerAdapterRecord<B:Backend> {
    version:u32,
    base_id:String,
    layers:usize,
    projections:Vec<ExpertAdapterProjectionEntry>,
    parameters:Vec<ParameterEntry<B>>,
}
impl<B:Backend> Record<B> for ExpertTransformerAdapterRecord<B> {
    type Item<S:PrecisionSettings> = (u32,String,usize,Vec<ExpertAdapterProjectionEntry>,
        Vec<(ParameterKey,[usize;3],DType,<<Param<Tensor<B,3>> as Module<B>>::Record as Record<B>>::Item<S>)>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.base_id,self.layers,self.projections,self.parameters.into_iter().map(|parameter|parameter.into_item::<S>()).collect())
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,base_id:item.1,layers:item.2,projections:item.3,
            parameters:item.4.into_iter().map(|parameter|ParameterEntry::<B>::from_item::<S>(parameter,device)).collect()}
    }
}
struct ParameterOccurrences(BTreeMap<ParameterKey,usize>);
impl<B:Backend> ModuleVisitor<B> for ParameterOccurrences {
    fn visit_float<const D:usize>(&mut self,parameter:&Param<Tensor<B,D>>) {
        let key=(parameter.id.val(),parameter.val().is_require_grad());*self.0.entry(key).or_default()+=1;
    }
}
fn check_isolated<B:Backend,M:Module<B>>(model:&M,expected:&BTreeMap<ParameterKey,usize>) -> Result<(),RecorderError> {
    let mut actual=ParameterOccurrences(BTreeMap::new());model.visit(&mut actual);
    for (key,count) in expected {if actual.0.get(key)!=Some(count) {
        return Err(invalid("expert A/B is tied to an omitted non-expert parameter; use a full model record"));}}
    Ok(())
}
fn local_sources<B:Backend,P:TransformerProjectionShape<B>,E:ExpertAdapterProjections<B>>(model:&Nf4MoeTransformerModel<B,P,E>)
    -> Vec<(ExpertAdapterPath,ExpertAdapterProjectionRef<'_,B>)> {
    let mut sources=Vec::new();for (index,layer) in model.layers.iter().enumerate() {if let Nf4MoeTransformerLayer::Packed(block)=layer {
        for (role,projection) in block.routed.experts.expert_adapter_projections() {sources.push((ExpertAdapterPath {layer:index,role},projection));}
    }}sources
}
impl<B:Backend> ExpertTransformerAdapterRecord<B> {
    /// Capture all actual expert adapters, retaining source paths/tying without downloading base values.
    /// Full-precision recorder settings are required for exact saved A/B values.
    pub fn capture<P:TransformerProjectionShape<B>,E:ExpertAdapterProjections<B>>(model:&Nf4MoeTransformerModel<B,P,E>,base_id:&str)
        -> Result<Self,RecorderError> {
        Self::capture_sources(model,base_id,model.layers.len(),local_sources(model))
    }
    pub(super) fn capture_sources<M:Module<B>>(model:&M,base_id:&str,layers:usize,
        sources:Vec<(ExpertAdapterPath,ExpertAdapterProjectionRef<'_,B>)>) -> Result<Self,RecorderError> {
        if base_id.is_empty() {return Err(invalid("complete original base identity must be supplied"));}
        let mut projections=Vec::new();let mut parameters=BTreeMap::<ParameterKey,ParameterEntry<B>>::new();
        let mut occurrences=BTreeMap::new();let mut devices=BTreeMap::new();
        for (path,projection) in sources {
                let schema=projection.schema(base_id)?;let leaves=projection.parameters();let bindings=[key(leaves[0]),key(leaves[1])];
                for parameter in leaves {
                    let binding=key(parameter);let value=parameter.val();*occurrences.entry(binding).or_default()+=1;
                    if let Some(previous)=parameters.get(&binding) {
                        if previous.shape!=value.dims() || previous.dtype!=value.dtype() || devices.get(&binding)!=Some(&value.device()) {
                            return Err(invalid("shared expert A/B has inconsistent shape, dtype or device"));}
                    } else {devices.insert(binding,value.device());parameters.insert(binding,ParameterEntry::capture(parameter));}
                }
                projections.push(ExpertAdapterProjectionEntry {path,schema,parameters:bindings});
        }
        if projections.is_empty() {return Err(invalid("model has no actual expert adapters"));}
        check_isolated::<B,_>(model,&occurrences)?;
        Ok(Self {version:1,base_id:base_id.into(),layers,projections,parameters:parameters.into_values().collect()})
    }
    /// Actual stored layer/role paths and their original projection contracts.
    pub fn targets(&self) -> impl Iterator<Item=(&ExpertAdapterPath,&ExpertLoRAAdapterSchema)> {
        self.projections.iter().map(|entry|(&entry.path,&entry.schema))
    }
    /// Number of canonical stored A/B cubes; tied occurrences are not counted twice.
    pub fn parameter_count(&self) -> usize {self.parameters.len()}
    /// Validate every target and saved shared-parameter contract before replacing any adapter values.
    pub fn validate_for<P:TransformerProjectionShape<B>,E:ExpertAdapterProjections<B>>(&self,model:&Nf4MoeTransformerModel<B,P,E>,base_id:&str)
        -> Result<(),RecorderError> {
        self.validate_sources(model,base_id,model.layers.len(),local_sources(model))
    }
    pub(super) fn validate_sources<M:Module<B>>(&self,model:&M,base_id:&str,layers:usize,
        sources:Vec<(ExpertAdapterPath,ExpertAdapterProjectionRef<'_,B>)>) -> Result<(),RecorderError> {
        if self.version!=1 || base_id.is_empty() || self.base_id!=base_id || self.layers!=layers || self.projections.is_empty() {
            return Err(invalid("version, complete original base identity or layer topology differs"));}
        let mut entries=BTreeMap::new();for entry in &self.projections {
            if entries.insert(entry.path,entry).is_some() {return Err(invalid("duplicate original expert projection path"));}}
        let mut parameters=BTreeMap::new();for entry in &self.parameters {
            if entry.key.0!=entry.record.id.val() || !matches!(entry.dtype,DType::F16|DType::BF16|DType::F32)
                || parameters.insert(entry.key,entry).is_some() {return Err(invalid("invalid or duplicate canonical A/B parameter"));}}
        let mut occurrences=BTreeMap::new();let mut used=BTreeSet::new();let mut count=0;
        for (path,projection) in sources {
                count+=1;let entry=entries.get(&path).ok_or_else(||invalid("missing original expert projection path"))?;
                projection.validate_schema(&entry.schema,base_id)?;
                for (leaf,binding) in projection.parameters().into_iter().zip(entry.parameters) {
                    let value=leaf.val();let old=key(leaf);let saved=parameters.get(&binding).ok_or_else(||invalid("expert A/B references an absent parameter"))?;
                    if binding.1!=value.is_require_grad() || saved.shape!=value.dims() || saved.dtype!=value.dtype() {
                        return Err(invalid("canonical expert A/B geometry/storage/flags differ"));}
                    *occurrences.entry(old).or_default()+=1;used.insert(binding);
                }
        }
        if count!=entries.len() || used.len()!=parameters.len() {return Err(invalid("unexpected expert target or unused A/B parameter"));}
        check_isolated::<B,_>(model,&occurrences)?;
        let mut actual=ParameterOccurrences(BTreeMap::new());model.visit(&mut actual);
        for binding in used {if actual.0.get(&binding).copied().unwrap_or(0)!=occurrences.get(&binding).copied().unwrap_or(0) {
            return Err(invalid("saved expert A/B ID collides with an unchanged non-expert parameter"));}}
        Ok(())
    }
    /// Save only actual canonical expert A/B and their source metadata using an existing recorder.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load the actual A/B table onto a selected device without constructing base weights or a replacement model.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}
    /// Restore original A/B IDs and shared native leaves, retaining every non-expert/base value, flag and parameter mapper.
    /// Optimizer/pending-gradient state is restored separately after this operation.
    pub fn restore_into<P:TransformerProjectionShape<B>,E:ExpertAdapterProjections<B>>(self,mut model:Nf4MoeTransformerModel<B,P,E>,base_id:&str)
        -> Result<Nf4MoeTransformerModel<B,P,E>,RecorderError> {
        self.validate_for(&model,base_id)?;
        let mut mapper=self.into_mapper();
        model.layers=model.layers.into_iter().enumerate().map(|(index,layer)| {
            mapper.layer=index;match layer {
                Nf4MoeTransformerLayer::Packed(mut block)=>{block.routed.experts=block.routed.experts.map_expert_adapters(&mut mapper)?;Ok(Nf4MoeTransformerLayer::Packed(block))},
                original=>Ok(original),
            }
        }).collect::<Result<_,RecorderError>>()?;Ok(model)
    }
    pub(super) fn into_mapper(self) -> Restore<B> {
        Restore {layer:0,entries:self.projections.into_iter().map(|entry|(entry.path,entry)).collect(),
            parameters:self.parameters.into_iter().map(|entry|(entry.key,entry)).collect(),leaves:BTreeMap::new()}
    }
}
pub(super) struct Restore<B:Backend> {
    pub(super) layer:usize,
    entries:BTreeMap<ExpertAdapterPath,ExpertAdapterProjectionEntry>,
    parameters:BTreeMap<ParameterKey,ParameterEntry<B>>,
    leaves:BTreeMap<ParameterKey,Tensor<B,3>>,
}
impl<B:Backend> Restore<B> {
    fn parameter(&mut self,original:Param<Tensor<B,3>>,binding:ParameterKey) -> Result<Param<Tensor<B,3>>,RecorderError> {
        let entry=self.parameters.get(&binding).ok_or_else(||invalid("missing validated A/B parameter"))?;
        let device=original.val().device();let loaded=original.load_record(entry.record.clone()).fork(&device)
            .map(|value|value.cast(entry.dtype).detach().set_require_grad(binding.1));
        let (id,value,parameter_mapper)=loaded.consume();
        if value.dims()!=entry.shape || value.dtype()!=entry.dtype || value.device()!=device || value.is_require_grad()!=binding.1 {
            return Err(invalid("A/B load mapper changed original geometry/storage/device/flags"));}
        let canonical=self.leaves.entry(binding).or_insert(value).clone();Ok(Param::from_mapped_value(id,canonical,parameter_mapper))
    }
}
impl<B:Backend> ExpertAdapterMapper<B> for Restore<B> {
    fn map<Base:ExpertAdapterRecordBase<B>>(&mut self,role:ExpertAdapterTarget,mut layer:PackedExpertLoRA<B,Base>)
        -> Result<PackedExpertLoRA<B,Base>,RecorderError> {
        let path=ExpertAdapterPath {layer:self.layer,role};let bindings=self.entries.get(&path).ok_or_else(||invalid("missing validated expert target"))?.parameters;
        layer.adapter_a.weight=self.parameter(layer.adapter_a.weight,bindings[0])?;
        layer.adapter_b.weight=self.parameter(layer.adapter_b.weight,bindings[1])?;layer.validate();Ok(layer)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:ExpertAdapterProjections<B>> Nf4MoeTransformerModel<B,P,E> {
    /// Export all actual expert adapters with exact source roles, preserving ties without serializing base tensors.
    pub fn expert_adapter_record(&self,base_id:&str) -> Result<ExpertTransformerAdapterRecord<B>,RecorderError> {
        ExpertTransformerAdapterRecord::capture(self,base_id)
    }
}
