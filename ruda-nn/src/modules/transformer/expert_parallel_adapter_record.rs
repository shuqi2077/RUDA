use alloc::{format,vec::Vec};
use ruda_model::{record::{Record,PrecisionSettings,Recorder,RecorderError},serde::{Serialize,Deserialize},tensor::backend::Backend};
use crate::{ExpertAdapterProjections,ExpertAdapterProjectionRef,ExpertLoRAAdapterSchema,expert_parallel::ExpertParallelGeometry};
use super::{TransformerProjectionShape,ExpertParallelTransformerModel,ExpertParallelTransformerLayer,ExpertTransformerAdapterRecord,ExpertAdapterPath};

/// Exact original layer ownership attached to a rank-local expert A/B record.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub struct ExpertAdapterOwnershipEntry {
    /// Actual zero-based model layer index.
    pub layer:usize,
    /// Complete original expert-world prefix, including zero-expert owners.
    pub prefix:Vec<usize>,
    /// Original expert-world rank, independent of data/tensor parallel rank.
    pub rank:usize,
}
fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid expert-owned model adapter record: {reason}"))}
fn ownership<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>>(model:&ExpertParallelTransformerModel<B,P,E>)
    -> Vec<ExpertAdapterOwnershipEntry> {
    let mut result=Vec::new();for (index,layer) in model.layers.iter().enumerate() {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
        result.push(ExpertAdapterOwnershipEntry {layer:index,prefix:block.routed.experts.ownership().prefix().to_vec(),rank:block.routed.experts.rank()});
    }}result
}
fn sources<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>+ExpertAdapterProjections<B>>(
    model:&ExpertParallelTransformerModel<B,P,E>) -> Vec<(ExpertAdapterPath,ExpertAdapterProjectionRef<'_,B>)> {
    let mut result=Vec::new();for (index,layer) in model.layers.iter().enumerate() {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
        for (role,projection) in block.routed.experts.expert_adapter_projections() {result.push((ExpertAdapterPath {layer:index,role},projection));}
    }}result
}
/// Actual rank-local expert A/B-only continuation, with exact full expert-world ownership per layer.
/// Stores each original shared A/B leaf once; excludes base payloads and optimizer/transport state.
/// It does not repartition adapters onto a different world or rank.
pub struct ExpertParallelTransformerAdapterRecord<B:Backend> {
    version:u32,
    ownership:Vec<ExpertAdapterOwnershipEntry>,
    adapters:ExpertTransformerAdapterRecord<B>,
}
impl<B:Backend> Record<B> for ExpertParallelTransformerAdapterRecord<B> {
    type Item<S:PrecisionSettings> = (u32,Vec<ExpertAdapterOwnershipEntry>,<ExpertTransformerAdapterRecord<B> as Record<B>>::Item<S>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {(self.version,self.ownership,self.adapters.into_item::<S>())}
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,ownership:item.1,adapters:ExpertTransformerAdapterRecord::from_item::<S>(item.2,device)}
    }
}
impl<B:Backend> ExpertParallelTransformerAdapterRecord<B> {
    /// Capture actual owned floating/NF4/AWQ expert A/B, retaining ties and exact rank ownership.
    /// Use full-precision recorder settings for exact saved A/B values.
    pub fn capture<P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>+ExpertAdapterProjections<B>>(
        model:&ExpertParallelTransformerModel<B,P,E>,base_id:&str) -> Result<Self,RecorderError> {
        let adapters=ExpertTransformerAdapterRecord::capture_sources(model,base_id,model.layers.len(),sources(model))?;
        Ok(Self {version:1,ownership:ownership(model),adapters})
    }
    /// Actual complete expert-world topology and original rank for every owned layer.
    pub fn ownership(&self) -> &[ExpertAdapterOwnershipEntry] {&self.ownership}
    /// Actual stored expert A/B roles and their original floating/quantized continuation contracts.
    pub fn targets(&self) -> impl Iterator<Item=(&ExpertAdapterPath,&ExpertLoRAAdapterSchema)> {self.adapters.targets()}
    /// Actual canonical stored A/B cubes, without counting tied occurrences twice.
    pub fn parameter_count(&self) -> usize {self.adapters.parameter_count()}
    /// Validate ownership, source windows, exact target set and shared A/B before replacing any leaves.
    pub fn validate_for<P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>+ExpertAdapterProjections<B>>(
        &self,model:&ExpertParallelTransformerModel<B,P,E>,base_id:&str) -> Result<(),RecorderError> {
        if self.version!=1 || self.ownership!=ownership(model) {return Err(invalid("original expert-world prefix, rank or owned layer order differs"));}
        self.adapters.validate_sources(model,base_id,model.layers.len(),sources(model))
    }
    /// Save only actual owned A/B and their exact source/ownership metadata with an existing recorder.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load actual owned A/B onto the selected device without constructing base coefficients or topology.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}
    /// Restore canonical actual A/B IDs/flags and shared leaves while retaining original base and non-expert values.
    /// Pending gradients, optimizer state and transport state remain caller-owned separate records.
    pub fn restore_into<P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>+ExpertAdapterProjections<B>>(
        self,mut model:ExpertParallelTransformerModel<B,P,E>,base_id:&str) -> Result<ExpertParallelTransformerModel<B,P,E>,RecorderError> {
        self.validate_for(&model,base_id)?;let mut mapper=self.adapters.into_mapper();
        model.layers=model.layers.into_iter().enumerate().map(|(index,layer)| {
            mapper.layer=index;match layer {
                ExpertParallelTransformerLayer::Parallel(mut block)=>{
                    block.routed.experts=block.routed.experts.map_expert_adapters(&mut mapper)?;Ok(ExpertParallelTransformerLayer::Parallel(block))
                },
                original=>Ok(original),
            }
        }).collect::<Result<_,RecorderError>>()?;Ok(model)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>+ExpertAdapterProjections<B>> ExpertParallelTransformerModel<B,P,E> {
    /// Export actual rank-owned expert A/B only, with exact original world topology and source-block contracts.
    pub fn expert_adapter_record(&self,base_id:&str) -> Result<ExpertParallelTransformerAdapterRecord<B>,RecorderError> {
        ExpertParallelTransformerAdapterRecord::capture(self,base_id)
    }
}
