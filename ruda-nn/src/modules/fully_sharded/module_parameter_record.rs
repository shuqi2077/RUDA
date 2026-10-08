use super::*;
use ruda_model::{module::{ModuleMapper,ModuleVisitor,ModuleDisplay},record::{Record,PrecisionSettings}};
use alloc::collections::BTreeSet;
use crate::hybrid_sharded::{FullyShardedColumnParallelLinear,FullyShardedRowParallelLinear,FullyShardedColumnParallelLoRA,
    FullyShardedRowParallelLoRA,FullyShardedTensorParallelGatedMlp,FullyShardedVocabParallelEmbedding,FullyShardedVocabParallelProjection};

mod delta;
pub use delta::FullyShardedModuleDeltaRecord;
mod packed;
pub use packed::{FullyShardedPackedParameterRecord,FullyShardedStorageRecord,FullyShardedStorageDeltaRecord};

/// Explicit logical-shard visitation alongside native local-parameter module visitation.
/// Implementations must enumerate every actual sharded leaf, including repeated shared roles.
pub trait FullyShardedModule<B:Backend>:Module<B> {
    /// Visit each actual logical parameter together with its exact rank/topology metadata.
    fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F);
    /// Visit exact immutable packed integer slices separately from floating leaves.
    /// Existing floating-only modules retain an empty packed schema.
    fn visit_packed_shards<F:FnMut(&ShardedPackedParameter<B>)>(&self,_visitor:&mut F) {}
}

fn require_float_only<B:Backend,M:FullyShardedModule<B>>(module:&M) -> Result<(),FullyShardedParameterError> {
    let mut packed=false;module.visit_packed_shards(&mut |_|packed=true);
    if packed {return Err(FullyShardedParameterError::Geometry("packed modules require FullyShardedStorageRecord; floating-only records omit packed base words"));}
    Ok(())
}

/// Canonical local parameter-only checkpoint for a complete native sharded module.
/// Architecture/options remain in the caller's prepared module; optimizer, pending gradients,
/// scheduler, RNG and data continuation require their matching existing separate records.
#[derive(Clone)]
pub struct FullyShardedModuleParameterRecord<B:Backend> {
    version:u32,
    parameters:Vec<FullyShardedParameterRecord<B>>,
}

fn schema<B:Backend,M:FullyShardedModule<B>>(module:&M) -> Result<BTreeMap<ParamId,ShardedParameter<B>>,FullyShardedParameterError> {
    let mut shards=BTreeMap::<ParamId,ShardedParameter<B>>::new();let mut error=None;
    module.visit_shards(&mut |parameter| {
        if error.is_some() {return;}
        if let Err(reason)=parameter.parameter_record() {error=Some(reason);return;}
        if let Some(previous)=shards.get(&parameter.local.id) {
            let first=previous.local.val();let current=parameter.local.val();
            if previous.logical_shape!=parameter.logical_shape || previous.rank!=parameter.rank || previous.world_size!=parameter.world_size {
                error=Some(FullyShardedParameterError::Record);
            } else if first.dtype()!=current.dtype() {error=Some(FullyShardedParameterError::DType);
            } else if first.is_require_grad()!=current.is_require_grad() {error=Some(FullyShardedParameterError::Trainability);
            } else if first.device()!=current.device() {error=Some(FullyShardedParameterError::Device);}
        } else {shards.insert(parameter.local.id,parameter.clone());}
    });
    if let Some(error)=error {return Err(error);}
    struct Check<'a,B:Backend> {shards:&'a BTreeMap<ParamId,ShardedParameter<B>>,seen:BTreeSet<ParamId>,error:Option<FullyShardedParameterError>}
    impl<B:Backend> ModuleVisitor<B> for Check<'_,B> {
        fn visit_float<const D:usize>(&mut self,parameter:&Param<Tensor<B,D>>) {
            let Some(shard)=self.shards.get(&parameter.id) else {self.error=Some(FullyShardedParameterError::Record);return;};
            if D!=1 || parameter.val().shape().num_elements()!=shard.local.val().dims()[0] {
                self.error=Some(FullyShardedParameterError::Geometry("module contains non-local parameter storage"));
            }
            if parameter.val().dtype()!=shard.local.val().dtype() {self.error=Some(FullyShardedParameterError::DType);}
            if parameter.val().is_require_grad()!=shard.local.val().is_require_grad() {self.error=Some(FullyShardedParameterError::Trainability);}
            if parameter.val().device()!=shard.local.val().device() {self.error=Some(FullyShardedParameterError::Device);}
            self.seen.insert(parameter.id);
        }
    }
    let mut check=Check {shards:&shards,seen:BTreeSet::new(),error:None};module.visit(&mut check);
    if let Some(error)=check.error {return Err(error);}
    if check.seen.len()!=shards.len() {return Err(FullyShardedParameterError::Record);}
    Ok(shards)
}

fn restore_values<B:Backend,M:FullyShardedModule<B>>(module:M,values:BTreeMap<ParamId,Tensor<B,1>>) -> Result<M,FullyShardedParameterError> {
    struct Restore<B:Backend> {values:BTreeMap<ParamId,Tensor<B,1>>,seen:BTreeSet<ParamId>}
    impl<B:Backend> ModuleMapper<B> for Restore<B> {
        fn map_float<const D:usize>(&mut self,parameter:Param<Tensor<B,D>>) -> Param<Tensor<B,D>> {
            let Some(value)=self.values.get(&parameter.id).cloned() else {return parameter;};
            assert_eq!(D,1,"validated sharded module leaf rank changed");self.seen.insert(parameter.id);
            // Primitive rewrapping preserves the exact canonical node, unlike a per-alias reshape/cast.
            parameter.map(|_|Tensor::<B,D>::from_primitive(value.into_primitive()))
        }
    }
    let mut restore=Restore {values,seen:BTreeSet::new()};let module=module.map(&mut restore);
    if restore.seen.len()!=restore.values.len() {return Err(FullyShardedParameterError::Record);}
    Ok(module)
}

impl<B:Backend> FullyShardedModuleParameterRecord<B> {
    /// Capture each canonical actual local leaf exactly once, retaining original logical/topology metadata.
    pub fn capture<M:FullyShardedModule<B>>(module:&M) -> Result<Self,FullyShardedParameterError> {
        require_float_only(module)?;
        let parameters=schema(module)?.into_values().map(|parameter|parameter.parameter_record()).collect::<Result<Vec<_>,_>>()?;
        Ok(Self {version:1,parameters})
    }
    /// Inspect actual canonical local parameter records without materializing global weights.
    pub fn parameters(&self) -> &[FullyShardedParameterRecord<B>] {&self.parameters}
    /// Number of distinct actual parameter identities, not repeated tied role count.
    pub fn parameter_count(&self) -> usize {self.parameters.len()}
    /// Validate every actual saved local interval and reject ambiguous duplicate identities.
    pub fn validate(&self) -> Result<(),FullyShardedParameterError> {
        if self.version!=1 {return Err(FullyShardedParameterError::Record);}
        let mut ids=BTreeSet::new();
        for parameter in &self.parameters {
            parameter.validate()?;
            if !ids.insert(parameter.id()) {return Err(FullyShardedParameterError::Record);}
        }
        Ok(())
    }
    /// Validate exact prepared architecture parameter IDs, logical layouts, topology, storage and trainability.
    pub fn validate_for<M:FullyShardedModule<B>>(&self,module:&M) -> Result<(),FullyShardedParameterError> {
        require_float_only(module)?;
        self.validate_float_for(module)
    }

    fn validate_float_for<M:FullyShardedModule<B>>(&self,module:&M) -> Result<(),FullyShardedParameterError> {
        self.validate()?;let shards=schema(module)?;
        if shards.len()!=self.parameters.len() {return Err(FullyShardedParameterError::Record);}
        for saved in &self.parameters {
            let target=shards.get(&saved.id()).ok_or(FullyShardedParameterError::Record)?;
            if target.logical_shape!=saved.logical_shape() || target.rank!=saved.rank() || target.world_size!=saved.world_size() {
                return Err(FullyShardedParameterError::Record);
            }
            if target.local.val().dtype()!=saved.local().val().dtype() {return Err(FullyShardedParameterError::DType);}
            if B::ad_enabled(&target.local.val().device()) && target.local.val().is_require_grad()!=saved.is_trainable() {
                return Err(FullyShardedParameterError::Trainability);
            }
        }
        Ok(())
    }
    /// Restore actual local values and one canonical resumed AD leaf per source ID across every tied role.
    /// Preserve destination Param mappers and prepared nonparameter architecture; no full weights are gathered.
    pub fn restore_into<M:FullyShardedModule<B>>(self,module:M) -> Result<M,FullyShardedParameterError> {
        self.validate_for(&module)?;let targets=schema(&module)?;
        let mut values=BTreeMap::new();
        for saved in self.parameters {
            let original=targets.get(&saved.id()).ok_or(FullyShardedParameterError::Record)?.local.val();
            let value=saved.local().val().to_device(&original.device()).detach().set_require_grad(original.is_require_grad());
            values.insert(saved.id(),value);
        }
        restore_values(module,values)
    }
    /// Offline complete-rank-set conversion of all original local values to a new data topology.
    /// Source records must already share the chosen device; matching optimizer/gradient conversion remains separate.
    pub fn repartition_from_ranks(sources:&[Self],rank:usize,world:usize) -> Result<Self,FullyShardedParameterError> {
        if world==0 || rank>=world {return Err(FullyShardedParameterError::Geometry("valid destination rank/world is required"));}
        let first=sources.first().ok_or(FullyShardedParameterError::Geometry("complete module rank set is empty"))?;
        let mut ranks=Vec::with_capacity(sources.len());
        for source in sources {
            source.validate()?;
            if source.parameters.len()!=first.parameters.len() {return Err(FullyShardedParameterError::Record);}
            ranks.push(source.parameters.iter().map(|record|(record.id(),record)).collect::<BTreeMap<_,_>>());
        }
        let mut parameters=Vec::with_capacity(first.parameters.len());
        for record in &first.parameters {
            let records=ranks.iter().map(|source|source.get(&record.id()).ok_or(FullyShardedParameterError::Record)
                .map(|saved|(*saved).clone())).collect::<Result<Vec<_>,_>>()?;
            parameters.push(FullyShardedParameterRecord::repartition_from_ranks(&records,rank,world)?);
        }
        Ok(Self {version:1,parameters})
    }
}

impl<B:Backend> Record<B> for FullyShardedModuleParameterRecord<B> {
    type Item<P:PrecisionSettings>=(u32,Vec<<FullyShardedParameterRecord<B> as Record<B>>::Item<P>>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.parameters.into_iter().map(|parameter|parameter.into_item::<P>()).collect())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        Self {version:item.0,parameters:item.1.into_iter().map(|parameter|FullyShardedParameterRecord::<B>::from_item::<P>(parameter,device)).collect()}
    }
}

macro_rules! visit_fields {
    ($module:ident,$($field:ident),+ $(,)?) => {
        impl<B:Backend> FullyShardedModule<B> for $module<B> {
            fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {$(self.$field.visit_shards(visitor);)+}
            fn visit_packed_shards<F:FnMut(&ShardedPackedParameter<B>)>(&self,visitor:&mut F) {$(self.$field.visit_packed_shards(visitor);)+}
        }
    };
}
impl<B:Backend> FullyShardedModule<B> for ShardedParameter<B> {
    fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {visitor(self);}
}
impl<B:Backend,M:FullyShardedModule<B>> FullyShardedModule<B> for Option<M> {
    fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {if let Some(module)=self {module.visit_shards(visitor);}}
    fn visit_packed_shards<F:FnMut(&ShardedPackedParameter<B>)>(&self,visitor:&mut F) {if let Some(module)=self {module.visit_packed_shards(visitor);}}
}
impl<B:Backend,M:FullyShardedModule<B>> FullyShardedModule<B> for Vec<M> {
    fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {for module in self {module.visit_shards(visitor);}}
    fn visit_packed_shards<F:FnMut(&ShardedPackedParameter<B>)>(&self,visitor:&mut F) {for module in self {module.visit_packed_shards(visitor);}}
}
visit_fields!(FullyShardedLinear,weight,bias);
visit_fields!(FullyShardedEmbedding,weight);
visit_fields!(FullyShardedProjection,weight,bias);
visit_fields!(FullyShardedLoRALinear,base,adapter_a,adapter_b);
visit_fields!(FullyShardedGatedMLP,gate,up,down);
visit_fields!(FullyShardedLayerNorm,gamma,beta);
visit_fields!(FullyShardedRmsNorm,gamma);
visit_fields!(FullyShardedPRelu,alpha);
visit_fields!(FullyShardedSwiGlu,inner,outer);
visit_fields!(FullyShardedGroupedQueryAttention,query,key,value,output);
visit_fields!(FullyShardedFeedForward,up,gate,down,activation);
visit_fields!(FullyShardedTransformerBlock,attention,feed_forward,attention_norm,feed_forward_norm);
visit_fields!(FullyShardedCrossAttentionBlock,attention,query_norm,memory_norm);
visit_fields!(FullyShardedEncoderDecoderLayer,backbone,cross_attention);
visit_fields!(FullyShardedTransformerStack,blocks);
visit_fields!(FullyShardedEncoderDecoderStack,layers);
visit_fields!(FullyShardedTransformerEmbeddings,token,position,token_type,normalization);
visit_fields!(FullyShardedTransformerHead,projection,normalization);
visit_fields!(FullyShardedTransformerModel,embeddings,backbone,normalization,head);
visit_fields!(FullyShardedEncoderDecoderModel,source_embeddings,encoder,encoder_normalization,target_embeddings,decoder,decoder_normalization,head);
visit_fields!(FullyShardedColumnParallelLinear,local);
visit_fields!(FullyShardedRowParallelLinear,local);
visit_fields!(FullyShardedColumnParallelLoRA,base,adapter_a,adapter_b);
visit_fields!(FullyShardedRowParallelLoRA,base,adapter_a,adapter_b);
visit_fields!(FullyShardedTensorParallelGatedMlp,gate,up,down);
visit_fields!(FullyShardedVocabParallelEmbedding,weight);
visit_fields!(FullyShardedVocabParallelProjection,weight,bias);
visit_fields!(FullyShardedAwqLinear,qweight,qzeros,scales,bias);
visit_fields!(FullyShardedAwqLoRALinear,base,adapter_a,adapter_b);
macro_rules! visit_projected_fields {
    ($module:ident,$($field:ident),+ $(,)?) => {
        impl<B:Backend,P:FullyShardedModule<B>+ModuleDisplay> FullyShardedModule<B> for $module<B,P> {
            fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {$(self.$field.visit_shards(visitor);)+}
            fn visit_packed_shards<F:FnMut(&ShardedPackedParameter<B>)>(&self,visitor:&mut F) {$(self.$field.visit_packed_shards(visitor);)+}
        }
    };
}
visit_projected_fields!(FullyShardedAwqAttention,query,key,value,output);
visit_projected_fields!(FullyShardedAwqFeedForward,up,gate,down,activation);
visit_projected_fields!(FullyShardedAwqTransformerBlock,attention,feed_forward,attention_norm,feed_forward_norm);
visit_projected_fields!(FullyShardedAwqTransformerStack,blocks);
visit_projected_fields!(FullyShardedAwqTransformerHead,projection,normalization);
visit_projected_fields!(FullyShardedAwqTransformerModel,embeddings,backbone,normalization,head);
visit_projected_fields!(FullyShardedProjectedCrossAttention,attention,query_norm,memory_norm);
visit_projected_fields!(FullyShardedProjectedDecoderLayer,backbone,cross_attention);
visit_projected_fields!(FullyShardedProjectedDecoderStack,layers);
visit_projected_fields!(FullyShardedProjectedEncoderDecoderModel,source_embeddings,encoder,encoder_normalization,target_embeddings,decoder,decoder_normalization,head);
visit_fields!(FullyShardedNf4Linear,packed,scales,codebook,bias);
visit_fields!(FullyShardedNf4LoRALinear,base,adapter_a,adapter_b);
impl<B:Backend> FullyShardedModule<B> for ShardedPackedParameter<B> {
    fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,_visitor:&mut F) {}
    fn visit_packed_shards<F:FnMut(&ShardedPackedParameter<B>)>(&self,visitor:&mut F) {visitor(self);}
}

macro_rules! visit_variants {
    ($module:ident,$($variant:ident),+ $(,)?) => {
        impl<B:Backend> FullyShardedModule<B> for $module<B> {
            fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {match self {$(Self::$variant(module)=>module.visit_shards(visitor),)+}}
            fn visit_packed_shards<F:FnMut(&ShardedPackedParameter<B>)>(&self,visitor:&mut F) {match self {$(Self::$variant(module)=>module.visit_packed_shards(visitor),)+}}
        }
    };
}
visit_variants!(FullyShardedAdaptedProjection,Dense,LoRA);
visit_variants!(FullyShardedAwqProjection,Dense,LoRA,Awq,AwqLoRA);
visit_variants!(FullyShardedNf4Projection,Dense,LoRA,Nf4,Nf4LoRA);
visit_variants!(FullyShardedMixedProjection,Dense,LoRA,Awq,AwqLoRA,Nf4,Nf4LoRA);
visit_variants!(FullyShardedTransformerNorm,Layer,Rms);
visit_variants!(FullyShardedHeadProjection,Column,RowMajor);
impl<B:Backend> FullyShardedModule<B> for FullyShardedActivation<B> {
    fn visit_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {
        match self {Self::Stateless(_)=>{},Self::PRelu(module)=>module.visit_shards(visitor),Self::SwiGlu(module)=>module.visit_shards(visitor)}
    }
}
