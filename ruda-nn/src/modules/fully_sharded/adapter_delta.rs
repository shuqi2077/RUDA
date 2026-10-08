use super::*;
use alloc::collections::BTreeSet;
use ruda_model::module::ModuleDisplay;

/// Explicit actual low-rank leaf visitation, separate from whole-model storage visitation.
/// Base, normalization, activation and table leaves remain in the complete storage schema.
pub trait FullyShardedAdapterModule<B:Backend>:FullyShardedModule<B> {
    /// Visit only actual A/B leaves, retaining every repeated original shared role.
    fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F);
    /// Convert only actual A/B local storage, preserving canonical tied IDs/nodes,
    /// trainability and Param mappers. Original NF4 FP32 scales/codebook, AWQ scale
    /// storage, packed bytes/words, biases, norms and tables remain unchanged.
    /// Converted trainable values are new leaves; optimizer/pending state must match this storage.
    fn to_adapter_dtype(self,dtype:FloatDType) -> Result<Self,FullyShardedParameterError> {
        if !matches!(DType::from(dtype),DType::F16|DType::BF16|DType::F32) {return Err(FullyShardedParameterError::DType);}
        let _=FullyShardedStorageRecord::capture(&self)?;
        let mut selected=BTreeSet::new();self.visit_adapter_shards(&mut |parameter| {selected.insert(parameter.local.id);});
        Ok(self.to_dtype_selected(dtype,&selected.into_iter().collect::<Vec<_>>()))
    }
    /// Capture exact rank-local A/B updates and the complete original packed/floating schema.
    /// Caller base identity is explicit; no frozen-base hash or numerical content check is invented.
    fn adapter_delta_record(&self,base_id:&str) -> Result<FullyShardedStorageDeltaRecord<B>,FullyShardedParameterError> {
        let mut selected=BTreeSet::new();self.visit_adapter_shards(&mut |parameter| {selected.insert(parameter.local.id);});
        if selected.is_empty() {return Err(FullyShardedParameterError::Geometry("module has no actual low-rank adapter leaves"));}
        let mut omitted=false;
        self.visit_shards(&mut |parameter| {omitted|=parameter.local.val().is_require_grad() && !selected.contains(&parameter.local.id);});
        if omitted {return Err(FullyShardedParameterError::Geometry("trainable non-adapter state requires a complete storage checkpoint"));}
        let ids=selected.into_iter().collect::<Vec<_>>();FullyShardedStorageDeltaRecord::capture(self,base_id,&ids)
    }
    /// Restore only actual A/B updates into the matching original prepared base/topology.
    /// All native bytes/scales/norm/table values remain untouched; matching optimizer,
    /// pending gradients, RNG and data continuation use their existing separate records.
    fn load_adapter_delta(self,record:FullyShardedStorageDeltaRecord<B>,base_id:&str) -> Result<Self,FullyShardedParameterError> {
        let expected=self.adapter_delta_record(base_id)?;
        let mut saved=BTreeSet::new();
        for parameter in record.updates().floating() {saved.insert(parameter.id());}
        let mut actual=BTreeSet::new();
        for parameter in expected.updates().floating() {actual.insert(parameter.id());}
        if saved!=actual || !record.updates().packed().is_empty() {return Err(FullyShardedParameterError::Record);}
        record.restore_into(self,base_id)
    }
}

macro_rules! adapter_leaves {
    ($module:ident) => {
        impl<B:Backend> FullyShardedAdapterModule<B> for $module<B> {
            fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {
                self.adapter_a.visit_shards(visitor);self.adapter_b.visit_shards(visitor);
            }
        }
    };
}
adapter_leaves!(FullyShardedLoRALinear);
adapter_leaves!(FullyShardedAwqLoRALinear);
adapter_leaves!(FullyShardedNf4LoRALinear);

macro_rules! no_adapters {
    ($module:ident) => {
        impl<B:Backend> FullyShardedAdapterModule<B> for $module<B> {
            fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,_visitor:&mut F) {}
        }
    };
}
no_adapters!(FullyShardedLinear);
no_adapters!(FullyShardedAwqLinear);
no_adapters!(FullyShardedNf4Linear);

impl<B:Backend,M:FullyShardedAdapterModule<B>> FullyShardedAdapterModule<B> for Option<M> {
    fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {if let Some(module)=self {module.visit_adapter_shards(visitor);}}
}
impl<B:Backend,M:FullyShardedAdapterModule<B>> FullyShardedAdapterModule<B> for Vec<M> {
    fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {for module in self {module.visit_adapter_shards(visitor);}}
}
macro_rules! adapter_variants {
    ($module:ident,[$($variant:ident),+]) => {
        impl<B:Backend> FullyShardedAdapterModule<B> for $module<B> {
            fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {
                match self {$(Self::$variant(module)=>module.visit_adapter_shards(visitor),)+}
            }
        }
    };
}
adapter_variants!(FullyShardedAwqProjection,[Dense,LoRA,Awq,AwqLoRA]);
adapter_variants!(FullyShardedNf4Projection,[Dense,LoRA,Nf4,Nf4LoRA]);
adapter_variants!(FullyShardedMixedProjection,[Dense,LoRA,Awq,AwqLoRA,Nf4,Nf4LoRA]);

macro_rules! adapter_components {
    ($module:ident,[$($field:ident),+]) => {
        impl<B:Backend,P:FullyShardedAdapterModule<B>+ModuleDisplay> FullyShardedAdapterModule<B> for $module<B,P> {
            fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {$(self.$field.visit_adapter_shards(visitor);)+}
        }
    };
}
adapter_components!(FullyShardedAwqAttention,[query,key,value,output]);
adapter_components!(FullyShardedAwqFeedForward,[up,gate,down]);
adapter_components!(FullyShardedAwqTransformerBlock,[attention,feed_forward]);
adapter_components!(FullyShardedAwqTransformerStack,[blocks]);
adapter_components!(FullyShardedAwqTransformerHead,[projection]);
adapter_components!(FullyShardedAwqTransformerModel,[backbone,head]);
adapter_components!(FullyShardedProjectedCrossAttention,[attention]);
adapter_components!(FullyShardedProjectedDecoderLayer,[backbone,cross_attention]);
adapter_components!(FullyShardedProjectedDecoderStack,[layers]);
adapter_components!(FullyShardedProjectedEncoderDecoderModel,[encoder,decoder,head]);
adapter_components!(FullyShardedNativeMoeLayer,[router]);
adapter_components!(FullyShardedNativeMoeFeedForward,[routed,shared]);
adapter_components!(FullyShardedNativeMoeTransformerBlock,[attention,feed_forward]);
adapter_components!(FullyShardedNativeMoeTransformerStack,[layers]);
adapter_components!(FullyShardedNativeMoeTransformerModel,[backbone,head]);
impl<B:Backend,P:FullyShardedAdapterModule<B>+ModuleDisplay> FullyShardedAdapterModule<B> for FullyShardedNativeMoeTransformerLayer<B,P> {
    fn visit_adapter_shards<F:FnMut(&ShardedParameter<B>)>(&self,visitor:&mut F) {match self {Self::Dense(block)=>block.visit_adapter_shards(visitor),Self::Routed(block)=>block.visit_adapter_shards(visitor)}}
}
