use super::*;
use alloc::{format,string::String,vec::Vec};
use super::super::{adapter_record::frozen,adapter_aliases::{self,AdapterTree}};
use crate::transformer::{StackAdapterRecord,EncoderDecoderAdapterRecord,TransformerHeadAdapterRecord};
use super::super::super::VocabParallelHeadAdapterRecord;
use ruda_model::record::{PrecisionSettings,Record,Recorder,RecorderError};

/// Explicit actual source, target-input and output vocabulary placements for one local paired model.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct TensorParallelPairedVocabularies {
    /// Source token-table storage, real classes and original rank order.
    pub source:VocabParallelLossLayout,
    /// Owning source vocabulary rank.
    pub source_rank:usize,
    /// Independent target-input token-table storage and real classes.
    pub target:VocabParallelLossLayout,
    /// Owning target-input vocabulary rank.
    pub target_rank:usize,
    /// Actual target output classes/storage, independent for an untied output head.
    pub output:VocabParallelLossLayout,
    /// Owning output vocabulary rank.
    pub output_rank:usize,
}

type Placement = (Vec<usize>,usize,usize);
fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid paired parallel model adapter record: {reason}"))}

fn placements<B:Backend>(model:&TensorParallelEncoderDecoderModel<B>,vocabularies:&TensorParallelPairedVocabularies)
    -> Result<[Placement;3],RecorderError> {
    for (table,layout,rank) in [(&model.source_embeddings,&vocabularies.source,vocabularies.source_rank),
        (&model.target_embeddings,&vocabularies.target,vocabularies.target_rank)] {
        if rank >= layout.world_size() || table.token.vocabulary_size != layout.vocabulary_size()
            || table.token.vocabulary_start != layout.interval(rank).start || table.token.local.weight.val().dims()[0] != layout.interval(rank).len() {
            return Err(invalid("source/target token storage or owning vocabulary rank/layout differs"));
        }
    }
    if vocabularies.output_rank >= vocabularies.output.world_size()
        || model.head.local_classes() != vocabularies.output.interval(vocabularies.output_rank).len() {
        return Err(invalid("actual output head vocabulary rank/storage differs"));
    }
    let metadata = |layout:&VocabParallelLossLayout,rank|((0..layout.world_size()).map(|owner|layout.interval(owner).len()).collect(),layout.vocabulary_size(),rank);
    Ok([metadata(&vocabularies.source,vocabularies.source_rank),metadata(&vocabularies.target,vocabularies.target_rank),
        metadata(&vocabularies.output,vocabularies.output_rank)])
}

/// Complete native paired encoder/decoder/head A/B-only continuation record.
/// Exact vocabulary placements and shared adapter topology accompany original native projection schemas.
/// Frozen embeddings, source memory, base weights and final norms are not retained in this payload.
pub struct TensorParallelEncoderDecoderAdapterRecord<B:Backend> {
    version:u32,
    base_id:String,
    partition_id:String,
    vocabularies:[Placement;3],
    aliases:Vec<usize>,
    encoder:Option<StackAdapterRecord<B>>,
    decoder:Option<EncoderDecoderAdapterRecord<B>>,
    linear_head:Option<TransformerHeadAdapterRecord<B>>,
    vocabulary_head:Option<VocabParallelHeadAdapterRecord<B>>,
}

impl<B:Backend> Record<B> for TensorParallelEncoderDecoderAdapterRecord<B> {
    type Item<S:PrecisionSettings> = (u32,String,String,[Placement;3],Vec<usize>,
        Option<<StackAdapterRecord<B> as Record<B>>::Item<S>>,Option<<EncoderDecoderAdapterRecord<B> as Record<B>>::Item<S>>,
        Option<<TransformerHeadAdapterRecord<B> as Record<B>>::Item<S>>,Option<<VocabParallelHeadAdapterRecord<B> as Record<B>>::Item<S>>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.base_id,self.partition_id,self.vocabularies,self.aliases,self.encoder.map(|record|record.into_item::<S>()),
            self.decoder.map(|record|record.into_item::<S>()),self.linear_head.map(|record|record.into_item::<S>()),self.vocabulary_head.map(|record|record.into_item::<S>()))
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,base_id:item.1,partition_id:item.2,vocabularies:item.3,aliases:item.4,
            encoder:item.5.map(|record|StackAdapterRecord::<B>::from_item::<S>(record,device)),
            decoder:item.6.map(|record|EncoderDecoderAdapterRecord::<B>::from_item::<S>(record,device)),
            linear_head:item.7.map(|record|TransformerHeadAdapterRecord::<B>::from_item::<S>(record,device)),
            vocabulary_head:item.8.map(|record|VocabParallelHeadAdapterRecord::<B>::from_item::<S>(record,device))}
    }
}

impl<B:Backend> TensorParallelEncoderDecoderAdapterRecord<B> {
    /// Capture every actual selected native adapter, without silently freezing omitted model state.
    /// The partition identity binds actual encoder/self/cross/FFN placements and forward configuration.
    pub fn capture(model:&TensorParallelEncoderDecoderModel<B>,base_id:&str,partition_id:&str,vocabularies:&TensorParallelPairedVocabularies)
        -> Result<Self,RecorderError> {
        if base_id.is_empty() || partition_id.is_empty() {return Err(invalid("complete original base and partition identities are required"));}
        let vocabularies_metadata = placements(model,vocabularies)?;
        frozen(&model.source_embeddings)?;frozen(&model.target_embeddings)?;frozen(&model.encoder_normalization)?;frozen(&model.decoder_normalization)?;
        let encoder = if model.encoder.adapters().is_empty() {frozen(&model.encoder)?;None} else {Some(model.encoder.adapter_record(base_id)?)};
        let decoder = if model.decoder.adapters().is_empty() {frozen(&model.decoder)?;None} else {Some(model.decoder.adapter_record(base_id)?)};
        let (linear_head,vocabulary_head) = match &model.head {
            TensorParallelOutputHead::AdaptedLinear(head)=>(Some(head.adapter_record(base_id)?),None),
            TensorParallelOutputHead::AdaptedVocabulary(head)=>(None,Some(head.adapter_record(&vocabularies.output,vocabularies.output_rank,base_id)?)),
            _=>{frozen(&model.head)?;(None,None)},
        };
        let aliases = adapter_aliases::capture(model)?;if aliases.is_empty() {return Err(invalid("paired model has no actual adapters"));}
        Ok(Self {version:1,base_id:base_id.into(),partition_id:partition_id.into(),vocabularies:vocabularies_metadata,
            aliases,encoder,decoder,linear_head,vocabulary_head})
    }

    /// Validate all original target/schema/placement/sharing contracts before replacing any A/B values.
    pub fn validate_for(&self,model:&TensorParallelEncoderDecoderModel<B>,base_id:&str,partition_id:&str,vocabularies:&TensorParallelPairedVocabularies)
        -> Result<(),RecorderError> {
        if self.version != 1 || base_id.is_empty() || partition_id.is_empty() || self.base_id != base_id || self.partition_id != partition_id {
            return Err(invalid("format or complete original base/partition identity differs"));
        }
        if self.vocabularies != placements(model,vocabularies)? {return Err(invalid("physical source/target/output vocabulary placements differ"));}
        frozen(&model.source_embeddings)?;frozen(&model.target_embeddings)?;frozen(&model.encoder_normalization)?;frozen(&model.decoder_normalization)?;
        match (&self.encoder,model.encoder.adapters().is_empty()) {
            (Some(record),false)=>record.validate_for(&model.encoder.clone().into_local_stack(),base_id)?,
            (None,true)=>frozen(&model.encoder)?,_=>return Err(invalid("actual encoder adapter selection differs")),
        }
        match (&self.decoder,model.decoder.adapters().is_empty()) {
            (Some(record),false)=>record.validate_for(&model.decoder.clone().into_local_stack(),base_id)?,
            (None,true)=>frozen(&model.decoder)?,_=>return Err(invalid("actual self/cross/FFN decoder adapter selection differs")),
        }
        match (&model.head,&self.linear_head,&self.vocabulary_head) {
            (TensorParallelOutputHead::AdaptedLinear(head),Some(record),None)=>record.validate_for(&head.local,base_id)?,
            (TensorParallelOutputHead::AdaptedVocabulary(head),None,Some(record))=>record.validate_for(head,&vocabularies.output,vocabularies.output_rank,base_id)?,
            (TensorParallelOutputHead::Linear(_)|TensorParallelOutputHead::Vocabulary(_),None,None)=>frozen(&model.head)?,
            _=>return Err(invalid("actual native output head kind/adapter selection differs")),
        }
        if self.aliases.is_empty() || self.aliases != adapter_aliases::capture(model)? {return Err(invalid("original shared adapter topology differs"));}
        Ok(())
    }

    /// Save actual native A/B payloads only; recorder precision controls stored floating values.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load original native adapter state on the explicit device, without rebuilding a frozen base.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}

    /// Restore all original A/B IDs/dtypes/flags and one leaf for each actual shared adapter.
    /// Frozen source/target/head base ties remain untouched; optimizer/pending-gradient state is separate.
    pub fn restore_into(self,mut model:TensorParallelEncoderDecoderModel<B>,base_id:&str,partition_id:&str,vocabularies:&TensorParallelPairedVocabularies)
        -> Result<TensorParallelEncoderDecoderModel<B>,RecorderError> {
        self.validate_for(&model,base_id,partition_id,vocabularies)?;
        if let Some(record) = self.encoder {model.encoder = model.encoder.restore_adapter_record(record,base_id)?;}
        if let Some(record) = self.decoder {model.decoder = model.decoder.restore_adapter_record(record,base_id)?;}
        model.head = match (model.head,self.linear_head,self.vocabulary_head) {
            (TensorParallelOutputHead::AdaptedLinear(head),Some(record),None)=>TensorParallelOutputHead::AdaptedLinear(head.restore_adapter(record,base_id)?),
            (TensorParallelOutputHead::AdaptedVocabulary(head),None,Some(record))=>TensorParallelOutputHead::AdaptedVocabulary(record.restore_into(head,&vocabularies.output,vocabularies.output_rank,base_id)?),
            (head,None,None)=>head,_=>return Err(invalid("unconsumed native paired head adapter state")),
        };
        adapter_aliases::rejoin(model,&self.aliases)
    }
}

impl<B:Backend> TensorParallelEncoderDecoderModel<B> {
    /// Capture this actual complete local paired model's native adapter-only continuation state.
    pub fn adapter_record(&self,base_id:&str,partition_id:&str,vocabularies:&TensorParallelPairedVocabularies)
        -> Result<TensorParallelEncoderDecoderAdapterRecord<B>,RecorderError> {
        TensorParallelEncoderDecoderAdapterRecord::capture(self,base_id,partition_id,vocabularies)
    }
    /// Restore actual selected encoder/self/cross/FFN/head adapters without replacing frozen components.
    pub fn restore_adapter_record(self,record:TensorParallelEncoderDecoderAdapterRecord<B>,base_id:&str,partition_id:&str,vocabularies:&TensorParallelPairedVocabularies)
        -> Result<Self,RecorderError> {record.restore_into(self,base_id,partition_id,vocabularies)}
}
