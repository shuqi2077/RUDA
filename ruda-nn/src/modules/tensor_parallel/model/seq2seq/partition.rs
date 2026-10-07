use super::*;
use alloc::{collections::BTreeMap,vec::Vec};
use core::ops::Range;
use crate::{Embedding,activation::Activation,transformer::{TransformerEmbeddings,AdaptedTransformerStack,AdaptedEncoderDecoderStack,
    DenseTransformerStack,DenseEncoderDecoderStack,AdaptedStackLayer,AdaptedEncoderDecoderLayer}};
use ruda_model::module::{ModuleVisitor,Param,ParamId};
use super::super::partition::head_weight;
use super::super::super::{TensorParallelTransformerPartition,TensorParallelEncoderDecoderPartition,VocabParallelEmbedding};

/// Exact native source/self/cross/FFN and vocabulary placement for one complete paired-model partition.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct TensorParallelEncoderDecoderModelPartition {
    /// Original-order explicit source encoder head/FFN partitions.
    pub encoder:Vec<TensorParallelTransformerPartition>,
    /// Original-order independent target self/cross/FFN partitions.
    pub decoder:Vec<TensorParallelEncoderDecoderPartition>,
    /// Actual source, target-input and output vocabulary rank/layout assignments.
    pub vocabularies:TensorParallelPairedVocabularies,
    /// Explicit native source lookup padding-derivative policy.
    pub source_padding_index:Option<usize>,
    /// Independent explicit native target lookup padding-derivative policy.
    pub target_padding_index:Option<usize>,
}

fn same_table<B:Backend>(source:&Embedding<B>,target:&Embedding<B>) {
    let source_value = source.weight.val();let target_value = target.weight.val();
    assert!(source.weight.id == target.weight.id && source_value.dims() == target_value.dims() && source_value.dtype() == target_value.dtype()
        && source_value.device() == target_value.device() && source_value.is_require_grad() == target_value.is_require_grad(),
        "full paired shared token weights must identify the same actual parameter contract");
}

fn tied_placement(left:&VocabParallelLossLayout,left_rank:usize,right:&VocabParallelLossLayout,right_rank:usize) {
    assert!(left == right && left_rank == right_rank,"shared native paired vocabulary weights require the same actual owning rank/layout");
}

struct Sharing {counts:BTreeMap<ParamId,usize>,vocabulary_weights:[ParamId;3]}
impl<B:Backend> ModuleVisitor<B> for Sharing {
    fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {
        let count = self.counts.entry(param.id).or_insert(0);*count += 1;
        let limit = self.vocabulary_weights.iter().filter(|id|**id == param.id).count().max(1);
        assert!(*count <= limit,"other shared full paired model roles require explicitly prepared tie-aware local loading");
    }
}

fn table_and_head<B:Backend>(tables:TransformerEmbeddings<B>,head:TensorParallelOutputHead<B>,layout:&VocabParallelLossLayout,rank:usize,
    padding:Option<usize>) -> (TensorParallelTransformerEmbeddings<B>,TensorParallelOutputHead<B>) {
    match head {
        TensorParallelOutputHead::Vocabulary(head)=>{
            let (tables,head) = TensorParallelTransformerEmbeddings::from_full_tied_head(tables,head,layout,rank,padding);
            (tables,TensorParallelOutputHead::Vocabulary(head))
        },
        TensorParallelOutputHead::AdaptedVocabulary(head)=>{
            let (tables,head) = TensorParallelTransformerEmbeddings::from_full_tied_adapted_head(tables,head,layout,rank,padding);
            (tables,TensorParallelOutputHead::AdaptedVocabulary(head))
        },
        _=>panic!("shared full column-major paired output roles require explicitly prepared compatible local weights"),
    }
}

fn shared_tables<B:Backend>(tables:TransformerEmbeddings<B>,token:&VocabParallelEmbedding<B>,layout:&VocabParallelLossLayout,rank:usize,
    padding:Option<usize>) -> TensorParallelTransformerEmbeddings<B> {
    let weight = tables.token.weight.map(|_|token.local.weight.val());
    TensorParallelTransformerEmbeddings::from_tables(VocabParallelEmbedding::from_shard_with_layout(Embedding {weight},layout,rank,padding),
        tables.position,tables.token_type,tables.normalization,tables.dropout)
}

impl<B:Backend> TensorParallelEncoderDecoderModel<B> {
    /// Partition an actual fully loaded floating source/target model using every declared native range.
    /// Original dense/adapter selections, norms, dtypes, flags and parameters are not reinitialized.
    /// Shared source/target/output row-major storage is sliced once and retains one local leaf, including
    /// independent lookup padding policies. Other shared roles and packed partial weights use explicit local loading.
    /// Actual original FFN activations are supplied to separate source/target partition callbacks.
    pub fn from_full<E,F>(source_embeddings:TransformerEmbeddings<B>,encoder:AdaptedTransformerStack<B>,encoder_normalization:Option<DenseTransformerNorm<B>>,
        target_embeddings:TransformerEmbeddings<B>,decoder:AdaptedEncoderDecoderStack<B>,decoder_normalization:Option<DenseTransformerNorm<B>>,
        head:TensorParallelOutputHead<B>,partition:&TensorParallelEncoderDecoderModelPartition,encoder_activation:E,decoder_activation:F) -> Self
        where E:FnMut(usize,Activation<B>,Range<usize>)->Activation<B>,F:FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        assert_eq!(partition.encoder.len(),encoder.layers.len(),"paired encoder plans must cover every actual original layer");
        assert_eq!(partition.decoder.len(),decoder.layers.len(),"paired decoder plans must cover every actual original layer");
        let vocab = &partition.vocabularies;vocab.source.interval(vocab.source_rank);vocab.target.interval(vocab.target_rank);vocab.output.interval(vocab.output_rank);
        if let Some(padding) = partition.source_padding_index {assert!(padding < vocab.source.vocabulary_size(),"source padding row is outside real source classes");}
        if let Some(padding) = partition.target_padding_index {assert!(padding < vocab.target.vocabulary_size(),"target padding row is outside real target classes");}
        let source_id = source_embeddings.token.weight.id;let target_id = target_embeddings.token.weight.id;let output_id = head_weight(&head);
        let shared_inputs = source_id == target_id;let shared_source_output = source_id == output_id;let shared_target_output = target_id == output_id;
        if shared_inputs {same_table(&source_embeddings.token,&target_embeddings.token);tied_placement(&vocab.source,vocab.source_rank,&vocab.target,vocab.target_rank);}
        if shared_source_output {tied_placement(&vocab.source,vocab.source_rank,&vocab.output,vocab.output_rank);}
        if shared_target_output {tied_placement(&vocab.target,vocab.target_rank,&vocab.output,vocab.output_rank);}
        let mut sharing = Sharing {counts:BTreeMap::new(),vocabulary_weights:[source_id,target_id,output_id]};
        source_embeddings.visit(&mut sharing);encoder.visit(&mut sharing);encoder_normalization.visit(&mut sharing);
        target_embeddings.visit(&mut sharing);decoder.visit(&mut sharing);decoder_normalization.visit(&mut sharing);head.visit(&mut sharing);
        let (source_embeddings,target_embeddings,head) = if shared_inputs {
            let (source,head) = if shared_source_output {table_and_head(source_embeddings,head,&vocab.source,vocab.source_rank,partition.source_padding_index)}
                else {(TensorParallelTransformerEmbeddings::from_full(source_embeddings,&vocab.source,vocab.source_rank,partition.source_padding_index),head.from_full(&vocab.output,vocab.output_rank))};
            let target = shared_tables(target_embeddings,&source.token,&vocab.target,vocab.target_rank,partition.target_padding_index);
            (source,target,head)
        } else if shared_source_output {
            let (source,head) = table_and_head(source_embeddings,head,&vocab.source,vocab.source_rank,partition.source_padding_index);
            (source,TensorParallelTransformerEmbeddings::from_full(target_embeddings,&vocab.target,vocab.target_rank,partition.target_padding_index),head)
        } else if shared_target_output {
            let (target,head) = table_and_head(target_embeddings,head,&vocab.target,vocab.target_rank,partition.target_padding_index);
            (TensorParallelTransformerEmbeddings::from_full(source_embeddings,&vocab.source,vocab.source_rank,partition.source_padding_index),target,head)
        } else {
            (TensorParallelTransformerEmbeddings::from_full(source_embeddings,&vocab.source,vocab.source_rank,partition.source_padding_index),
                TensorParallelTransformerEmbeddings::from_full(target_embeddings,&vocab.target,vocab.target_rank,partition.target_padding_index),head.from_full(&vocab.output,vocab.output_rank))
        };
        Self::from_parts(source_embeddings,TensorParallelAdaptedTransformerStack::from_full_stack(encoder,&partition.encoder,encoder_activation),encoder_normalization,
            target_embeddings,TensorParallelAdaptedEncoderDecoderStack::from_full_stack(decoder,&partition.decoder,decoder_activation),decoder_normalization,head)
    }

    /// Partition actual all-dense source/target backbones without selecting or initializing any adapters.
    pub fn from_full_dense<E,F>(source_embeddings:TransformerEmbeddings<B>,encoder:DenseTransformerStack<B>,encoder_normalization:Option<DenseTransformerNorm<B>>,
        target_embeddings:TransformerEmbeddings<B>,decoder:DenseEncoderDecoderStack<B>,decoder_normalization:Option<DenseTransformerNorm<B>>,
        head:TensorParallelOutputHead<B>,partition:&TensorParallelEncoderDecoderModelPartition,encoder_activation:E,decoder_activation:F) -> Self
        where E:FnMut(usize,Activation<B>,Range<usize>)->Activation<B>,F:FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        let encoder = AdaptedTransformerStack::new(encoder.blocks.into_iter().map(AdaptedStackLayer::Dense).collect());
        let decoder = AdaptedEncoderDecoderStack::new(decoder.layers.into_iter().map(AdaptedEncoderDecoderLayer::dense).collect());
        Self::from_full(source_embeddings,encoder,encoder_normalization,target_embeddings,decoder,decoder_normalization,head,partition,encoder_activation,decoder_activation)
    }
}
