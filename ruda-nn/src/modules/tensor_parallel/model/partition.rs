use super::*;
use alloc::{collections::BTreeMap,vec::Vec};
use core::ops::Range;
use crate::{activation::Activation,transformer::{TransformerEmbeddings,AdaptedTransformerStack,DenseTransformerStack}};
use ruda_model::module::{ModuleVisitor,Param,ParamId};
use super::super::{TensorParallelTransformerPartition,VocabParallelTransformerHead,VocabParallelAdaptedTransformerHead,
    TensorParallelTransformerHead,TensorParallelAdaptedTransformerHead};

/// Actual complete-model local placement, with no inferred equal shard sizes or model-family geometry.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct TensorParallelTransformerModelPartition {
    /// Original-order attention/FFN partition for every actual backbone layer.
    pub layers:Vec<TensorParallelTransformerPartition>,
    /// Actual full token-table storage and real input vocabulary.
    pub input_vocabulary:VocabParallelLossLayout,
    /// Owning rank in the explicit input vocabulary group.
    pub input_rank:usize,
    /// Actual full output storage and logical class vocabulary, independent for untied heads.
    pub output_vocabulary:VocabParallelLossLayout,
    /// Owning rank in the explicit output vocabulary group.
    pub output_rank:usize,
    /// Actual global input padding row, if it has the native frozen-lookup derivative policy.
    pub padding_index:Option<usize>,
}

fn head_weight<B:Backend>(head:&TensorParallelOutputHead<B>) -> ParamId {
    match head {
        TensorParallelOutputHead::Linear(head)=>head.local.projection.weight.id,
        TensorParallelOutputHead::AdaptedLinear(head)=>head.local.projection.base.weight.id,
        TensorParallelOutputHead::Vocabulary(head)=>head.projection.weight.id,
        TensorParallelOutputHead::AdaptedVocabulary(head)=>head.projection.base.weight.id,
    }
}

struct Sharing {counts:BTreeMap<ParamId,usize>,token:ParamId,tied:bool}
impl<B:Backend> ModuleVisitor<B> for Sharing {
    fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {
        let count = self.counts.entry(param.id).or_insert(0);*count += 1;
        let limit = if self.tied && param.id == self.token {2} else {1};
        assert!(*count <= limit,"other shared full model roles require explicit tie-aware local loading");
    }
}

impl<B:Backend> TensorParallelOutputHead<B> {
    /// Partition this actual full native head, retaining its original storage kind and adapters.
    /// This independent-head entry point does not jointly slice a separate embedding parameter.
    pub fn from_full(self,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        match self {
            Self::Linear(head)=>Self::Linear(TensorParallelTransformerHead::from_full(head.local,layout,rank)),
            Self::AdaptedLinear(head)=>Self::AdaptedLinear(TensorParallelAdaptedTransformerHead::from_full(head.local,layout,rank)),
            Self::Vocabulary(head)=>Self::Vocabulary(VocabParallelTransformerHead::from_full(head,layout,rank)),
            Self::AdaptedVocabulary(head)=>Self::AdaptedVocabulary(VocabParallelAdaptedTransformerHead::from_full(head,layout,rank)),
        }
    }
}

impl<B:Backend> TensorParallelTransformerModel<B> {
    /// Partition an actual loaded full floating native model into one declared complete local model.
    /// Dense/adapter selections, all original IDs/flags/dtypes and norm/dropout order are retained.
    /// An actual row-major input/head tie is sliced once into the same local leaf. Other shared roles
    /// or packed partial weights use from_parts with explicitly tie-aware native local loading.
    /// The activation callback partitions each original FFN activation; optimizer state is separate.
    pub fn from_full<F>(embeddings:TransformerEmbeddings<B>,backbone:AdaptedTransformerStack<B>,
        final_normalization:Option<DenseTransformerNorm<B>>,head:TensorParallelOutputHead<B>,
        partition:&TensorParallelTransformerModelPartition,activation:F) -> Self
        where F:FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        assert_eq!(partition.layers.len(),backbone.layers.len(),"full parallel model plans must cover every actual layer exactly once");
        partition.input_vocabulary.interval(partition.input_rank);partition.output_vocabulary.interval(partition.output_rank);
        if let Some(padding) = partition.padding_index {assert!(padding < partition.input_vocabulary.vocabulary_size(),"full model padding row is outside real input classes");}
        let token = embeddings.token.weight.id;let tied = token == head_weight(&head);
        if tied {
            assert!(matches!(&head,TensorParallelOutputHead::Vocabulary(_)|TensorParallelOutputHead::AdaptedVocabulary(_)),
                "shared full column-major output roles require explicitly prepared compatible local weights");
            assert!(partition.input_vocabulary == partition.output_vocabulary && partition.input_rank == partition.output_rank,
                "tied native input/output storage must use the same actual rank/layout");
        }
        let mut sharing = Sharing {counts:BTreeMap::new(),token,tied};
        embeddings.visit(&mut sharing);backbone.visit(&mut sharing);final_normalization.visit(&mut sharing);head.visit(&mut sharing);
        let (embeddings,head) = match head {
            TensorParallelOutputHead::Vocabulary(head) if tied=>{
                let (embeddings,head) = TensorParallelTransformerEmbeddings::from_full_tied_head(embeddings,head,
                    &partition.input_vocabulary,partition.input_rank,partition.padding_index);
                (embeddings,TensorParallelOutputHead::Vocabulary(head))
            },
            TensorParallelOutputHead::AdaptedVocabulary(head) if tied=>{
                let (embeddings,head) = TensorParallelTransformerEmbeddings::from_full_tied_adapted_head(embeddings,head,
                    &partition.input_vocabulary,partition.input_rank,partition.padding_index);
                (embeddings,TensorParallelOutputHead::AdaptedVocabulary(head))
            },
            head=>(TensorParallelTransformerEmbeddings::from_full(embeddings,&partition.input_vocabulary,partition.input_rank,partition.padding_index),
                head.from_full(&partition.output_vocabulary,partition.output_rank)),
        };
        Self::from_parts(embeddings,TensorParallelAdaptedTransformerStack::from_full_stack(backbone,&partition.layers,activation),final_normalization,head)
    }

    /// Complete all-dense full-model partitioning without initializing or selecting any adapters.
    pub fn from_full_dense<F>(embeddings:TransformerEmbeddings<B>,backbone:DenseTransformerStack<B>,
        final_normalization:Option<DenseTransformerNorm<B>>,head:TensorParallelOutputHead<B>,
        partition:&TensorParallelTransformerModelPartition,activation:F) -> Self
        where F:FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        let layers = backbone.blocks.into_iter().map(crate::transformer::AdaptedStackLayer::Dense).collect();
        Self::from_full(embeddings,AdaptedTransformerStack::new(layers),final_normalization,head,partition,activation)
    }
}
