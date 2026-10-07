use ruda_model::{module::Module,tensor::{Tensor,Int,FloatDType,backend::Backend}};
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy};
use crate::{Dropout,transformer::DenseTransformerNorm,cache::{ProjectedKvCache,TransformerKvCache},
    attention::PackedSequenceLayout};
use super::{BroadcastTensorCollective,VocabParallelLossLayout,TensorParallelTransformerEmbeddings,
    TensorParallelTransformerStack,TensorParallelAdaptedStackLayer,TensorParallelAdaptedTransformerStack,TensorParallelOutputHead};

mod input;
pub use input::TensorParallelTransformerInput;
mod hidden;
mod training;
mod inference;
mod adapter_record;
mod adapter_aliases;
pub use adapter_record::TensorParallelModelAdapterRecord;
mod packed;
mod cached;
mod partition;
pub use partition::TensorParallelTransformerModelPartition;
#[cfg(feature="std")]
mod batches;
mod seq2seq;
pub use seq2seq::{TensorParallelEncoderDecoderModel,TensorParallelEncoderDecoderAdapterRecord,TensorParallelPairedVocabularies,
    TensorParallelEncoderDecoderModelPartition};

/// Complete native single-stream Transformer assembled from actual local model partitions.
/// Input tables, exact selected/unselected layer order, optional final normalization and
/// output projection remain ordinary recorded modules. Transports/layouts are call arguments.
#[derive(Module,Debug)]
pub struct TensorParallelTransformerModel<B:Backend> {
    /// Original token/position/type lookup, input normalization and dropout.
    pub embeddings:TensorParallelTransformerEmbeddings<B>,
    /// Actual ordered dense/adapter layer choices, retaining all parameter identities.
    pub backbone:TensorParallelAdaptedTransformerStack<B>,
    /// Explicit backbone-final normalization, separate from the head's own normalization.
    pub final_normalization:Option<DenseTransformerNorm<B>>,
    /// Actual row-major or column-major native local output projection.
    pub head:TensorParallelOutputHead<B>,
}

impl<B:Backend> TensorParallelTransformerModel<B> {
    /// Connect actual prepared modules without selecting architectures, tying/fixing parameters,
    /// initializing weights, changing trainable flags or installing transport endpoints.
    pub fn from_parts(embeddings:TensorParallelTransformerEmbeddings<B>,backbone:TensorParallelAdaptedTransformerStack<B>,
        final_normalization:Option<DenseTransformerNorm<B>>,head:TensorParallelOutputHead<B>) -> Self {
        let width = embeddings.token.local.weight.val().dims()[1];
        assert_eq!(head.hidden_width(),width,"parallel model input/output hidden widths differ");
        if let Some(norm) = &final_normalization {assert_eq!(norm.width(),width,"parallel model final normalization width differs");}
        for layer in &backbone.layers {
            let (attention,feed_forward) = match layer {
                TensorParallelAdaptedStackLayer::Dense(block)=>(&block.attention_norm,&block.feed_forward_norm),
                TensorParallelAdaptedStackLayer::Adapted(block)=>(&block.attention_norm,&block.feed_forward_norm),
            };
            assert_eq!(attention.width(),width,"parallel model layer attention width differs");
            assert_eq!(feed_forward.width(),width,"parallel model layer feed-forward width differs");
        }
        Self {embeddings,backbone,final_normalization,head}
    }

    /// Move an existing all-dense partitioned backbone into the same complete model container.
    /// No adapters or new parameter copies are created.
    pub fn from_dense_parts(embeddings:TensorParallelTransformerEmbeddings<B>,backbone:TensorParallelTransformerStack<B>,
        final_normalization:Option<DenseTransformerNorm<B>>,head:TensorParallelOutputHead<B>) -> Self {
        let layers = backbone.blocks.into_iter().map(TensorParallelAdaptedStackLayer::Dense).collect();
        Self::from_parts(embeddings,TensorParallelAdaptedTransformerStack::new(layers),final_normalization,head)
    }

    /// Cache metadata for the exact original layer count, without allocating global heads.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {self.backbone.new_kv_cache(initial_capacity)}

    fn finish<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {
        assert!(D > 0,"parallel model hidden tensor requires a feature axis");
        assert_eq!(hidden.dims()[D-1],self.head.hidden_width(),"parallel model backbone changed its hidden width");
        if let Some(norm) = &self.final_normalization {norm.forward(hidden)} else {hidden}
    }
}
