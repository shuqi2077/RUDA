use super::*;
use super::super::{TensorParallelAdaptedEncoderDecoderStack,TensorParallelAdaptedEncoderDecoderLayer,TensorParallelAdaptedDecoderCrossAttention};
use crate::{cache::EncoderDecoderKvCache,transformer::AdaptedProjection};

mod training;
mod inference;

/// Complete native paired model with independently declared source/target tables and vocabularies.
/// Every encoder and decoder stage is the actual locally loaded dense/adapted module.
#[derive(Module,Debug)]
pub struct TensorParallelEncoderDecoderModel<B:Backend> {
    /// Original source token/learned-position/type tables and combined-input transform.
    pub source_embeddings:TensorParallelTransformerEmbeddings<B>,
    /// Actual ordered source self-attention/FFN layers.
    pub encoder:TensorParallelAdaptedTransformerStack<B>,
    /// Explicit encoder-final normalization before any cross-memory projection.
    pub encoder_normalization:Option<DenseTransformerNorm<B>>,
    /// Original target token/learned-position/type tables, including any explicit base ties.
    pub target_embeddings:TensorParallelTransformerEmbeddings<B>,
    /// Actual self-attention -> source cross-attention -> FFN decoder layer order.
    pub decoder:TensorParallelAdaptedEncoderDecoderStack<B>,
    /// Explicit decoder-final normalization, independent of native head normalization.
    pub decoder_normalization:Option<DenseTransformerNorm<B>>,
    /// Original target output head, retaining its actual row/column storage and adapters.
    pub head:TensorParallelOutputHead<B>,
}

fn layer_width<B:Backend>(layer:&TensorParallelAdaptedStackLayer<B>) -> usize {
    let (attention,feed) = match layer {
        TensorParallelAdaptedStackLayer::Dense(block)=>(&block.attention_norm,&block.feed_forward_norm),
        TensorParallelAdaptedStackLayer::Adapted(block)=>(&block.attention_norm,&block.feed_forward_norm),
    };
    assert_eq!(attention.width(),feed.width(),"paired model self-attention/FFN widths differ");attention.width()
}

fn projection_width<B:Backend>(projection:&AdaptedProjection<B>) -> usize {
    match projection {AdaptedProjection::Dense(layer)=>layer.weight.val().dims()[0],AdaptedProjection::LoRA(layer)=>layer.base.weight.val().dims()[0]}
}

impl<B:Backend> TensorParallelEncoderDecoderModel<B> {
    /// Assemble actual prepared local modules without initializing/tieing weights or choosing an architecture.
    /// Source and decoder hidden widths may differ exactly as their original cross projections declare.
    pub fn from_parts(source_embeddings:TensorParallelTransformerEmbeddings<B>,encoder:TensorParallelAdaptedTransformerStack<B>,
        encoder_normalization:Option<DenseTransformerNorm<B>>,target_embeddings:TensorParallelTransformerEmbeddings<B>,
        decoder:TensorParallelAdaptedEncoderDecoderStack<B>,decoder_normalization:Option<DenseTransformerNorm<B>>,head:TensorParallelOutputHead<B>) -> Self {
        let source = source_embeddings.token.local.weight.val().dims()[1];let target = target_embeddings.token.local.weight.val().dims()[1];
        assert_eq!(head.hidden_width(),target,"paired model target/output head widths differ");
        if let Some(norm) = &encoder_normalization {assert_eq!(norm.width(),source,"paired model encoder final norm width differs");}
        if let Some(norm) = &decoder_normalization {assert_eq!(norm.width(),target,"paired model decoder final norm width differs");}
        for layer in &encoder.layers {assert_eq!(layer_width(layer),source,"paired model encoder layer width differs");}
        for layer in &decoder.layers {
            assert_eq!(layer_width(&layer.backbone),target,"paired model decoder layer width differs");
            let (query,memory) = match &layer.cross_attention {
                TensorParallelAdaptedDecoderCrossAttention::Dense(block)=>(block.attention.local.query.weight.val().dims()[0],block.attention.local.key.weight.val().dims()[0]),
                TensorParallelAdaptedDecoderCrossAttention::Adapted(block)=>(projection_width(&block.attention.local.query),projection_width(&block.attention.local.key)),
            };
            assert_eq!((query,memory),(target,source),"paired model actual cross query/source widths differ");
        }
        Self {source_embeddings,encoder,encoder_normalization,target_embeddings,decoder,decoder_normalization,head}
    }

    fn source_finish<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {
        assert!(D > 0 && hidden.dims()[D-1] == self.source_embeddings.token.local.weight.val().dims()[1],"paired encoder changed source hidden width");
        if let Some(norm) = &self.encoder_normalization {norm.forward(hidden)} else {hidden}
    }
    fn target_finish<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {
        assert!(D > 0 && hidden.dims()[D-1] == self.head.hidden_width(),"paired decoder changed target hidden width");
        if let Some(norm) = &self.decoder_normalization {norm.forward(hidden)} else {hidden}
    }
    fn dense_memory(&self,input:&Tensor<B,2,Int>,memory:&Tensor<B,3>) {
        assert_eq!(memory.dims()[0],input.dims()[0],"paired source/target actual batch rows differ");
        assert_eq!(memory.dims()[2],self.source_embeddings.token.local.weight.val().dims()[1],"paired actual source memory width differs");
        assert_eq!(memory.device(),input.device(),"paired actual source/target devices differ");
    }
    fn packed_memory(&self,input:&Tensor<B,1,Int>,memory:&Tensor<B,2>,source:&PackedSequenceLayout,target:&PackedSequenceLayout) {
        assert_eq!(source.documents(),target.documents(),"paired packed source/target document counts differ");
        assert_eq!(memory.dims(),[source.tokens(),self.source_embeddings.token.local.weight.val().dims()[1]],"paired packed source memory geometry differs");
        assert_eq!(input.dims(),[target.tokens()],"paired packed target IDs/layout differ");
        assert_eq!(memory.device(),input.device(),"paired packed source/target devices differ");
    }

    /// Native source encoding and original final normalization, performed once when preparing memory.
    pub fn encode_inference_with<C,F>(&self,input:TensorParallelTransformerInput<B>,communicator:C,layout:&VocabParallelLossLayout,layer:F)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        let hidden = input.embed_inference(&self.source_embeddings,communicator,layout)?;
        self.encoder.forward_inference_with(hidden,layer).map(|hidden|self.source_finish(hidden))
    }

    /// Native independent packed source documents without materializing a padded source batch.
    pub fn encode_packed_inference_with<C,F>(&self,input:TensorParallelTransformerInput<B,1>,packed:&PackedSequenceLayout,
        communicator:C,layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<B,2>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        let hidden = input.into_batched(packed).embed_inference(&self.source_embeddings,communicator,layout)?;
        let width = hidden.dims()[2];self.encoder.forward_packed_inference_with(hidden.reshape([packed.tokens(),width]),layer).map(|hidden|self.source_finish(hidden))
    }

    /// Actual native target input tables, self/cross/FFN stages and decoder-final normalization.
    pub fn decode_hidden_inference_with<C,F>(&self,input:TensorParallelTransformerInput<B>,memory:Tensor<B,3>,communicator:C,
        layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        self.dense_memory(&input.tokens,&memory);let hidden = input.embed_inference(&self.target_embeddings,communicator,layout)?;
        self.decoder.forward_inference_with(hidden,memory,layer).map(|hidden|self.target_finish(hidden))
    }

    /// Native flat target states over exactly paired independent source/target document layouts.
    pub fn decode_packed_hidden_inference_with<C,F>(&self,input:TensorParallelTransformerInput<B,1>,memory:Tensor<B,2>,source:&PackedSequenceLayout,
        target:&PackedSequenceLayout,communicator:C,layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<B,2>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        self.packed_memory(&input.tokens,&memory,source,target);let hidden = input.into_batched(target).embed_inference(&self.target_embeddings,communicator,layout)?;
        let width = hidden.dims()[2];self.decoder.forward_inference_with(hidden.reshape([target.tokens(),width]),memory,layer).map(|hidden|self.target_finish(hidden))
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelEncoderDecoderModel<Autodiff<B,S>> {
    /// Original complete native source graph, including embedding and encoder-final norm derivatives.
    pub fn encode_with<C,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,communicator:C,layout:&VocabParallelLossLayout,layer:F)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let hidden = input.embed(&self.source_embeddings,communicator,layout)?;
        self.encoder.forward_with(hidden,layer).map(|hidden|self.source_finish(hidden))
    }

    /// Complete native packed source graph with caller-owned independent-document positions/masks.
    pub fn encode_packed_with<C,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,packed:&PackedSequenceLayout,
        communicator:C,layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        let hidden = input.into_batched(packed).embed(&self.source_embeddings,communicator,layout)?;
        let width = hidden.dims()[2];self.encoder.forward_packed_with(hidden.reshape([packed.tokens(),width]),layer).map(|hidden|self.source_finish(hidden))
    }

    /// Original target graph retaining actual source derivatives through every cross-memory stage.
    pub fn decode_hidden_with<C,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,memory:Tensor<Autodiff<B,S>,3>,communicator:C,
        layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)
            ->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        self.dense_memory(&input.tokens,&memory);let hidden = input.embed(&self.target_embeddings,communicator,layout)?;
        self.decoder.forward_with(hidden,memory,layer).map(|hidden|self.target_finish(hidden))
    }

    /// Native target/source packed graph over real paired boundaries, with no detached source fallback.
    pub fn decode_packed_hidden_with<C,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,memory:Tensor<Autodiff<B,S>,2>,source:&PackedSequenceLayout,
        target:&PackedSequenceLayout,communicator:C,layout:&VocabParallelLossLayout,layer:F) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)
            ->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        self.packed_memory(&input.tokens,&memory,source,target);let hidden = input.into_batched(target).embed(&self.target_embeddings,communicator,layout)?;
        let width = hidden.dims()[2];self.decoder.forward_with(hidden.reshape([target.tokens(),width]),memory,layer).map(|hidden|self.target_finish(hidden))
    }
}
