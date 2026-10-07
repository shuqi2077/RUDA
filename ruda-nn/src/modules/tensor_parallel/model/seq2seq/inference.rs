use super::*;
use super::super::super::{VocabParallelGreedySelection,VocabParallelTopKSelection};
use crate::{pool::SequencePooling,transformer::SequenceHeadOutput};
use ruda_model::tensor::Bool;

impl<B:Backend> TensorParallelEncoderDecoderModel<B> {
    /// Complete native paired hidden output, before the original target head.
    pub fn forward_hidden_inference_with<C,T,E,F>(&self,source:TensorParallelTransformerInput<B>,target:TensorParallelTransformerInput<B>,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        let memory = self.encode_inference_with(source,source_group,source_layout,encoder)?;
        self.decode_hidden_inference_with(target,memory,target_group,target_layout,decoder)
    }

    /// Complete packed paired hidden inference, retaining independent actual document lengths.
    pub fn forward_packed_hidden_inference_with<C,T,E,F>(&self,source:TensorParallelTransformerInput<B,1>,target:TensorParallelTransformerInput<B,1>,
        source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F) -> Result<Tensor<B,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        assert_eq!(source_packed.documents(),target_packed.documents(),"native packed paired hidden document counts differ");
        let memory = self.encode_packed_inference_with(source,source_packed,source_group,source_layout,encoder)?;
        self.decode_packed_hidden_inference_with(target,memory,source_packed,target_packed,target_group,target_layout,decoder)
    }

    /// Paired target-sequence classification using the original native real-token pooling policy.
    pub fn forward_sequence_inference_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<B>,target:TensorParallelTransformerInput<B>,
        visible:Tensor<B,2,Bool>,pooling:SequencePooling,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<SequenceHeadOutput<B>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert_eq!(visible.dims(),target.tokens.dims(),"native paired sequence visibility/decoder rows differ");
        let hidden = self.forward_hidden_inference_with(source,target,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_sequence_inference(hidden,visible,pooling,output_group,output_layout,gather_output)
    }

    /// Native packed paired document classification, preserving empty-document visibility and counts.
    pub fn forward_packed_sequences_inference_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<B,1>,target:TensorParallelTransformerInput<B,1>,
        source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,visible:Option<Tensor<B,1,Bool>>,pooling:SequencePooling,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<SequenceHeadOutput<B>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        let hidden = self.forward_packed_hidden_inference_with(source,target,source_packed,target_packed,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_packed_sequences_inference(hidden,target_packed,visible,pooling,output_group,output_layout,gather_output)
    }

    /// Complete paired prompt-to-global greedy IDs, projecting only the actual last target row.
    /// No source/target tokenization, EOS termination or sampling policy is selected.
    pub fn forward_greedy_last_inference_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<B>,target:TensorParallelTransformerInput<B>,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,visible:Option<Tensor<B,1,Bool>>) -> Result<VocabParallelGreedySelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert!(target.tokens.dims()[1] > 0,"native paired prompt greedy requires an actual target token");
        let hidden = self.forward_hidden_inference_with(source,target,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_greedy_last_inference(hidden,output_group,output_layout,visible)
    }

    /// Complete paired prompt-to-global top-k candidates, without projecting/gathering full prompt logits.
    pub fn forward_topk_last_inference_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<B>,target:TensorParallelTransformerInput<B>,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,k:usize,visible:Option<Tensor<B,1,Bool>>) -> Result<VocabParallelTopKSelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert!(target.tokens.dims()[1] > 0,"native paired prompt top-k requires an actual target token");
        let hidden = self.forward_hidden_inference_with(source,target,source_group,source_layout,encoder,target_group,target_layout,decoder)?;
        self.head.forward_topk_last_inference(hidden,output_group,output_layout,k,visible)
    }

    /// Native complete source/target inference with original independent per-layer policies.
    pub fn forward_inference_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<B>,target:TensorParallelTransformerInput<B>,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        let memory = self.encode_inference_with(source,source_group,source_layout,encoder)?;
        let hidden = self.decode_hidden_inference_with(target,memory,target_group,target_layout,decoder)?;
        self.head.forward_inference(hidden,output_group,output_layout,gather_output)
    }

    /// Native paired flat-document inference with real independent source/target boundaries.
    pub fn forward_packed_inference_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<B,1>,target:TensorParallelTransformerInput<B,1>,
        source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<B,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        assert_eq!(source_packed.documents(),target_packed.documents(),"native packed paired model document counts differ");
        let memory = self.encode_packed_inference_with(source,source_packed,source_group,source_layout,encoder)?;
        let hidden = self.decode_packed_hidden_inference_with(target,memory,source_packed,target_packed,target_group,target_layout,decoder)?;
        self.head.forward_inference(hidden,output_group,output_layout,gather_output)
    }

    /// Encode the actual source once, then prepare each real decoder layer's positioned source K/V.
    /// The callback explicitly supplies persistent visibility and original cross-memory position policy.
    pub fn prepare_kv_cache_inference_with<C,E,F>(&self,source:TensorParallelTransformerInput<B>,source_group:C,
        source_layout:&VocabParallelLossLayout,encoder:E,initial_capacity:usize,prepare:F) -> Result<EncoderDecoderKvCache<B>,C::Error>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedDecoderCrossAttention<B>,Tensor<B,3>)->Result<ProjectedKvCache<B>,C::Error> {
        let memory = self.encode_inference_with(source,source_group,source_layout,encoder)?;
        self.decoder.prepare_kv_cache_inference_with(memory,initial_capacity,prepare)
    }

    /// Actual new target rows over original prepared source/decoder history, without rerunning the encoder.
    pub fn decode_cached_hidden_inference_with<C,F>(&self,target:TensorParallelTransformerInput<B>,cache:&mut EncoderDecoderKvCache<B>,
        target_group:C,target_layout:&VocabParallelLossLayout,decoder:F) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)
            ->Result<Tensor<B,3>,C::Error> {
        let hidden = target.embed_inference(&self.target_embeddings,target_group,target_layout)?;
        self.decoder.forward_cached_inference_with(hidden,cache,decoder).map(|hidden|self.target_finish(hidden))
    }

    /// Complete native cached target decoding and actual local output projection.
    /// Existing cache completion follows the decoder contract; head transport is a subsequent stage.
    pub fn forward_cached_inference_with<C,O,F>(&self,target:TensorParallelTransformerInput<B>,cache:&mut EncoderDecoderKvCache<B>,
        target_group:C,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,C::Error> {
        let hidden = self.decode_cached_hidden_inference_with(target,cache,target_group,target_layout,decoder)?;
        self.head.forward_inference(hidden,output_group,output_layout,gather_output)
    }

    /// Native cached target-to-global-greedy output, projecting only the actual last new decoder row.
    pub fn forward_cached_greedy_inference_with<C,O,F>(&self,target:TensorParallelTransformerInput<B>,cache:&mut EncoderDecoderKvCache<B>,
        target_group:C,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,
        visible:Option<Tensor<B,1,ruda_model::tensor::Bool>>) -> Result<VocabParallelGreedySelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,C::Error> {
        assert!(target.tokens.dims()[1] > 0,"cached paired greedy model requires an actual last target token");
        let hidden = self.decode_cached_hidden_inference_with(target,cache,target_group,target_layout,decoder)?;
        self.head.forward_greedy_last_inference(hidden,output_group,output_layout,visible)
    }

    /// Native paired cached decoder-to-global-top-k candidates over already prepared actual source memory.
    pub fn forward_cached_topk_inference_with<C,O,F>(&self,target:TensorParallelTransformerInput<B>,cache:&mut EncoderDecoderKvCache<B>,
        target_group:C,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,
        k:usize,visible:Option<Tensor<B,1,ruda_model::tensor::Bool>>) -> Result<VocabParallelTopKSelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,C::Error> {
        assert!(target.tokens.dims()[1] > 0,"cached paired top-k model requires an actual last target token");
        let hidden = self.decode_cached_hidden_inference_with(target,cache,target_group,target_layout,decoder)?;
        self.head.forward_topk_last_inference(hidden,output_group,output_layout,k,visible)
    }
}
