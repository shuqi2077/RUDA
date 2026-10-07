use super::*;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},pool::SequencePooling,transformer::SequenceHeadOutput};
use ruda_model::tensor::Bool;
use super::super::VocabParallelGreedySelection;

impl<B:Backend> TensorParallelTransformerModel<B> {
    /// Native complete model inference using original per-layer positions, masks and transports.
    /// Use the native inference backend/valid model for inference dropout semantics.
    pub fn forward_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B>,input_group:C,input_layout:&VocabParallelLossLayout,
        layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        let hidden = self.forward_hidden_inference_with(input,input_group,input_layout,layer)?;
        self.head.forward_inference(hidden,output_group,output_layout,gather_output)
    }

    /// One explicit shared TP group with original attention policy and no added Q/K transform.
    pub fn forward_inference<C:BroadcastTensorCollective<B>>(&self,input:TensorParallelTransformerInput<B>,masks:DenseAttentionMask<B>,
        options:DenseAttentionOptions,communicator:C,input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,
        gather_output:bool) -> Result<Tensor<B,3>,C::Error> {
        self.forward_inference_with_positions(input,masks,options,communicator,input_layout,output_layout,gather_output,|_,query,key|(query,key))
    }

    /// Complete native model with actual caller-owned per-layer local-head position transforms.
    pub fn forward_inference_with_positions<C,P>(&self,input:TensorParallelTransformerInput<B>,masks:DenseAttentionMask<B>,
        options:DenseAttentionOptions,communicator:C,input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,
        gather_output:bool,mut positions:P) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_inference_with(input,communicator.clone(),input_layout,|index,block,hidden|
            block.forward_inference(hidden,masks.clone(),options,communicator.clone(),|query,key|positions(index,query,key)),
            communicator.clone(),output_layout,gather_output)
    }

    /// Complete native flat-token inference, retaining the actual packed layer policy.
    pub fn forward_packed_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B,1>,packed:&PackedSequenceLayout,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<B,2>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        let hidden = self.forward_packed_hidden_inference_with(input,packed,input_group,input_layout,layer)?;
        self.head.forward_inference(hidden,output_group,output_layout,gather_output)
    }

    /// Actual new cached rows through input tables, native history, final norm and local output head.
    /// A completed backbone cache remains advanced if the subsequent output transport fails.
    pub fn forward_cached_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B>,cache:&mut TransformerKvCache<B>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Result<Tensor<B,3>,C::Error> {
        let hidden = self.forward_cached_hidden_inference_with(input,cache,input_group,input_layout,layer)?;
        self.head.forward_inference(hidden,output_group,output_layout,gather_output)
    }

    /// Native cached model-to-global-greedy IDs, projecting only each row's actual last new token.
    /// No full global logits gather, sampler, automatic EOS policy or extra generation step is added.
    pub fn forward_cached_greedy_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B>,cache:&mut TransformerKvCache<B>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,
        visible:Option<Tensor<B,1,Bool>>) -> Result<VocabParallelGreedySelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Result<Tensor<B,3>,C::Error> {
        assert!(input.tokens.dims()[1] > 0,"cached parallel greedy model requires an actual last input token");
        let hidden = self.forward_cached_hidden_inference_with(input,cache,input_group,input_layout,layer)?;
        self.head.forward_greedy_last_inference(hidden,output_group,output_layout,visible)
    }

    /// Full-sequence native input/backbone followed by only the actual last-token greedy projection.
    pub fn forward_greedy_last_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,visible:Option<Tensor<B,1,Bool>>)
        -> Result<VocabParallelGreedySelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert!(input.tokens.dims()[1] > 0,"parallel greedy model requires an actual last input token");
        let hidden = self.forward_hidden_inference_with(input,input_group,input_layout,layer)?;
        self.head.forward_greedy_last_inference(hidden,output_group,output_layout,visible)
    }

    /// Actual local per-token log probabilities normalized over every real output class.
    pub fn forward_log_probabilities_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,visible:Option<Tensor<B,2,Bool>>)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        let logits = self.forward_inference_with(input,input_group,input_layout,layer,output_group.clone(),output_layout,false)?;
        output_layout.log_softmax_tokens_inference(logits,output_group,visible)
    }

    /// Actual complete-class normalized per-token probabilities, retaining local output storage.
    pub fn forward_probabilities_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,visible:Option<Tensor<B,2,Bool>>)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        let logits = self.forward_inference_with(input,input_group,input_layout,layer,output_group.clone(),output_layout,false)?;
        output_layout.softmax_tokens_inference(logits,output_group,visible)
    }

    /// Native complete-model sequence classification with explicit visible-token pooling.
    pub fn forward_sequence_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B>,visible:Tensor<B,2,Bool>,pooling:SequencePooling,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<SequenceHeadOutput<B>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert_eq!(visible.dims(),input.tokens.dims(),"parallel native sequence pool/input rows differ");
        let hidden = self.forward_hidden_inference_with(input,input_group,input_layout,layer)?;
        self.head.forward_sequence_inference(hidden,visible,pooling,output_group,output_layout,gather_output)
    }

    /// Independent actual packed-document classification with original empty-row visibility/counts.
    pub fn forward_packed_sequences_inference_with<C,O,F>(&self,input:TensorParallelTransformerInput<B,1>,packed:&PackedSequenceLayout,
        visible:Option<Tensor<B,1,Bool>>,pooling:SequencePooling,input_group:C,input_layout:&VocabParallelLossLayout,layer:F,
        output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<SequenceHeadOutput<B>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        let hidden = self.forward_packed_hidden_inference_with(input,packed,input_group,input_layout,layer)?;
        self.head.forward_packed_sequences_inference(hidden,packed,visible,pooling,output_group,output_layout,gather_output)
    }
}
