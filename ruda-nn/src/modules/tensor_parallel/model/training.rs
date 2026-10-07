use super::*;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},
    loss::{CausalCrossEntropyConfig,CausalLoss,LossTerms,KLDivLoss},pool::SequencePooling,transformer::SequenceHeadOutput};
use ruda_model::tensor::Bool;
use super::super::{AttentionParallelGroups,VocabParallelCrossEntropy};

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerModel<Autodiff<B,S>> {
    /// Complete input-to-logits graph with caller-owned per-layer positions/masks/groups.
    /// Input and output vocabularies/transports are independent actual placements.
    pub fn forward_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let hidden = self.forward_hidden_with(input,input_group,input_layout,layer)?;
        self.head.forward(hidden,output_group,output_layout,gather_output)
    }

    /// Explicit shared input/head/adapter dropout and original per-layer architecture callbacks.
    /// Full output gathering occurs only when requested; native parameter ties stay connected.
    pub fn forward_with_dropouts<C,O,F,I,H,A>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,input_group:C,
        input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool,
        input_dropout:I,head_dropout:H,adapter_dropout:A) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            H:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            A:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let hidden = self.forward_hidden_with_dropout(input,input_group,input_layout,layer,input_dropout)?;
        self.head.forward_with_dropouts(hidden,output_group,output_layout,gather_output,head_dropout,adapter_dropout)
    }

    /// One explicit shared TP group, original attention policy and unchanged native Q/K positions.
    /// This does not infer a causal mask from the output head or model name.
    pub fn forward<C,K>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,masks:DenseAttentionMask<Autodiff<B,S>>,
        options:DenseAttentionOptions,groups:&AttentionParallelGroups<C,K>,input_layout:&VocabParallelLossLayout,
        output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,K:BroadcastTensorCollective<B,Error=C::Error> {
        self.forward_with_positions(input,masks,options,groups,input_layout,output_layout,gather_output,|_,query,key|(query,key))
    }

    /// Complete shared-group graph with actual per-layer rotary/relative/custom Q/K transforms.
    pub fn forward_with_positions<C,K,P>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,masks:DenseAttentionMask<Autodiff<B,S>>,
        options:DenseAttentionOptions,groups:&AttentionParallelGroups<C,K>,input_layout:&VocabParallelLossLayout,
        output_layout:&VocabParallelLossLayout,gather_output:bool,mut positions:P) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,K:BroadcastTensorCollective<B,Error=C::Error>,
            P:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_with(input,groups.heads.clone(),input_layout,|index,block,hidden|
            block.forward(hidden,masks.clone(),options,groups,|query,key|positions(index,query,key)),
            groups.heads.clone(),output_layout,gather_output)
    }

    /// Full actual model-to-causal-objective graph, projecting local logits in native criterion chunks.
    /// Original label shift, ignore sentinel and real-class smoothing remain criterion-owned.
    pub fn forward_causal_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(labels.dims(),input.tokens.dims(),"parallel causal model labels/input rows differ");
        assert_eq!(labels.device(),input.tokens.device(),"parallel causal model labels/input devices differ");
        let hidden = self.forward_hidden_with(input,input_group,input_layout,layer)?;
        self.head.forward_causal_loss(hidden,labels,criterion,output_group,output_layout,label_smoothing)
    }

    /// Explicit shared input and chunk-wise head/adapter dropout for the same causal objective.
    /// Callback invocation/chunk geometry follows the actual native criterion, not a synthetic full-logit pass.
    pub fn forward_causal_with_dropouts<C,O,F,I,H,A>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,input_dropout:I,mut head_dropout:H,mut adapter_dropout:A)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            H:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,
            A:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        assert_eq!(labels.dims(),input.tokens.dims(),"parallel causal model labels/input rows differ");
        assert_eq!(labels.device(),input.tokens.device(),"parallel causal model labels/input devices differ");
        let hidden = self.forward_hidden_with_dropout(input,input_group,input_layout,layer,input_dropout)?;
        criterion.forward_sharded_hidden(hidden,labels,output_layout,output_group,|rows,group|
            self.head.forward_with_dropouts(rows,group.clone(),output_layout,false,&mut head_dropout,&mut adapter_dropout),label_smoothing)
    }

    /// Actual packed tokens through lookup, all selected layers, final norm and local output logits.
    pub fn forward_packed_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,packed:&PackedSequenceLayout,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        let hidden = self.forward_packed_hidden_with(input,packed,input_group,input_layout,layer)?;
        self.head.forward(hidden,output_group,output_layout,gather_output)
    }

    /// Packed full-model causal objective; native boundary masking prevents cross-document targets.
    pub fn forward_packed_causal_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        packed:&PackedSequenceLayout,input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,
        output_layout:&VocabParallelLossLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(labels.dims(),input.tokens.dims(),"parallel packed causal model labels/input lengths differ");
        assert_eq!(labels.device(),input.tokens.device(),"parallel packed causal model labels/input devices differ");
        let hidden = self.forward_packed_hidden_with(input,packed,input_group,input_layout,layer)?;
        self.head.forward_packed_causal_loss(hidden,labels,packed,criterion,output_group,output_layout,label_smoothing)
    }

    /// End-to-end actual packed causal graph with shared lookup and chunk-wise head dropout.
    pub fn forward_packed_causal_with_dropouts<C,O,F,I,H,A>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        packed:&PackedSequenceLayout,input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,
        output_layout:&VocabParallelLossLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,
        input_dropout:I,mut head_dropout:H,mut adapter_dropout:A) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            I:FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            H:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,
            A:FnMut(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        assert_eq!(labels.dims(),input.tokens.dims(),"parallel packed causal model labels/input lengths differ");
        assert_eq!(labels.device(),input.tokens.device(),"parallel packed causal model labels/input devices differ");
        let hidden = self.forward_packed_hidden_with_dropout(input,packed,input_group,input_layout,layer,input_dropout)?;
        criterion.forward_sharded_packed_hidden(hidden,labels,packed,output_layout,output_group,|rows,group|
            self.head.forward_with_dropouts(rows,group.clone(),output_layout,false,&mut head_dropout,&mut adapter_dropout),label_smoothing)
    }

    /// Native end-to-end per-token hard-label terms for explicitly aligned labels, without a causal shift.
    pub fn forward_token_terms_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,criterion:&VocabParallelCrossEntropy,
        visible:Option<Tensor<Autodiff<B,S>,2,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(labels.dims(),input.tokens.dims(),"parallel token model labels/input rows differ");
        let hidden = self.forward_hidden_with(input,input_group,input_layout,layer)?;
        self.head.forward_token_terms(hidden,labels,criterion,output_group,visible,weights)
    }

    /// Actual local soft targets through the complete native model and original sharded criterion.
    /// Teacher targets remain differentiable when supplied that way; no detachment/renormalization is inferred.
    pub fn forward_soft_token_terms_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,targets:Tensor<Autodiff<B,S>,3>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,criterion:&VocabParallelCrossEntropy,
        visible:Option<Tensor<Autodiff<B,S>,2,Bool>>,weights:Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let rows = input.tokens.dims();assert_eq!([targets.dims()[0],targets.dims()[1]],rows,"parallel soft targets/input rows differ");
        let hidden = self.forward_hidden_with(input,input_group,input_layout,layer)?;
        self.head.forward_soft_token_terms(hidden,targets,criterion,output_group,visible,weights)
    }

    /// Native complete-model KL terms in the caller's explicit probability or log-target space.
    pub fn forward_token_kl_terms_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,targets:Tensor<Autodiff<B,S>,3>,
        input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&KLDivLoss,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>) -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let rows = input.tokens.dims();assert_eq!([targets.dims()[0],targets.dims()[1]],rows,"parallel KL targets/input rows differ");
        let hidden = self.forward_hidden_with(input,input_group,input_layout,layer)?;
        self.head.forward_token_kl_terms(hidden,targets,criterion,output_group,output_layout,visible)
    }

    /// Complete input/backbone/final-norm sequence classification with explicit real-token pooling.
    pub fn forward_sequence_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,visible:Tensor<Autodiff<B,S>,2,Bool>,
        pooling:SequencePooling,input_group:C,input_layout:&VocabParallelLossLayout,layer:F,output_group:O,
        output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<SequenceHeadOutput<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(visible.dims(),input.tokens.dims(),"parallel sequence pool/input rows differ");
        let hidden = self.forward_hidden_with(input,input_group,input_layout,layer)?;
        self.head.forward_sequence(hidden,visible,pooling,output_group,output_layout,gather_output)
    }

    /// Actual packed document classification, retaining native empty-document visibility/counts.
    pub fn forward_packed_sequences_with<C,O,F>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,packed:&PackedSequenceLayout,
        visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,pooling:SequencePooling,input_group:C,input_layout:&VocabParallelLossLayout,
        layer:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<SequenceHeadOutput<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        let hidden = self.forward_packed_hidden_with(input,packed,input_group,input_layout,layer)?;
        self.head.forward_packed_sequences(hidden,packed,visible,pooling,output_group,output_layout,gather_output)
    }
}
