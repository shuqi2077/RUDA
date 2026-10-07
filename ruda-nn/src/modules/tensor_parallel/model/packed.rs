use super::*;
use super::super::AttentionParallelGroups;
use crate::{attention::{PackedAttentionOptions,PackedDocumentAttentionMask},loss::{CausalCrossEntropyConfig,CausalLoss}};

impl<B:Backend> TensorParallelTransformerModel<B> {
    /// Complete native packed model under one actual TP group and explicit document attention policy.
    pub fn forward_packed_inference<C:BroadcastTensorCollective<B>>(&self,input:TensorParallelTransformerInput<B,1>,packed:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,communicator:C,
        input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,2>,C::Error> {
        self.forward_packed_inference_with_positions(input,packed,masks,options,communicator,input_layout,output_layout,gather_output,|_,query,key|(query,key))
    }

    /// Actual flat-token Q/K position transforms per layer; documents are never implicitly repadded.
    pub fn forward_packed_inference_with_positions<C,P>(&self,input:TensorParallelTransformerInput<B,1>,packed:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,communicator:C,
        input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,gather_output:bool,mut positions:P)
        -> Result<Tensor<B,2>,C::Error>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_packed_inference_with(input,packed,communicator.clone(),input_layout,|index,block,hidden|
            block.forward_packed_inference(hidden,packed,masks,options,communicator.clone(),|query,key|positions(index,query,key)),
            communicator.clone(),output_layout,gather_output)
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerModel<Autodiff<B,S>> {
    /// Complete actual packed model graph using the declared group and unchanged Q/K positions.
    pub fn forward_packed<C,K>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,packed:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options:PackedAttentionOptions,groups:&AttentionParallelGroups<C,K>,
        input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,K:BroadcastTensorCollective<B,Error=C::Error> {
        self.forward_packed_with_positions(input,packed,masks,options,groups,input_layout,output_layout,gather_output,|_,query,key|(query,key))
    }

    /// Native packed lookup, per-layer local Q/K positions, selected adapters and final output head.
    pub fn forward_packed_with_positions<C,K,P>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,packed:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options:PackedAttentionOptions,groups:&AttentionParallelGroups<C,K>,
        input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,gather_output:bool,mut positions:P)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,K:BroadcastTensorCollective<B,Error=C::Error>,
            P:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        self.forward_packed_with(input,packed,groups.heads.clone(),input_layout,|index,block,hidden|
            block.forward_packed(hidden,packed,masks,options,groups,|query,key|positions(index,query,key)),
            groups.heads.clone(),output_layout,gather_output)
    }

    /// End-to-end native causal packed training under explicit independent-document attention.
    /// The criterion projects only local vocabulary chunks and retains its original boundary-aware shift.
    pub fn forward_packed_causal_with_positions<C,K,P>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        packed:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options:PackedAttentionOptions,
        groups:&AttentionParallelGroups<C,K>,input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,mut positions:P) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,K:BroadcastTensorCollective<B,Error=C::Error>,
            P:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        self.forward_packed_causal_with(input,labels,packed,groups.heads.clone(),input_layout,|index,block,hidden|
            block.forward_packed(hidden,packed,masks,options,groups,|query,key|positions(index,query,key)),
            groups.heads.clone(),output_layout,criterion,label_smoothing)
    }
}
