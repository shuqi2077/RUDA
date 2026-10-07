use super::*;
use super::super::{AttentionParallelGroups,VocabParallelGreedySelection};
use crate::attention::{DenseAttentionMask,DenseAttentionOptions};
use ruda_model::tensor::Bool;

impl<B:Backend> TensorParallelTransformerModel<B> {
    /// Complete native new-row cached inference with each actual layer's absolute Q/K positions.
    /// Attention visibility/masks are supplied explicitly; learned position IDs remain in the input.
    pub fn forward_cached_inference_with_positions<C,P>(&self,input:TensorParallelTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,
        input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,gather_output:bool,mut positions:P)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_cached_inference_with(input,cache,communicator.clone(),input_layout,|index,block,hidden,history|
            block.forward_cached_inference(hidden,visible.clone(),history,masks.clone(),options,communicator.clone(),
                |query,key,offset|positions(index,query,key,offset)),communicator.clone(),output_layout,gather_output)
    }

    /// Native cached model generation entry point with explicit projected absolute positions.
    /// Only the actual last new token is projected; returned valid flags retain native all-NaN/excluded behavior.
    pub fn forward_cached_greedy_inference_with_positions<C,P>(&self,input:TensorParallelTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,
        input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,output_visible:Option<Tensor<B,1,Bool>>,mut positions:P)
        -> Result<VocabParallelGreedySelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_cached_greedy_inference_with(input,cache,communicator.clone(),input_layout,|index,block,hidden,history|
            block.forward_cached_inference(hidden,visible.clone(),history,masks.clone(),options,communicator.clone(),
                |query,key,offset|positions(index,query,key,offset)),communicator.clone(),output_layout,output_visible)
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerModel<Autodiff<B,S>> {
    /// Original selected-layer cached graph and local output head, with native detached history.
    /// Inference mode remains backend-owned; no_grad alone does not disable dropout.
    pub fn forward_cached_with_positions<C,K,P>(&self,input:TensorParallelTransformerInput<Autodiff<B,S>>,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache:&mut TransformerKvCache<Autodiff<B,S>>,masks:DenseAttentionMask<Autodiff<B,S>>,options:DenseAttentionOptions,
        groups:&AttentionParallelGroups<C,K>,input_layout:&VocabParallelLossLayout,output_layout:&VocabParallelLossLayout,
        gather_output:bool,mut positions:P) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,K:BroadcastTensorCollective<B,Error=C::Error>,
            P:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let hidden = self.forward_cached_hidden_with(input,cache,groups.heads.clone(),input_layout,|index,block,hidden,history|
            block.forward_cached(hidden,visible.clone(),history,masks.clone(),options,groups,|query,key,offset|positions(index,query,key,offset)))?;
        self.head.forward(hidden,groups.heads.clone(),output_layout,gather_output)
    }
}
