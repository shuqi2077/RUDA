use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,TensorParallelTransformerBlock,
    Tensor,DenseAttentionMask,DenseAttentionOptions,residual};
use ruda_model::tensor::Bool;
use crate::cache::ProjectedKvCache;

impl<B: Backend> TensorParallelTransformerBlock<B> {
    /// Native cached self-attention alone; prepared source attention/FFN follow explicitly.
    pub fn forward_cached_attention_inference<C,F>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        residual(input,&self.attention_norm,self.norm_first,|source|
            self.attention.forward_cached_inference(source,visible,cache,masks,options,communicator,positions),
            |branch|self.residual_dropout.forward(branch))
    }

    /// Native inference processes only actual new rows through attention and FFN.
    pub fn forward_cached_inference<C,F>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source|
            self.attention.forward_cached_inference(source,visible,cache,masks,options,communicator.clone(),positions),
            |branch|self.residual_dropout.forward(branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward_inference(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelTransformerBlock<Autodiff<B,S>> {
    /// Detached-history inference on real local head caches; new rows alone run FFN.
    pub fn forward_cached<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_feed_forward(self.forward_cached_attention(input,visible,cache,masks,options,groups,positions)?,groups.heads.clone())
    }

    /// Cached self-attention stage alone, before actual immutable encoder memory.
    pub fn forward_cached_attention<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        residual(input,&self.attention_norm,self.norm_first,|source|
            self.attention.forward_cached(source,visible,cache,masks,options,groups,positions),
            |branch|self.residual_dropout.forward(branch))
    }
}
