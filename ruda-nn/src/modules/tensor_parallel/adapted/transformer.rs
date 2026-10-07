use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Dropout,Module,Tensor,geometry};
use super::{TensorParallelAdaptedGroupedQueryAttention,TensorParallelAdaptedFeedForward};
use super::super::transformer::residual;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},cache::ProjectedKvCache,
    transformer::{AdaptedTransformerBlock,DenseTransformerNorm,AttentionAdapterTarget,FeedForwardAdapterTarget}};
use ruda_model::tensor::Bool;

/// Original native adapter block, with dense/LoRA partitions on explicit projection targets.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedTransformerBlock<B: Backend> {
    /// Actual dense/adapter head projections and output rows.
    pub attention: TensorParallelAdaptedGroupedQueryAttention<B>,
    /// Actual selected intermediate and output adapters.
    pub feed_forward: TensorParallelAdaptedFeedForward<B>,
    /// Original replicated attention normalization.
    pub attention_norm: DenseTransformerNorm<B>,
    /// Original independent replicated FFN normalization.
    pub feed_forward_norm: DenseTransformerNorm<B>,
    /// Original residual branch dropout, with caller-owned matching replica values.
    pub residual_dropout: Dropout,
    /// Original pre/post norm ordering.
    pub norm_first: bool,
}

impl<B: Backend> TensorParallelAdaptedTransformerBlock<B> {
    /// Consume actual locally sharded base/adapters without resetting IDs or freezing new targets.
    pub fn from_sharded_block(block: AdaptedTransformerBlock<B>) -> Self {
        let width = geometry(&block.attention.query)[0];
        assert_eq!(geometry(&block.attention.key)[0],width,"adapted parallel self-memory width differs");
        assert_eq!(geometry(&block.feed_forward.up)[0],width,"adapted parallel FFN residual width differs");
        assert_eq!(block.attention_norm.width(),width,"adapted parallel attention norm width differs");
        assert_eq!(block.feed_forward_norm.width(),width,"adapted parallel FFN norm width differs");
        Self {attention:TensorParallelAdaptedGroupedQueryAttention::from_shard(block.attention),
            feed_forward:TensorParallelAdaptedFeedForward::from_shard(block.feed_forward),attention_norm:block.attention_norm,
            feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }

    /// Return the original local module container for its existing adapter-record APIs.
    /// No adapter merge, optimizer conversion or parameter value copy occurs.
    pub fn into_local_block(self) -> AdaptedTransformerBlock<B> {
        AdaptedTransformerBlock {attention:self.attention.local,feed_forward:self.feed_forward.local,
            attention_norm:self.attention_norm,feed_forward_norm:self.feed_forward_norm,residual_dropout:self.residual_dropout,norm_first:self.norm_first}
    }

    /// Native inference with unmerged actual adapter state and original normalization order.
    pub fn forward_inference<C,F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,
        communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.local.project(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted native positions changed actual local heads");
            self.attention.forward_projected_inference(query,key,value,masks,options,communicator.clone())
        },|branch|self.residual_dropout.forward(branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward_inference(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }

    /// Native inference appends only actual new adapted K/V, and runs FFN only on new rows.
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

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedTransformerBlock<Autodiff<B,S>> {
    /// Actual native adapter/residual order and explicitly supplied head/KV replica groups.
    pub fn forward<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_with_adapter_dropout(input,masks,options,groups,positions,
            |_,module,input|module.forward(input),|_,module,input|module.forward(input))
    }

    /// Caller-owned actual per-target dropout at each adapter A dtype. Column input
    /// dropout and full-residual dropout must follow the declared replica semantics.
    pub fn forward_with_adapter_dropout<C,K,F,A,G>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F,mut attention_dropout: A,feed_forward_dropout: G) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            A: FnMut(AttentionAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,
            G: FnMut(FeedForwardAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_with_adapter_dropout(source.clone(),source.clone(),source,groups,&mut attention_dropout)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted parallel positions changed actual head geometry");
            self.attention.forward_projected_with_adapter_dropout(query,key,value,masks,options,groups.heads.clone(),
                |module,input|attention_dropout(AttentionAdapterTarget::Output,module,input))
        },|branch|self.residual_dropout.forward(branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,
            |source|self.feed_forward.forward_with_adapter_dropout(source,groups.heads.clone(),feed_forward_dropout),
            |branch|self.residual_dropout.forward(branch))
    }

    /// Detached-history inference with actual selected adapters and no base merge/replay.
    pub fn forward_cached<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project(source.clone(),source.clone(),source,groups)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key,cache.position());
            assert_eq!((query.dims(),key.dims()),geometry,"adapted parallel cached positions changed actual heads");
            self.attention.forward_cached_projected(query,key,value,visible,cache,masks,options,groups.heads.clone())
        },|branch|self.residual_dropout.forward(branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward(source,groups.heads.clone()),
            |branch|self.residual_dropout.forward(branch))
    }
}
