use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy};
use ruda_model::{module::Module,tensor::{Tensor,backend::Backend}};
use crate::{Dropout,attention::{DenseAttentionMask,DenseAttentionOptions},
    transformer::{DenseTransformerBlock,DenseTransformerNorm}};
use super::{AttentionParallelGroups,TensorParallelGroupedQueryAttention,TensorParallelFeedForward,BroadcastTensorCollective};

mod cached;
mod packed;
mod stack;
pub use stack::*;

/// Actual full-residual stage receiving an explicit caller-owned shared dropout mask.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum TensorParallelResidualStage {
    /// Reduced attention output, before the original residual addition.
    Attention,
    /// Reduced feed-forward output, before its original residual addition.
    FeedForward,
}

/// One actual native transformer block with local attention/FFN projection shards.
/// Norm parameters and full residual tensors are replicated. Local probability
/// and intermediate dropout use the original backend modules; residual dropout
/// must produce matching values across the model-parallel group.
#[derive(Module,Debug)]
pub struct TensorParallelTransformerBlock<B: Backend> {
    /// Actual local head projections and row-sharded output.
    pub attention: TensorParallelGroupedQueryAttention<B>,
    /// Actual intermediate columns and output rows.
    pub feed_forward: TensorParallelFeedForward<B>,
    /// Original replicated attention affine normalization.
    pub attention_norm: DenseTransformerNorm<B>,
    /// Original independent replicated FFN affine normalization.
    pub feed_forward_norm: DenseTransformerNorm<B>,
    /// Native branch dropout; caller owns replicated RNG/mask semantics.
    pub residual_dropout: Dropout,
    /// Exactly the supplied pre/post normalization ordering.
    pub norm_first: bool,
}

pub(super) fn residual<B: Backend,E,F,R,const D: usize>(input: Tensor<B,D>,norm: &DenseTransformerNorm<B>,norm_first: bool,
    branch: F,dropout: R) -> Result<Tensor<B,D>,E>
    where F: FnOnce(Tensor<B,D>)->Result<Tensor<B,D>,E>,R: FnOnce(Tensor<B,D>)->Tensor<B,D> {
    let source = if norm_first {norm.forward(input.clone())} else {input.clone()};
    let output = input+dropout(branch(source)?);
    Ok(if norm_first {output} else {norm.forward(output)})
}

impl<B: Backend> TensorParallelTransformerBlock<B> {
    /// Wrap an actual caller-sharded block, preserving all IDs, weights and norm choices.
    /// The source is already locally partitioned; no global weights are loaded or guessed.
    pub fn from_sharded_block(block: DenseTransformerBlock<B>) -> Self {
        let width = block.attention.query.weight.val().dims()[0];
        assert_eq!(block.attention.key.weight.val().dims()[0],width,"self-attention memory/residual width differs");
        assert_eq!(block.feed_forward.up.weight.val().dims()[0],width,"parallel FFN/residual width differs");
        assert_eq!(block.attention_norm.width(),width,"parallel attention norm/residual width differs");
        assert_eq!(block.feed_forward_norm.width(),width,"parallel FFN norm/residual width differs");
        Self {attention:TensorParallelGroupedQueryAttention::from_shard(block.attention),
            feed_forward:TensorParallelFeedForward::from_shard(block.feed_forward),attention_norm:block.attention_norm,
            feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }

    /// Return the original local module container without weight gathering or ID changes.
    pub fn into_local_block(self) -> DenseTransformerBlock<B> {
        DenseTransformerBlock {attention:self.attention.local,feed_forward:self.feed_forward.local,
            attention_norm:self.attention_norm,feed_forward_norm:self.feed_forward_norm,residual_dropout:self.residual_dropout,norm_first:self.norm_first}
    }

    /// Native inference using existing backend attention/FFN and explicit output collectives.
    /// Positions are applied to actual local Q/K heads, never to a synthetic full model.
    pub fn forward_inference<C,F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,
        communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.local.project(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"parallel inference positions changed actual head geometry");
            self.attention.forward_projected_inference(query,key,value,masks,options,communicator.clone())
        },|branch|self.residual_dropout.forward(branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward_inference(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelTransformerBlock<Autodiff<B,S>> {
    /// Original pre/post norm and residual order, with no implicit positional transform.
    pub fn forward<C,K>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.forward_with_positions(input,masks,options,groups,|query,key|(query,key))
    }

    /// Caller-owned local-head positions; native probability/intermediate dropout remains local.
    pub fn forward_with_positions<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_with_residual(input,masks,options,groups,positions,|_,branch|self.residual_dropout.forward(branch))
    }

    /// Explicit replicated residual transform/dropout after each reduced branch.
    /// Use actual shared masks when per-rank native RNG streams differ. No new
    /// RNG seed, random synchronization, gradient detach or residual rule is inferred.
    pub fn forward_with_residual<C,K,F,R>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F,mut branch_output: R) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            R: FnMut(TensorParallelResidualStage,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_self(source,groups)?;
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"parallel transformer positions changed local head geometry");
            self.attention.forward_projected(query,key,value,masks,options,groups.heads.clone())
        },|branch|branch_output(TensorParallelResidualStage::Attention,branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward(source,groups.heads.clone()),
            |branch|branch_output(TensorParallelResidualStage::FeedForward,branch))
    }

    /// Attention/residual/norm only, before an explicit encoder-memory stage.
    pub fn forward_attention_with_positions<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,
        options: DenseAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_self(source,groups)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"parallel self-attention positions changed local geometry");
            self.attention.forward_projected(query,key,value,masks,options,groups.heads.clone())
        },|branch|self.residual_dropout.forward(branch))
    }

    /// Actual FFN/residual/norm stage without repeating any attention computation.
    pub fn forward_feed_forward<C: BroadcastTensorCollective<B>>(&self,input: Tensor<Autodiff<B,S>,3>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        residual(input,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }
}
