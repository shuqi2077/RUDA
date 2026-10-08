use super::*;
use ruda_model::tensor::{Bool,IntegerTensorCollective};
use crate::transformer::TransformerProjection;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::{ProjectedKvCache,TransformerKvCache},
    transformer::{AwqTransformerProjection,AwqGroupedQueryAttention,AwqFeedForward,AwqTransformerBlock,AwqTransformerStack}};

/// Exact native projection choice with both packed base and floating leaves locally sharded.
#[derive(Module,Debug)]
pub enum FullyShardedAwqProjection<B:Backend> {
    /// Original dense values/flags, not an automatically frozen base.
    Dense(FullyShardedLinear<B>),
    /// Original native dense-base LoRA.
    LoRA(FullyShardedLoRALinear<B>),
    /// Actual native immutable packed AWQ base.
    Awq(FullyShardedAwqLinear<B>),
    /// Actual native packed base with separately sharded trainable A/B.
    AwqLoRA(FullyShardedAwqLoRALinear<B>),
}

/// Actual native mixed-projection GQA with only rank-local persistent parameters.
#[derive(Module,Debug)]
pub struct FullyShardedAwqAttention<B:Backend,P:Module<B>=FullyShardedAwqProjection<B>> {
    /// Backend identity, without tensor storage or persistent parameters.
    #[module(skip)]
    pub backend:core::marker::PhantomData<B>,
    /// Original selected query projection.
    pub query:P,
    /// Original selected key projection.
    pub key:P,
    /// Original selected value projection.
    pub value:P,
    /// Original selected output projection.
    pub output:P,
    /// Original attention probability dropout.
    pub dropout:crate::Dropout,
    /// Actual original query head count.
    pub query_heads:usize,
    /// Actual original KV head count.
    pub kv_heads:usize,
    /// Actual original head feature width.
    pub head_dimension:usize,
}

/// Actual ordinary/gated native FFN with independently selected sharded projections.
#[derive(Module,Debug)]
pub struct FullyShardedAwqFeedForward<B:Backend,P:Module<B>=FullyShardedAwqProjection<B>> {
    /// Original loaded up/value projection.
    pub up:P,
    /// Original optional independent gate.
    pub gate:Option<P>,
    /// Original loaded down projection.
    pub down:P,
    /// Original actual activation, including locally sharded affine leaves.
    pub activation:FullyShardedActivation<B>,
    /// Original intermediate dropout.
    pub dropout:crate::Dropout,
}

/// Complete original native block with fully sharded packed bases and adapters.
#[derive(Module,Debug)]
pub struct FullyShardedAwqTransformerBlock<B:Backend,P:Module<B>=FullyShardedAwqProjection<B>> {
    /// Original independent projection choices and exact head geometry.
    pub attention:FullyShardedAwqAttention<B,P>,
    /// Original native ordinary/gated FFN.
    pub feed_forward:FullyShardedAwqFeedForward<B,P>,
    /// Original actual attention affine norm/epsilon.
    pub attention_norm:FullyShardedTransformerNorm<B>,
    /// Original actual independent FFN affine norm/epsilon.
    pub feed_forward_norm:FullyShardedTransformerNorm<B>,
    /// Original residual branch dropout.
    pub residual_dropout:crate::Dropout,
    /// Original normalization/residual ordering.
    pub norm_first:bool,
}

/// Exact native loaded layer order with local-only persistent integer/floating storage.
#[derive(Module,Debug)]
pub struct FullyShardedAwqTransformerStack<B:Backend,P:Module<B>=FullyShardedAwqProjection<B>> {
    /// Every actual loaded block, including unchanged dense projection choices.
    pub blocks:Vec<FullyShardedAwqTransformerBlock<B,P>>,
}

impl<B:Backend> ShardingContext<B> {
    /// Partition exactly the caller-selected native projection variant.
    pub fn awq_projection<P:ShardTransformerProjection<B>>(&mut self,projection:P) -> P::Sharded {projection.shard(self)}
    /// Preserve every actual original Q/K/V/output leaf, dropout and head geometry.
    pub fn awq_attention<P:ShardTransformerProjection<B>>(&mut self,attention:AwqGroupedQueryAttention<B,P>) -> FullyShardedAwqAttention<B,P::Sharded> {
        FullyShardedAwqAttention {backend:core::marker::PhantomData,query:self.awq_projection(attention.query),key:self.awq_projection(attention.key),
            value:self.awq_projection(attention.value),output:self.awq_projection(attention.output),dropout:attention.dropout,
            query_heads:attention.query_heads,kv_heads:attention.kv_heads,head_dimension:attention.head_dimension}
    }
    /// Preserve each real native ordinary/gated FFN role and its actual activation.
    pub fn awq_feed_forward<P:ShardTransformerProjection<B>>(&mut self,feed:AwqFeedForward<B,P>) -> FullyShardedAwqFeedForward<B,P::Sharded> {
        FullyShardedAwqFeedForward {up:self.awq_projection(feed.up),gate:feed.gate.map(|gate|self.awq_projection(gate)),
            down:self.awq_projection(feed.down),activation:self.activation(feed.activation),dropout:feed.dropout}
    }
    /// Build a complete sharded native block through this original shared-ID context.
    pub fn awq_transformer<P:ShardTransformerProjection<B>>(&mut self,block:AwqTransformerBlock<B,P>) -> FullyShardedAwqTransformerBlock<B,P::Sharded> {
        FullyShardedAwqTransformerBlock {attention:self.awq_attention(block.attention),feed_forward:self.awq_feed_forward(block.feed_forward),
            attention_norm:self.normalization(block.attention_norm),feed_forward_norm:self.normalization(block.feed_forward_norm),
            residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }
    /// Preserve original complete block order and one canonical local leaf for tied IDs.
    pub fn awq_transformer_stack<P:ShardTransformerProjection<B>>(&mut self,stack:AwqTransformerStack<B,P>) -> FullyShardedAwqTransformerStack<B,P::Sharded> {
        FullyShardedAwqTransformerStack {blocks:stack.blocks.into_iter().map(|block|self.awq_transformer(block)).collect()}
    }
}

macro_rules! gather_awq_components {
    ($backend:ty,[$($generics:tt)*],$gather:ident) => {
        impl<$($generics)*> FullyShardedAwqProjection<$backend> {
            /// Transient original native choice over actual packed and floating values.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqTransformerProjection<$backend>,C::Error> {
                match self {
                    Self::Dense(layer)=>layer.$gather(communicator).map(AwqTransformerProjection::Dense),
                    Self::LoRA(layer)=>layer.$gather(communicator).map(AwqTransformerProjection::LoRA),
                    Self::Awq(layer)=>layer.$gather(communicator).map(AwqTransformerProjection::Awq),
                    Self::AwqLoRA(layer)=>layer.$gather(communicator).map(AwqTransformerProjection::AwqLoRA),
                }
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqAttention<$backend,P> {
            /// Gather actual original projections without merging or dequantizing bases.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqGroupedQueryAttention<$backend,P::Gathered>,C::Error> {
                Ok(AwqGroupedQueryAttention::from_projections(self.query.gather_projection(communicator.clone())?,self.key.gather_projection(communicator.clone())?,
                    self.value.gather_projection(communicator.clone())?,self.output.gather_projection(communicator)?,self.query_heads,self.kv_heads,self.head_dimension,self.dropout.clone()))
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqFeedForward<$backend,P> {
            /// Original native ordinary/gated FFN with actual gathered local leaves.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqFeedForward<$backend,P::Gathered>,C::Error> {
                Ok(AwqFeedForward::from_projections(self.up.gather_projection(communicator.clone())?,
                    self.gate.as_ref().map(|gate|gate.gather_projection(communicator.clone())).transpose()?,self.down.gather_projection(communicator.clone())?,
                    self.activation.$gather(communicator)?,self.dropout.clone()))
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqTransformerBlock<$backend,P> {
            /// Gather only this native block, retaining original local persistent storage.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqTransformerBlock<$backend,P::Gathered>,C::Error> {
                Ok(AwqTransformerBlock {attention:self.attention.$gather(communicator.clone())?,feed_forward:self.feed_forward.$gather(communicator.clone())?,
                    attention_norm:self.attention_norm.$gather(communicator.clone())?,feed_forward_norm:self.feed_forward_norm.$gather(communicator)?,
                    residual_dropout:self.residual_dropout.clone(),norm_first:self.norm_first})
            }
        }
    };
}
gather_awq_components!(B,[B:Backend],gather_inference);
gather_awq_components!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);

impl<B:Backend,P:Module<B>> FullyShardedAwqTransformerStack<B,P> {
    /// Shard caller-loaded native blocks; reuse an external context for embedding/head ties.
    pub fn from_full<Q:ShardTransformerProjection<B,Sharded=P>>(stack:AwqTransformerStack<B,Q>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).awq_transformer_stack(stack)}
    /// Prepare the actual original cache topology without allocating model weights.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.blocks.len(),initial_capacity)}
}

macro_rules! execute_awq_blocks {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$forward:ident,$packed:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqTransformerBlock<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Actual original block graph with native packed input and adapter derivatives.
            pub fn $forward<C,F>(&self,input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?.forward_with_positions(input,masks,options,positions)
                    .map_err(FullyShardedAwqError::Projection)
            }
            /// Original packed documents/masks, without padded hidden rows or dense AWQ shadows.
            pub fn $packed<C,F>(&self,input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?.forward_packed_with_positions(input,layout,masks,options,positions)
                    .map_err(FullyShardedAwqError::Projection)
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqTransformerStack<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Execute actual original block order, gathering only the current block.
            pub fn $forward<C,F>(&self,mut input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                for (index,block) in self.blocks.iter().enumerate() {input=block.$forward(input,masks.clone(),options,communicator.clone(),|query,key|positions(index,query,key))?;}
                Ok(input)
            }
            /// Execute the exact original packed document graph through every real block.
            pub fn $packed<C,F>(&self,mut input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                for (index,block) in self.blocks.iter().enumerate() {input=block.$packed(input,layout,masks,options,communicator.clone(),|query,key|positions(index,query,key))?;}
                Ok(input)
            }
        }
    };
}
execute_awq_blocks!(B,[B:Backend],gather_inference,forward_inference,forward_packed_inference);
execute_awq_blocks!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward,forward_packed);

impl<B:Backend,P:GatherTransformerProjection<B,B>> FullyShardedAwqTransformerBlock<B,P>
    where P::Gathered:TransformerProjection<B> {
    /// Native incremental inference with the actual original positioned KV cache.
    pub fn forward_cached_inference<C,F>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,positions:F)
        -> Result<Tensor<B,3>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<B>>::Error>>
        where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.gather_inference(communicator).map_err(FullyShardedAwqError::Collective)?
            .forward_cached_with_positions(input,new_visible,cache,masks,options,positions).map_err(FullyShardedAwqError::Projection)
    }
}
