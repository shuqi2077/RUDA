use super::*;
use ruda_model::tensor::{Bool,FrozenAwqOps,IntegerTensorCollective};
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
pub struct FullyShardedAwqAttention<B:Backend> {
    /// Original selected query projection.
    pub query:FullyShardedAwqProjection<B>,
    /// Original selected key projection.
    pub key:FullyShardedAwqProjection<B>,
    /// Original selected value projection.
    pub value:FullyShardedAwqProjection<B>,
    /// Original selected output projection.
    pub output:FullyShardedAwqProjection<B>,
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
pub struct FullyShardedAwqFeedForward<B:Backend> {
    /// Original loaded up/value projection.
    pub up:FullyShardedAwqProjection<B>,
    /// Original optional independent gate.
    pub gate:Option<FullyShardedAwqProjection<B>>,
    /// Original loaded down projection.
    pub down:FullyShardedAwqProjection<B>,
    /// Original actual activation, including locally sharded affine leaves.
    pub activation:FullyShardedActivation<B>,
    /// Original intermediate dropout.
    pub dropout:crate::Dropout,
}

/// Complete original native block with fully sharded packed bases and adapters.
#[derive(Module,Debug)]
pub struct FullyShardedAwqTransformerBlock<B:Backend> {
    /// Original independent projection choices and exact head geometry.
    pub attention:FullyShardedAwqAttention<B>,
    /// Original native ordinary/gated FFN.
    pub feed_forward:FullyShardedAwqFeedForward<B>,
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
pub struct FullyShardedAwqTransformerStack<B:Backend> {
    /// Every actual loaded block, including unchanged dense projection choices.
    pub blocks:Vec<FullyShardedAwqTransformerBlock<B>>,
}

impl<B:Backend> ShardingContext<B> {
    /// Partition exactly the caller-selected native projection variant.
    pub fn awq_projection(&mut self,projection:AwqTransformerProjection<B>) -> FullyShardedAwqProjection<B> {
        match projection {
            AwqTransformerProjection::Dense(layer)=>FullyShardedAwqProjection::Dense(self.linear(layer)),
            AwqTransformerProjection::LoRA(layer)=>FullyShardedAwqProjection::LoRA(self.lora(layer)),
            AwqTransformerProjection::Awq(layer)=>FullyShardedAwqProjection::Awq(self.awq(layer)),
            AwqTransformerProjection::AwqLoRA(layer)=>FullyShardedAwqProjection::AwqLoRA(self.awq_lora(layer)),
        }
    }
    /// Preserve every actual original Q/K/V/output leaf, dropout and head geometry.
    pub fn awq_attention(&mut self,attention:AwqGroupedQueryAttention<B>) -> FullyShardedAwqAttention<B> {
        FullyShardedAwqAttention {query:self.awq_projection(attention.query),key:self.awq_projection(attention.key),
            value:self.awq_projection(attention.value),output:self.awq_projection(attention.output),dropout:attention.dropout,
            query_heads:attention.query_heads,kv_heads:attention.kv_heads,head_dimension:attention.head_dimension}
    }
    /// Preserve each real native ordinary/gated FFN role and its actual activation.
    pub fn awq_feed_forward(&mut self,feed:AwqFeedForward<B>) -> FullyShardedAwqFeedForward<B> {
        FullyShardedAwqFeedForward {up:self.awq_projection(feed.up),gate:feed.gate.map(|gate|self.awq_projection(gate)),
            down:self.awq_projection(feed.down),activation:self.activation(feed.activation),dropout:feed.dropout}
    }
    /// Build a complete sharded native block through this original shared-ID context.
    pub fn awq_transformer(&mut self,block:AwqTransformerBlock<B>) -> FullyShardedAwqTransformerBlock<B> {
        FullyShardedAwqTransformerBlock {attention:self.awq_attention(block.attention),feed_forward:self.awq_feed_forward(block.feed_forward),
            attention_norm:self.normalization(block.attention_norm),feed_forward_norm:self.normalization(block.feed_forward_norm),
            residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }
    /// Preserve original complete block order and one canonical local leaf for tied IDs.
    pub fn awq_transformer_stack(&mut self,stack:AwqTransformerStack<B>) -> FullyShardedAwqTransformerStack<B> {
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
        impl<$($generics)*> FullyShardedAwqAttention<$backend> {
            /// Gather actual original projections without merging or dequantizing bases.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqGroupedQueryAttention<$backend>,C::Error> {
                Ok(AwqGroupedQueryAttention::from_projections(self.query.$gather(communicator.clone())?,self.key.$gather(communicator.clone())?,
                    self.value.$gather(communicator.clone())?,self.output.$gather(communicator)?,self.query_heads,self.kv_heads,self.head_dimension,self.dropout.clone()))
            }
        }
        impl<$($generics)*> FullyShardedAwqFeedForward<$backend> {
            /// Original native ordinary/gated FFN with actual gathered local leaves.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqFeedForward<$backend>,C::Error> {
                Ok(AwqFeedForward::from_projections(self.up.$gather(communicator.clone())?,
                    self.gate.as_ref().map(|gate|gate.$gather(communicator.clone())).transpose()?,self.down.$gather(communicator.clone())?,
                    self.activation.$gather(communicator)?,self.dropout.clone()))
            }
        }
        impl<$($generics)*> FullyShardedAwqTransformerBlock<$backend> {
            /// Gather only this native block, retaining original local persistent storage.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqTransformerBlock<$backend>,C::Error> {
                Ok(AwqTransformerBlock {attention:self.attention.$gather(communicator.clone())?,feed_forward:self.feed_forward.$gather(communicator.clone())?,
                    attention_norm:self.attention_norm.$gather(communicator.clone())?,feed_forward_norm:self.feed_forward_norm.$gather(communicator)?,
                    residual_dropout:self.residual_dropout.clone(),norm_first:self.norm_first})
            }
        }
    };
}
gather_awq_components!(B,[B:Backend],gather_inference);
gather_awq_components!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);

impl<B:Backend> FullyShardedAwqTransformerStack<B> {
    /// Shard caller-loaded native blocks; reuse an external context for embedding/head ties.
    pub fn from_full(stack:AwqTransformerStack<B>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).awq_transformer_stack(stack)}
    /// Prepare the actual original cache topology without allocating model weights.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.blocks.len(),initial_capacity)}
}

macro_rules! execute_awq_blocks {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$forward:ident,$packed:ident) => {
        impl<$($generics)*> FullyShardedAwqTransformerBlock<$backend> {
            /// Actual original block graph with native packed input and adapter derivatives.
            pub fn $forward<C,F>(&self,input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedAwqError<C::Error,<$backend as FrozenAwqOps>::AwqError>>
                where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?.forward_with_positions(input,masks,options,positions)
                    .map_err(FullyShardedAwqError::Projection)
            }
            /// Original packed documents/masks, without padded hidden rows or dense AWQ shadows.
            pub fn $packed<C,F>(&self,input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedAwqError<C::Error,<$backend as FrozenAwqOps>::AwqError>>
                where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?.forward_packed_with_positions(input,layout,masks,options,positions)
                    .map_err(FullyShardedAwqError::Projection)
            }
        }
        impl<$($generics)*> FullyShardedAwqTransformerStack<$backend> {
            /// Execute actual original block order, gathering only the current block.
            pub fn $forward<C,F>(&self,mut input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedAwqError<C::Error,<$backend as FrozenAwqOps>::AwqError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                for (index,block) in self.blocks.iter().enumerate() {input=block.$forward(input,masks.clone(),options,communicator.clone(),|query,key|positions(index,query,key))?;}
                Ok(input)
            }
            /// Execute the exact original packed document graph through every real block.
            pub fn $packed<C,F>(&self,mut input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedAwqError<C::Error,<$backend as FrozenAwqOps>::AwqError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                for (index,block) in self.blocks.iter().enumerate() {input=block.$packed(input,layout,masks,options,communicator.clone(),|query,key|positions(index,query,key))?;}
                Ok(input)
            }
        }
    };
}
execute_awq_blocks!(B,[B:FrozenAwqOps],gather_inference,forward_inference,forward_packed_inference);
execute_awq_blocks!(Autodiff<B,S>,[B:FrozenAwqOps,S:CheckpointStrategy],gather,forward,forward_packed);

impl<B:FrozenAwqOps> FullyShardedAwqTransformerBlock<B> {
    /// Native incremental inference with the actual original positioned KV cache.
    pub fn forward_cached_inference<C,F>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,positions:F)
        -> Result<Tensor<B,3>,FullyShardedAwqError<C::Error,B::AwqError>>
        where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.gather_inference(communicator).map_err(FullyShardedAwqError::Collective)?
            .forward_cached_with_positions(input,new_visible,cache,masks,options,positions).map_err(FullyShardedAwqError::Projection)
    }
}
