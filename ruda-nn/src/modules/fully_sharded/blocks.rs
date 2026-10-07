use super::*;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::ProjectedKvCache,transformer::{DenseTransformerBlock,AdaptedTransformerBlock,AdaptedStackLayer,DenseCrossAttentionBlock,
        AdaptedCrossAttentionBlock,DecoderCrossAttention,DenseEncoderDecoderLayer,AdaptedEncoderDecoderLayer,
        AdaptedProjection,AdaptedGroupedQueryAttention,DenseFeedForward,DenseTransformerNorm}};
use ruda_model::tensor::Bool;

/// Actual original dense/adapted transformer block with locally sharded persistent parameters.
#[derive(Module,Debug)]
pub struct FullyShardedTransformerBlock<B:Backend> {
    /// Actual original self-attention projection choices and head geometry.
    pub attention:FullyShardedGroupedQueryAttention<B>,
    /// Actual original ordinary/gated FFN, including its activation parameters.
    pub feed_forward:FullyShardedFeedForward<B>,
    /// Original independent attention affine normalization.
    pub attention_norm:FullyShardedTransformerNorm<B>,
    /// Original independent FFN affine normalization.
    pub feed_forward_norm:FullyShardedTransformerNorm<B>,
    /// Original residual-branch dropout.
    pub residual_dropout:crate::Dropout,
    /// Exactly the original pre/post normalization order.
    pub norm_first:bool,
    /// Retain the original dense-vs-adapted native execution/record choice.
    pub original_dense:bool,
}

/// Actual dense/adapted encoder-memory attention with locally sharded affine/projection leaves.
#[derive(Module,Debug)]
pub struct FullyShardedCrossAttentionBlock<B:Backend> {
    /// Actual original attention, including independent source and target widths.
    pub attention:FullyShardedGroupedQueryAttention<B>,
    /// Original query/residual normalization.
    pub query_norm:FullyShardedTransformerNorm<B>,
    /// Original optional encoder-memory normalization, neither added nor omitted.
    pub memory_norm:Option<FullyShardedTransformerNorm<B>>,
    /// Original cross-attention residual dropout.
    pub residual_dropout:crate::Dropout,
    /// Original query pre/post norm selection.
    pub norm_first:bool,
    /// Exact original dense/adapted memory-stage choice.
    pub original_dense:bool,
}

/// Original self-attention, source-memory attention and FFN in that exact order.
#[derive(Module,Debug)]
pub struct FullyShardedEncoderDecoderLayer<B:Backend> {
    /// Original independently configured self-attention and FFN stages.
    pub backbone:FullyShardedTransformerBlock<B>,
    /// Original independently configured encoder-memory stage.
    pub cross_attention:FullyShardedCrossAttentionBlock<B>,
}

impl<B:Backend> ShardingContext<B> {
    /// Partition a complete loaded dense block with one shared local leaf for each actual source ID.
    pub fn transformer(&mut self,block:DenseTransformerBlock<B>) -> FullyShardedTransformerBlock<B> {
        FullyShardedTransformerBlock {attention:self.grouped_attention(block.attention),feed_forward:self.feed_forward(block.feed_forward),
            attention_norm:self.normalization(block.attention_norm),feed_forward_norm:self.normalization(block.feed_forward_norm),
            residual_dropout:block.residual_dropout,norm_first:block.norm_first,original_dense:true}
    }
    /// Partition existing selected A/B roles, retaining unselected/frozen parameters and the original source choice.
    pub fn adapted_transformer(&mut self,block:AdaptedTransformerBlock<B>) -> FullyShardedTransformerBlock<B> {
        FullyShardedTransformerBlock {attention:self.adapted_attention(block.attention),feed_forward:self.adapted_feed_forward(block.feed_forward),
            attention_norm:self.normalization(block.attention_norm),feed_forward_norm:self.normalization(block.feed_forward_norm),
            residual_dropout:block.residual_dropout,norm_first:block.norm_first,original_dense:false}
    }
    /// Preserve an existing stack's actual dense/adapted selection without injecting adapters.
    pub fn transformer_choice(&mut self,block:AdaptedStackLayer<B>) -> FullyShardedTransformerBlock<B> {
        match block {AdaptedStackLayer::Dense(block)=>self.transformer(block),AdaptedStackLayer::Adapted(block)=>self.adapted_transformer(block)}
    }
    /// Partition actual dense encoder-memory parameters with their original independent input widths.
    pub fn cross_attention(&mut self,block:DenseCrossAttentionBlock<B>) -> FullyShardedCrossAttentionBlock<B> {
        FullyShardedCrossAttentionBlock {attention:self.grouped_attention(block.attention),query_norm:self.normalization(block.query_norm),
            memory_norm:block.memory_norm.map(|norm|self.normalization(norm)),residual_dropout:block.residual_dropout,
            norm_first:block.norm_first,original_dense:true}
    }
    /// Retain actual independently selected cross-attention adapters and optional source normalization.
    pub fn adapted_cross_attention(&mut self,block:AdaptedCrossAttentionBlock<B>) -> FullyShardedCrossAttentionBlock<B> {
        FullyShardedCrossAttentionBlock {attention:self.adapted_attention(block.attention),query_norm:self.normalization(block.query_norm),
            memory_norm:block.memory_norm.map(|norm|self.normalization(norm)),residual_dropout:block.residual_dropout,
            norm_first:block.norm_first,original_dense:false}
    }
    /// Preserve the real dense/adapted memory-stage selection.
    pub fn cross_attention_choice(&mut self,block:DecoderCrossAttention<B>) -> FullyShardedCrossAttentionBlock<B> {
        match block {DecoderCrossAttention::Dense(block)=>self.cross_attention(block),DecoderCrossAttention::Adapted(block)=>self.adapted_cross_attention(block)}
    }
    /// Partition a loaded complete native decoder layer without changing stage order or initialization.
    pub fn encoder_decoder_layer(&mut self,layer:DenseEncoderDecoderLayer<B>) -> FullyShardedEncoderDecoderLayer<B> {
        FullyShardedEncoderDecoderLayer {backbone:self.transformer(layer.backbone),cross_attention:self.cross_attention(layer.cross_attention)}
    }
    /// Partition the original independent decoder/source-memory adapter selections through the same alias context.
    pub fn adapted_encoder_decoder_layer(&mut self,layer:AdaptedEncoderDecoderLayer<B>) -> FullyShardedEncoderDecoderLayer<B> {
        FullyShardedEncoderDecoderLayer {backbone:self.transformer_choice(layer.backbone),cross_attention:self.cross_attention_choice(layer.cross_attention)}
    }
}

fn dense_projection<B:Backend>(projection:AdaptedProjection<B>) -> crate::Linear<B> {
    match projection {AdaptedProjection::Dense(layer)=>layer,AdaptedProjection::LoRA(_)=>panic!("original dense block contains an adapter choice")}
}
fn dense_attention<B:Backend>(attention:AdaptedGroupedQueryAttention<B>) -> crate::attention::GroupedQueryAttention<B> {
    crate::attention::GroupedQueryAttention {query:dense_projection(attention.query),key:dense_projection(attention.key),
        value:dense_projection(attention.value),output:dense_projection(attention.output),dropout:attention.dropout,
        query_heads:attention.query_heads,kv_heads:attention.kv_heads,head_dimension:attention.head_dimension}
}
fn block_choice<B:Backend>(block:AdaptedTransformerBlock<B>,dense:bool) -> AdaptedStackLayer<B> {
    if dense {
        AdaptedStackLayer::Dense(DenseTransformerBlock {attention:dense_attention(block.attention),
            feed_forward:DenseFeedForward {up:dense_projection(block.feed_forward.up),gate:block.feed_forward.gate.map(dense_projection),
                down:dense_projection(block.feed_forward.down),activation:block.feed_forward.activation,dropout:block.feed_forward.dropout},
            attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
    } else {AdaptedStackLayer::Adapted(block)}
}

macro_rules! gathered_blocks {
    ($backend:ty,[$($generics:tt)*],$gather:ident) => {
        impl<$($generics)*> FullyShardedTransformerBlock<$backend> {
            /// Transient actual native block; parameter values keep local collective derivatives in training.
            /// Full-value retention remains the selected original checkpoint strategy's behavior.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<AdaptedStackLayer<$backend>,C::Error> {
                let block = AdaptedTransformerBlock {attention:self.attention.$gather(communicator.clone())?,
                    feed_forward:self.feed_forward.$gather(communicator.clone())?,attention_norm:self.attention_norm.$gather(communicator.clone())?,
                    feed_forward_norm:self.feed_forward_norm.$gather(communicator)?,residual_dropout:self.residual_dropout.clone(),norm_first:self.norm_first};
                Ok(block_choice(block,self.original_dense))
            }
        }
        impl<$($generics)*> FullyShardedCrossAttentionBlock<$backend> {
            /// Transient original dense/adapted source-memory module; no source graph is detached.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<DecoderCrossAttention<$backend>,C::Error> {
                let attention = self.attention.$gather(communicator.clone())?;
                let query_norm = self.query_norm.$gather(communicator.clone())?;
                let memory_norm = self.memory_norm.as_ref().map(|norm|norm.$gather(communicator)).transpose()?;
                Ok(if self.original_dense {
                    DecoderCrossAttention::Dense(DenseCrossAttentionBlock {attention:dense_attention(attention),query_norm,memory_norm,
                        residual_dropout:self.residual_dropout.clone(),norm_first:self.norm_first})
                } else {DecoderCrossAttention::Adapted(AdaptedCrossAttentionBlock {attention,query_norm,memory_norm,
                    residual_dropout:self.residual_dropout.clone(),norm_first:self.norm_first})})
            }
        }
        impl<$($generics)*> FullyShardedEncoderDecoderLayer<$backend> {
            /// Transient native decoder over actual gathered self/cross/FFN values and original choices.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<AdaptedEncoderDecoderLayer<$backend>,C::Error> {
                Ok(AdaptedEncoderDecoderLayer {backbone:self.backbone.$gather(communicator.clone())?,cross_attention:self.cross_attention.$gather(communicator)?})
            }
        }
    };
}
gathered_blocks!(B,[B:Backend],gather_inference);
gathered_blocks!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);

macro_rules! block_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$forward:ident,$packed:ident,$masked:ident) => {
        impl<$($generics)*> FullyShardedTransformerBlock<$backend> {
            /// Original complete self-attention/FFN graph with explicit actual Q/K position policy.
            pub fn $forward<C,F>(&self,input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,
                communicator:C,positions:F) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                Ok(self.$gather(communicator)?.forward_with_positions(input,masks,options,positions))
            }
            /// Original packed independent-document attention and FFN, without padded hidden tokens.
            pub fn $packed<C,F>(&self,input:Tensor<$backend,2>,layout:&PackedSequenceLayout,options:PackedAttentionOptions,
                communicator:C,positions:F) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                Ok(self.$gather(communicator)?.forward_packed_with_positions(input,layout,options,positions))
            }
            /// Original native per-document query/key visibility and score bias, retained exactly.
            pub fn $masked<C,F>(&self,input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:&[PackedDocumentAttentionMask<$backend>],
                options:PackedAttentionOptions,communicator:C,positions:F) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                Ok(self.$gather(communicator)?.forward_packed_masked_with_positions(input,layout,masks,options,positions))
            }
        }
        impl<$($generics)*> FullyShardedEncoderDecoderLayer<$backend> {
            /// Complete original self -> memory -> FFN graph, retaining source derivatives and independent positions.
            pub fn $forward<C,F,G>(&self,input:Tensor<$backend,3>,memory:Tensor<$backend,3>,self_masks:DenseAttentionMask<$backend>,
                self_options:DenseAttentionOptions,cross_masks:DenseAttentionMask<$backend>,cross_options:DenseAttentionOptions,communicator:C,
                self_positions:F,cross_positions:G) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>),
                    G:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                Ok(self.$gather(communicator)?.forward_with_positions(input,memory,self_masks,self_options,cross_masks,cross_options,self_positions,cross_positions))
            }
            /// Actual paired packed target/source graph over independently declared real document boundaries.
            pub fn $packed<C,F,G>(&self,input:Tensor<$backend,2>,memory:Tensor<$backend,2>,query_layout:&PackedSequenceLayout,memory_layout:&PackedSequenceLayout,
                self_options:PackedAttentionOptions,cross_options:PackedAttentionOptions,communicator:C,self_positions:F,cross_positions:G)
                -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>),
                    G:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                Ok(self.$gather(communicator)?.forward_packed_with_positions(input,memory,query_layout,memory_layout,self_options,cross_options,self_positions,cross_positions))
            }
            /// Original separate self/cross per-document masks, without a global packed score mask.
            pub fn $masked<C,F,G>(&self,input:Tensor<$backend,2>,memory:Tensor<$backend,2>,query_layout:&PackedSequenceLayout,memory_layout:&PackedSequenceLayout,
                self_masks:&[PackedDocumentAttentionMask<$backend>],self_options:PackedAttentionOptions,cross_masks:&[PackedDocumentAttentionMask<$backend>],
                cross_options:PackedAttentionOptions,communicator:C,self_positions:F,cross_positions:G) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>),
                    G:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                Ok(self.$gather(communicator)?.forward_packed_masked_with_positions(input,memory,query_layout,memory_layout,
                    self_masks,self_options,cross_masks,cross_options,self_positions,cross_positions))
            }
        }
    };
}
block_execution!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward,forward_packed,forward_packed_masked);
block_execution!(B,[B:Backend],gather_inference,forward_inference,forward_packed_inference,forward_packed_masked_inference);

impl<B:Backend> FullyShardedTransformerBlock<B> {
    /// Native cached inference over actual new Q/K rows, preserving original positions and cache append behavior.
    pub fn forward_cached_inference<C,F>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,positions:F) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let block = self.gather_inference(communicator)?;
        Ok(block.forward_cached_with_positions(input,new_visible,cache,masks,options,positions))
    }
}
