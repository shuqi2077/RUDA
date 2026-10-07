use super::*;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions},
    cache::{TransformerKvCache,ProjectedKvCache},transformer::{DenseTransformerStack,AdaptedTransformerStack,DenseEncoderDecoderStack,AdaptedEncoderDecoderStack}};
use ruda_model::tensor::Bool;

/// Exact original layer order with only rank-local persistent parameter/gradient storage.
#[derive(Module,Debug)]
pub struct FullyShardedTransformerStack<B:Backend> {
    /// Every actual original dense/adapted layer, without skipped or initialized replacements.
    pub blocks:Vec<FullyShardedTransformerBlock<B>>,
}

/// Exact original paired decoder self/cross/FFN layer order with local persistent parameters.
#[derive(Module,Debug)]
pub struct FullyShardedEncoderDecoderStack<B:Backend> {
    /// Original independently selected self-attention and source-memory stages.
    pub layers:Vec<FullyShardedEncoderDecoderLayer<B>>,
}

impl<B:Backend> ShardingContext<B> {
    /// Partition every original dense layer through one shared alias context.
    pub fn transformer_stack(&mut self,stack:DenseTransformerStack<B>) -> FullyShardedTransformerStack<B> {
        FullyShardedTransformerStack {blocks:stack.blocks.into_iter().map(|block|self.transformer(block)).collect()}
    }
    /// Retain exact original dense/adapted selections and cross-layer shared leaves.
    pub fn adapted_transformer_stack(&mut self,stack:AdaptedTransformerStack<B>) -> FullyShardedTransformerStack<B> {
        FullyShardedTransformerStack {blocks:stack.layers.into_iter().map(|block|self.transformer_choice(block)).collect()}
    }
    /// Partition the complete loaded decoder through the same real parameter alias context.
    pub fn encoder_decoder_stack(&mut self,stack:DenseEncoderDecoderStack<B>) -> FullyShardedEncoderDecoderStack<B> {
        FullyShardedEncoderDecoderStack {layers:stack.layers.into_iter().map(|layer|self.encoder_decoder_layer(layer)).collect()}
    }
    /// Retain the original independent per-layer backbone/memory adapter selections.
    pub fn adapted_encoder_decoder_stack(&mut self,stack:AdaptedEncoderDecoderStack<B>) -> FullyShardedEncoderDecoderStack<B> {
        FullyShardedEncoderDecoderStack {layers:stack.layers.into_iter().map(|layer|self.adapted_encoder_decoder_layer(layer)).collect()}
    }
}

impl<B:Backend> FullyShardedTransformerStack<B> {
    /// Partition actual loaded dense weights; reuse ShardingContext for ties to other model components.
    pub fn from_full(stack:DenseTransformerStack<B>,rank:usize,world:usize) -> Self {
        ShardingContext::new(rank,world).transformer_stack(stack)
    }
    /// Preserve actual loaded dense/adapted choices without creating new adapters.
    pub fn from_full_adapted(stack:AdaptedTransformerStack<B>,rank:usize,world:usize) -> Self {
        ShardingContext::new(rank,world).adapted_transformer_stack(stack)
    }
    /// Real per-layer execution; architecture-owned visibility/positions/transports stay explicit.
    pub fn forward_with<E,F>(&self,mut input:Tensor<B,3>,mut layer:F) -> Result<Tensor<B,3>,E>
        where F:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,3>)->Result<Tensor<B,3>,E> {
        for (index,block) in self.blocks.iter().enumerate() {input = layer(index,block,input)?;}
        Ok(input)
    }
    /// Actual flat independent-document rows through every original layer, without padding.
    pub fn forward_packed_with<E,F>(&self,mut input:Tensor<B,2>,mut layer:F) -> Result<Tensor<B,2>,E>
        where F:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,2>)->Result<Tensor<B,2>,E> {
        for (index,block) in self.blocks.iter().enumerate() {input = layer(index,block,input)?;}
        Ok(input)
    }
    /// Allocate original native cache metadata for the exact actual block count.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {
        TransformerKvCache::new(self.blocks.len(),initial_capacity)
    }
    /// Native cached execution over the actual per-layer history; only complete chunks advance
    /// the stack boundary. Partial failures may leave advanced layer caches: restore the original
    /// completed cache record before replay, not a fabricated partially completed stack position.
    pub fn forward_cached_inference_with<E,F>(&self,mut input:Tensor<B,3>,cache:&mut TransformerKvCache<B>,mut layer:F)
        -> Result<Tensor<B,3>,E>
        where F:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Result<Tensor<B,3>,E> {
        cache.validate_layers(self.blocks.len());
        let rows = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(rows.1).expect("fully sharded cached stack position overflow");
        for (index,block) in self.blocks.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"fully sharded cached layer changed actual chunk rows");
        }
        cache.finish_chunk(next);Ok(input)
    }
}

impl<B:Backend> FullyShardedEncoderDecoderStack<B> {
    /// Partition the actual complete dense decoder without resetting any native source weights.
    pub fn from_full(stack:DenseEncoderDecoderStack<B>,rank:usize,world:usize) -> Self {
        ShardingContext::new(rank,world).encoder_decoder_stack(stack)
    }
    /// Partition actual dense/adapted paired layers, retaining their original choices and ties.
    pub fn from_full_adapted(stack:AdaptedEncoderDecoderStack<B>,rank:usize,world:usize) -> Self {
        ShardingContext::new(rank,world).adapted_encoder_decoder_stack(stack)
    }
    /// Complete native paired graph; each original layer receives the actual same source memory.
    /// Training keeps original encoder derivatives; this does not detach or reproject synthetic memory.
    pub fn forward_with<E,F>(&self,mut input:Tensor<B,3>,memory:Tensor<B,3>,mut layer:F) -> Result<Tensor<B,3>,E>
        where F:FnMut(usize,&FullyShardedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }
    /// Original paired flat-document graph, retaining complete source-memory derivatives.
    pub fn forward_packed_with<E,F>(&self,mut input:Tensor<B,2>,memory:Tensor<B,2>,mut layer:F) -> Result<Tensor<B,2>,E>
        where F:FnMut(usize,&FullyShardedEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }
}

macro_rules! stack_execution {
    ($backend:ty,[$($generics:tt)*],$block_forward:ident,$block_packed:ident,$forward:ident,$packed:ident) => {
        impl<$($generics)*> FullyShardedTransformerStack<$backend> {
            /// Original whole stack with shared explicit attention policy and per-layer actual Q/K positions.
            /// Only each current layer is gathered by this runner; persistent modules retain local storage.
            pub fn $forward<C,F>(&self,input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,
                communicator:C,mut positions:F) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.forward_with(input,|index,block,input|block.$block_forward(input,masks.clone(),options,communicator.clone(),
                    |query,key|positions(index,query,key)))
            }
            /// Actual whole packed stack with original document-local visibility/alignment and positions.
            pub fn $packed<C,F>(&self,input:Tensor<$backend,2>,layout:&PackedSequenceLayout,options:PackedAttentionOptions,
                communicator:C,mut positions:F) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                self.forward_packed_with(input,|index,block,input|block.$block_packed(input,layout,options,communicator.clone(),
                    |query,key|positions(index,query,key)))
            }
        }
        impl<$($generics)*> FullyShardedEncoderDecoderStack<$backend> {
            /// Original paired stack and separate self/cross policies, without changing source gradients or stage order.
            pub fn $forward<C,F,G>(&self,input:Tensor<$backend,3>,memory:Tensor<$backend,3>,self_masks:DenseAttentionMask<$backend>,
                self_options:DenseAttentionOptions,cross_masks:DenseAttentionMask<$backend>,cross_options:DenseAttentionOptions,communicator:C,
                mut self_positions:F,mut cross_positions:G) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>),
                    G:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.forward_with(input,memory,|index,block,input,memory|block.$block_forward(input,memory,self_masks.clone(),self_options,
                    cross_masks.clone(),cross_options,communicator.clone(),|query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
            }
            /// Complete original paired packed stack with independent source/target document boundaries.
            pub fn $packed<C,F,G>(&self,input:Tensor<$backend,2>,memory:Tensor<$backend,2>,query_layout:&PackedSequenceLayout,memory_layout:&PackedSequenceLayout,
                self_options:PackedAttentionOptions,cross_options:PackedAttentionOptions,communicator:C,mut self_positions:F,mut cross_positions:G)
                -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>),
                    G:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                self.forward_packed_with(input,memory,|index,block,input,memory|block.$block_packed(input,memory,query_layout,memory_layout,
                    self_options,cross_options,communicator.clone(),|query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
            }
        }
    };
}
stack_execution!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],forward,forward_packed,forward,forward_packed);
stack_execution!(B,[B:Backend],forward_inference,forward_packed_inference,forward_inference,forward_packed_inference);

impl<B:Backend> FullyShardedTransformerStack<B> {
    /// Original whole-stack native cached inference with actual absolute per-layer new-token positions.
    pub fn forward_cached_inference<C,F>(&self,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut TransformerKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,mut positions:F) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_cached_inference_with(input,cache,|index,block,input,cache|
            block.forward_cached_inference(input,visible.clone(),cache,masks.clone(),options,communicator.clone(),|query,key,offset|positions(index,query,key,offset)))
    }
}
