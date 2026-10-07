use ruda_model::tensor::{Bool,Tensor,backend::Backend};
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},cache::{ProjectedKvCache,TransformerKvCache}};
use super::{DenseTransformerBlock,AdaptedTransformerBlock,AdaptedStackLayer,DenseTransformerStack,AdaptedTransformerStack};
use super::dense::residual_branch;

impl<B: Backend> DenseTransformerBlock<B> {
    /// Native incremental inference: append actual new K/V, then run FFN only on new tokens.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with_positions(input,new_visible,cache,masks,options,|query,key,_|(query,key))
    }

    /// Actual new Q/K positional transforms with unchanged residual/norm/FFN order.
    pub fn forward_cached_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_feed_forward(self.forward_cached_attention_with_positions(input,new_visible,cache,masks,options,positions))
    }

    /// Incremental self-attention/residual/norm only, before a decoder memory stage.
    pub fn forward_cached_attention_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source|
            self.attention.forward_cached_with_positions(source,new_visible,cache,masks,options,positions))
    }
}

impl<B: Backend> AdaptedTransformerBlock<B> {
    /// Incremental inference with original dense weights and actual selected native adapters.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with_positions(input,new_visible,cache,masks,options,|query,key,_|(query,key))
    }

    /// Native adapter inference on all new tokens, without merging or recomputing old FFNs.
    pub fn forward_cached_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_feed_forward(self.forward_cached_attention_with_positions(input,new_visible,cache,masks,options,positions))
    }

    /// Actual adapted self-attention stage alone, for incremental encoder-decoder inference.
    pub fn forward_cached_attention_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source|
            self.attention.forward_cached_with_positions(source,new_visible,cache,masks,options,positions))
    }
}

impl<B: Backend> AdaptedStackLayer<B> {
    /// Incremental inference through the actual original/adapted layer type.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        match self {
            Self::Dense(block)=>block.forward_cached(input,new_visible,cache,masks,options),
            Self::Adapted(block)=>block.forward_cached(input,new_visible,cache,masks,options),
        }
    }

    /// Caller-owned new Q/K position transforms on either actual layer type.
    pub fn forward_cached_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        match self {
            Self::Dense(block)=>block.forward_cached_with_positions(input,new_visible,cache,masks,options,positions),
            Self::Adapted(block)=>block.forward_cached_with_positions(input,new_visible,cache,masks,options,positions),
        }
    }

    /// Actual cached self-attention stage before inserting an encoder-memory stage.
    pub fn forward_cached_attention_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        match self {
            Self::Dense(block)=>block.forward_cached_attention_with_positions(input,new_visible,cache,masks,options,positions),
            Self::Adapted(block)=>block.forward_cached_attention_with_positions(input,new_visible,cache,masks,options,positions),
        }
    }
}

impl<B: Backend> DenseTransformerStack<B> {
    /// Allocate no tensors; prepare caches for the exact actual native block count.
    pub fn new_kv_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {
        TransformerKvCache::new(self.blocks.len(),initial_capacity)
    }

    /// Incremental inference with explicit shared masks/visibility/alignment across layers.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut TransformerKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with(input,cache,|_,block,input,cache|
            block.forward_cached(input,new_visible.clone(),cache,masks.clone(),options))
    }

    /// Actual per-layer positions/masks/bias; each callback must append the complete new chunk.
    pub fn forward_cached_with<F>(&self,mut input: Tensor<B,3>,cache: &mut TransformerKvCache<B>,mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&DenseTransformerBlock<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Tensor<B,3> {
        cache.validate_layers(self.blocks.len());
        let shape = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(shape.1).expect("cached transformer position overflow");
        for (index,block) in self.blocks.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index]);
            assert_eq!((input.dims()[0],input.dims()[1]),shape,"cached native block changed actual input rows");
        }
        cache.finish_chunk(next);
        input
    }
}

impl<B: Backend> AdaptedTransformerStack<B> {
    /// Prepare exactly the selected/unselected layer count without allocating cache payloads.
    pub fn new_kv_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {
        TransformerKvCache::new(self.layers.len(),initial_capacity)
    }

    /// Incremental adapter inference with exact actual per-layer base and A/B parameters.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut TransformerKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with(input,cache,|_,block,input,cache|
            block.forward_cached(input,new_visible.clone(),cache,masks.clone(),options))
    }

    /// Caller-supplied layer positions/visibility/bias, preserving actual complete chunks.
    pub fn forward_cached_with<F>(&self,mut input: Tensor<B,3>,cache: &mut TransformerKvCache<B>,mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&AdaptedStackLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Tensor<B,3> {
        cache.validate_layers(self.layers.len());
        let shape = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(shape.1).expect("cached adapter stack position overflow");
        for (index,block) in self.layers.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index]);
            assert_eq!((input.dims()[0],input.dims()[1]),shape,"cached adapted block changed actual input rows");
        }
        cache.finish_chunk(next);
        input
    }
}
