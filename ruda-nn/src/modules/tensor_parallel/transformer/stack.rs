use alloc::vec::Vec;
use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,TensorParallelTransformerBlock,
    Tensor,DenseAttentionMask,DenseAttentionOptions,Module};
use ruda_model::tensor::Bool;
use crate::{cache::{TransformerKvCache,ProjectedKvCache},transformer::DenseTransformerStack};

/// Actual ordered parallel blocks, without guessed embeddings, head or final normalization.
#[derive(Module,Debug)]
pub struct TensorParallelTransformerStack<B: Backend> {
    /// Caller-loaded local model partitions in original layer order.
    pub blocks: Vec<TensorParallelTransformerBlock<B>>,
}

impl<B: Backend> TensorParallelTransformerStack<B> {
    /// Connect actual already partitioned blocks and retain all original parameter IDs.
    pub fn new(blocks: Vec<TensorParallelTransformerBlock<B>>) -> Self {Self {blocks}}
    /// Wrap existing local native blocks without rebuilding normalization or projection state.
    pub fn from_sharded_stack(stack: DenseTransformerStack<B>) -> Self {
        Self::new(stack.blocks.into_iter().map(TensorParallelTransformerBlock::from_sharded_block).collect())
    }
    /// Allocate only cache metadata for actual local KV layers; no full global heads are created.
    pub fn new_kv_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.blocks.len(),initial_capacity)}

    /// Native inference with explicit per-layer local positions/masks/options.
    pub fn forward_inference_with<E,F>(&self,mut input: Tensor<B,3>,mut layer: F) -> Result<Tensor<B,3>,E>
        where F: FnMut(usize,&TensorParallelTransformerBlock<B>,Tensor<B,3>)->Result<Tensor<B,3>,E> {
        for (index,block) in self.blocks.iter().enumerate() {input = layer(index,block,input)?;}
        Ok(input)
    }

    /// Native cached inference commits the stack position only after every actual layer succeeds.
    /// Transport/model failures can leave advanced local layers; restore a completed record.
    pub fn forward_cached_inference_with<E,F>(&self,mut input: Tensor<B,3>,cache: &mut TransformerKvCache<B>,mut layer: F)
        -> Result<Tensor<B,3>,E>
        where F: FnMut(usize,&TensorParallelTransformerBlock<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Result<Tensor<B,3>,E> {
        cache.validate_layers(self.blocks.len());
        let rows = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(rows.1).expect("parallel cached stack position overflow");
        for (index,block) in self.blocks.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"parallel cached layer changed actual chunk rows");
        }
        cache.finish_chunk(next);
        Ok(input)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelTransformerStack<Autodiff<B,S>> {
    /// Shared explicit attention options/groups; each local block uses its original weights.
    pub fn forward<C,K>(&self,mut input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        for block in &self.blocks {input = block.forward(input,masks.clone(),options,groups)?;}
        Ok(input)
    }

    /// Fallible per-layer positions, masks and transport choices remain architecture-owned.
    pub fn forward_with<E,F>(&self,mut input: Tensor<Autodiff<B,S>,3>,mut layer: F) -> Result<Tensor<Autodiff<B,S>,3>,E>
        where F: FnMut(usize,&TensorParallelTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,E> {
        for (index,block) in self.blocks.iter().enumerate() {input = layer(index,block,input)?;}
        Ok(input)
    }

    /// Detached local history, unchanged layer order and original per-layer parameters.
    pub fn forward_cached<C,K>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut TransformerKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.forward_cached_with(input,cache,|_,block,input,cache|
            block.forward_cached(input,visible.clone(),cache,masks.clone(),options,groups,|query,key,_|(query,key)))
    }

    /// Actual per-layer cached callback; only complete chunks advance the stack boundary.
    pub fn forward_cached_with<E,F>(&self,mut input: Tensor<Autodiff<B,S>,3>,cache: &mut TransformerKvCache<Autodiff<B,S>>,mut layer: F)
        -> Result<Tensor<Autodiff<B,S>,3>,E>
        where F: FnMut(usize,&TensorParallelTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,&mut ProjectedKvCache<Autodiff<B,S>>)
            -> Result<Tensor<Autodiff<B,S>,3>,E> {
        cache.validate_layers(self.blocks.len());
        let rows = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(rows.1).expect("parallel cached stack position overflow");
        for (index,block) in self.blocks.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"parallel cached block changed actual input rows");
        }
        cache.finish_chunk(next);
        Ok(input)
    }
}
