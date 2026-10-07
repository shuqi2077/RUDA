use super::*;
use crate::cache::{EncoderDecoderKvCache,ProjectedKvCache};

impl<B:Backend> FullyShardedCrossAttentionBlock<B> {
    /// Prepare actual normalized/positioned source K/V using the original dense/adapted native module.
    /// Native query/output weights are not executed; encoder memory projections run once, not each decode step.
    pub fn prepare_cached_memory_inference<C,F>(&self,memory:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,
        start_position:usize,communicator:C,positions:F) -> Result<ProjectedKvCache<B>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        Ok(self.gather_inference(communicator)?.prepare_cached_memory(memory,visible,start_position,positions))
    }
}

impl<B:Backend> FullyShardedEncoderDecoderLayer<B> {
    /// Original incremental self -> immutable cached source -> FFN stages on actual new target rows.
    /// Selected adapters stay independent; source K/V/norm expressions are not recomputed here.
    pub fn forward_cached_inference<C,F,G>(&self,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut ProjectedKvCache<B>,memory:&ProjectedKvCache<B>,self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,
        cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,communicator:C,self_positions:F,cross_positions:G)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),
            G:FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        Ok(self.gather_inference(communicator)?.forward_cached_with_positions(input,visible,cache,memory,self_masks,self_options,
            cross_masks,cross_options,self_positions,cross_positions))
    }
}

impl<B:Backend> FullyShardedEncoderDecoderStack<B> {
    /// Prepare original actual per-layer encoder K/V and an empty target cache, retaining source validity/offset.
    pub fn prepare_kv_cache_inference<C,F>(&self,memory:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,start_position:usize,
        initial_capacity:usize,communicator:C,mut positions:F) -> Result<EncoderDecoderKvCache<B>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        let memory=self.layers.iter().enumerate().map(|(index,layer)|layer.cross_attention.prepare_cached_memory_inference(
            memory.clone(),visible.clone(),start_position,communicator.clone(),|key,offset|positions(index,key,offset))).collect::<Result<Vec<_>,_>>()?;
        Ok(EncoderDecoderKvCache::new(TransformerKvCache::new(self.layers.len(),initial_capacity),memory))
    }
    /// Execute actual new target rows against immutable original source memory in real layer order.
    /// Finish the stack position only after every layer succeeds; a partial failure requires the caller's
    /// last complete native cache record before replay, not an assumed automatic GPU rollback.
    pub fn forward_cached_inference_with<E,F>(&self,mut input:Tensor<B,3>,cache:&mut EncoderDecoderKvCache<B>,mut layer:F)
        -> Result<Tensor<B,3>,E>
        where F:FnMut(usize,&FullyShardedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,E> {
        let (decoder,memory)=cache.parts_mut();decoder.validate_layers(self.layers.len());
        assert_eq!(memory.len(),self.layers.len(),"sharded paired cache/source layer counts differ");
        let rows=(input.dims()[0],input.dims()[1]);let next=decoder.position().checked_add(rows.1).expect("sharded paired cache position overflow");
        for (index,block) in self.layers.iter().enumerate() {
            input=layer(index,block,input,&mut decoder.layers_mut()[index],&memory[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"sharded cached paired layer changed actual new target rows");
        }
        decoder.finish_chunk(next);Ok(input)
    }
    /// Complete original paired cached stack with separate actual self Q/K and cross Q positional transforms.
    pub fn forward_cached_inference<C,F,G>(&self,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut EncoderDecoderKvCache<B>,
        self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,
        communicator:C,mut self_positions:F,mut cross_positions:G) -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),
            G:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        self.forward_cached_inference_with(input,cache,|index,block,hidden,cache,memory|block.forward_cached_inference(
            hidden,visible.clone(),cache,memory,self_masks.clone(),self_options,cross_masks.clone(),cross_options,communicator.clone(),
            |query,key,offset|self_positions(index,query,key,offset),|query,offset|cross_positions(index,query,offset)))
    }
}

impl<B:Backend> FullyShardedEncoderDecoderModel<B> {
    /// Encode actual source inputs once, then cache each original decoder layer's actual projected source K/V.
    /// Source key positions and physical offset stay independent of subsequent target query positions.
    pub fn prepare_kv_cache_inference_with<C,E,P>(&self,source:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        source_start:usize,initial_capacity:usize,communicator:C,encoder:E,source_positions:P) -> Result<EncoderDecoderKvCache<B>,C::Error>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            P:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        let memory=self.encode_inference_with(source,communicator.clone(),encoder)?;
        self.decoder.prepare_kv_cache_inference(memory,visible,source_start,initial_capacity,communicator,source_positions)
    }
    /// Complete new-target-row backbone/final-norm execution using actual paired cached memory.
    pub fn forward_cached_hidden_inference<C,F,G>(&self,target:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut EncoderDecoderKvCache<B>,self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,
        cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,communicator:C,self_positions:F,cross_positions:G)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),
            G:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        let hidden=self.target_embeddings.forward_inference(target,communicator.clone())?;
        let hidden=self.decoder.forward_cached_inference(hidden,visible,cache,self_masks,self_options,cross_masks,cross_options,communicator.clone(),self_positions,cross_positions)?;
        Ok(if let Some(norm)=&self.decoder_normalization {norm.gather_inference(communicator)?.forward(hidden)} else {hidden})
    }
    /// Complete cached actual target logits without re-encoding source rows or replaying old target tokens.
    pub fn forward_cached_inference<C,F,G>(&self,target:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut EncoderDecoderKvCache<B>,self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,
        cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,communicator:C,self_positions:F,cross_positions:G)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),
            G:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        let hidden=self.forward_cached_hidden_inference(target,visible,cache,self_masks,self_options,cross_masks,cross_options,communicator.clone(),self_positions,cross_positions)?;
        self.head.forward_inference(hidden,communicator)
    }
    /// Actual cached paired next-row candidates, with no sampling/EOS policy or data-rank score reduction.
    pub fn forward_cached_topk_inference<C,F,G>(&self,target:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut EncoderDecoderKvCache<B>,self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,
        cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,communicator:C,self_positions:F,cross_positions:G,
        k:usize,row_visibility:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedTopKSelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),
            G:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        assert!(target.tokens.dims()[1]>0,"native sharded cached paired candidates need actual new target tokens");
        let hidden=self.forward_cached_hidden_inference(target,visible,cache,self_masks,self_options,cross_masks,cross_options,communicator.clone(),self_positions,cross_positions)?;
        self.head.forward_topk_last_inference(hidden,communicator,k,row_visibility)
    }
}
