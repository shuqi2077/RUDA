use super::{Backend,BroadcastTensorCollective,Tensor,Bool,DenseAttentionMask,DenseAttentionOptions,ProjectedKvCache,EncoderDecoderKvCache,Vec};
use super::{TensorParallelCrossAttentionBlock,TensorParallelEncoderDecoderLayer,TensorParallelEncoderDecoderStack};

impl<B: Backend> TensorParallelCrossAttentionBlock<B> {
    /// Native immutable source preparation with the original memory norm and key positions.
    pub fn prepare_memory_inference<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,positions: F) -> ProjectedKvCache<B>
        where F: FnOnce(Tensor<B,4>)->Tensor<B,4> {
        let (key,value) = self.attention.project_memory_inference(self.memory(memory));
        let geometry = key.dims();let key = positions(key);
        assert_eq!(key.dims(),geometry,"native parallel prepared memory positions changed actual keys");
        ProjectedKvCache::from_projected(key,value,visible,0)
    }

    /// Native query-only continuation over retained positioned source K/V.
    pub fn forward_cached_inference<C,F>(&self,input: Tensor<B,3>,memory: &ProjectedKvCache<B>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,communicator: C,position: usize,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let query = self.attention.project_query_inference(self.source(input.clone()));
        let geometry = query.dims();let query = positions(query,position);
        assert_eq!(query.dims(),geometry,"native parallel cached cross positions changed actual query heads");
        Ok(self.finish(input,self.attention.forward_cached_memory_inference(query,memory,masks,options,communicator)?))
    }
}

impl<B: Backend> TensorParallelEncoderDecoderLayer<B> {
    /// Original native self -> cross -> FFN order with explicitly independent transports.
    pub fn forward_inference<C,D,F,G>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions,self_communicator: C,cross_communicator: D,
        self_positions: F,cross_positions: G) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,D: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),G: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = self.backbone.forward_attention_inference(input,self_masks,self_options,self_communicator.clone(),self_positions)?;
        let hidden = self.cross_attention.forward_inference(hidden,memory,memory_masks,memory_options,cross_communicator,cross_positions)?;
        self.backbone.forward_feed_forward_inference(hidden,self_communicator)
    }

    /// Actual native decoder chunks reuse each layer's immutable projected encoder memory.
    pub fn forward_cached_inference<C,D,F,G>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,
        decoder: &mut ProjectedKvCache<B>,memory: &ProjectedKvCache<B>,self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions,self_communicator: C,cross_communicator: D,
        self_positions: F,cross_positions: G) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,D: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),G: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let position = decoder.position();
        let hidden = self.backbone.forward_cached_attention_inference(input,visible,decoder,self_masks,self_options,self_communicator.clone(),self_positions)?;
        let hidden = self.cross_attention.forward_cached_inference(hidden,memory,memory_masks,memory_options,cross_communicator,position,cross_positions)?;
        self.backbone.forward_feed_forward_inference(hidden,self_communicator)
    }
}

impl<B: Backend> TensorParallelEncoderDecoderStack<B> {
    /// Native inference over the exact local layer sequence and original source rows.
    pub fn forward_inference_with<E,F>(&self,mut input: Tensor<B,3>,memory: Tensor<B,3>,mut layer: F) -> Result<Tensor<B,3>,E>
        where F: FnMut(usize,&TensorParallelEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }

    /// Actual per-layer native prepared source state paired with empty local decoder history.
    pub fn prepare_kv_cache_inference_with<E,F>(&self,memory: Tensor<B,3>,initial_capacity: usize,mut layer: F) -> Result<EncoderDecoderKvCache<B>,E>
        where F: FnMut(usize,&TensorParallelCrossAttentionBlock<B>,Tensor<B,3>)->Result<ProjectedKvCache<B>,E> {
        let mut prepared = Vec::with_capacity(self.layers.len());
        for (index,block) in self.layers.iter().enumerate() {prepared.push(layer(index,&block.cross_attention,memory.clone())?);}
        Ok(EncoderDecoderKvCache::new(self.new_decoder_cache(initial_capacity),prepared))
    }

    /// Native paired continuation commits only a complete chunk on every actual layer.
    /// Errors do not roll back partially appended decoder layers; restore a complete record.
    pub fn forward_cached_inference_with<E,F>(&self,mut input: Tensor<B,3>,cache: &mut EncoderDecoderKvCache<B>,mut layer: F)
        -> Result<Tensor<B,3>,E>
        where F: FnMut(usize,&TensorParallelEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,E> {
        assert!(cache.is_consistent(),"native parallel source/decoder cache pairing is inconsistent");
        cache.decoder().validate_layers(self.layers.len());
        let rows = (input.dims()[0],input.dims()[1]);let next = cache.position().checked_add(rows.1).expect("native parallel decoder position overflow");
        let (decoder,memory) = cache.parts_mut();
        for (index,block) in self.layers.iter().enumerate() {
            input = layer(index,block,input,&mut decoder.layers_mut()[index],&memory[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"native parallel decoder changed actual chunk rows");
        }
        decoder.finish_chunk(next);
        Ok(input)
    }
}
