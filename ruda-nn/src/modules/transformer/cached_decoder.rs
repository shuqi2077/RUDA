use alloc::vec::Vec;
use ruda_model::tensor::{Bool,Tensor,backend::Backend};
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},cache::{ProjectedKvCache,TransformerKvCache,EncoderDecoderKvCache}};
use super::{DenseCrossAttentionBlock,AdaptedCrossAttentionBlock,DecoderCrossAttention,DenseEncoderDecoderLayer,
    AdaptedEncoderDecoderLayer,DenseEncoderDecoderStack,AdaptedEncoderDecoderStack};
use super::dense::residual_branch;

impl<B: Backend> DenseCrossAttentionBlock<B> {
    /// Normalize/project actual encoder memory once using the original K/V weights.
    /// The caller owns its key positions, validity and absolute physical slot offset.
    pub fn prepare_cached_memory<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,
        start_position: usize,positions: F) -> ProjectedKvCache<B>
    where F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        let (key,value) = self.attention.project_key_value(memory.clone(),memory);
        let shape = key.dims();
        let key = positions(key,start_position);
        assert_eq!(key.dims(),shape,"prepared cross key positions changed actual memory geometry");
        ProjectedKvCache::from_projected(key,value,visible,start_position)
    }

    /// Cross-attend actual prepared encoder memory without repeating its norm/K/V projections.
    pub fn forward_cached_memory<F>(&self,input: Tensor<B,3>,memory: &ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,query_position: usize,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let query = self.attention.project_query(source);
            let shape = query.dims();
            let query = positions(query,query_position);
            assert_eq!(query.dims(),shape,"cached cross query positions changed actual new geometry");
            self.attention.forward_cached_memory(query,memory,masks,options)
        })
    }
}

impl<B: Backend> AdaptedCrossAttentionBlock<B> {
    /// Prepare original/adapted encoder K/V once; do not run an unrelated query projection.
    pub fn prepare_cached_memory<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,
        start_position: usize,positions: F) -> ProjectedKvCache<B>
    where F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        let (key,value) = self.attention.project_key_value(memory.clone(),memory);
        let shape = key.dims();
        let key = positions(key,start_position);
        assert_eq!(key.dims(),shape,"prepared adapted cross key positions changed actual memory geometry");
        ProjectedKvCache::from_projected(key,value,visible,start_position)
    }

    /// Native selected query/output adapters attend immutable already-positioned memory.
    pub fn forward_cached_memory<F>(&self,input: Tensor<B,3>,memory: &ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,query_position: usize,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let query = self.attention.project_query(source);
            let shape = query.dims();
            let query = positions(query,query_position);
            assert_eq!(query.dims(),shape,"cached adapted cross positions changed actual query geometry");
            self.attention.forward_cached_memory(query,memory,masks,options)
        })
    }
}

impl<B: Backend> DecoderCrossAttention<B> {
    /// Prepare actual original/adapted memory parameters without changing any module state.
    pub fn prepare_cached_memory<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,
        start_position: usize,positions: F) -> ProjectedKvCache<B>
    where F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        match self {
            Self::Dense(block)=>block.prepare_cached_memory(memory,visible,start_position,positions),
            Self::Adapted(block)=>block.prepare_cached_memory(memory,visible,start_position,positions),
        }
    }

    /// Actual dense/adapted query/residual/norm stage with immutable prepared memory.
    pub fn forward_cached_memory<F>(&self,input: Tensor<B,3>,memory: &ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,query_position: usize,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        match self {
            Self::Dense(block)=>block.forward_cached_memory(input,memory,masks,options,query_position,positions),
            Self::Adapted(block)=>block.forward_cached_memory(input,memory,masks,options,query_position,positions),
        }
    }
}

impl<B: Backend> DenseEncoderDecoderLayer<B> {
    /// Incremental self-attention, immutable memory attention, then FFN only on new tokens.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        memory: &ProjectedKvCache<B>,self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        cross_masks: DenseAttentionMask<B>,cross_options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with_positions(input,new_visible,cache,memory,self_masks,self_options,cross_masks,cross_options,
            |query,key,_|(query,key),|query,_|query)
    }

    /// Distinct caller-owned new self Q/K and cross Q position transforms.
    pub fn forward_cached_with_positions<F,G>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,memory: &ProjectedKvCache<B>,self_masks: DenseAttentionMask<B>,
        self_options: DenseAttentionOptions,cross_masks: DenseAttentionMask<B>,cross_options: DenseAttentionOptions,
        self_positions: F,cross_positions: G) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),
        G: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let position = cache.position();
        let hidden = self.backbone.forward_cached_attention_with_positions(input,new_visible,cache,self_masks,self_options,self_positions);
        let hidden = self.cross_attention.forward_cached_memory(hidden,memory,cross_masks,cross_options,position,cross_positions);
        self.backbone.forward_feed_forward(hidden)
    }
}

impl<B: Backend> AdaptedEncoderDecoderLayer<B> {
    /// Incremental inference using actual self/cross/FFN adapters without merging them.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        memory: &ProjectedKvCache<B>,self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        cross_masks: DenseAttentionMask<B>,cross_options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with_positions(input,new_visible,cache,memory,self_masks,self_options,cross_masks,cross_options,
            |query,key,_|(query,key),|query,_|query)
    }

    /// Only actual new self keys are appended; encoder keys retain their original transform.
    pub fn forward_cached_with_positions<F,G>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,memory: &ProjectedKvCache<B>,self_masks: DenseAttentionMask<B>,
        self_options: DenseAttentionOptions,cross_masks: DenseAttentionMask<B>,cross_options: DenseAttentionOptions,
        self_positions: F,cross_positions: G) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),
        G: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let position = cache.position();
        let hidden = self.backbone.forward_cached_attention_with_positions(input,new_visible,cache,self_masks,self_options,self_positions);
        let hidden = self.cross_attention.forward_cached_memory(hidden,memory,cross_masks,cross_options,position,cross_positions);
        self.backbone.forward_feed_forward(hidden)
    }
}

impl<B: Backend> DenseEncoderDecoderStack<B> {
    /// Prepare actual paired source/decoder state for native incremental generation.
    pub fn prepare_kv_cache<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,start_position: usize,
        initial_capacity: usize,positions: F) -> EncoderDecoderKvCache<B>
    where F: FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        EncoderDecoderKvCache::new(self.new_kv_cache(initial_capacity),self.prepare_cached_memory(memory,visible,start_position,positions))
    }

    /// Execute actual paired native state, retaining both sides for beam reorder and resume.
    pub fn forward_cached_pair(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut EncoderDecoderKvCache<B>,
        self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,cross_masks: DenseAttentionMask<B>,
        cross_options: DenseAttentionOptions) -> Tensor<B,3> {
        let (decoder,memory) = cache.parts_mut();
        self.forward_cached(input,new_visible,decoder,memory,self_masks,self_options,cross_masks,cross_options)
    }

    /// Prepare actual immutable encoder K/V independently for every original decoder layer.
    pub fn prepare_cached_memory<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,
        start_position: usize,mut positions: F) -> Vec<ProjectedKvCache<B>>
    where F: FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        self.layers.iter().enumerate().map(|(index,layer)|layer.cross_attention.prepare_cached_memory(
            memory.clone(),visible.clone(),start_position,|key,position|positions(index,key,position))).collect()
    }

    /// Prepare exactly the actual decoder self-attention layer count without allocation.
    pub fn new_kv_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {
        TransformerKvCache::new(self.layers.len(),initial_capacity)
    }

    /// Native new-token inference with prepared memory and explicit self/cross masks/options.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut TransformerKvCache<B>,
        memory: &[ProjectedKvCache<B>],self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        cross_masks: DenseAttentionMask<B>,cross_options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with(input,cache,memory,|_,layer,input,cache,memory|
            layer.forward_cached(input,new_visible.clone(),cache,memory,self_masks.clone(),self_options,cross_masks.clone(),cross_options))
    }

    /// Per-layer actual self/cross positions and score biases with unchanged architecture order.
    pub fn forward_cached_with<F>(&self,mut input: Tensor<B,3>,cache: &mut TransformerKvCache<B>,
        memory: &[ProjectedKvCache<B>],mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&DenseEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Tensor<B,3> {
        cache.validate_layers(self.layers.len());
        assert_eq!(memory.len(),self.layers.len(),"actual cached encoder memory and decoder layer counts differ");
        let shape = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(shape.1).expect("cached decoder position overflow");
        for (index,block) in self.layers.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index],&memory[index]);
            assert_eq!((input.dims()[0],input.dims()[1]),shape,"cached decoder layer changed actual new rows");
        }
        cache.finish_chunk(next);
        input
    }
}

impl<B: Backend> AdaptedEncoderDecoderStack<B> {
    /// Prepare paired actual dense/adapted source and decoder caches without merging weights.
    pub fn prepare_kv_cache<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,start_position: usize,
        initial_capacity: usize,positions: F) -> EncoderDecoderKvCache<B>
    where F: FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        EncoderDecoderKvCache::new(self.new_kv_cache(initial_capacity),self.prepare_cached_memory(memory,visible,start_position,positions))
    }

    /// Native adapter inference against paired source/decoder state without implicit policies.
    pub fn forward_cached_pair(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut EncoderDecoderKvCache<B>,
        self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,cross_masks: DenseAttentionMask<B>,
        cross_options: DenseAttentionOptions) -> Tensor<B,3> {
        let (decoder,memory) = cache.parts_mut();
        self.forward_cached(input,new_visible,decoder,memory,self_masks,self_options,cross_masks,cross_options)
    }

    /// Prepare actual original/adapted encoder K/V once for every decoder layer.
    pub fn prepare_cached_memory<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,
        start_position: usize,mut positions: F) -> Vec<ProjectedKvCache<B>>
    where F: FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        self.layers.iter().enumerate().map(|(index,layer)|layer.cross_attention.prepare_cached_memory(
            memory.clone(),visible.clone(),start_position,|key,position|positions(index,key,position))).collect()
    }

    /// Prepare self-attention caches for actual selected/unselected decoder layers.
    pub fn new_kv_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {
        TransformerKvCache::new(self.layers.len(),initial_capacity)
    }

    /// Reuse real encoder history while executing native new-token adapter inference.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut TransformerKvCache<B>,
        memory: &[ProjectedKvCache<B>],self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        cross_masks: DenseAttentionMask<B>,cross_options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with(input,cache,memory,|_,layer,input,cache,memory|
            layer.forward_cached(input,new_visible.clone(),cache,memory,self_masks.clone(),self_options,cross_masks.clone(),cross_options))
    }

    /// Caller-owned per-layer positions/visibility/bias on the actual adapter-bearing stages.
    pub fn forward_cached_with<F>(&self,mut input: Tensor<B,3>,cache: &mut TransformerKvCache<B>,
        memory: &[ProjectedKvCache<B>],mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&AdaptedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Tensor<B,3> {
        cache.validate_layers(self.layers.len());
        assert_eq!(memory.len(),self.layers.len(),"actual adapted memory cache and decoder layer counts differ");
        let shape = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(shape.1).expect("cached adapted decoder position overflow");
        for (index,block) in self.layers.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index],&memory[index]);
            assert_eq!((input.dims()[0],input.dims()[1]),shape,"cached adapted decoder changed actual new rows");
        }
        cache.finish_chunk(next);
        input
    }
}
