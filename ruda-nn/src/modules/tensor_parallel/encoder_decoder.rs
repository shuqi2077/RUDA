use alloc::vec::Vec;
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy};
use ruda_model::{module::Module,tensor::{Bool,Tensor,backend::Backend}};
use crate::{Dropout,attention::{DenseAttentionMask,DenseAttentionOptions},cache::{ProjectedKvCache,TransformerKvCache,EncoderDecoderKvCache},
    transformer::{DenseCrossAttentionBlock,DenseTransformerNorm,DenseEncoderDecoderLayer,DenseEncoderDecoderStack}};
use super::{AttentionParallelGroups,TensorParallelGroupedQueryAttention,TensorParallelTransformerBlock,BroadcastTensorCollective};

mod native;
mod packed;

/// Original residual cross-attention with actual local Q/K/V and output-row shards.
/// Query and memory widths, independent norms and pre/post order are unchanged.
#[derive(Module,Debug)]
pub struct TensorParallelCrossAttentionBlock<B: Backend> {
    /// Local query/source head projections and full-residual output reduction.
    pub attention: TensorParallelGroupedQueryAttention<B>,
    /// Original replicated query norm.
    pub query_norm: DenseTransformerNorm<B>,
    /// Original optional replicated source-memory norm.
    pub memory_norm: Option<DenseTransformerNorm<B>>,
    /// Original branch dropout; values must match residual replicas across ranks.
    pub residual_dropout: Dropout,
    /// Exactly the original pre/post query normalization choice.
    pub norm_first: bool,
}

impl<B: Backend> TensorParallelCrossAttentionBlock<B> {
    /// Preserve actual asymmetric widths and all original parameter identities.
    pub fn from_sharded_block(block: DenseCrossAttentionBlock<B>) -> Self {
        assert_eq!(block.query_norm.width(),block.attention.query.weight.val().dims()[0],"parallel cross query norm width differs");
        if let Some(norm) = &block.memory_norm {assert_eq!(norm.width(),block.attention.key.weight.val().dims()[0],"parallel cross memory norm width differs");}
        Self {attention:TensorParallelGroupedQueryAttention::from_shard(block.attention),query_norm:block.query_norm,
            memory_norm:block.memory_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }

    /// Return the original local cross-attention container without collecting or copying weights.
    pub fn into_local_block(self) -> DenseCrossAttentionBlock<B> {
        DenseCrossAttentionBlock {attention:self.attention.local,query_norm:self.query_norm,memory_norm:self.memory_norm,
            residual_dropout:self.residual_dropout,norm_first:self.norm_first}
    }

    fn source(&self,input: Tensor<B,3>) -> Tensor<B,3> {
        if self.norm_first {self.query_norm.forward(input)} else {input}
    }
    fn memory(&self,input: Tensor<B,3>) -> Tensor<B,3> {
        if let Some(norm) = &self.memory_norm {norm.forward(input)} else {input}
    }
    fn finish(&self,input: Tensor<B,3>,branch: Tensor<B,3>) -> Tensor<B,3> {
        let output = input+self.residual_dropout.forward(branch);
        if self.norm_first {output} else {self.query_norm.forward(output)}
    }

    /// Native inference with independent actual query/source position transforms.
    pub fn forward_inference<C,F>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,
        communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let source = self.source(input.clone());let memory = self.memory(memory);
        let (query,key,value) = self.attention.local.project(source,memory.clone(),memory);
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"parallel cross positions changed actual query/source heads");
        Ok(self.finish(input,self.attention.forward_projected_inference(query,key,value,masks,options,communicator)?))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelCrossAttentionBlock<Autodiff<B,S>> {
    /// Query and memory gradients are SUMmed over their actual head contributions;
    /// optional KV replica groups SUM only corresponding K/V parameter gradients.
    pub fn forward<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,memory: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,
        options: DenseAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let query = self.attention.project_query(self.source(input.clone()),groups.heads.clone())?;
        let (key,value) = self.attention.project_memory(self.memory(memory),groups)?;
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"parallel cross positions changed actual query/memory head geometry");
        Ok(self.finish(input,self.attention.forward_projected(query,key,value,masks,options,groups.heads.clone())?))
    }

    /// Prepare this rank's immutable encoder memory once. Optional source normalization
    /// and positioned K/V are the actual original operations, not decoder-position guesses.
    pub fn prepare_memory<C,K,F>(&self,memory: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<ProjectedKvCache<Autodiff<B,S>>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>)->Tensor<Autodiff<B,S>,4> {
        let (key,value) = self.attention.project_memory(self.memory(memory),groups)?;
        let geometry = key.dims();let key = positions(key);
        assert_eq!(key.dims(),geometry,"parallel prepared source positions changed actual key geometry");
        Ok(ProjectedKvCache::from_projected(key,value,visible,0))
    }

    /// Project only new queries/output on immutable already-prepared encoder K/V.
    /// Position callbacks receive the actual decoder offset, independent of source length.
    pub fn forward_cached<C,F>(&self,input: Tensor<Autodiff<B,S>,3>,memory: &ProjectedKvCache<Autodiff<B,S>>,
        masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,communicator: C,position: usize,positions: F)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<Autodiff<B,S>,4>,usize)->Tensor<Autodiff<B,S>,4> {
        let query = self.attention.project_query(self.source(input.clone()),communicator.clone())?;
        let geometry = query.dims();let query = positions(query,position);
        assert_eq!(query.dims(),geometry,"parallel cached cross positions changed actual query geometry");
        Ok(self.finish(input,self.attention.forward_cached_memory(query,memory,masks,options,communicator)?))
    }
}

/// Original self-attention -> source cross-attention -> FFN order, all on local shards.
#[derive(Module,Debug)]
pub struct TensorParallelEncoderDecoderLayer<B: Backend> {
    /// Original decoder backbone; its FFN runs only after actual cross-attention.
    pub backbone: TensorParallelTransformerBlock<B>,
    /// Original asymmetric source-memory block.
    pub cross_attention: TensorParallelCrossAttentionBlock<B>,
}

impl<B: Backend> TensorParallelEncoderDecoderLayer<B> {
    /// Wrap actual already partitioned layers without adding/removing any model stage.
    pub fn from_sharded_layer(layer: DenseEncoderDecoderLayer<B>) -> Self {
        assert_eq!(layer.backbone.attention.query.weight.val().dims()[0],layer.cross_attention.attention.query.weight.val().dims()[0],
            "parallel decoder/cross residual widths differ");
        Self {backbone:TensorParallelTransformerBlock::from_sharded_block(layer.backbone),
            cross_attention:TensorParallelCrossAttentionBlock::from_sharded_block(layer.cross_attention)}
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelEncoderDecoderLayer<Autodiff<B,S>> {
    /// Actual source/target inputs and masks with explicitly independent self/cross groups.
    pub fn forward<C,K,L,F,G>(&self,input: Tensor<Autodiff<B,S>,3>,memory: Tensor<Autodiff<B,S>,3>,
        self_masks: DenseAttentionMask<Autodiff<B,S>>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<Autodiff<B,S>>,memory_options: DenseAttentionOptions,
        self_groups: &AttentionParallelGroups<C,K>,memory_groups: &AttentionParallelGroups<C,L>,self_positions: F,cross_positions: G)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,L: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            G: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let hidden = self.backbone.forward_attention_with_positions(input,self_masks,self_options,self_groups,self_positions)?;
        let hidden = self.cross_attention.forward(hidden,memory,memory_masks,memory_options,memory_groups,cross_positions)?;
        self.backbone.forward_feed_forward(hidden,self_groups.heads.clone())
    }

    /// Native decoder-prefix cache plus prepared immutable local encoder memory.
    pub fn forward_cached<C,K,F,G>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        decoder: &mut ProjectedKvCache<Autodiff<B,S>>,memory: &ProjectedKvCache<Autodiff<B,S>>,
        self_masks: DenseAttentionMask<Autodiff<B,S>>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<Autodiff<B,S>>,memory_options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,cross_communicator: C,self_positions: F,cross_positions: G) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            G: FnOnce(Tensor<Autodiff<B,S>,4>,usize)->Tensor<Autodiff<B,S>,4> {
        let position = decoder.position();
        let hidden = self.backbone.forward_cached_attention(input,visible,decoder,self_masks,self_options,groups,self_positions)?;
        let hidden = self.cross_attention.forward_cached(hidden,memory,memory_masks,memory_options,cross_communicator,position,cross_positions)?;
        self.backbone.forward_feed_forward(hidden,groups.heads.clone())
    }
}

/// Ordered actual parallel encoder-decoder layers and paired native source/history caches.
#[derive(Module,Debug)]
pub struct TensorParallelEncoderDecoderStack<B: Backend> {
    /// The original architecture's layer order with unchanged local parameter IDs.
    pub layers: Vec<TensorParallelEncoderDecoderLayer<B>>,
}

impl<B: Backend> TensorParallelEncoderDecoderStack<B> {
    /// Connect already wrapped local decoder layers without guessing a model family.
    pub fn new(layers: Vec<TensorParallelEncoderDecoderLayer<B>>) -> Self {Self {layers}}
    /// Preserve every actual source/query norm, head geometry and residual choice.
    pub fn from_sharded_stack(stack: DenseEncoderDecoderStack<B>) -> Self {
        Self::new(stack.layers.into_iter().map(TensorParallelEncoderDecoderLayer::from_sharded_layer).collect())
    }
    /// Exactly this stack's local self-attention layer count, with no tensor allocations.
    pub fn new_decoder_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.layers.len(),initial_capacity)}
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelEncoderDecoderStack<Autodiff<B,S>> {
    /// Actual per-layer self/cross positions, layouts and groups through a fallible callback.
    pub fn forward_with<E,F>(&self,mut input: Tensor<Autodiff<B,S>,3>,memory: Tensor<Autodiff<B,S>,3>,mut layer: F)
        -> Result<Tensor<Autodiff<B,S>,3>,E>
        where F: FnMut(usize,&TensorParallelEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)
            -> Result<Tensor<Autodiff<B,S>,3>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }

    /// Prepare actual layer-specific source projection/norm/positions and pair with decoder history.
    pub fn prepare_kv_cache_with<E,F>(&self,memory: Tensor<Autodiff<B,S>,3>,initial_capacity: usize,mut layer: F)
        -> Result<EncoderDecoderKvCache<Autodiff<B,S>>,E>
        where F: FnMut(usize,&TensorParallelCrossAttentionBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)
            -> Result<ProjectedKvCache<Autodiff<B,S>>,E> {
        let mut prepared = Vec::with_capacity(self.layers.len());
        for (index,block) in self.layers.iter().enumerate() {prepared.push(layer(index,&block.cross_attention,memory.clone())?);}
        Ok(EncoderDecoderKvCache::new(self.new_decoder_cache(initial_capacity),prepared))
    }

    /// Paired source/decoder continuation, committing only complete actual target chunks.
    pub fn forward_cached_with<E,F>(&self,mut input: Tensor<Autodiff<B,S>,3>,cache: &mut EncoderDecoderKvCache<Autodiff<B,S>>,mut layer: F)
        -> Result<Tensor<Autodiff<B,S>,3>,E>
        where F: FnMut(usize,&TensorParallelEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,
            &mut ProjectedKvCache<Autodiff<B,S>>,&ProjectedKvCache<Autodiff<B,S>>)->Result<Tensor<Autodiff<B,S>,3>,E> {
        assert!(cache.is_consistent(),"parallel source/decoder cache pairing is inconsistent");
        cache.decoder().validate_layers(self.layers.len());
        let rows = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(rows.1).expect("parallel encoder-decoder position overflow");
        let (decoder,memory) = cache.parts_mut();
        for (index,block) in self.layers.iter().enumerate() {
            input = layer(index,block,input,&mut decoder.layers_mut()[index],&memory[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"parallel decoder changed actual target chunk rows");
        }
        decoder.finish_chunk(next);
        Ok(input)
    }
}
