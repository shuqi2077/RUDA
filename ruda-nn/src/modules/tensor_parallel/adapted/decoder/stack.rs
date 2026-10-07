use alloc::vec::Vec;
use super::{Autodiff,Backend,CheckpointStrategy,Module,Tensor,ProjectedKvCache,TensorParallelAdaptedDecoderCrossAttention,TensorParallelAdaptedEncoderDecoderLayer};
use crate::{cache::{TransformerKvCache,EncoderDecoderKvCache},transformer::{AdaptedEncoderDecoderStack,EncoderDecoderAdapterRecord}};
use ruda_model::record::RecorderError;

/// Ordered actual selected/unselected parallel source/target layers and native A/B-only state.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedEncoderDecoderStack<B: Backend> {
    /// Original layer order and exact independently selected native adapter targets.
    pub layers: Vec<TensorParallelAdaptedEncoderDecoderLayer<B>>,
}

fn cached<B: Backend,E,F>(mut input: Tensor<B,3>,cache: &mut EncoderDecoderKvCache<B>,layers: &[TensorParallelAdaptedEncoderDecoderLayer<B>],mut layer: F)
    -> Result<Tensor<B,3>,E>
    where F: FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,E> {
    assert!(cache.is_consistent(),"adapted parallel source/decoder cache pairing is inconsistent");
    cache.decoder().validate_layers(layers.len());
    let rows = (input.dims()[0],input.dims()[1]);let next = cache.position().checked_add(rows.1).expect("adapted parallel decoder position overflow");
    let (decoder,memory) = cache.parts_mut();
    for (index,block) in layers.iter().enumerate() {
        input = layer(index,block,input,&mut decoder.layers_mut()[index],&memory[index])?;
        assert_eq!((input.dims()[0],input.dims()[1]),rows,"adapted parallel decoder changed actual target chunk rows");
    }
    decoder.finish_chunk(next);
    Ok(input)
}

impl<B: Backend> TensorParallelAdaptedEncoderDecoderStack<B> {
    /// Connect actual prepared local layers without reinitializing parameters or selecting targets.
    pub fn new(layers: Vec<TensorParallelAdaptedEncoderDecoderLayer<B>>) -> Self {Self {layers}}
    /// Preserve the actual original dense/adapted stage choices and parameter identities.
    pub fn from_sharded_stack(stack: AdaptedEncoderDecoderStack<B>) -> Self {
        Self::new(stack.layers.into_iter().map(TensorParallelAdaptedEncoderDecoderLayer::from_sharded_layer).collect())
    }
    /// Restore original native local containers for existing model and adapter state integrations.
    pub fn into_local_stack(self) -> AdaptedEncoderDecoderStack<B> {
        AdaptedEncoderDecoderStack::new(self.layers.into_iter().map(TensorParallelAdaptedEncoderDecoderLayer::into_local_layer).collect())
    }
    /// Capture actual self/cross/FFN A/B-only records with the original frozen-base contract.
    pub fn adapter_record(&self,base_id: &str) -> Result<EncoderDecoderAdapterRecord<B>,RecorderError> {
        self.clone().into_local_stack().adapter_record(base_id)
    }
    /// Restore the native exact selected A/B record without adapter merging or global weight gathers.
    pub fn restore_adapter_record(self,record: EncoderDecoderAdapterRecord<B>,base_id: &str) -> Result<Self,RecorderError> {
        record.restore_into(self.into_local_stack(),base_id).map(Self::from_sharded_stack)
    }
    /// Metadata-only actual local decoder layer count, without allocating global head tensors.
    pub fn new_decoder_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.layers.len(),initial_capacity)}

    /// Native actual dense or packed layer sequence with explicitly supplied per-layer policies.
    pub fn forward_inference_with<E,F,const D: usize>(&self,mut input: Tensor<B,D>,memory: Tensor<B,D>,mut layer: F) -> Result<Tensor<B,D>,E>
        where F: FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,D>,Tensor<B,D>)->Result<Tensor<B,D>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }

    /// Actual native layer-specific projected source state paired with empty local decoder history.
    pub fn prepare_kv_cache_inference_with<E,F>(&self,memory: Tensor<B,3>,initial_capacity: usize,mut layer: F) -> Result<EncoderDecoderKvCache<B>,E>
        where F: FnMut(usize,&TensorParallelAdaptedDecoderCrossAttention<B>,Tensor<B,3>)->Result<ProjectedKvCache<B>,E> {
        let mut prepared = Vec::with_capacity(self.layers.len());
        for (index,block) in self.layers.iter().enumerate() {prepared.push(layer(index,&block.cross_attention,memory.clone())?);}
        Ok(EncoderDecoderKvCache::new(self.new_decoder_cache(initial_capacity),prepared))
    }

    /// Native completed-chunk continuation on paired actual adapted source/decoder caches.
    /// Errors can leave partial decoder updates; restore a completed native cache record.
    pub fn forward_cached_inference_with<E,F>(&self,input: Tensor<B,3>,cache: &mut EncoderDecoderKvCache<B>,layer: F) -> Result<Tensor<B,3>,E>
        where F: FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,E> {
        cached(input,cache,&self.layers,layer)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedEncoderDecoderStack<Autodiff<B,S>> {
    /// Original selected stage sequence, with actual dense/packed source rows and native gradients.
    pub fn forward_with<E,F,const D: usize>(&self,mut input: Tensor<Autodiff<B,S>,D>,memory: Tensor<Autodiff<B,S>,D>,mut layer: F) -> Result<Tensor<Autodiff<B,S>,D>,E>
        where F: FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,D>,Tensor<Autodiff<B,S>,D>)->Result<Tensor<Autodiff<B,S>,D>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }

    /// Per-layer actual source projection/norm/positions, retaining original adapter storage and scale.
    pub fn prepare_kv_cache_with<E,F>(&self,memory: Tensor<Autodiff<B,S>,3>,initial_capacity: usize,mut layer: F) -> Result<EncoderDecoderKvCache<Autodiff<B,S>>,E>
        where F: FnMut(usize,&TensorParallelAdaptedDecoderCrossAttention<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<ProjectedKvCache<Autodiff<B,S>>,E> {
        let mut prepared = Vec::with_capacity(self.layers.len());
        for (index,block) in self.layers.iter().enumerate() {prepared.push(layer(index,&block.cross_attention,memory.clone())?);}
        Ok(EncoderDecoderKvCache::new(self.new_decoder_cache(initial_capacity),prepared))
    }

    /// Inference-only detached actual source/decoder history, committing only complete target chunks.
    pub fn forward_cached_with<E,F>(&self,input: Tensor<Autodiff<B,S>,3>,cache: &mut EncoderDecoderKvCache<Autodiff<B,S>>,layer: F) -> Result<Tensor<Autodiff<B,S>,3>,E>
        where F: FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,&mut ProjectedKvCache<Autodiff<B,S>>,&ProjectedKvCache<Autodiff<B,S>>)
            -> Result<Tensor<Autodiff<B,S>,3>,E> {
        cached(input,cache,&self.layers,layer)
    }
}
