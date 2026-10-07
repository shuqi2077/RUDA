use alloc::vec::Vec;
use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Module,Tensor};
use super::TensorParallelAdaptedTransformerBlock;
use super::super::TensorParallelTransformerBlock;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},cache::{TransformerKvCache,ProjectedKvCache},
    transformer::{AdaptedStackLayer,AdaptedTransformerStack,StackAdapterRecord}};
use ruda_model::{record::RecorderError,tensor::Bool};

/// Actual selected/unselected layer kinds, with no blanket base freezing.
#[derive(Module,Debug)]
pub enum TensorParallelAdaptedStackLayer<B: Backend> {
    /// Original locally partitioned dense layer.
    Dense(TensorParallelTransformerBlock<B>),
    /// Actual selected projection adapters on their original local base.
    Adapted(TensorParallelAdaptedTransformerBlock<B>),
}

impl<B: Backend> TensorParallelAdaptedStackLayer<B> {
    /// Preserve exact native layer selection and all original parameter identities.
    pub fn from_sharded_layer(layer: AdaptedStackLayer<B>) -> Self {
        match layer {AdaptedStackLayer::Dense(block)=>Self::Dense(TensorParallelTransformerBlock::from_sharded_block(block)),
            AdaptedStackLayer::Adapted(block)=>Self::Adapted(TensorParallelAdaptedTransformerBlock::from_sharded_block(block))}
    }
    /// Return original native local containers without merging adapters or collecting full weights.
    pub fn into_local_layer(self) -> AdaptedStackLayer<B> {
        match self {Self::Dense(block)=>AdaptedStackLayer::Dense(block.into_local_block()),Self::Adapted(block)=>AdaptedStackLayer::Adapted(block.into_local_block())}
    }

    /// Native non-autodiff inference on either actual layer kind and unchanged positions.
    pub fn forward_inference<C,F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F)
        -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_inference(input,masks,options,communicator,positions),
            Self::Adapted(block)=>block.forward_inference(input,masks,options,communicator,positions)}
    }

    /// Native cached inference with the selected layer's actual local K/V heads.
    pub fn forward_cached_inference<C,F>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_cached_inference(input,visible,cache,masks,options,communicator,positions),
            Self::Adapted(block)=>block.forward_cached_inference(input,visible,cache,masks,options,communicator,positions)}
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedStackLayer<Autodiff<B,S>> {
    /// Original per-layer dense/adapter graph, with explicit local-head positions/groups.
    pub fn forward<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        match self {Self::Dense(block)=>block.forward_with_positions(input,masks,options,groups,positions),
            Self::Adapted(block)=>block.forward(input,masks,options,groups,positions)}
    }

    /// Detached-history cached path using this actual layer's original selected adapters.
    pub fn forward_cached<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        match self {Self::Dense(block)=>block.forward_cached(input,visible,cache,masks,options,groups,positions),
            Self::Adapted(block)=>block.forward_cached(input,visible,cache,masks,options,groups,positions)}
    }
}

/// Ordered actual selected/unselected model partitions with native adapter-only records.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedTransformerStack<B: Backend> {
    /// Original layer order and explicit projection selections.
    pub layers: Vec<TensorParallelAdaptedStackLayer<B>>,
}

impl<B: Backend> TensorParallelAdaptedTransformerStack<B> {
    /// Connect actual already wrapped local layers without changing any trainable flags.
    pub fn new(layers: Vec<TensorParallelAdaptedStackLayer<B>>) -> Self {Self {layers}}
    /// Wrap each original layer choice; adapters are not initialized or selected again.
    pub fn from_sharded_stack(stack: AdaptedTransformerStack<B>) -> Self {
        Self::new(stack.layers.into_iter().map(TensorParallelAdaptedStackLayer::from_sharded_layer).collect())
    }
    /// Restore the original local module containers for existing save/load integrations.
    pub fn into_local_stack(self) -> AdaptedTransformerStack<B> {
        AdaptedTransformerStack::new(self.layers.into_iter().map(TensorParallelAdaptedStackLayer::into_local_layer).collect())
    }
    /// Original A/B-only native record, excluding base weights and preserving exact layer paths.
    pub fn adapter_record(&self,base_id: &str) -> Result<StackAdapterRecord<B>,RecorderError> {
        self.clone().into_local_stack().adapter_record(base_id)
    }
    /// Restore the existing native A/B-only record against this exact local frozen base.
    /// Native record validation retains its layer/target/dtype/scale and trainable-flag contracts.
    pub fn restore_adapter_record(self,record: StackAdapterRecord<B>,base_id: &str) -> Result<Self,RecorderError> {
        record.restore_into(self.into_local_stack(),base_id).map(Self::from_sharded_stack)
    }
    /// Actual layer-count cache metadata, with no full global-head allocations.
    pub fn new_kv_cache(&self,initial_capacity: usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.layers.len(),initial_capacity)}

    /// Native inference callback with actual per-layer positions, masks and groups.
    pub fn forward_inference_with<E,F>(&self,mut input: Tensor<B,3>,mut layer: F) -> Result<Tensor<B,3>,E>
        where F: FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>)->Result<Tensor<B,3>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input)?;}
        Ok(input)
    }

    /// Native inference advances the stack boundary only after all actual local caches succeed.
    /// A failed callback can leave partial layer updates; a completed cache record is the recovery point.
    pub fn forward_cached_inference_with<E,F>(&self,mut input: Tensor<B,3>,cache: &mut TransformerKvCache<B>,mut layer: F)
        -> Result<Tensor<B,3>,E>
        where F: FnMut(usize,&TensorParallelAdaptedStackLayer<B>,Tensor<B,3>,&mut ProjectedKvCache<B>)->Result<Tensor<B,3>,E> {
        cache.validate_layers(self.layers.len());
        let rows = (input.dims()[0],input.dims()[1]);
        let next = cache.position().checked_add(rows.1).expect("adapted native parallel stack position overflow");
        for (index,block) in self.layers.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"adapted native cached layer changed actual chunk rows");
        }
        cache.finish_chunk(next);
        Ok(input)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedTransformerStack<Autodiff<B,S>> {
    /// Original layer sequence with no guessed architecture/head/normalization additions.
    pub fn forward_with<E,F>(&self,mut input: Tensor<Autodiff<B,S>,3>,mut layer: F) -> Result<Tensor<Autodiff<B,S>,3>,E>
        where F: FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input)?;}
        Ok(input)
    }

    /// Commit the actual decoder boundary only after all selected/unselected layers complete.
    pub fn forward_cached_with<E,F>(&self,mut input: Tensor<Autodiff<B,S>,3>,cache: &mut TransformerKvCache<Autodiff<B,S>>,mut layer: F)
        -> Result<Tensor<Autodiff<B,S>,3>,E>
        where F: FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,&mut ProjectedKvCache<Autodiff<B,S>>)
            -> Result<Tensor<Autodiff<B,S>,3>,E> {
        cache.validate_layers(self.layers.len());
        let rows = (input.dims()[0],input.dims()[1]);let next = cache.position().checked_add(rows.1).expect("adapted parallel stack position overflow");
        for (index,block) in self.layers.iter().enumerate() {
            input = layer(index,block,input,&mut cache.layers_mut()[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"adapted parallel cached layer changed actual chunk rows");
        }
        cache.finish_chunk(next);
        Ok(input)
    }
}
