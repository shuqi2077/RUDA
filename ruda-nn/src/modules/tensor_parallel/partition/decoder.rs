use super::{Backend,Activation,Range,TensorParallelHeadPartition,TensorParallelTransformerPartition,check_independent_module,
    TensorParallelGroupedQueryAttention,TensorParallelAdaptedGroupedQueryAttention,TensorParallelTransformerBlock,TensorParallelAdaptedStackLayer};
use super::super::{TensorParallelCrossAttentionBlock,TensorParallelEncoderDecoderLayer,TensorParallelEncoderDecoderStack,
    TensorParallelAdaptedCrossAttentionBlock,TensorParallelAdaptedDecoderCrossAttention,TensorParallelAdaptedEncoderDecoderLayer,TensorParallelAdaptedEncoderDecoderStack};
use crate::transformer::{DenseCrossAttentionBlock,AdaptedCrossAttentionBlock,DecoderCrossAttention,DenseEncoderDecoderLayer,
    AdaptedEncoderDecoderLayer,DenseEncoderDecoderStack,AdaptedEncoderDecoderStack};

/// Explicit independently declared native self/FFN and encoder-memory projection partitions.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct TensorParallelEncoderDecoderPartition {
    /// Actual original self-attention and FFN ranges for this decoder layer.
    pub backbone: TensorParallelTransformerPartition,
    /// Actual original cross query/KV head ranges, independent of self-attention geometry.
    pub cross_attention: TensorParallelHeadPartition,
}

impl TensorParallelEncoderDecoderPartition {
    /// Caller-owned source/target head placement; no communicator groups or model names are inferred.
    pub fn new(backbone: TensorParallelTransformerPartition,cross_attention: TensorParallelHeadPartition) -> Self {Self {backbone,cross_attention}}
}

impl<B: Backend> TensorParallelCrossAttentionBlock<B> {
    /// Partition actual loaded floating cross projections while retaining asymmetric query/source widths.
    pub fn from_full_block(block: DenseCrossAttentionBlock<B>,partition: &TensorParallelHeadPartition) -> Self {
        check_independent_module(&block);
        Self::from_sharded_block(DenseCrossAttentionBlock {attention:TensorParallelGroupedQueryAttention::from_full(block.attention,partition).local,
            query_norm:block.query_norm,memory_norm:block.memory_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
    }
}

impl<B: Backend> TensorParallelAdaptedCrossAttentionBlock<B> {
    /// Partition actual loaded selected cross adapters, preserving original A/B ranks, storage and scales.
    pub fn from_full_block(block: AdaptedCrossAttentionBlock<B>,partition: &TensorParallelHeadPartition) -> Self {
        check_independent_module(&block);
        Self::from_sharded_block(AdaptedCrossAttentionBlock {attention:TensorParallelAdaptedGroupedQueryAttention::from_full(block.attention,partition).local,
            query_norm:block.query_norm,memory_norm:block.memory_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first})
    }
}

impl<B: Backend> TensorParallelAdaptedDecoderCrossAttention<B> {
    /// Partition the actual original dense/adapted source stage without reselecting projection targets.
    pub fn from_full_stage(stage: DecoderCrossAttention<B>,partition: &TensorParallelHeadPartition) -> Self {
        match stage {DecoderCrossAttention::Dense(block)=>Self::Dense(TensorParallelCrossAttentionBlock::from_full_block(block,partition)),
            DecoderCrossAttention::Adapted(block)=>Self::Adapted(TensorParallelAdaptedCrossAttentionBlock::from_full_block(block,partition))}
    }
}

impl<B: Backend> TensorParallelEncoderDecoderLayer<B> {
    /// Partition actual native floating self/cross/FFN weights with independent source/target head plans.
    pub fn from_full_layer<F>(layer: DenseEncoderDecoderLayer<B>,partition: &TensorParallelEncoderDecoderPartition,activation: F) -> Self
        where F: FnOnce(Activation<B>,Range<usize>)->Activation<B> {
        check_independent_module(&layer);
        Self::from_sharded_layer(DenseEncoderDecoderLayer {backbone:TensorParallelTransformerBlock::from_full_block(layer.backbone,&partition.backbone,activation).into_local_block(),
            cross_attention:TensorParallelCrossAttentionBlock::from_full_block(layer.cross_attention,&partition.cross_attention).into_local_block()})
    }
}

impl<B: Backend> TensorParallelAdaptedEncoderDecoderLayer<B> {
    /// Partition original selected/unselected stages, passing the actual activation to its explicit local partition.
    pub fn from_full_layer<F>(layer: AdaptedEncoderDecoderLayer<B>,partition: &TensorParallelEncoderDecoderPartition,activation: F) -> Self
        where F: FnOnce(Activation<B>,Range<usize>)->Activation<B> {
        check_independent_module(&layer);
        Self::from_sharded_layer(AdaptedEncoderDecoderLayer {backbone:TensorParallelAdaptedStackLayer::from_full_layer(layer.backbone,&partition.backbone,activation).into_local_layer(),
            cross_attention:TensorParallelAdaptedDecoderCrossAttention::from_full_stage(layer.cross_attention,&partition.cross_attention).into_local_stage()})
    }
}

impl<B: Backend> TensorParallelEncoderDecoderStack<B> {
    /// Partition every original floating source/target layer in order, with exact per-layer plans.
    /// Tied/packed-quantized global checkpoints use explicit local loading instead of implicit conversion.
    pub fn from_full_stack<F>(stack: DenseEncoderDecoderStack<B>,partitions: &[TensorParallelEncoderDecoderPartition],mut activation: F) -> Self
        where F: FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        assert_eq!(partitions.len(),stack.layers.len(),"parallel source/target plans must cover every actual layer exactly once");
        check_independent_module(&stack);
        Self::new(stack.layers.into_iter().zip(partitions).enumerate().map(|(index,(layer,partition))|
            TensorParallelEncoderDecoderLayer::from_full_layer(layer,partition,|module,features|activation(index,module,features))).collect())
    }
}

impl<B: Backend> TensorParallelAdaptedEncoderDecoderStack<B> {
    /// Partition actual independently selected native self/cross/FFN adapters without rebuilding their base.
    /// Optimizer states and shared global parameter placement are not silently reinterpreted as local states.
    pub fn from_full_stack<F>(stack: AdaptedEncoderDecoderStack<B>,partitions: &[TensorParallelEncoderDecoderPartition],mut activation: F) -> Self
        where F: FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        assert_eq!(partitions.len(),stack.layers.len(),"adapted source/target plans must cover every actual original layer exactly once");
        check_independent_module(&stack);
        Self::new(stack.layers.into_iter().zip(partitions).enumerate().map(|(index,(layer,partition))|
            TensorParallelAdaptedEncoderDecoderLayer::from_full_layer(layer,partition,|module,features|activation(index,module,features))).collect())
    }
}
