use core::ops::Range;
use alloc::{collections::BTreeSet,vec::Vec};
use ruda_model::{module::{Module,ModuleVisitor,Param,ParamId},tensor::{DType,Tensor,backend::Backend}};
use crate::{Linear,activation::Activation,attention::GroupedQueryAttention,
    transformer::{AdaptedProjection,AdaptedGroupedQueryAttention,DenseFeedForward,AdaptedFeedForward,
        DenseTransformerBlock,AdaptedTransformerBlock,DenseTransformerStack,AdaptedTransformerStack,AdaptedStackLayer}};
use super::{ColumnParallelLinear,RowParallelLinear,TensorParallelGroupedQueryAttention,TensorParallelAdaptedGroupedQueryAttention,
    TensorParallelFeedForward,TensorParallelAdaptedFeedForward,TensorParallelTransformerBlock,TensorParallelAdaptedTransformerBlock,
    TensorParallelTransformerStack,TensorParallelAdaptedTransformerStack,TensorParallelAdaptedStackLayer};

mod decoder;
pub use decoder::*;
mod vocabulary;

/// Explicit feature axis of a loaded full projection, not a guessed rank/world layout.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum TensorParallelProjectionAxis {
    /// Output columns and their local output bias; input features remain replicated.
    Column,
    /// Input rows, retaining the complete original output bias for one post-SUM addition.
    Row,
}

fn check_range(range: &Range<usize>,length: usize) {
    assert!(range.start < range.end && range.end <= length,"parallel projection range must be a nonempty actual feature slice");
}

fn check_storage<B: Backend,const D: usize>(value: &Tensor<B,D>) {
    assert!(!matches!(value.dtype(),DType::QFloat(_)),"packed quantized projections must be loaded as explicit native local shards");
}

fn partition_linear<B: Backend>(mut layer: Linear<B>,axis: TensorParallelProjectionAxis,range: Range<usize>) -> Linear<B> {
    let shape = layer.weight.val().dims();let dimension = if axis == TensorParallelProjectionAxis::Column {1} else {0};
    check_range(&range,shape[dimension]);
    if let Some(bias) = &layer.bias {assert_eq!(bias.val().dims(),[shape[1]],"full projection output bias width differs");}
    if range == (0..shape[dimension]) {return layer;}
    check_storage(&layer.weight.val());
    if axis == TensorParallelProjectionAxis::Column {if let Some(bias) = &layer.bias {check_storage(&bias.val());}}
    layer.weight = layer.weight.map(|value| {
        let trainable = value.is_require_grad();
        value.slice_dim(dimension,range.clone()).detach().set_require_grad(trainable)
    });
    if axis == TensorParallelProjectionAxis::Column {
        layer.bias = layer.bias.map(|bias|bias.map(|value| {
            let trainable = value.is_require_grad();
            value.slice_dim(0,range).detach().set_require_grad(trainable)
        }));
    }
    layer
}

impl<B: Backend> ColumnParallelLinear<B> {
    /// Slice actual loaded floating weights/bias without initialization or dtype changes.
    /// Parameter IDs/mappers/flags survive; this does not convert global optimizer state.
    /// Packed quantized bases use from_shard instead of implicit full dequantization.
    pub fn from_full(layer: Linear<B>,columns: Range<usize>) -> Self {
        Self::from_shard(partition_linear(layer,TensorParallelProjectionAxis::Column,columns))
    }
}

impl<B: Backend> RowParallelLinear<B> {
    /// Slice actual loaded floating input rows and retain the original replicated output bias.
    pub fn from_full(layer: Linear<B>,rows: Range<usize>) -> Self {
        Self::from_shard(partition_linear(layer,TensorParallelProjectionAxis::Row,rows))
    }
}

/// Partition actual selected dense/LoRA projection storage, retaining scale/dropout/flags.
/// Column slices base/B outputs and keeps A; row slices base/A inputs and keeps B.
/// No adapter merge, rank initialization, quantized repacking or optimizer conversion occurs.
pub fn partition_parallel_projection<B: Backend>(layer: AdaptedProjection<B>,axis: TensorParallelProjectionAxis,range: Range<usize>) -> AdaptedProjection<B> {
    match layer {
        AdaptedProjection::Dense(layer)=>AdaptedProjection::Dense(partition_linear(layer,axis,range)),
        AdaptedProjection::LoRA(mut layer)=>{
            let [input,output] = layer.base.weight.val().dims();let rank = layer.adapter_a.weight.val().dims()[1];
            assert_eq!(layer.adapter_a.weight.val().dims(),[input,rank],"full adapter A geometry differs");
            assert_eq!(layer.adapter_b.weight.val().dims(),[rank,output],"full adapter B geometry differs");
            assert!(rank > 0 && layer.scale.is_finite(),"invalid full adapter rank/scale");
            layer.base = partition_linear(layer.base,axis,range.clone());
            if axis == TensorParallelProjectionAxis::Column {layer.adapter_b = partition_linear(layer.adapter_b,axis,range);}
            else {layer.adapter_a = partition_linear(layer.adapter_a,axis,range);}
            AdaptedProjection::LoRA(layer)
        }
    }
}

/// Explicit corresponding contiguous query/KV head ranges in an actual loaded full attention.
/// Repeated KV ranges are allowed across ranks, but their replica communicator stays caller-owned.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct TensorParallelHeadPartition {
    /// Actual global query-head range, not a padded logical head count.
    pub query: Range<usize>,
    /// Actual global KV-head range corresponding to these queries under original GQA mapping.
    pub key_value: Range<usize>,
}

impl TensorParallelHeadPartition {
    /// Explicit head ranges; geometry is checked against actual weights before slicing.
    pub fn new(query: Range<usize>,key_value: Range<usize>) -> Self {Self {query,key_value}}

    /// Validate every local query/KV grouping against the original global GQA mapping.
    /// Supports uneven valid head partitions and query subsets sharing a replicated single KV head.
    pub fn validate(&self,query_heads: usize,kv_heads: usize,head_dimension: usize) {
        assert!(kv_heads > 0 && query_heads > 0 && head_dimension > 0 && query_heads.is_multiple_of(kv_heads),"invalid global GQA geometry");
        check_range(&self.query,query_heads);check_range(&self.key_value,kv_heads);
        let queries = self.query.end-self.query.start;let keys = self.key_value.end-self.key_value.start;
        assert!(queries.is_multiple_of(keys),"local query/KV head counts do not form native GQA groups");
        let global_group = query_heads/kv_heads;let local_group = queries/keys;
        for key in 0..keys {
            let first = self.query.start+key*local_group;let last = first+local_group-1;
            assert!(first/global_group == self.key_value.start+key && last/global_group == self.key_value.start+key,
                "local GQA partition changes the original query-to-KV head mapping");
        }
        self.query.end.checked_mul(head_dimension).expect("parallel full query feature range overflow");
        self.key_value.end.checked_mul(head_dimension).expect("parallel full KV feature range overflow");
    }

    fn query_features(&self,width: usize) -> Range<usize> {self.query.start*width..self.query.end*width}
    fn kv_features(&self,width: usize) -> Range<usize> {self.key_value.start*width..self.key_value.end*width}
}

fn check_independent<B: Backend>(layers: &[&Linear<B>]) {
    let mut ids: Vec<ParamId> = Vec::new();
    for layer in layers {
        for id in core::iter::once(&layer.weight.id).chain(layer.bias.iter().map(|bias|&bias.id)) {
            assert!(!ids.contains(id),"tied full projection parameters require explicit tie-aware local shards");
            ids.push(id.clone());
        }
    }
}

fn adapted_linears<'a,B: Backend>(layers: &[&'a AdaptedProjection<B>]) -> Vec<&'a Linear<B>> {
    let mut values = Vec::new();
    for layer in layers {match layer {AdaptedProjection::Dense(layer)=>values.push(layer),
        AdaptedProjection::LoRA(layer)=>{values.push(&layer.base);values.push(&layer.adapter_a);values.push(&layer.adapter_b);}}}
    values
}

impl<B: Backend> TensorParallelGroupedQueryAttention<B> {
    /// Partition actual loaded Q/K/V columns and matching output rows, with bias added once.
    /// Full tied projections and packed quantized bases require explicit from_shard loading.
    pub fn from_full(base: GroupedQueryAttention<B>,partition: &TensorParallelHeadPartition) -> Self {
        let mut base = Self::from_shard(base).local;
        partition.validate(base.query_heads,base.kv_heads,base.head_dimension);
        check_independent(&[&base.query,&base.key,&base.value,&base.output]);
        let query = partition.query_features(base.head_dimension);let key_value = partition.kv_features(base.head_dimension);
        base.query = ColumnParallelLinear::from_full(base.query,query.clone()).local;
        base.key = ColumnParallelLinear::from_full(base.key,key_value.clone()).local;
        base.value = ColumnParallelLinear::from_full(base.value,key_value).local;
        base.output = RowParallelLinear::from_full(base.output,query).local;
        base.query_heads = partition.query.end-partition.query.start;base.kv_heads = partition.key_value.end-partition.key_value.start;
        Self::from_shard(base)
    }
}

impl<B: Backend> TensorParallelAdaptedGroupedQueryAttention<B> {
    /// Partition actual loaded native adapter heads without changing A/B ranks, dtypes or scales.
    pub fn from_full(base: AdaptedGroupedQueryAttention<B>,partition: &TensorParallelHeadPartition) -> Self {
        let mut base = Self::from_shard(base).local;
        partition.validate(base.query_heads,base.kv_heads,base.head_dimension);
        check_independent(&adapted_linears(&[&base.query,&base.key,&base.value,&base.output]));
        let query = partition.query_features(base.head_dimension);let key_value = partition.kv_features(base.head_dimension);
        base.query = partition_parallel_projection(base.query,TensorParallelProjectionAxis::Column,query.clone());
        base.key = partition_parallel_projection(base.key,TensorParallelProjectionAxis::Column,key_value.clone());
        base.value = partition_parallel_projection(base.value,TensorParallelProjectionAxis::Column,key_value);
        base.output = partition_parallel_projection(base.output,TensorParallelProjectionAxis::Row,query);
        base.query_heads = partition.query.end-partition.query.start;base.kv_heads = partition.key_value.end-partition.key_value.start;
        Self::from_shard(base)
    }
}

impl<B: Backend> TensorParallelFeedForward<B> {
    /// Partition actual loaded up/gate columns and down rows, including original projection biases.
    /// The callback consumes the actual original activation and supplies its explicit local partition;
    /// a globally channel-mixing activation is not silently replaced with a pointwise one.
    pub fn from_full<F>(base: DenseFeedForward<B>,features: Range<usize>,activation: F) -> Self
        where F: FnOnce(Activation<B>,Range<usize>)->Activation<B> {
        let base = Self::from_shard(base).local;
        let mut layers = alloc::vec![&base.up,&base.down];if let Some(gate) = &base.gate {layers.push(gate);}
        check_independent(&layers);check_range(&features,base.up.weight.val().dims()[1]);
        Self::from_shard(DenseFeedForward {up:ColumnParallelLinear::from_full(base.up,features.clone()).local,
            gate:base.gate.map(|gate|ColumnParallelLinear::from_full(gate,features.clone()).local),
            down:RowParallelLinear::from_full(base.down,features.clone()).local,activation:activation(base.activation,features),dropout:base.dropout})
    }
}

impl<B: Backend> TensorParallelAdaptedFeedForward<B> {
    /// Partition actual dense/LoRA FFN weights and keep the original explicitly partitioned activation.
    pub fn from_full<F>(base: AdaptedFeedForward<B>,features: Range<usize>,activation: F) -> Self
        where F: FnOnce(Activation<B>,Range<usize>)->Activation<B> {
        let base = Self::from_shard(base).local;
        let mut layers = alloc::vec![&base.up,&base.down];if let Some(gate) = &base.gate {layers.push(gate);}
        check_independent(&adapted_linears(&layers));
        let width = match &base.up {AdaptedProjection::Dense(layer)=>layer.weight.val().dims()[1],AdaptedProjection::LoRA(layer)=>layer.base.weight.val().dims()[1]};
        check_range(&features,width);
        Self::from_shard(AdaptedFeedForward {up:partition_parallel_projection(base.up,TensorParallelProjectionAxis::Column,features.clone()),
            gate:base.gate.map(|gate|partition_parallel_projection(gate,TensorParallelProjectionAxis::Column,features.clone())),
            down:partition_parallel_projection(base.down,TensorParallelProjectionAxis::Row,features.clone()),
            activation:activation(base.activation,features),dropout:base.dropout})
    }
}

/// Explicit native block partition with independently specified attention and FFN ranges.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct TensorParallelTransformerPartition {
    /// Actual corresponding query/KV head ranges from the original full block.
    pub attention: TensorParallelHeadPartition,
    /// Actual intermediate feature range, shared by up/gate outputs and down inputs.
    pub feed_forward: Range<usize>,
}

impl TensorParallelTransformerPartition {
    /// Caller-declared native ranges; does not infer communicator membership or model architecture.
    pub fn new(attention: TensorParallelHeadPartition,feed_forward: Range<usize>) -> Self {Self {attention,feed_forward}}
}

struct IndependentParameters {ids: BTreeSet<ParamId>}
impl<B: Backend> ModuleVisitor<B> for IndependentParameters {
    fn visit_float<const D: usize>(&mut self,param: &Param<Tensor<B,D>>) {
        assert!(self.ids.insert(param.id),"tied full model parameters require explicit tie-aware local shards");
    }
}

fn check_independent_module<B: Backend,M: Module<B>>(module: &M) {module.visit(&mut IndependentParameters {ids:BTreeSet::new()});}

impl<B: Backend> TensorParallelTransformerBlock<B> {
    /// Partition actual loaded floating block projections, retaining original norms/residual order.
    /// The callback partitions the actual activation; no optimizer state or tied global model is converted.
    pub fn from_full_block<F>(block: DenseTransformerBlock<B>,partition: &TensorParallelTransformerPartition,activation: F) -> Self
        where F: FnOnce(Activation<B>,Range<usize>)->Activation<B> {
        check_independent_module(&block);
        Self::from_sharded_block(DenseTransformerBlock {attention:TensorParallelGroupedQueryAttention::from_full(block.attention,&partition.attention).local,
            feed_forward:TensorParallelFeedForward::from_full(block.feed_forward,partition.feed_forward.clone(),activation).local,
            attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,
            residual_dropout:block.residual_dropout,norm_first:block.norm_first})
    }
}

impl<B: Backend> TensorParallelAdaptedTransformerBlock<B> {
    /// Partition original selected adapters/base projections without reselection or reinitialization.
    pub fn from_full_block<F>(block: AdaptedTransformerBlock<B>,partition: &TensorParallelTransformerPartition,activation: F) -> Self
        where F: FnOnce(Activation<B>,Range<usize>)->Activation<B> {
        check_independent_module(&block);
        Self::from_sharded_block(AdaptedTransformerBlock {attention:TensorParallelAdaptedGroupedQueryAttention::from_full(block.attention,&partition.attention).local,
            feed_forward:TensorParallelAdaptedFeedForward::from_full(block.feed_forward,partition.feed_forward.clone(),activation).local,
            attention_norm:block.attention_norm,feed_forward_norm:block.feed_forward_norm,
            residual_dropout:block.residual_dropout,norm_first:block.norm_first})
    }
}

impl<B: Backend> TensorParallelAdaptedStackLayer<B> {
    /// Partition the actual layer kind, with no blanket freezing or new adapters on dense layers.
    pub fn from_full_layer<F>(layer: AdaptedStackLayer<B>,partition: &TensorParallelTransformerPartition,activation: F) -> Self
        where F: FnOnce(Activation<B>,Range<usize>)->Activation<B> {
        match layer {AdaptedStackLayer::Dense(block)=>Self::Dense(TensorParallelTransformerBlock::from_full_block(block,partition,activation)),
            AdaptedStackLayer::Adapted(block)=>Self::Adapted(TensorParallelAdaptedTransformerBlock::from_full_block(block,partition,activation))}
    }
}

impl<B: Backend> TensorParallelTransformerStack<B> {
    /// Partition every actual native layer from loaded full floating weights using its explicit plan.
    /// Preserves layer order/normalization and passes the original activation to each local partition callback.
    pub fn from_full_stack<F>(stack: DenseTransformerStack<B>,partitions: &[TensorParallelTransformerPartition],mut activation: F) -> Self
        where F: FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        assert_eq!(partitions.len(),stack.blocks.len(),"native parallel plans must cover every actual layer exactly once");
        check_independent_module(&stack);
        Self::new(stack.blocks.into_iter().zip(partitions).enumerate().map(|(index,(block,partition))|
            TensorParallelTransformerBlock::from_full_block(block,partition,|module,features|activation(index,module,features))).collect())
    }
}

impl<B: Backend> TensorParallelAdaptedTransformerStack<B> {
    /// Partition original selected/unselected layers, retaining all actual A/B dtypes and scales.
    /// Complete global optimizer state and tied/packed-quantized checkpoint conversion remain explicit separate operations.
    pub fn from_full_stack<F>(stack: AdaptedTransformerStack<B>,partitions: &[TensorParallelTransformerPartition],mut activation: F) -> Self
        where F: FnMut(usize,Activation<B>,Range<usize>)->Activation<B> {
        assert_eq!(partitions.len(),stack.layers.len(),"native adapter parallel plans must cover every actual layer exactly once");
        check_independent_module(&stack);
        Self::new(stack.layers.into_iter().zip(partitions).enumerate().map(|(index,(layer,partition))|
            TensorParallelAdaptedStackLayer::from_full_layer(layer,partition,|module,features|activation(index,module,features))).collect())
    }
}
