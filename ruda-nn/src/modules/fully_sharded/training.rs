use super::*;
use crate::loss::CausalCrossEntropyConfig;
use crate::attention::{PackedSequenceLayout,DenseAttentionMask,DenseAttentionOptions,PackedAttentionOptions};
use ruda_autodiff::collective::{CollectiveScope,ScopedTensorCollective,ScopedCollectiveError};
use ruda_model::tensor::{IntDType,TensorData,ElementConversion};

mod paired;

/// Actual original local loss graph, globally effective integer count and detached native global loss metrics.
/// Normalize the local SUM before backward: gathered parameters already SUM/reduce-scatter their derivatives.
/// No second gradient all-reduce, implicit loss scaler, optimizer skip policy or model-family rule is added.
pub struct FullyShardedLoss<B:Backend,S:CheckpointStrategy> {
    /// Actual local sum, completed with only globally required zero collective dependencies.
    pub loss_sum:Tensor<Autodiff<B,S>,1>,
    /// Actual native local I64 effective token/example/element count.
    pub local_count:Tensor<B,1,Int>,
    /// Exact checked global count; not a count accumulated in FP32.
    pub global_count:u64,
    /// Actual FP32 global loss SUM for metrics, without a differentiable replicated-loss reduction.
    pub global_loss_sum:Tensor<B,1>,
}
impl<B:Backend,S:CheckpointStrategy> FullyShardedLoss<B,S> {
    /// Original FP32 integer-count mean, with an empty global count using the same clamp-to-one loss convention.
    pub fn mean(&self) -> Tensor<Autodiff<B,S>,1> {self.normalized(self.global_count)}
    /// Normalize this actual local SUM by a separately coordinated whole accumulation-window count.
    /// This avoids averaging differently sized microbatch means; it does not invent the window's weights.
    pub fn normalized(&self,global_window_count:u64) -> Tensor<Autodiff<B,S>,1> {
        self.loss_sum.clone()/global_window_count.max(1) as f64
    }
    /// Actual globally weighted detached FP32 metric, not the local backward objective multiplied by world size.
    pub fn global_mean(&self) -> Tensor<B,1> {self.global_loss_sum.clone()/self.global_count.max(1) as f64}
}

/// Complete the actual sharded loss graph and integer normalization/metrics on the explicit original data group.
/// Only four exact 16-bit count words per rank are decoded on the host as coordination metadata;
/// all model/loss/gradient arithmetic remains on its original tensor backend and precision path.
pub fn complete_fully_sharded_loss<B,S,C>(scope:&CollectiveScope<B,S>,loss_sum:Tensor<Autodiff<B,S>,1>,
    local_count:Tensor<Autodiff<B,S>,1,Int>,communicator:C) -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
    where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
    let loss_sum=scope.complete(loss_sum,communicator.clone())?;
    if local_count.dims()!=[1] || local_count.device()!=loss_sum.device() {return Err(ScopedCollectiveError::Protocol("actual loss count shape/device differs"));}
    let local_count=local_count.inner().cast(IntDType::I64);let device=loss_sum.device();
    let mask=Tensor::<B,1,Int>::from_data(TensorData::new(alloc::vec![65535_i64],[1]),(&device,DType::I64));
    let mut parts=Vec::with_capacity(4);
    for word in 0..4 {
        let part=local_count.clone().bitwise_right_shift_scalar(((word*16) as i32).elem()).bitwise_and(mask.clone());
        parts.push(part.cast(FloatDType::F32));
    }
    let words=Tensor::<B,1>::cat(parts,0);
    let words=if communicator.world_size()==1 {words} else {Tensor::from_primitive(TensorPrimitive::Float(
        communicator.all_gather_float(words.into_primitive().tensor()).map_err(ScopedCollectiveError::Collective)?))};
    let expected=(communicator.world_size() as usize).checked_mul(4).ok_or(ScopedCollectiveError::Protocol("count gather size overflows"))?;
    if words.dims()!=[expected] || words.dtype()!=DType::F32 || words.device()!=device {return Err(ScopedCollectiveError::Protocol("count transport changed original shape/storage/device"));}
    let words=words.into_data().to_vec::<f32>().map_err(|_|ScopedCollectiveError::Protocol("count metadata cannot be decoded"))?;
    let mut global_count=0u64;
    for rank in 0..communicator.world_size() as usize {
        let mut count=0u64;
        for word in 0..4 {
            let part=words[rank*4+word];
            if !part.is_finite() || part<0.0 || part>65535.0 || part as u32 as f32!=part {return Err(ScopedCollectiveError::Protocol("invalid exact count word"));}
            count|=(part as u64)<<(word*16);
        }
        if count>i64::MAX as u64 {return Err(ScopedCollectiveError::Protocol("a native effective count is negative or overflowed"));}
        global_count=global_count.checked_add(count).ok_or(ScopedCollectiveError::Protocol("global effective count overflows u64"))?;
    }
    let metric=loss_sum.clone().inner().cast(DType::F32);
    let global_loss_sum=if communicator.world_size()==1 {metric} else {Tensor::<B,1>::from_primitive(TensorPrimitive::Float(
        communicator.all_reduce_sum(metric.into_primitive().tensor()).map_err(ScopedCollectiveError::Collective)?))};
    if global_loss_sum.dims()!=[1] || global_loss_sum.dtype()!=DType::F32 || global_loss_sum.device()!=device {return Err(ScopedCollectiveError::Protocol("loss metric transport changed original shape/storage/device"));}
    Ok(FullyShardedLoss {loss_sum,local_count,global_count,global_loss_sum})
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedTransformerModel<Autodiff<B,S>> {
    /// Complete real causal model SUM and exact global effective count using actual original per-layer policies.
    /// The scoped transport is passed into each architecture hook so all actual parameter gathers participate.
    pub fn forward_causal_with<C,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut layer:F)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(input.tokens.dims(),labels.dims(),"actual sharded causal input/label geometry differs");
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden_with(input,transport.clone(),|index,block,hidden|layer(index,block,hidden,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        let loss=criterion.forward_hidden_with_smoothing(hidden,labels,|rows|head.forward(rows),label_smoothing);
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator)
    }
    /// Actual full model causal loss with explicit per-layer Q/K positions and visibility/options.
    pub fn forward_causal_with_positions<C,P>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        masks:DenseAttentionMask<Autodiff<B,S>>,options:DenseAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut positions:P)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_causal_with(input,labels,criterion,label_smoothing,communicator,|index,block,hidden,transport|
            block.forward(hidden,masks.clone(),options,transport,|query,key|positions(index,query,key)))
    }
    /// Actual flat independent-document model/loss, retaining original packed shift/sentinel/smoothing rules.
    pub fn forward_packed_causal_with<C,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut layer:F)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(input.tokens.dims(),labels.dims(),"actual sharded packed causal input/label geometry differs");
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden_with(input,layout,transport.clone(),|index,block,hidden|layer(index,block,hidden,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        let loss=criterion.forward_packed_hidden_with_smoothing(hidden,labels,layout,|rows|head.forward(rows),label_smoothing);
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator)
    }
    /// Original actual packed causal loss and separately caller-declared layer positions/attention policy.
    pub fn forward_packed_causal_with_positions<C,P>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,options:PackedAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut positions:P)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        self.forward_packed_causal_with(input,labels,layout,criterion,label_smoothing,communicator,|index,block,hidden,transport|
            block.forward_packed(hidden,layout,options,transport,|query,key|positions(index,query,key)))
    }
}
