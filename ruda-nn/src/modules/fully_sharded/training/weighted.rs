use super::*;
use crate::{pool::SequencePooling,transformer::SequenceHeadOutput};
use ruda_model::tensor::Bool;

/// Original globally weighted native objective, including fractional class/sample weights.
/// Loss derivatives remain local sums whose sharded parameter gathers SUM/reduce-scatter them.
pub struct FullyShardedWeightedLoss<B:Backend,S:CheckpointStrategy> {
    /// Actual local sum and exact integer selected-element statistics.
    pub statistics:FullyShardedLoss<B,S>,
    /// Actual detached native local effective weight; not physical row/storage count.
    pub local_weight:Tensor<B,1>,
    /// Actual detached native global effective weight on the original data group.
    pub global_weight:Tensor<B,1>,
}
impl<B:Backend,S:CheckpointStrategy> FullyShardedWeightedLoss<B,S> {
    /// Original weighted mean; a zero denominator produces a graph-connected zero.
    /// Fractional nonzero weights are not clamped to one and no denominator derivative is taken.
    pub fn mean(&self) -> Tensor<Autodiff<B,S>,1> {self.normalized(self.global_weight.clone())}
    /// Normalize by an explicitly coordinated accumulation-window weight, without averaging batch means.
    pub fn normalized(&self,weight:Tensor<B,1>) -> Tensor<Autodiff<B,S>,1> {
        assert_eq!(weight.dims(),[1],"sharded window weight must be scalar");
        assert_eq!(weight.device(),self.statistics.loss_sum.device(),"sharded window weight device differs");
        let dtype=self.statistics.loss_sum.dtype();
        let weight=Tensor::<Autodiff<B,S>,1>::from_inner(weight.cast(dtype));
        let empty=weight.clone().equal_elem(0);
        self.statistics.loss_sum.clone().mask_fill(empty.clone(),0)/weight.mask_fill(empty,1)
    }
    /// Actual globally weighted metric, with the same zero/fractional denominator semantics.
    pub fn global_mean(&self) -> Tensor<B,1> {
        let empty=self.global_weight.clone().equal_elem(0);
        self.statistics.global_loss_sum.clone().mask_fill(empty.clone(),0)/self.global_weight.clone().mask_fill(empty,1)
    }
}

/// Complete original unreduced classification/binary/regression/KL terms on their actual data group.
/// This preserves explicit class/sample weights, ignored-target selection and F64 work precision.
pub fn complete_fully_sharded_terms<B,S,C,const D:usize>(scope:&CollectiveScope<B,S>,terms:LossTerms<Autodiff<B,S>,D>,communicator:C)
    -> Result<FullyShardedWeightedLoss<B,S>,ScopedCollectiveError<C::Error>>
    where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
    if terms.values.dims()!=terms.normalizers.dims() || terms.values.dims()!=terms.valid.dims()
        || terms.values.device()!=terms.normalizers.device() || terms.values.device()!=terms.valid.device() {
        return Err(ScopedCollectiveError::Protocol("actual loss term/weight/visibility geometry differs"));
    }
    if !matches!(terms.values.dtype(),DType::F32|DType::F64) || !matches!(terms.normalizers.dtype(),DType::F32|DType::F64) {
        return Err(ScopedCollectiveError::Protocol("original unreduced losses require F32/F64 work precision"));
    }
    let work=if terms.values.dtype()==DType::F64 || terms.normalizers.dtype()==DType::F64 {DType::F64} else {DType::F32};
    let local_weight=terms.normalizers.clone().cast(work).sum().inner();
    let statistics=complete_fully_sharded_loss(scope,terms.values.clone().cast(work).sum(),terms.valid_count(),communicator.clone())?;
    let global_weight=if communicator.world_size()==1 {local_weight.clone()} else {Tensor::<B,1>::from_primitive(TensorPrimitive::Float(
        communicator.all_reduce_sum(local_weight.clone().into_primitive().tensor()).map_err(ScopedCollectiveError::Collective)?))};
    if global_weight.dims()!=[1] || global_weight.dtype()!=work || global_weight.device()!=local_weight.device() {
        return Err(ScopedCollectiveError::Protocol("effective weight transport changed original shape/storage/device"));
    }
    Ok(FullyShardedWeightedLoss {statistics,local_weight,global_weight})
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedTransformerModel<Autodiff<B,S>> {
    /// Complete actual model and caller-owned native objective; projection can remain token-chunked.
    /// The objective sees the original gathered head, not precomputed full-vocabulary token logits.
    pub fn forward_loss_with<C,F,O,const D:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,communicator:C,mut layer:F,objective:O)
        -> Result<FullyShardedWeightedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            O:FnOnce(Tensor<Autodiff<B,S>,3>,GatheredFullyShardedTransformerHead<Autodiff<B,S>>)->LossTerms<Autodiff<B,S>,D> {
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden_with(input,transport.clone(),|index,block,hidden|layer(index,block,hidden,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        complete_fully_sharded_terms(&scope,objective(hidden,head),communicator)
    }
    /// Actual independent-document model and custom native per-token/sequence objective.
    pub fn forward_packed_loss_with<C,F,O,const D:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,layout:&PackedSequenceLayout,communicator:C,mut layer:F,objective:O)
        -> Result<FullyShardedWeightedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            O:FnOnce(Tensor<Autodiff<B,S>,2>,GatheredFullyShardedTransformerHead<Autodiff<B,S>>)->LossTerms<Autodiff<B,S>,D> {
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden_with(input,layout,transport.clone(),|index,block,hidden|layer(index,block,hidden,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        complete_fully_sharded_terms(&scope,objective(hidden,head),communicator)
    }
    /// Real pooled output and its visible-row/count metadata, followed by the caller's actual loss terms.
    pub fn forward_sequence_loss_with<C,F,O,const D:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,visible:Tensor<Autodiff<B,S>,2,Bool>,
        pooling:SequencePooling,communicator:C,layer:F,objective:O) -> Result<FullyShardedWeightedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            O:FnOnce(SequenceHeadOutput<Autodiff<B,S>>)->LossTerms<Autodiff<B,S>,D> {
        self.forward_loss_with(input,communicator,layer,|hidden,head|objective(head.forward_sequence(hidden,visible,pooling)))
    }
    /// Original independent packed sequence pooling, including empty-document validity, before the native loss.
    pub fn forward_packed_sequence_loss_with<C,F,O,const D:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,layout:&PackedSequenceLayout,
        visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,pooling:SequencePooling,communicator:C,layer:F,objective:O)
        -> Result<FullyShardedWeightedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            O:FnOnce(SequenceHeadOutput<Autodiff<B,S>>)->LossTerms<Autodiff<B,S>,D> {
        self.forward_packed_loss_with(input,layout,communicator,layer,|hidden,head|objective(head.forward_packed_sequences(hidden,layout,visible,pooling)))
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedEncoderDecoderModel<Autodiff<B,S>> {
    /// Complete original encoder-memory gradients and custom native decoder/token/sequence objective.
    pub fn forward_loss_with<C,E,F,O,const D:usize>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>>,target:FullyShardedTransformerInput<Autodiff<B,S>>,
        communicator:C,mut encoder:E,mut decoder:F,objective:O) -> Result<FullyShardedWeightedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&FullyShardedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            O:FnOnce(Tensor<Autodiff<B,S>,3>,GatheredFullyShardedTransformerHead<Autodiff<B,S>>)->LossTerms<Autodiff<B,S>,D> {
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden_with(source,target,transport.clone(),|index,block,hidden|encoder(index,block,hidden,transport.clone()),
            |index,block,hidden,memory|decoder(index,block,hidden,memory,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        complete_fully_sharded_terms(&scope,objective(hidden,head),communicator)
    }
    /// Original separately bounded packed source/target model and caller-selected unreduced decoder loss.
    pub fn forward_packed_loss_with<C,E,F,O,const D:usize>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>,1>,target:FullyShardedTransformerInput<Autodiff<B,S>,1>,
        source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,communicator:C,mut encoder:E,mut decoder:F,objective:O)
        -> Result<FullyShardedWeightedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&FullyShardedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            O:FnOnce(Tensor<Autodiff<B,S>,2>,GatheredFullyShardedTransformerHead<Autodiff<B,S>>)->LossTerms<Autodiff<B,S>,D> {
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden_with(source,target,source_layout,target_layout,transport.clone(),|index,block,hidden|encoder(index,block,hidden,transport.clone()),
            |index,block,hidden,memory|decoder(index,block,hidden,memory,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        complete_fully_sharded_terms(&scope,objective(hidden,head),communicator)
    }
}
