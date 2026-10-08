use super::*;
use ruda_autodiff::collective::ScopedTensorCollective;
use ruda_model::tensor::{IntegerTensorCollective,VariableTensorCollective,MoeDispatchOps,MoeReceivedOps};
use crate::loss::{CausalCrossEntropyConfig,LossTerms};
use crate::expert_parallel::ExpertParallelReceived;
use crate::transformer::{TransformerProjection,ExpertParallelTransformerLayer,ExpertParallelTransformerError,ProjectedTransformerHead};
use crate::attention::PackedSequenceLayout;

pub type ExpertOwnedHybridTrainingError<D,E,G,N,P,X,O=core::convert::Infallible> =
    HybridModelTrainingError<FullyShardedExpertParallelModelError<D,E,P,X>,O,D,E,G,N>;

impl<B:MoeDispatchOps+MoeReceivedOps,S:CheckpointStrategy,P:GatherTransformerProjection<Autodiff<B,S>,B>,
    X:GatherOwnedExperts<Autodiff<B,S>,B>> FullyShardedExpertParallelTransformerModel<Autodiff<B,S>,P,X>
where P::Gathered:TransformerProjection<Autodiff<B,S>>,X::Gathered:ExpertParallelReceived<Autodiff<B,S>> {
    /// Every gather uses the original scoped data group; architecture callbacks
    /// receive the independent scoped expert transport. Both graph windows close
    /// before statistics on the explicitly selected normalization group.
    pub fn forward_hybrid_causal_with<D,E,G,N,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut layer:F)
        -> Result<HybridFullyShardedLoss<B,S>,ExpertOwnedHybridTrainingError<D::Error,E::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>
    where D:IntegerTensorCollective<B>,E:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<E,B,S>)
            -> Result<Tensor<Autodiff<B,S>,3>,ExpertParallelTransformerError<E::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>> {
        assert_eq!(input.tokens.dims(),labels.dims(),"hybrid causal token/label geometry differs");
        let expert=context.expert();
        let terms=self.forward_causal_terms_with(input,labels,criterion,label_smoothing,context.data(),
            |index,native,hidden,_|layer(index,native,hidden,expert.clone())).map_err(HybridModelTrainingError::Model)?;
        context.complete(terms.loss_sum,terms.valid_tokens).map_err(HybridModelTrainingError::Loss)
    }

    /// Keep original document-local shifts, sentinel/smoothing policy and true
    /// empty-rank packed paths. The native head is gathered once per loss window.
    pub fn forward_hybrid_packed_causal_with<D,E,G,N,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut layer:F)
        -> Result<HybridFullyShardedLoss<B,S>,ExpertOwnedHybridTrainingError<D::Error,E::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>
    where D:IntegerTensorCollective<B>,E:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<E,B,S>)
            -> Result<Tensor<Autodiff<B,S>,2>,ExpertParallelTransformerError<E::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>> {
        assert_eq!(input.tokens.dims(),labels.dims(),"hybrid packed causal token/label geometry differs");
        let expert=context.expert();
        let terms=self.forward_packed_causal_terms_with(input,labels,layout,criterion,label_smoothing,context.data(),
            |index,native,hidden,_|layer(index,native,hidden,expert.clone())).map_err(HybridModelTrainingError::Model)?;
        context.complete(terms.loss_sum,terms.valid_tokens).map_err(HybridModelTrainingError::Loss)
    }

    /// Caller-owned token/sequence/task loss terms see native hidden rows and
    /// the real gathered head. No full token-logit materialization is required.
    pub fn forward_hybrid_objective_with<D,E,G,N,F,O,Q,const K:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,
        context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut layer:F,objective:O)
        -> Result<HybridFullyShardedWeightedLoss<B,S>,ExpertOwnedHybridTrainingError<D::Error,E::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error,Q>>
    where D:IntegerTensorCollective<B>,E:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,Q:core::fmt::Debug,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<E,B,S>)
            -> Result<Tensor<Autodiff<B,S>,3>,ExpertParallelTransformerError<E::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>,
        O:FnOnce(Tensor<Autodiff<B,S>,3>,&ProjectedTransformerHead<Autodiff<B,S>,P::Gathered>)->Result<LossTerms<Autodiff<B,S>,K>,Q> {
        let expert=context.expert();
        let hidden=self.forward_hidden_with(input,context.data(),|index,native,hidden,_|layer(index,native,hidden,expert.clone()))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.data()).map_err(|error|HybridModelTrainingError::Model(FullyShardedExpertParallelModelError::Data(error)))?;
        let terms=objective(hidden,&head).map_err(HybridModelTrainingError::Objective)?;
        context.complete_terms(terms).map_err(HybridModelTrainingError::Loss)
    }
    pub fn forward_hybrid_packed_objective_with<D,E,G,N,F,O,Q,const K:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,
        layout:&PackedSequenceLayout,context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut layer:F,objective:O)
        -> Result<HybridFullyShardedWeightedLoss<B,S>,ExpertOwnedHybridTrainingError<D::Error,E::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error,Q>>
    where D:IntegerTensorCollective<B>,E:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,Q:core::fmt::Debug,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<E,B,S>)
            -> Result<Tensor<Autodiff<B,S>,2>,ExpertParallelTransformerError<E::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>,
        O:FnOnce(Tensor<Autodiff<B,S>,2>,&ProjectedTransformerHead<Autodiff<B,S>,P::Gathered>)->Result<LossTerms<Autodiff<B,S>,K>,Q> {
        let expert=context.expert();
        let hidden=self.forward_packed_hidden_with(input,layout,context.data(),|index,native,hidden,_|layer(index,native,hidden,expert.clone()))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.data()).map_err(|error|HybridModelTrainingError::Model(FullyShardedExpertParallelModelError::Data(error)))?;
        let terms=objective(hidden,&head).map_err(HybridModelTrainingError::Objective)?;
        context.complete_terms(terms).map_err(HybridModelTrainingError::Loss)
    }

    /// Select the original parameter-gather slot explicitly. Each architecture
    /// callback selects its real per-layer expert slot from the same context,
    /// so a stack may retain independently configured expert groups per layer.
    pub fn forward_grouped_causal_with<C,G,N,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,data_slot:usize,context:GroupedCollectiveLossContext<B,S,C,G,N>,mut layer:F)
        -> Result<HybridFullyShardedLoss<B,S>,ExpertOwnedHybridTrainingError<C::Error,C::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>
    where C:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,3>,&GroupedCollectiveLossContext<B,S,C,G,N>)
            -> Result<Tensor<Autodiff<B,S>,3>,ExpertParallelTransformerError<C::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>> {
        assert_eq!(input.tokens.dims(),labels.dims(),"grouped causal token/label geometry differs");
        let causal=self.forward_causal_terms_with(input,labels,criterion,label_smoothing,context.transport(data_slot),
            |index,native,hidden,_|layer(index,native,hidden,&context)).map_err(HybridModelTrainingError::Model)?;
        context.complete(causal.loss_sum,causal.valid_tokens).map_err(HybridModelTrainingError::Loss)
    }
    pub fn forward_grouped_packed_causal_with<C,G,N,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,data_slot:usize,context:GroupedCollectiveLossContext<B,S,C,G,N>,mut layer:F)
        -> Result<HybridFullyShardedLoss<B,S>,ExpertOwnedHybridTrainingError<C::Error,C::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>
    where C:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,2>,&GroupedCollectiveLossContext<B,S,C,G,N>)
            -> Result<Tensor<Autodiff<B,S>,2>,ExpertParallelTransformerError<C::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>> {
        assert_eq!(input.tokens.dims(),labels.dims(),"grouped packed causal token/label geometry differs");
        let causal=self.forward_packed_causal_terms_with(input,labels,layout,criterion,label_smoothing,context.transport(data_slot),
            |index,native,hidden,_|layer(index,native,hidden,&context)).map_err(HybridModelTrainingError::Model)?;
        context.complete(causal.loss_sum,causal.valid_tokens).map_err(HybridModelTrainingError::Loss)
    }
    pub fn forward_grouped_objective_with<C,G,N,F,O,Q,const K:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,data_slot:usize,
        context:GroupedCollectiveLossContext<B,S,C,G,N>,mut layer:F,objective:O)
        -> Result<HybridFullyShardedWeightedLoss<B,S>,ExpertOwnedHybridTrainingError<C::Error,C::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error,Q>>
    where C:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,Q:core::fmt::Debug,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,3>,&GroupedCollectiveLossContext<B,S,C,G,N>)
            -> Result<Tensor<Autodiff<B,S>,3>,ExpertParallelTransformerError<C::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>,
        O:FnOnce(Tensor<Autodiff<B,S>,3>,&ProjectedTransformerHead<Autodiff<B,S>,P::Gathered>)->Result<LossTerms<Autodiff<B,S>,K>,Q> {
        let hidden=self.forward_hidden_with(input,context.transport(data_slot),|index,native,hidden,_|layer(index,native,hidden,&context))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.transport(data_slot)).map_err(|error|HybridModelTrainingError::Model(FullyShardedExpertParallelModelError::Data(error)))?;
        let terms=objective(hidden,&head).map_err(HybridModelTrainingError::Objective)?;
        context.complete_terms(terms).map_err(HybridModelTrainingError::Loss)
    }
    pub fn forward_grouped_packed_objective_with<C,G,N,F,O,Q,const K:usize>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,
        layout:&PackedSequenceLayout,data_slot:usize,context:GroupedCollectiveLossContext<B,S,C,G,N>,mut layer:F,objective:O)
        -> Result<HybridFullyShardedWeightedLoss<B,S>,ExpertOwnedHybridTrainingError<C::Error,C::Error,G::Error,N::Error,
            <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error,Q>>
    where C:VariableTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,Q:core::fmt::Debug,
        F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P::Gathered,X::Gathered>,Tensor<Autodiff<B,S>,2>,&GroupedCollectiveLossContext<B,S,C,G,N>)
            -> Result<Tensor<Autodiff<B,S>,2>,ExpertParallelTransformerError<C::Error,
                <P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<X::Gathered as ExpertParallelReceived<Autodiff<B,S>>>::Error>>,
        O:FnOnce(Tensor<Autodiff<B,S>,2>,&ProjectedTransformerHead<Autodiff<B,S>,P::Gathered>)->Result<LossTerms<Autodiff<B,S>,K>,Q> {
        let hidden=self.forward_packed_hidden_with(input,layout,context.transport(data_slot),|index,native,hidden,_|layer(index,native,hidden,&context))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.transport(data_slot)).map_err(|error|HybridModelTrainingError::Model(FullyShardedExpertParallelModelError::Data(error)))?;
        let terms=objective(hidden,&head).map_err(HybridModelTrainingError::Objective)?;
        context.complete_terms(terms).map_err(HybridModelTrainingError::Loss)
    }
}
