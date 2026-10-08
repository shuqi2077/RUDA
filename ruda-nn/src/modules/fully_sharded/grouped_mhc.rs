use super::*;
use core::fmt;
use ruda_model::tensor::{Bool,IntegerTensorCollective};
use ruda_autodiff::collective::ScopedTensorCollective;
use crate::{attention::{CompressedAttentionProjection,CompressedAttentionOutput,PackedCompressedAttentionOutput,PackedSequenceLayout},
    transformer::{TransformerProjection,ProjectedTransformerHead},loss::{CausalCrossEntropyConfig,LossTerms}};

pub type MhcHybridTrainingError<D,E,G,N,R,H,O=core::convert::Infallible> =
    HybridModelTrainingError<FullyShardedMhcModelError<D,R,H>,O,D,E,G,N>;

impl<B:Backend,S:CheckpointStrategy,P:GatherTransformerProjection<Autodiff<B,S>,B>,F:GatherMhcResidualBranch<Autodiff<B,S>,B>,
    H:GatherTransformerProjection<Autodiff<B,S>,B>> FullyShardedMhcResidualModel<Autodiff<B,S>,P,F,H>
where P::Gathered:CompressedAttentionProjection<Autodiff<B,S>>,H::Gathered:TransformerProjection<Autodiff<B,S>> {
    /// Actual mHC/compressed/owned-expert graph, with independent scoped data
    /// gathers and branch transport. No expert topology or task denominator is
    /// inferred from the original attention width or data-world rank.
    pub fn try_forward_hybrid_causal_with<D,E,G,N,R,Q>(&self,tokens:Tensor<Autodiff<B,S>,2,Int>,labels:Tensor<Autodiff<B,S>,2,Int>,
        valid:Option<Tensor<Autodiff<B,S>,2,Bool>>,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,
        context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut branch:Q)
        -> Result<HybridFullyShardedLoss<B,S>,MhcHybridTrainingError<D::Error,E::Error,G::Error,N::Error,R,
            <H::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
    where D:IntegerTensorCollective<B>,E:BroadcastTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,R:fmt::Debug,
        Q:FnMut(usize,&F::Gathered,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<E,B,S>)->Result<Tensor<Autodiff<B,S>,3>,R> {
        assert_eq!(tokens.dims(),labels.dims(),"hybrid mHC causal token/label geometry differs");
        assert_eq!(tokens.device(),labels.device(),"hybrid mHC causal label device differs");
        let expert=context.expert();
        let hidden=self.try_forward_hidden_with(tokens,valid,context.data(),|index,feed,input,_|branch(index,feed,input,expert.clone()))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.data()).map_err(|error|HybridModelTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let causal=criterion.try_forward_hidden_with_smoothing(hidden,labels,|rows|head.forward(rows),label_smoothing)
            .map_err(|error|HybridModelTrainingError::Model(FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error))))?;
        context.complete(causal.loss_sum,causal.valid_tokens).map_err(HybridModelTrainingError::Loss)
    }

    /// Original independent document positions/masks and shifted causal labels,
    /// retaining native empty packed-rank participation and bounded head chunks.
    pub fn try_forward_hybrid_packed_causal_with<D,E,G,N,R,Q>(&self,tokens:Tensor<Autodiff<B,S>,1,Int>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,valid:Option<Tensor<Autodiff<B,S>,1,Bool>>,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,
        context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut branch:Q)
        -> Result<HybridFullyShardedLoss<B,S>,MhcHybridTrainingError<D::Error,E::Error,G::Error,N::Error,R,
            <H::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
    where D:IntegerTensorCollective<B>,E:BroadcastTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,R:fmt::Debug,
        Q:FnMut(usize,&F::Gathered,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<E,B,S>)->Result<Tensor<Autodiff<B,S>,3>,R> {
        assert_eq!(tokens.dims(),labels.dims(),"hybrid mHC packed token/label geometry differs");
        assert_eq!(tokens.device(),labels.device(),"hybrid mHC packed label device differs");
        let expert=context.expert();
        let hidden=self.try_forward_packed_hidden_with(tokens,layout,valid,context.data(),|index,feed,input,_|branch(index,feed,input,expert.clone()))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.data()).map_err(|error|HybridModelTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let causal=criterion.try_forward_packed_hidden_with_smoothing(hidden,labels,layout,|rows|head.forward(rows),label_smoothing)
            .map_err(|error|HybridModelTrainingError::Model(FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error))))?;
        context.complete(causal.loss_sum,causal.valid_tokens).map_err(HybridModelTrainingError::Loss)
    }

    /// Caller chooses task/class/sample weights and any actual indexer-loss
    /// coefficient before closing both groups. Native head projection is not
    /// rerun solely to obtain detached metrics or collective path flags.
    pub fn try_forward_hybrid_objective_with<D,E,G,N,R,Q,O,Z,const K:usize>(&self,tokens:Tensor<Autodiff<B,S>,2,Int>,
        valid:Option<Tensor<Autodiff<B,S>,2,Bool>>,indexer_warmup:bool,context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut branch:Q,objective:O)
        -> Result<HybridFullyShardedWeightedLoss<B,S>,MhcHybridTrainingError<D::Error,E::Error,G::Error,N::Error,R,
            <H::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,Z>>
    where D:IntegerTensorCollective<B>,E:BroadcastTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,R:fmt::Debug,Z:fmt::Debug,
        Q:FnMut(usize,&F::Gathered,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<E,B,S>)->Result<Tensor<Autodiff<B,S>,3>,R>,
        O:FnOnce(CompressedAttentionOutput<Autodiff<B,S>>,&ProjectedTransformerHead<Autodiff<B,S>,H::Gathered>)->Result<LossTerms<Autodiff<B,S>,K>,Z> {
        let expert=context.expert();
        let hidden=self.try_forward_hidden_with_aux(tokens,valid,indexer_warmup,context.data(),|index,feed,input,_|branch(index,feed,input,expert.clone()))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.data()).map_err(|error|HybridModelTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let terms=objective(hidden,&head).map_err(HybridModelTrainingError::Objective)?;
        context.complete_terms(terms).map_err(HybridModelTrainingError::Loss)
    }
    pub fn try_forward_hybrid_packed_objective_with<D,E,G,N,R,Q,O,Z,const K:usize>(&self,tokens:Tensor<Autodiff<B,S>,1,Int>,layout:&PackedSequenceLayout,
        valid:Option<Tensor<Autodiff<B,S>,1,Bool>>,indexer_warmup:bool,context:HybridCollectiveLossContext<B,S,D,E,G,N>,mut branch:Q,objective:O)
        -> Result<HybridFullyShardedWeightedLoss<B,S>,MhcHybridTrainingError<D::Error,E::Error,G::Error,N::Error,R,
            <H::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,Z>>
    where D:IntegerTensorCollective<B>,E:BroadcastTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B>,R:fmt::Debug,Z:fmt::Debug,
        Q:FnMut(usize,&F::Gathered,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<E,B,S>)->Result<Tensor<Autodiff<B,S>,3>,R>,
        O:FnOnce(PackedCompressedAttentionOutput<Autodiff<B,S>>,&ProjectedTransformerHead<Autodiff<B,S>,H::Gathered>)->Result<LossTerms<Autodiff<B,S>,K>,Z> {
        let expert=context.expert();
        let hidden=self.try_forward_packed_hidden_with_aux(tokens,layout,valid,indexer_warmup,context.data(),|index,feed,input,_|branch(index,feed,input,expert.clone()))
            .map_err(HybridModelTrainingError::Model)?;
        let head=self.head.gather(context.data()).map_err(|error|HybridModelTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let terms=objective(hidden,&head).map_err(HybridModelTrainingError::Objective)?;
        context.complete_terms(terms).map_err(HybridModelTrainingError::Loss)
    }
}
