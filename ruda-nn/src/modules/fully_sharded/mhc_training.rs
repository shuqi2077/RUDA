use super::*;
use core::fmt;
use ruda_model::tensor::{Bool, IntegerTensorCollective};
use ruda_autodiff::collective::{CollectiveScope, ScopedTensorCollective, ScopedCollectiveError};
use crate::{attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout},
    transformer::{TransformerProjection, ProjectedTransformerHead, MhcResidualBranch}, loss::{CausalCrossEntropyConfig, LossTerms}};

#[derive(Debug)]
pub enum FullyShardedMhcTrainingError<C: fmt::Debug, R: fmt::Debug, H: fmt::Debug, O: fmt::Debug> {
    Model(FullyShardedMhcModelError<C, R, H>),
    Objective(O),
    Loss(ScopedCollectiveError<C>),
}
impl<C: fmt::Debug, R: fmt::Debug, H: fmt::Debug, O: fmt::Debug> fmt::Display for FullyShardedMhcTrainingError<C, R, H, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Model(error) => write!(f, "{error}"), Self::Objective(error) => write!(f, "mHC training objective: {error:?}"),
            Self::Loss(error) => write!(f, "mHC distributed loss: {error}") }
    }
}
impl<C: fmt::Debug, R: fmt::Debug, H: fmt::Debug, O: fmt::Debug> core::error::Error for FullyShardedMhcTrainingError<C, R, H, O> {}

impl<B: Backend, S: CheckpointStrategy, P: GatherTransformerProjection<Autodiff<B, S>, B>,
    F: GatherMhcResidualBranch<Autodiff<B, S>, B>, H: GatherTransformerProjection<Autodiff<B, S>, B>>
    FullyShardedMhcResidualModel<Autodiff<B, S>, P, F, H>
where P::Gathered: CompressedAttentionProjection<Autodiff<B, S>>, H::Gathered: TransformerProjection<Autodiff<B, S>> {
    /// Gather each real layer once and the head once, not once per differently sized local vocabulary chunk.
    /// Exact global effective-token counts and globally used collective paths use the original explicit data group.
    pub fn try_forward_causal_with<C, R, G>(&self, tokens: Tensor<Autodiff<B, S>, 2, Int>, labels: Tensor<Autodiff<B, S>, 2, Int>,
        valid: Option<Tensor<Autodiff<B, S>, 2, Bool>>, criterion: &CausalCrossEntropyConfig, label_smoothing: f64, communicator: C, branch: G)
        -> Result<FullyShardedLoss<B, S>, FullyShardedMhcTrainingError<C::Error, R, <H::Gathered as TransformerProjection<Autodiff<B, S>>>::Error, core::convert::Infallible>>
    where C: IntegerTensorCollective<B>, R: fmt::Debug,
        G: FnMut(usize, &F::Gathered, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R> {
        assert_eq!(tokens.dims(), labels.dims(), "sharded mHC causal token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "sharded mHC causal label device differs");
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_hidden_with(tokens, valid, transport.clone(), branch).map_err(FullyShardedMhcTrainingError::Model)?;
        let head = self.head.gather(transport).map_err(|error| FullyShardedMhcTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let loss = criterion.try_forward_hidden_with_smoothing(hidden, labels, |rows| head.forward(rows), label_smoothing)
            .map_err(|error| FullyShardedMhcTrainingError::Model(FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error))))?;
        complete_fully_sharded_loss(&scope, loss.loss_sum, loss.valid_tokens, communicator).map_err(FullyShardedMhcTrainingError::Loss)
    }

    /// Local document counts/token lengths may differ; all ranks execute the same actual parameter-gather order.
    pub fn try_forward_packed_causal_with<C, R, G>(&self, tokens: Tensor<Autodiff<B, S>, 1, Int>, labels: Tensor<Autodiff<B, S>, 1, Int>,
        layout: &PackedSequenceLayout, valid: Option<Tensor<Autodiff<B, S>, 1, Bool>>, criterion: &CausalCrossEntropyConfig,
        label_smoothing: f64, communicator: C, branch: G)
        -> Result<FullyShardedLoss<B, S>, FullyShardedMhcTrainingError<C::Error, R, <H::Gathered as TransformerProjection<Autodiff<B, S>>>::Error, core::convert::Infallible>>
    where C: IntegerTensorCollective<B>, R: fmt::Debug,
        G: FnMut(usize, &F::Gathered, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R> {
        assert_eq!(tokens.dims(), labels.dims(), "sharded mHC packed causal token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "sharded mHC packed causal label device differs");
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_packed_hidden_with(tokens, layout, valid, transport.clone(), branch).map_err(FullyShardedMhcTrainingError::Model)?;
        let head = self.head.gather(transport).map_err(|error| FullyShardedMhcTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let loss = criterion.try_forward_packed_hidden_with_smoothing(hidden, labels, layout, |rows| head.forward(rows), label_smoothing)
            .map_err(|error| FullyShardedMhcTrainingError::Model(FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error))))?;
        complete_fully_sharded_loss(&scope, loss.loss_sum, loss.valid_tokens, communicator).map_err(FullyShardedMhcTrainingError::Loss)
    }

    /// Caller forms the actual weighted task/indexer objective before graph completion and global denominator reduction.
    /// No auxiliary coefficient, class weights, ignored targets or optimizer/reduction policy is inferred.
    pub fn try_forward_objective_with<C, R, G, O, Q, const D: usize>(&self, tokens: Tensor<Autodiff<B, S>, 2, Int>,
        valid: Option<Tensor<Autodiff<B, S>, 2, Bool>>, indexer_warmup: bool, communicator: C, branch: G, objective: O)
        -> Result<FullyShardedWeightedLoss<B, S>, FullyShardedMhcTrainingError<C::Error, R, <H::Gathered as TransformerProjection<Autodiff<B, S>>>::Error, Q>>
    where C: IntegerTensorCollective<B>, R: fmt::Debug, Q: fmt::Debug,
        G: FnMut(usize, &F::Gathered, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R>,
        O: FnOnce(CompressedAttentionOutput<Autodiff<B, S>>, &ProjectedTransformerHead<Autodiff<B, S>, H::Gathered>) -> Result<LossTerms<Autodiff<B, S>, D>, Q> {
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_hidden_with_aux(tokens, valid, indexer_warmup, transport.clone(), branch).map_err(FullyShardedMhcTrainingError::Model)?;
        let head = self.head.gather(transport).map_err(|error| FullyShardedMhcTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let terms = objective(hidden, &head).map_err(FullyShardedMhcTrainingError::Objective)?;
        complete_fully_sharded_terms(&scope, terms, communicator).map_err(FullyShardedMhcTrainingError::Loss)
    }

    pub fn try_forward_packed_objective_with<C, R, G, O, Q, const D: usize>(&self, tokens: Tensor<Autodiff<B, S>, 1, Int>,
        layout: &PackedSequenceLayout, valid: Option<Tensor<Autodiff<B, S>, 1, Bool>>, indexer_warmup: bool,
        communicator: C, branch: G, objective: O)
        -> Result<FullyShardedWeightedLoss<B, S>, FullyShardedMhcTrainingError<C::Error, R, <H::Gathered as TransformerProjection<Autodiff<B, S>>>::Error, Q>>
    where C: IntegerTensorCollective<B>, R: fmt::Debug, Q: fmt::Debug,
        G: FnMut(usize, &F::Gathered, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R>,
        O: FnOnce(PackedCompressedAttentionOutput<Autodiff<B, S>>, &ProjectedTransformerHead<Autodiff<B, S>, H::Gathered>) -> Result<LossTerms<Autodiff<B, S>, D>, Q> {
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_packed_hidden_with_aux(tokens, layout, valid, indexer_warmup, transport.clone(), branch).map_err(FullyShardedMhcTrainingError::Model)?;
        let head = self.head.gather(transport).map_err(|error| FullyShardedMhcTrainingError::Model(
            FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error))))?;
        let terms = objective(hidden, &head).map_err(FullyShardedMhcTrainingError::Objective)?;
        complete_fully_sharded_terms(&scope, terms, communicator).map_err(FullyShardedMhcTrainingError::Loss)
    }
}

impl<B: Backend, S: CheckpointStrategy, P: GatherTransformerProjection<Autodiff<B, S>, B>,
    F: GatherMhcResidualBranch<Autodiff<B, S>, B>, H: GatherTransformerProjection<Autodiff<B, S>, B>>
    FullyShardedMhcResidualModel<Autodiff<B, S>, P, F, H>
where P::Gathered: CompressedAttentionProjection<Autodiff<B, S>>, F::Gathered: MhcResidualBranch<Autodiff<B, S>>,
    H::Gathered: TransformerProjection<Autodiff<B, S>> {
    pub fn forward_causal<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<Autodiff<B, S>, 2, Int>, labels: Tensor<Autodiff<B, S>, 2, Int>,
        valid: Option<Tensor<Autodiff<B, S>, 2, Bool>>, criterion: &CausalCrossEntropyConfig, label_smoothing: f64, communicator: C)
        -> Result<FullyShardedLoss<B, S>, FullyShardedMhcTrainingError<C::Error, <F::Gathered as MhcResidualBranch<Autodiff<B, S>>>::Error,
            <H::Gathered as TransformerProjection<Autodiff<B, S>>>::Error, core::convert::Infallible>> {
        self.try_forward_causal_with(tokens, labels, valid, criterion, label_smoothing, communicator, |_, feed, input, _| feed.forward_branch(input))
    }
    pub fn forward_packed_causal<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<Autodiff<B, S>, 1, Int>, labels: Tensor<Autodiff<B, S>, 1, Int>,
        layout: &PackedSequenceLayout, valid: Option<Tensor<Autodiff<B, S>, 1, Bool>>, criterion: &CausalCrossEntropyConfig, label_smoothing: f64, communicator: C)
        -> Result<FullyShardedLoss<B, S>, FullyShardedMhcTrainingError<C::Error, <F::Gathered as MhcResidualBranch<Autodiff<B, S>>>::Error,
            <H::Gathered as TransformerProjection<Autodiff<B, S>>>::Error, core::convert::Infallible>> {
        self.try_forward_packed_causal_with(tokens, labels, layout, valid, criterion, label_smoothing, communicator, |_, feed, input, _| feed.forward_branch(input))
    }
}
