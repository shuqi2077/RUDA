use core::fmt;
use ruda_model::tensor::{Tensor, Bool, Int, VariableTensorCollective, backend::Backend};
use ruda_autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy,
    collective::{CollectiveScope, ScopedTensorCollective, ScopedCollectiveError}};
use crate::{attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout},
    loss::{CausalCrossEntropyConfig, LossTerms}, fully_sharded::{FullyShardedLoss, FullyShardedWeightedLoss,
        complete_fully_sharded_loss, complete_fully_sharded_terms}};
use super::{MhcResidualModel, MhcResidualModelError, MhcResidualBranchShape, TransformerProjection, ProjectedTransformerHead};

/// Original model/objective errors or actual exact-count and collective-graph completion failure.
#[derive(Debug)]
pub enum MhcDistributedObjectiveError<R: fmt::Debug, C: fmt::Debug, O: fmt::Debug> {
    Model(R),
    Objective(O),
    Collective(ScopedCollectiveError<C>),
}
impl<R: fmt::Debug, C: fmt::Debug, O: fmt::Debug> fmt::Display for MhcDistributedObjectiveError<R, C, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Model(error) => write!(f, "mHC distributed model: {error:?}"),
            Self::Objective(error) => write!(f, "mHC distributed objective: {error:?}"),
            Self::Collective(error) => write!(f, "mHC distributed loss: {error}") }
    }
}
impl<R: fmt::Debug, C: fmt::Debug, O: fmt::Debug> core::error::Error for MhcDistributedObjectiveError<R, C, O> {}

impl<B: Backend, S: CheckpointStrategy, P: CompressedAttentionProjection<Autodiff<B, S>>,
    F: MhcResidualBranchShape<Autodiff<B, S>>, H: TransformerProjection<Autodiff<B, S>>> MhcResidualModel<Autodiff<B, S>, P, F, H> {
    /// One explicit expert-world loss scope. Branches use the supplied original scoped transport.
    /// This completes globally used communication paths on locally unused ranks, not gradient-reduction policies.
    pub fn try_distributed_causal_with<C, R, G>(&self, tokens: Tensor<Autodiff<B, S>, 2, Int>,
        labels: Tensor<Autodiff<B, S>, 2, Int>, valid: Option<Tensor<Autodiff<B, S>, 2, Bool>>,
        criterion: &CausalCrossEntropyConfig, label_smoothing: f64, communicator: C, mut branch: G)
        -> Result<FullyShardedLoss<B, S>, MhcDistributedObjectiveError<MhcResidualModelError<R, H::Error>, C::Error, core::convert::Infallible>>
    where C: VariableTensorCollective<B>, R: fmt::Debug,
        G: FnMut(usize, &F, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R> {
        assert_eq!(tokens.dims(), labels.dims(), "mHC distributed causal token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "mHC distributed causal label device differs");
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_hidden_with(tokens, valid, |index, feed, input| branch(index, feed, input, transport.clone()))
            .map_err(|error| MhcDistributedObjectiveError::Model(MhcResidualModelError::Branch(error)))?;
        let causal = criterion.try_forward_hidden_with_smoothing(hidden, labels, |rows| self.head.forward(rows), label_smoothing)
            .map_err(|error| MhcDistributedObjectiveError::Model(MhcResidualModelError::Head(error)))?;
        complete_fully_sharded_loss(&scope, causal.loss_sum, causal.valid_tokens, communicator).map_err(MhcDistributedObjectiveError::Collective)
    }

    pub fn try_distributed_packed_causal_with<C, R, G>(&self, tokens: Tensor<Autodiff<B, S>, 1, Int>,
        labels: Tensor<Autodiff<B, S>, 1, Int>, layout: &PackedSequenceLayout, valid: Option<Tensor<Autodiff<B, S>, 1, Bool>>,
        criterion: &CausalCrossEntropyConfig, label_smoothing: f64, communicator: C, mut branch: G)
        -> Result<FullyShardedLoss<B, S>, MhcDistributedObjectiveError<MhcResidualModelError<R, H::Error>, C::Error, core::convert::Infallible>>
    where C: VariableTensorCollective<B>, R: fmt::Debug,
        G: FnMut(usize, &F, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R> {
        assert_eq!(tokens.dims(), labels.dims(), "mHC distributed packed token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "mHC distributed packed label device differs");
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_packed_hidden_with(tokens, layout, valid,
            |index, feed, input| branch(index, feed, input, transport.clone()))
            .map_err(|error| MhcDistributedObjectiveError::Model(MhcResidualModelError::Branch(error)))?;
        let causal = criterion.try_forward_packed_hidden_with_smoothing(hidden, labels, layout, |rows| self.head.forward(rows), label_smoothing)
            .map_err(|error| MhcDistributedObjectiveError::Model(MhcResidualModelError::Head(error)))?;
        complete_fully_sharded_loss(&scope, causal.loss_sum, causal.valid_tokens, communicator).map_err(MhcDistributedObjectiveError::Collective)
    }

    /// Caller chooses the actual weighted criterion and any indexer coefficient before graph completion.
    /// Head projection stays bounded/native inside the objective; full vocabulary logits are not imposed.
    pub fn try_distributed_objective_with<C, R, G, O, Q, const D: usize>(&self, tokens: Tensor<Autodiff<B, S>, 2, Int>,
        valid: Option<Tensor<Autodiff<B, S>, 2, Bool>>, indexer_warmup: bool, communicator: C, mut branch: G, objective: O)
        -> Result<FullyShardedWeightedLoss<B, S>, MhcDistributedObjectiveError<R, C::Error, Q>>
    where C: VariableTensorCollective<B>, R: fmt::Debug, Q: fmt::Debug,
        G: FnMut(usize, &F, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R>,
        O: FnOnce(CompressedAttentionOutput<Autodiff<B, S>>, &ProjectedTransformerHead<Autodiff<B, S>, H>) -> Result<LossTerms<Autodiff<B, S>, D>, Q> {
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_hidden_with_aux(tokens, valid, indexer_warmup,
            |index, feed, input| branch(index, feed, input, transport.clone())).map_err(MhcDistributedObjectiveError::Model)?;
        let terms = objective(hidden, &self.head).map_err(MhcDistributedObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope, terms, communicator).map_err(MhcDistributedObjectiveError::Collective)
    }

    /// Independent document KL values remain available to the original weighted packed-token/sequence objective.
    /// Empty source ranks still enter every original parallel branch before scope/count completion.
    pub fn try_distributed_packed_objective_with<C, R, G, O, Q, const D: usize>(&self, tokens: Tensor<Autodiff<B, S>, 1, Int>,
        layout: &PackedSequenceLayout, valid: Option<Tensor<Autodiff<B, S>, 1, Bool>>, indexer_warmup: bool,
        communicator: C, mut branch: G, objective: O) -> Result<FullyShardedWeightedLoss<B, S>, MhcDistributedObjectiveError<R, C::Error, Q>>
    where C: VariableTensorCollective<B>, R: fmt::Debug, Q: fmt::Debug,
        G: FnMut(usize, &F, Tensor<Autodiff<B, S>, 3>, ScopedTensorCollective<C, B, S>) -> Result<Tensor<Autodiff<B, S>, 3>, R>,
        O: FnOnce(PackedCompressedAttentionOutput<Autodiff<B, S>>, &ProjectedTransformerHead<Autodiff<B, S>, H>) -> Result<LossTerms<Autodiff<B, S>, D>, Q> {
        let scope = CollectiveScope::<B, S>::new();
        let transport = scope.bind(communicator.clone());
        let hidden = self.try_forward_packed_hidden_with_aux(tokens, layout, valid, indexer_warmup,
            |index, feed, input| branch(index, feed, input, transport.clone())).map_err(MhcDistributedObjectiveError::Model)?;
        let terms = objective(hidden, &self.head).map_err(MhcDistributedObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope, terms, communicator).map_err(MhcDistributedObjectiveError::Collective)
    }
}
