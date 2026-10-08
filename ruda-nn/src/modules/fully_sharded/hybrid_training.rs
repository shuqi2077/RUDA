use super::*;
use ruda_model::tensor::{Bool, IntegerTensorCollective};
use ruda_autodiff::collective::{CollectiveScope, ScopedCollectiveError};
use crate::{attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout},
    loss::{CausalCrossEntropyConfig, LossTerms}};
use core::fmt;

#[derive(Debug)]
pub enum FullyShardedHybridObjectiveError<C: fmt::Debug, O: fmt::Debug> {
    Collective(C),
    Objective(O),
    Loss(ScopedCollectiveError<C>),
}
impl<C: fmt::Debug, O: fmt::Debug> fmt::Display for FullyShardedHybridObjectiveError<C, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Collective(error) => write!(f, "hybrid model transport: {error:?}"), Self::Objective(error) => write!(f, "hybrid native objective: {error:?}"),
            Self::Loss(error) => write!(f, "hybrid distributed objective: {error}") }
    }
}
impl<C: fmt::Debug, O: fmt::Debug> core::error::Error for FullyShardedHybridObjectiveError<C, O> {}

impl<B: Backend, S: CheckpointStrategy, P: GatherTransformerProjection<Autodiff<B, S>, B>> FullyShardedHybridLanguageModel<Autodiff<B, S>, P>
where P::Gathered: CompressedAttentionProjection<Autodiff<B, S>> {
    /// The true row-major tied table remains one local parameter; native logits are projected in original criterion chunks.
    pub fn forward_causal<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<Autodiff<B, S>, 2, Int>, labels: Tensor<Autodiff<B, S>, 2, Int>,
        valid: Option<Tensor<Autodiff<B, S>, 2, Bool>>, criterion: &CausalCrossEntropyConfig, label_smoothing: f64, communicator: C)
        -> Result<FullyShardedLoss<B, S>, ScopedCollectiveError<C::Error>> {
        assert_eq!(tokens.dims(), labels.dims(), "sharded hybrid causal token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "sharded hybrid causal label device differs");
        let scope = CollectiveScope::<B, S>::new(); let transport = scope.bind(communicator.clone());
        let hidden = self.backbone.forward(tokens, valid, transport.clone()).map_err(ScopedCollectiveError::Collective)?;
        let head = self.gather_head(transport).map_err(ScopedCollectiveError::Collective)?;
        let loss = criterion.forward_hidden_with_smoothing(hidden, labels, |rows| head.project(rows), label_smoothing);
        complete_fully_sharded_loss(&scope, loss.loss_sum, loss.valid_tokens, communicator)
    }
    pub fn forward_packed_causal<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<Autodiff<B, S>, 1, Int>, labels: Tensor<Autodiff<B, S>, 1, Int>,
        layout: &PackedSequenceLayout, valid: Option<Tensor<Autodiff<B, S>, 1, Bool>>, criterion: &CausalCrossEntropyConfig, label_smoothing: f64, communicator: C)
        -> Result<FullyShardedLoss<B, S>, ScopedCollectiveError<C::Error>> {
        assert_eq!(tokens.dims(), labels.dims(), "sharded hybrid packed token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "sharded hybrid packed label device differs");
        let scope = CollectiveScope::<B, S>::new(); let transport = scope.bind(communicator.clone());
        let hidden = self.backbone.forward_packed(tokens, layout, valid, transport.clone()).map_err(ScopedCollectiveError::Collective)?;
        let head = self.gather_head(transport).map_err(ScopedCollectiveError::Collective)?;
        let loss = criterion.forward_packed_hidden_with_smoothing(hidden, labels, layout, |rows| head.project(rows), label_smoothing);
        complete_fully_sharded_loss(&scope, loss.loss_sum, loss.valid_tokens, communicator)
    }
    /// The actual auxiliary coefficient/weighted loss is selected by the caller before completing the communication graph.
    pub fn forward_objective<C, O, Q, const D: usize>(&self, tokens: Tensor<Autodiff<B, S>, 2, Int>, valid: Option<Tensor<Autodiff<B, S>, 2, Bool>>,
        indexer_warmup: bool, communicator: C, objective: O) -> Result<FullyShardedWeightedLoss<B, S>, FullyShardedHybridObjectiveError<C::Error, Q>>
    where C: IntegerTensorCollective<B>, Q: fmt::Debug,
        O: FnOnce(CompressedAttentionOutput<Autodiff<B, S>>, &GatheredHybridHead<Autodiff<B, S>, P::Gathered>) -> Result<LossTerms<Autodiff<B, S>, D>, Q> {
        let scope = CollectiveScope::<B, S>::new(); let transport = scope.bind(communicator.clone());
        let hidden = self.backbone.forward_with_aux(tokens, valid, indexer_warmup, transport.clone()).map_err(FullyShardedHybridObjectiveError::Collective)?;
        let head = self.gather_head(transport).map_err(FullyShardedHybridObjectiveError::Collective)?;
        let terms = objective(hidden, &head).map_err(FullyShardedHybridObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope, terms, communicator).map_err(FullyShardedHybridObjectiveError::Loss)
    }
    pub fn forward_packed_objective<C, O, Q, const D: usize>(&self, tokens: Tensor<Autodiff<B, S>, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<Autodiff<B, S>, 1, Bool>>, indexer_warmup: bool, communicator: C, objective: O)
        -> Result<FullyShardedWeightedLoss<B, S>, FullyShardedHybridObjectiveError<C::Error, Q>>
    where C: IntegerTensorCollective<B>, Q: fmt::Debug,
        O: FnOnce(PackedCompressedAttentionOutput<Autodiff<B, S>>, &GatheredHybridHead<Autodiff<B, S>, P::Gathered>) -> Result<LossTerms<Autodiff<B, S>, D>, Q> {
        let scope = CollectiveScope::<B, S>::new(); let transport = scope.bind(communicator.clone());
        let hidden = self.backbone.forward_packed_with_aux(tokens, layout, valid, indexer_warmup, transport.clone()).map_err(FullyShardedHybridObjectiveError::Collective)?;
        let head = self.gather_head(transport).map_err(FullyShardedHybridObjectiveError::Collective)?;
        let terms = objective(hidden, &head).map_err(FullyShardedHybridObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope, terms, communicator).map_err(FullyShardedHybridObjectiveError::Loss)
    }
}
