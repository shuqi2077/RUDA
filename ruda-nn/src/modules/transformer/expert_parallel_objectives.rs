use core::fmt;
use ruda_model::tensor::{Tensor,Bool,VariableTensorCollective,MoeDispatchOps,MoeReceivedOps};
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,collective::{CollectiveScope,ScopedTensorCollective,ScopedCollectiveError}};
use crate::{expert_parallel::ExpertParallelReceived,attention::PackedSequenceLayout,pool::SequencePooling,loss::LossTerms,
    fully_sharded::{FullyShardedWeightedLoss,complete_fully_sharded_terms}};
use super::{TransformerProjection,ProjectedTransformerHead,ProjectedTransformerInput,SequenceHeadOutput,
    ExpertParallelTransformerModel,ExpertParallelTransformerLayer,ExpertParallelTransformerError};

/// Original local loss sum, exact global selected count and actual floating effective-weight denominator.
/// Expert-owned derivatives remain local sums; non-expert gradient synchronization groups remain explicit.
pub type ExpertParallelWeightedLoss<B,S> = FullyShardedWeightedLoss<B,S>;
/// Original model/transport error, caller-selected objective failure or weighted loss completion failure.
#[derive(Debug)]
pub enum ExpertParallelObjectiveError<C:fmt::Debug,P:fmt::Debug,M:fmt::Debug,O:fmt::Debug> {
    /// Actual original model, selected expert or native transport failure.
    Model(ExpertParallelTransformerError<C,P,M>),
    /// Actual caller-selected objective error; no substitute criterion is run.
    Objective(O),
    /// Original exact-count/graph-completion/weight collective failure.
    Loss(ScopedCollectiveError<C>),
}
impl<C:fmt::Debug,P:fmt::Debug,M:fmt::Debug,O:fmt::Debug> fmt::Display for ExpertParallelObjectiveError<C,P,M,O> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {
        Self::Model(error)=>write!(f,"expert model: {error}"),Self::Objective(error)=>write!(f,"expert objective: {error:?}"),Self::Loss(error)=>write!(f,"expert weighted loss: {error}")}}
}
impl<C:fmt::Debug,P:fmt::Debug,M:fmt::Debug,O:fmt::Debug> core::error::Error for ExpertParallelObjectiveError<C,P,M,O> {}

impl<B:MoeDispatchOps+MoeReceivedOps,S:CheckpointStrategy,P:TransformerProjection<Autodiff<B,S>>,E:ExpertParallelReceived<Autodiff<B,S>>>
    ExpertParallelTransformerModel<Autodiff<B,S>,P,E> {
    /// Run the complete original model and a fallible caller-owned weighted objective on actual hidden states/head.
    /// The head is not precomputed: callers retain bounded token projection and the original native storage/error contract.
    pub fn forward_loss_with<C,F,O,Q,const D:usize>(&self,input:ProjectedTransformerInput<Autodiff<B,S>>,communicator:C,layer:F,objective:O)
        -> Result<ExpertParallelWeightedLoss<B,S>,ExpertParallelObjectiveError<C::Error,P::Error,E::Error,Q>>
        where C:VariableTensorCollective<B>,Q:fmt::Debug,
            F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P,E>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)
                -> Result<Tensor<Autodiff<B,S>,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>,
            O:FnOnce(Tensor<Autodiff<B,S>,3>,&ProjectedTransformerHead<Autodiff<B,S>,P>)->Result<LossTerms<Autodiff<B,S>,D>,Q> {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden_with(input,transport,layer).map_err(ExpertParallelObjectiveError::Model)?;
        let terms=objective(hidden,&self.head).map_err(ExpertParallelObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope,terms,communicator).map_err(ExpertParallelObjectiveError::Loss)
    }
    /// Run original independently packed documents and a weighted token/sequence objective without padding.
    /// Masks, target interpretation, class/sample weights and loss reduction are supplied by the actual objective.
    pub fn forward_packed_loss_with<C,F,O,Q,const D:usize>(&self,input:ProjectedTransformerInput<Autodiff<B,S>,1>,layout:&PackedSequenceLayout,
        communicator:C,layer:F,objective:O) -> Result<ExpertParallelWeightedLoss<B,S>,ExpertParallelObjectiveError<C::Error,P::Error,E::Error,Q>>
        where C:VariableTensorCollective<B>,Q:fmt::Debug,
            F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P,E>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)
                -> Result<Tensor<Autodiff<B,S>,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>,
            O:FnOnce(Tensor<Autodiff<B,S>,2>,&ProjectedTransformerHead<Autodiff<B,S>,P>)->Result<LossTerms<Autodiff<B,S>,D>,Q> {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden_with(input,layout,transport,layer).map_err(ExpertParallelObjectiveError::Model)?;
        let terms=objective(hidden,&self.head).map_err(ExpertParallelObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope,terms,communicator).map_err(ExpertParallelObjectiveError::Loss)
    }
    /// Run real visible-token pooling, the original native head and a caller-selected sequence objective.
    /// Empty-row visibility and I64 real-token counts are passed unchanged to the objective, not guessed from labels.
    pub fn forward_sequence_loss_with<C,F,O,Q,const D:usize>(&self,input:ProjectedTransformerInput<Autodiff<B,S>>,visible:Tensor<Autodiff<B,S>,2,Bool>,
        pooling:SequencePooling,communicator:C,layer:F,objective:O)
        -> Result<ExpertParallelWeightedLoss<B,S>,ExpertParallelObjectiveError<C::Error,P::Error,E::Error,Q>>
        where C:VariableTensorCollective<B>,Q:fmt::Debug,
            F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P,E>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)
                -> Result<Tensor<Autodiff<B,S>,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>,
            O:FnOnce(SequenceHeadOutput<Autodiff<B,S>>)->Result<LossTerms<Autodiff<B,S>,D>,Q> {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let output=self.forward_sequence_with(input,visible,pooling,transport,layer).map_err(ExpertParallelObjectiveError::Model)?;
        let terms=objective(output).map_err(ExpertParallelObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope,terms,communicator).map_err(ExpertParallelObjectiveError::Loss)
    }
    /// Run each actual packed document's pooled native head and original sequence criterion independently.
    /// The global denominator is the actual selected effective weight, including fractional weights and excluded rows.
    pub fn forward_packed_sequence_loss_with<C,F,O,Q,const D:usize>(&self,input:ProjectedTransformerInput<Autodiff<B,S>,1>,layout:&PackedSequenceLayout,
        visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,pooling:SequencePooling,communicator:C,layer:F,objective:O)
        -> Result<ExpertParallelWeightedLoss<B,S>,ExpertParallelObjectiveError<C::Error,P::Error,E::Error,Q>>
        where C:VariableTensorCollective<B>,Q:fmt::Debug,
            F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P,E>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)
                -> Result<Tensor<Autodiff<B,S>,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>,
            O:FnOnce(SequenceHeadOutput<Autodiff<B,S>>)->Result<LossTerms<Autodiff<B,S>,D>,Q> {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let output=self.forward_packed_sequences_with(input,layout,visible,pooling,transport,layer).map_err(ExpertParallelObjectiveError::Model)?;
        let terms=objective(output).map_err(ExpertParallelObjectiveError::Objective)?;
        complete_fully_sharded_terms(&scope,terms,communicator).map_err(ExpertParallelObjectiveError::Loss)
    }
}
