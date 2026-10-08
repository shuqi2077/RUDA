use alloc::vec::Vec;
use ruda_model::{record::RecorderError, tensor::backend::Backend};
use crate::{attention::CompressedAttentionProjection, expert_parallel::ExpertParallelGeometry};
use super::{ExpertOwnedModelGeometry, ExpertAdapterOwnershipEntry, ExpertParallelModelStateRecord,
    MhcResidualStack, MhcResidualModel, MhcResidualBranchShape, MixedMhcFeedForward, TransformerProjectionShape};

impl<B: Backend, P: CompressedAttentionProjection<B>, L: MhcResidualBranchShape<B>, Q: TransformerProjectionShape<B>, E: ExpertParallelGeometry<B>>
    ExpertOwnedModelGeometry<B> for MhcResidualStack<B, P, MixedMhcFeedForward<B, L, Q, E>> {
    fn expert_model_layers(&self) -> usize { self.layers.len() }
    fn expert_model_ownership(&self) -> Vec<ExpertAdapterOwnershipEntry> {
        let mut ownership = Vec::new();
        for (layer, value) in self.layers.iter().enumerate() {
            if let MixedMhcFeedForward::Parallel(branch) = &value.feed_forward {
                ownership.push(ExpertAdapterOwnershipEntry { layer, prefix: branch.routed.experts.ownership().prefix().to_vec(),
                    rank: branch.routed.experts.rank() });
            }
        }
        ownership
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, L: MhcResidualBranchShape<B>, Q: TransformerProjectionShape<B>, E: ExpertParallelGeometry<B>>
    MhcResidualStack<B, P, MixedMhcFeedForward<B, L, Q, E>> {
    /// Actual rank-local stack leaves, including mHC mappings, compression/indexer projections and packed expert metadata.
    pub fn expert_model_state_record(&self, contract_id: &str) -> Result<ExpertParallelModelStateRecord<B>, RecorderError> {
        ExpertParallelModelStateRecord::capture_module(self, contract_id)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, L: MhcResidualBranchShape<B>, Q: TransformerProjectionShape<B>,
    E: ExpertParallelGeometry<B>, H: TransformerProjectionShape<B>> ExpertOwnedModelGeometry<B> for MhcResidualModel<B, P, MixedMhcFeedForward<B, L, Q, E>, H> {
    fn expert_model_layers(&self) -> usize { self.stack.expert_model_layers() }
    fn expert_model_ownership(&self) -> Vec<ExpertAdapterOwnershipEntry> { self.stack.expert_model_ownership() }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, L: MhcResidualBranchShape<B>, Q: TransformerProjectionShape<B>,
    E: ExpertParallelGeometry<B>, H: TransformerProjectionShape<B>> MhcResidualModel<B, P, MixedMhcFeedForward<B, L, Q, E>, H> {
    /// Full native model state reuses exact dtype/ID/tie restoration, including actual embedding and independent head.
    /// Optimizer, scheduler, RNG and data position stay in their existing separate training records.
    pub fn expert_model_state_record(&self, contract_id: &str) -> Result<ExpertParallelModelStateRecord<B>, RecorderError> {
        ExpertParallelModelStateRecord::capture_module(self, contract_id)
    }
}
