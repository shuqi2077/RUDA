use alloc::{collections::{BTreeMap, BTreeSet}, vec::Vec};
use ruda_model::{module::{Module, ParamId, list_param_ids}, tensor::{MoeOptions, MoeExpertStrategy, backend::Backend}};
use crate::{FrozenExpertGeometry, MixedExpertParallelSource, MixedOwnedExperts, PackedExpertPartitionContext,
    attention::CompressedAttentionProjection, expert_parallel::ExpertParallelMoeLayer};
use super::{MhcFeedForward, MhcResidualBlock, MhcResidualStack, MhcResidualModel, MhcResidualBranchShape,
    MixedMhcFeedForward, ExpertParallelMhcFeedForward, MixedExpertParallelLayerConfig, TransformerProjectionShape};
use super::super::mixed_expert_parallel::validate_ownership_aliases;

/// Mixed original local branches and actual rank-owned floating/NF4/AWQ expert chains.
pub type ExpertPartitionedMhcBranch<B, Q, E> = MixedMhcFeedForward<B, MhcFeedForward<B, Q, E>, Q, MixedOwnedExperts<B>>;
pub type ExpertPartitionedMhcStack<B, P, Q, E> = MhcResidualStack<B, P, ExpertPartitionedMhcBranch<B, Q, E>>;
pub type ExpertPartitionedMhcModel<B, P, Q, E, H> = MhcResidualModel<B, P, ExpertPartitionedMhcBranch<B, Q, E>, H>;

fn options_from_packed(routing: crate::Nf4MoeRouting) -> MoeOptions {
    MoeOptions { selection: routing.selection, weights: routing.weights, combine_backward: routing.combine_backward,
        forward: MoeExpertStrategy::Scalar, backward: MoeExpertStrategy::Scalar }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, Q: TransformerProjectionShape<B>,
    E: FrozenExpertGeometry<B> + Into<MixedExpertParallelSource<B>>> MhcResidualStack<B, P, MhcFeedForward<B, Q, E>> {
    /// Move real selected expert storage into explicit rank-owned windows; all other loaded parts remain unchanged.
    /// A single partition context preserves original ties between selected layers.
    pub fn into_mixed_expert_parallel(self, targets: &[MixedExpertParallelLayerConfig], rank: usize) -> ExpertPartitionedMhcStack<B, P, Q, E> {
        let mut selected = BTreeMap::new();
        let mut sources = Vec::with_capacity(targets.len());
        for config in targets {
            assert!(config.layer < self.layers.len(), "mHC expert ownership layer index exceeds loaded stack");
            assert!(selected.insert(config.layer, config).is_none(), "duplicate mHC owned-expert layer index");
            let branch = &self.layers[config.layer].feed_forward;
            branch.validate_branch();
            let (source, options) = match branch {
                MhcFeedForward::Dense(_) => panic!("mHC expert ownership cannot select an ordinary dense branch"),
                MhcFeedForward::Floating(value) => (MixedExpertParallelSource::Native(value.routed.experts.clone()), value.routed.options),
                MhcFeedForward::Packed(value) => (value.routed.experts.clone().into(), options_from_packed(value.routed.routing)),
            };
            source.validate();
            assert_eq!(source.dimensions()[0], config.ownership.experts(), "mHC actual expert count differs from declared ownership");
            config.ownership.range(rank);
            if let Some(adapters) = &config.adapters {
                assert_eq!(adapters.layer, config.layer, "mHC expert adapter and ownership layer indices differ");
                source.validate_adapter_targets(&adapters.adapter, &adapters.targets, adapters.adapter_dtype, options);
            }
            sources.push(source);
        }
        validate_ownership_aliases(&self, sources);
        let mut context = PackedExpertPartitionContext::new();
        let layers = self.layers.into_iter().enumerate().map(|(index, block)| {
            let branch = if let Some(config) = selected.get(&index) {
                let (router, source, correction_bias, options, router_input_dtype, shared) = match block.feed_forward {
                    MhcFeedForward::Dense(_) => unreachable!("all selected branch kinds were validated before partitioning"),
                    MhcFeedForward::Floating(value) => {
                        let routed = value.routed;
                        (routed.router, MixedExpertParallelSource::Native(routed.experts), routed.correction_bias,
                            routed.options, routed.router_input_dtype, value.shared)
                    },
                    MhcFeedForward::Packed(value) => {
                        let routed = value.routed;
                        (routed.router, routed.experts.into(), routed.correction_bias,
                            options_from_packed(routed.routing), routed.router_input_dtype, value.shared)
                    },
                };
                let mut experts = context.mixed_experts(source, config.ownership.clone(), rank);
                if let Some(adapters) = &config.adapters {
                    experts = experts.with_adapters(&adapters.adapter, &adapters.targets, adapters.adapter_dtype,
                        adapters.use_rslora, options, adapters.forward, adapters.backward);
                }
                let routed = ExpertParallelMoeLayer::from_expert_parts(router, experts, correction_bias, options, router_input_dtype);
                MixedMhcFeedForward::Parallel(ExpertParallelMhcFeedForward::from_parts(routed, shared))
            } else { MixedMhcFeedForward::Local(block.feed_forward) };
            MhcResidualBlock::from_parts(block.attention_connection, block.ffn_connection, block.attention,
                block.attention_norm, block.ffn_norm, branch, block.epsilon)
        }).collect();
        MhcResidualStack::from_parts(layers, self.final_norm, self.epsilon)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, Q: TransformerProjectionShape<B>,
    E: FrozenExpertGeometry<B> + Into<MixedExpertParallelSource<B>>, H: TransformerProjectionShape<B>>
    MhcResidualModel<B, P, MhcFeedForward<B, Q, E>, H> {
    /// Validate ties against embedding/head state too, before slicing or allocating any actual expert payload.
    pub fn into_mixed_expert_parallel(self, targets: &[MixedExpertParallelLayerConfig], rank: usize) -> ExpertPartitionedMhcModel<B, P, Q, E, H> {
        let mut indices = BTreeSet::new();
        let mut sources = Vec::with_capacity(targets.len());
        for target in targets {
            assert!(indices.insert(target.layer), "duplicate mHC model owned-expert layer index");
            let layer = self.stack.layers.get(target.layer).expect("mHC model owned-expert layer index exceeds loaded stack");
            sources.push(match &layer.feed_forward {
                MhcFeedForward::Dense(_) => panic!("mHC model expert ownership cannot select a dense branch"),
                MhcFeedForward::Floating(value) => MixedExpertParallelSource::Native(value.routed.experts.clone()),
                MhcFeedForward::Packed(value) => value.routed.experts.clone().into(),
            });
        }
        validate_ownership_aliases(&self, sources);
        MhcResidualModel::from_parts(self.embedding, self.stack.into_mixed_expert_parallel(targets, rank), self.head)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, Q: TransformerProjectionShape<B>, E: FrozenExpertGeometry<B>>
    ExpertPartitionedMhcStack<B, P, Q, E> {
    /// Canonical original local expert float/int/bool state IDs, with no lazy parameter readback.
    pub fn owned_expert_state_ids(&self) -> Vec<ParamId> {
        let mut ids = BTreeSet::new();
        for layer in &self.layers {
            if let MixedMhcFeedForward::Parallel(branch) = &layer.feed_forward {
                ids.extend(list_param_ids(&branch.routed.experts));
            }
        }
        ids.into_iter().collect()
    }

    /// Actual expert A/B identities, excluding quantized source payloads and non-expert adapters.
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids = BTreeSet::new();
        for layer in &self.layers {
            if let MixedMhcFeedForward::Parallel(branch) = &layer.feed_forward {
                ids.extend(branch.routed.experts.adapter_parameter_ids());
            }
        }
        ids.into_iter().collect()
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, Q: TransformerProjectionShape<B>, E: FrozenExpertGeometry<B>, H: TransformerProjectionShape<B>>
    ExpertPartitionedMhcModel<B, P, Q, E, H> {
    pub fn owned_expert_state_ids(&self) -> Vec<ParamId> { self.stack.owned_expert_state_ids() }
    pub fn expert_adapter_parameter_ids(&self) -> Vec<ParamId> { self.stack.expert_adapter_parameter_ids() }
    /// State outside the actual owned experts; membership alone does not imply replication or a reduction group.
    pub fn non_expert_state_ids(&self) -> Vec<ParamId> {
        let owned: BTreeSet<_> = self.owned_expert_state_ids().into_iter().collect();
        let all: BTreeSet<_> = list_param_ids(self).into_iter().collect();
        all.difference(&owned).copied().collect()
    }
}
