use alloc::vec::Vec;
use ruda_model::tensor::backend::Backend;
use crate::{Linear, attention::{CompressedAttentionProjection, CompressedAttentionProjectionRole}};
use super::{MhcResidualBlock, MhcResidualStack, MhcResidualModel, MhcResidualBranchShape, TransformerProjectionShape,
    TransformerAdapterConfig, AdaptedProjection, AdaptTransformerProjection, ProjectedTransformerHead,
    MhcBranchProjectionMap, MhcBranchProjectionRole};
use super::compressed_adapters::{check_options, planned_projection};

/// Actual physical compression/indexer projection in one original native mHC layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MhcAttentionAdapterTarget {
    pub layer: usize,
    pub role: CompressedAttentionProjectionRole,
}

/// Exact original ordinary/router/shared FFN role in one native residual layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MhcBranchAdapterTarget {
    pub layer: usize,
    pub role: MhcBranchProjectionRole,
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualBlock<B, P, F> {
    pub fn visit_attention_projections<'a>(&'a self, visitor: impl FnMut(CompressedAttentionProjectionRole, &'a P)) {
        self.attention.visit_projections(visitor);
    }

    pub fn map_attention_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(CompressedAttentionProjectionRole, P) -> Q) -> MhcResidualBlock<B, Q, F> {
        match self.try_map_attention_projections(|role, projection| Ok::<Q, core::convert::Infallible>(mapper(role, projection))) {
            Ok(block) => block, Err(error) => match error {},
        }
    }

    /// Change only the actual owned attention projections, retaining the exact local/parallel branch and residual leaves.
    pub fn try_map_attention_projections<Q: CompressedAttentionProjection<B>, R>(self,
        mapper: impl FnMut(CompressedAttentionProjectionRole, P) -> Result<Q, R>) -> Result<MhcResidualBlock<B, Q, F>, R> {
        let attention = self.attention.try_map_projections(mapper)?;
        Ok(MhcResidualBlock::from_parts(self.attention_connection, self.ffn_connection, attention,
            self.attention_norm, self.ffn_norm, self.feed_forward, self.epsilon))
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualStack<B, P, F> {
    pub fn visit_attention_projections<'a>(&'a self, mut visitor: impl FnMut(MhcAttentionAdapterTarget, &'a P)) {
        for (layer, block) in self.layers.iter().enumerate() {
            block.visit_attention_projections(|role, projection| visitor(MhcAttentionAdapterTarget { layer, role }, projection));
        }
    }

    pub fn map_attention_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(MhcAttentionAdapterTarget, P) -> Q) -> MhcResidualStack<B, Q, F> {
        match self.try_map_attention_projections(|role, projection| Ok::<Q, core::convert::Infallible>(mapper(role, projection))) {
            Ok(stack) => stack, Err(error) => match error {},
        }
    }

    pub fn try_map_attention_projections<Q: CompressedAttentionProjection<B>, R>(self,
        mut mapper: impl FnMut(MhcAttentionAdapterTarget, P) -> Result<Q, R>) -> Result<MhcResidualStack<B, Q, F>, R> {
        let layers = self.layers.into_iter().enumerate().map(|(layer, block)| block.try_map_attention_projections(|role, projection|
            mapper(MhcAttentionAdapterTarget { layer, role }, projection))).collect::<Result<_, R>>()?;
        Ok(MhcResidualStack::from_parts(layers, self.final_norm, self.epsilon))
    }
}

impl<B: Backend, F: MhcResidualBranchShape<B>> MhcResidualStack<B, Linear<B>, F> {
    /// Independent native LoRA/rsLoRA rank, scale, dropout and dtype on selected compression/indexer roles only.
    pub fn with_attention_adapter_plan(self, plan: &[(MhcAttentionAdapterTarget, TransformerAdapterConfig)])
        -> MhcResidualStack<B, AdaptedProjection<B>, F> {
        let mut actual = Vec::new();
        self.visit_attention_projections(|target, _| actual.push(target));
        for (index, (target, config)) in plan.iter().enumerate() {
            assert!(!plan[..index].iter().any(|(previous, _)| previous == target), "duplicate mHC attention adapter target");
            assert!(actual.contains(target), "selected mHC attention projection does not exist");
            check_options(config);
        }
        self.map_attention_projections(|target, projection| planned_projection(projection, &target, plan))
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B> + AdaptTransformerProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualStack<B, P, F> {
    /// Extend an existing mixed dense/LoRA projection graph without reinitializing any already loaded A/B.
    pub fn with_attention_adapters(self, plan: &[(MhcAttentionAdapterTarget, TransformerAdapterConfig)]) -> Self {
        for (index, (target, config)) in plan.iter().enumerate() {
            assert!(!plan[..index].iter().any(|(previous, _)| previous == target), "duplicate loaded mHC attention adapter target");
            let mut found = false;
            self.visit_attention_projections(|actual, projection| {
                if actual == *target { found = true; projection.validate_adapter(config); }
            });
            assert!(found, "selected loaded mHC attention projection does not exist");
        }
        self.map_attention_projections(|target, projection| match plan.iter().find(|(actual, _)| *actual == target) {
            Some((_, config)) => projection.with_adapter(config), None => projection,
        })
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>, H: TransformerProjectionShape<B>> MhcResidualModel<B, P, F, H> {
    pub fn try_map_attention_projections<Q: CompressedAttentionProjection<B>, R>(self,
        mapper: impl FnMut(MhcAttentionAdapterTarget, P) -> Result<Q, R>) -> Result<MhcResidualModel<B, Q, F, H>, R> {
        Ok(MhcResidualModel::from_parts(self.embedding, self.stack.try_map_attention_projections(mapper)?, self.head))
    }

    /// Independently map the actual task/vocabulary projection, preserving normalization and original head dropout.
    pub fn try_map_head<Q: TransformerProjectionShape<B>, R>(self, mapper: impl FnOnce(H) -> Result<Q, R>)
        -> Result<MhcResidualModel<B, P, F, Q>, R> {
        let head = ProjectedTransformerHead::from_projection(mapper(self.head.projection)?, self.head.normalization, self.head.dropout);
        Ok(MhcResidualModel::from_parts(self.embedding, self.stack, head))
    }
}

impl<B: Backend, F: MhcResidualBranchShape<B>, H: TransformerProjectionShape<B>> MhcResidualModel<B, Linear<B>, F, H> {
    pub fn with_attention_adapter_plan(self, plan: &[(MhcAttentionAdapterTarget, TransformerAdapterConfig)])
        -> MhcResidualModel<B, AdaptedProjection<B>, F, H> {
        MhcResidualModel::from_parts(self.embedding, self.stack.with_attention_adapter_plan(plan), self.head)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualModel<B, P, F, Linear<B>> {
    /// Attach only the explicit independent head A/B; embedding and unselected model flags are untouched.
    pub fn with_head_adapter(self, config: &TransformerAdapterConfig) -> MhcResidualModel<B, P, F, AdaptedProjection<B>> {
        check_options(config);
        let projection = AdaptedProjection::Dense(self.head.projection).with_adapter(config);
        let head = ProjectedTransformerHead::from_projection(projection, self.head.normalization, self.head.dropout);
        MhcResidualModel::from_parts(self.embedding, self.stack, head)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>, H: AdaptTransformerProjection<B>> MhcResidualModel<B, P, F, H> {
    pub fn with_head_adapter_in_place(self, config: &TransformerAdapterConfig) -> Self {
        self.head.projection.validate_adapter(config);
        let head = ProjectedTransformerHead::from_projection(self.head.projection.with_adapter(config), self.head.normalization, self.head.dropout);
        MhcResidualModel::from_parts(self.embedding, self.stack, head)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B> + AdaptTransformerProjection<B>, F: MhcResidualBranchShape<B>, H: TransformerProjectionShape<B>>
    MhcResidualModel<B, P, F, H> {
    pub fn with_attention_adapters(self, plan: &[(MhcAttentionAdapterTarget, TransformerAdapterConfig)]) -> Self {
        MhcResidualModel::from_parts(self.embedding, self.stack.with_attention_adapters(plan), self.head)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcBranchProjectionMap<B>> MhcResidualBlock<B, P, F> {
    pub fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mapper: impl FnMut(MhcBranchProjectionRole, F::Projection) -> Result<Q, R>) -> Result<MhcResidualBlock<B, P, F::Mapped<Q>>, R> {
        let feed = self.feed_forward.try_map_branch_projections(mapper)?;
        Ok(MhcResidualBlock::from_parts(self.attention_connection, self.ffn_connection, self.attention,
            self.attention_norm, self.ffn_norm, feed, self.epsilon))
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcBranchProjectionMap<B>> MhcResidualStack<B, P, F> {
    pub fn visit_branch_projections<'a>(&'a self, mut visitor: impl FnMut(MhcBranchAdapterTarget, &'a F::Projection)) {
        for (layer, block) in self.layers.iter().enumerate() {
            block.feed_forward.visit_branch_projections(|role, projection| visitor(MhcBranchAdapterTarget { layer, role }, projection));
        }
    }

    pub fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mut mapper: impl FnMut(MhcBranchAdapterTarget, F::Projection) -> Result<Q, R>) -> Result<MhcResidualStack<B, P, F::Mapped<Q>>, R> {
        let layers = self.layers.into_iter().enumerate().map(|(layer, block)| block.try_map_branch_projections(|role, projection|
            mapper(MhcBranchAdapterTarget { layer, role }, projection))).collect::<Result<_, R>>()?;
        Ok(MhcResidualStack::from_parts(layers, self.final_norm, self.epsilon))
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcBranchProjectionMap<B, Projection = Linear<B>>> MhcResidualStack<B, P, F> {
    /// Add real A/B to selected actual FFN/router/shared roles, leaving original floating/packed owned experts untouched.
    pub fn with_branch_adapter_plan(self, plan: &[(MhcBranchAdapterTarget, TransformerAdapterConfig)])
        -> MhcResidualStack<B, P, F::Mapped<AdaptedProjection<B>>> {
        let mut actual = Vec::new();
        self.visit_branch_projections(|target, _| actual.push(target));
        for (index, (target, config)) in plan.iter().enumerate() {
            assert!(!plan[..index].iter().any(|(previous, _)| previous == target), "duplicate mHC branch adapter target");
            assert!(actual.contains(target), "selected actual mHC branch projection does not exist");
            check_options(config);
        }
        match self.try_map_branch_projections(|target, projection| Ok::<_, core::convert::Infallible>(planned_projection(projection, &target, plan))) {
            Ok(stack) => stack, Err(error) => match error {},
        }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcBranchProjectionMap<B>> MhcResidualStack<B, P, F>
where F::Projection: AdaptTransformerProjection<B> {
    /// Preserve original packed projection formats and present adapters; reject re-adapting a selected loaded role.
    pub fn with_branch_adapters(self, plan: &[(MhcBranchAdapterTarget, TransformerAdapterConfig)]) -> MhcResidualStack<B, P, F::Mapped<F::Projection>> {
        for (index, (target, config)) in plan.iter().enumerate() {
            assert!(!plan[..index].iter().any(|(previous, _)| previous == target), "duplicate loaded mHC branch adapter target");
            let mut found = false;
            self.visit_branch_projections(|actual, projection| {
                if actual == *target { found = true; projection.validate_adapter(config); }
            });
            assert!(found, "selected loaded mHC branch projection does not exist");
        }
        match self.try_map_branch_projections(|target, projection| Ok::<_, core::convert::Infallible>(
            match plan.iter().find(|(actual, _)| *actual == target) { Some((_, config)) => projection.with_adapter(config), None => projection })) {
            Ok(stack) => stack, Err(error) => match error {},
        }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcBranchProjectionMap<B>, H: TransformerProjectionShape<B>> MhcResidualModel<B, P, F, H> {
    pub fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mapper: impl FnMut(MhcBranchAdapterTarget, F::Projection) -> Result<Q, R>) -> Result<MhcResidualModel<B, P, F::Mapped<Q>, H>, R> {
        Ok(MhcResidualModel::from_parts(self.embedding, self.stack.try_map_branch_projections(mapper)?, self.head))
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcBranchProjectionMap<B, Projection = Linear<B>>, H: TransformerProjectionShape<B>> MhcResidualModel<B, P, F, H> {
    pub fn with_branch_adapter_plan(self, plan: &[(MhcBranchAdapterTarget, TransformerAdapterConfig)])
        -> MhcResidualModel<B, P, F::Mapped<AdaptedProjection<B>>, H> {
        MhcResidualModel::from_parts(self.embedding, self.stack.with_branch_adapter_plan(plan), self.head)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcBranchProjectionMap<B>, H: TransformerProjectionShape<B>> MhcResidualModel<B, P, F, H>
where F::Projection: AdaptTransformerProjection<B> {
    pub fn with_branch_adapters(self, plan: &[(MhcBranchAdapterTarget, TransformerAdapterConfig)]) -> MhcResidualModel<B, P, F::Mapped<F::Projection>, H> {
        MhcResidualModel::from_parts(self.embedding, self.stack.with_branch_adapters(plan), self.head)
    }
}
