use ruda_model::tensor::backend::Backend;
use crate::{NativeMoeLayer, Nf4MoeLayer, FrozenExpertGeometry};
use super::{MhcResidualBranchShape, MhcFeedForward, PackedMhcFeedForward, NativeMoeFeedForward, ProjectedFeedForward,
    TransformerProjectionShape, FeedForwardAdapterTarget};

/// Actual ordinary FFN, router or shared FFN projection; expert cube adapters retain their existing separate contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MhcBranchProjectionRole {
    FeedForward(FeedForwardAdapterTarget),
    Router,
    Shared(FeedForwardAdapterTarget),
}

/// Visit/map original non-expert projections without changing resident expert payloads or dispatch policies.
pub trait MhcBranchProjectionMap<B: Backend>: MhcResidualBranchShape<B> {
    type Projection: TransformerProjectionShape<B>;
    type Mapped<Q: TransformerProjectionShape<B>>: MhcResidualBranchShape<B>;
    fn visit_branch_projections<'a>(&'a self, visitor: impl FnMut(MhcBranchProjectionRole, &'a Self::Projection));
    fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mapper: impl FnMut(MhcBranchProjectionRole, Self::Projection) -> Result<Q, R>) -> Result<Self::Mapped<Q>, R>;
}

pub(super) fn visit_feed<'a, B: Backend, P: TransformerProjectionShape<B>>(feed: &'a ProjectedFeedForward<B, P>, shared: bool,
    mut visitor: impl FnMut(MhcBranchProjectionRole, &'a P)) {
    let role = |target| if shared { MhcBranchProjectionRole::Shared(target) } else { MhcBranchProjectionRole::FeedForward(target) };
    visitor(role(FeedForwardAdapterTarget::Up), &feed.up);
    if let Some(gate) = &feed.gate { visitor(role(FeedForwardAdapterTarget::Gate), gate); }
    visitor(role(FeedForwardAdapterTarget::Down), &feed.down);
}

pub(super) fn map_feed<B: Backend, P: TransformerProjectionShape<B>, Q: TransformerProjectionShape<B>, R>(
    feed: ProjectedFeedForward<B, P>, shared: bool, mut mapper: impl FnMut(MhcBranchProjectionRole, P) -> Result<Q, R>)
    -> Result<ProjectedFeedForward<B, Q>, R> {
    let role = |target| if shared { MhcBranchProjectionRole::Shared(target) } else { MhcBranchProjectionRole::FeedForward(target) };
    let up = mapper(role(FeedForwardAdapterTarget::Up), feed.up)?;
    let gate = feed.gate.map(|gate| mapper(role(FeedForwardAdapterTarget::Gate), gate)).transpose()?;
    let down = mapper(role(FeedForwardAdapterTarget::Down), feed.down)?;
    Ok(ProjectedFeedForward::from_projections(up, gate, down, feed.activation, feed.dropout))
}

impl<B: Backend, P: TransformerProjectionShape<B>> MhcBranchProjectionMap<B> for ProjectedFeedForward<B, P> {
    type Projection = P;
    type Mapped<Q: TransformerProjectionShape<B>> = ProjectedFeedForward<B, Q>;
    fn visit_branch_projections<'a>(&'a self, visitor: impl FnMut(MhcBranchProjectionRole, &'a P)) { visit_feed(self, false, visitor); }
    fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mapper: impl FnMut(MhcBranchProjectionRole, P) -> Result<Q, R>) -> Result<Self::Mapped<Q>, R> { map_feed(self, false, mapper) }
}

impl<B: Backend, P: TransformerProjectionShape<B>> MhcBranchProjectionMap<B> for NativeMoeFeedForward<B, P> {
    type Projection = P;
    type Mapped<Q: TransformerProjectionShape<B>> = NativeMoeFeedForward<B, Q>;
    fn visit_branch_projections<'a>(&'a self, mut visitor: impl FnMut(MhcBranchProjectionRole, &'a P)) {
        visitor(MhcBranchProjectionRole::Router, &self.routed.router);
        if let Some(shared) = &self.shared { visit_feed(shared, true, visitor); }
    }
    fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mut mapper: impl FnMut(MhcBranchProjectionRole, P) -> Result<Q, R>) -> Result<Self::Mapped<Q>, R> {
        let routed = self.routed;
        let routed = NativeMoeLayer::from_parts(mapper(MhcBranchProjectionRole::Router, routed.router)?, routed.experts,
            routed.correction_bias, routed.options, routed.router_input_dtype);
        let shared = self.shared.map(|feed| map_feed(feed, true, mapper)).transpose()?;
        Ok(NativeMoeFeedForward::from_parts(routed, shared))
    }
}

impl<B: Backend, P: TransformerProjectionShape<B>, E: FrozenExpertGeometry<B>> MhcBranchProjectionMap<B> for PackedMhcFeedForward<B, P, E> {
    type Projection = P;
    type Mapped<Q: TransformerProjectionShape<B>> = PackedMhcFeedForward<B, Q, E>;
    fn visit_branch_projections<'a>(&'a self, mut visitor: impl FnMut(MhcBranchProjectionRole, &'a P)) {
        visitor(MhcBranchProjectionRole::Router, &self.routed.router);
        if let Some(shared) = &self.shared { visit_feed(shared, true, visitor); }
    }
    fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mut mapper: impl FnMut(MhcBranchProjectionRole, P) -> Result<Q, R>) -> Result<Self::Mapped<Q>, R> {
        let routed = self.routed;
        let routed = Nf4MoeLayer::from_parts(mapper(MhcBranchProjectionRole::Router, routed.router)?, routed.experts,
            routed.correction_bias, routed.routing, routed.router_input_dtype);
        let shared = self.shared.map(|feed| map_feed(feed, true, mapper)).transpose()?;
        Ok(PackedMhcFeedForward::from_parts(routed, shared))
    }
}

impl<B: Backend, P: TransformerProjectionShape<B>, E: FrozenExpertGeometry<B>> MhcBranchProjectionMap<B> for MhcFeedForward<B, P, E> {
    type Projection = P;
    type Mapped<Q: TransformerProjectionShape<B>> = MhcFeedForward<B, Q, E>;
    fn visit_branch_projections<'a>(&'a self, visitor: impl FnMut(MhcBranchProjectionRole, &'a P)) {
        match self { Self::Dense(value) => value.visit_branch_projections(visitor),
            Self::Floating(value) => value.visit_branch_projections(visitor), Self::Packed(value) => value.visit_branch_projections(visitor) }
    }
    fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
        mapper: impl FnMut(MhcBranchProjectionRole, P) -> Result<Q, R>) -> Result<Self::Mapped<Q>, R> {
        Ok(match self { Self::Dense(value) => MhcFeedForward::Dense(value.try_map_branch_projections(mapper)?),
            Self::Floating(value) => MhcFeedForward::Floating(value.try_map_branch_projections(mapper)?),
            Self::Packed(value) => MhcFeedForward::Packed(value.try_map_branch_projections(mapper)?) })
    }
}

#[cfg(feature = "tensor-parallel")]
mod parallel {
    use super::*;
    use crate::expert_parallel::{ExpertParallelGeometry, ExpertParallelMoeLayer};
    use super::super::{ExpertParallelMhcFeedForward, MixedMhcFeedForward};

    impl<B: Backend, P: TransformerProjectionShape<B>, E: ExpertParallelGeometry<B>> MhcBranchProjectionMap<B> for ExpertParallelMhcFeedForward<B, P, E> {
        type Projection = P;
        type Mapped<Q: TransformerProjectionShape<B>> = ExpertParallelMhcFeedForward<B, Q, E>;
        fn visit_branch_projections<'a>(&'a self, mut visitor: impl FnMut(MhcBranchProjectionRole, &'a P)) {
            visitor(MhcBranchProjectionRole::Router, &self.routed.router);
            if let Some(shared) = &self.shared { visit_feed(shared, true, visitor); }
        }
        fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
            mut mapper: impl FnMut(MhcBranchProjectionRole, P) -> Result<Q, R>) -> Result<Self::Mapped<Q>, R> {
            let routed = self.routed;
            let routed = ExpertParallelMoeLayer::from_expert_parts(mapper(MhcBranchProjectionRole::Router, routed.router)?, routed.experts,
                routed.correction_bias, routed.options, routed.router_input_dtype);
            let shared = self.shared.map(|feed| map_feed(feed, true, mapper)).transpose()?;
            Ok(ExpertParallelMhcFeedForward::from_parts(routed, shared))
        }
    }

    impl<B: Backend, P: TransformerProjectionShape<B>, E: ExpertParallelGeometry<B>, L: MhcBranchProjectionMap<B, Projection = P>>
        MhcBranchProjectionMap<B> for MixedMhcFeedForward<B, L, P, E> {
        type Projection = P;
        type Mapped<Q: TransformerProjectionShape<B>> = MixedMhcFeedForward<B, L::Mapped<Q>, Q, E>;
        fn visit_branch_projections<'a>(&'a self, visitor: impl FnMut(MhcBranchProjectionRole, &'a P)) {
            match self { Self::Local(value) => value.visit_branch_projections(visitor), Self::Parallel(value) => value.visit_branch_projections(visitor) }
        }
        fn try_map_branch_projections<Q: TransformerProjectionShape<B>, R>(self,
            mapper: impl FnMut(MhcBranchProjectionRole, P) -> Result<Q, R>) -> Result<Self::Mapped<Q>, R> {
            Ok(match self { Self::Local(value) => MixedMhcFeedForward::Local(value.try_map_branch_projections(mapper)?),
                Self::Parallel(value) => MixedMhcFeedForward::Parallel(value.try_map_branch_projections(mapper)?) })
        }
    }
}
