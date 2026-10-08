use super::*;
use ruda_model::{module::ModuleDisplay, tensor::IntegerTensorCollective};
use crate::transformer::{MhcResidualBranchShape, DenseFeedForward, AdaptedFeedForward,
    ProjectedFeedForward, NativeMoeFeedForward};

/// Partition every actual branch parameter through the same source-ID context as attention/mHC/embedding.
pub trait ShardMhcResidualBranch<B: Backend>: MhcResidualBranchShape<B> {
    type Sharded: FullyShardedModule<B> + ModuleDisplay;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded;
}

/// Reconstruct only the current branch's exact native graph on the actual execution backend.
pub trait GatherMhcResidualBranch<AB: Backend, B: Backend>: FullyShardedModule<AB> + ModuleDisplay {
    type Gathered: MhcResidualBranchShape<AB>;
    fn gather_branch<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error>;
}

impl<B: Backend> ShardMhcResidualBranch<B> for DenseFeedForward<B> {
    type Sharded = FullyShardedFeedForward<B>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.feed_forward(self) }
}
impl<B: Backend> ShardMhcResidualBranch<B> for AdaptedFeedForward<B> {
    type Sharded = FullyShardedFeedForward<B>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.adapted_feed_forward(self) }
}
impl<B: Backend, P: ShardTransformerProjection<B>> ShardMhcResidualBranch<B> for ProjectedFeedForward<B, P> {
    type Sharded = FullyShardedProjectedFeedForward<B, P::Sharded>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.awq_feed_forward(self) }
}
impl<B: Backend, P: ShardTransformerProjection<B>> ShardMhcResidualBranch<B> for NativeMoeFeedForward<B, P> {
    type Sharded = FullyShardedNativeMoeFeedForward<B, P::Sharded>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.moe_feed_forward(self) }
}

macro_rules! gather_mhc_branches {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*> GatherMhcResidualBranch<$backend, B> for FullyShardedFeedForward<$backend> {
            type Gathered = AdaptedFeedForward<$backend>;
            fn gather_branch<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> { self.$gather(communicator) }
        }
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> GatherMhcResidualBranch<$backend, B> for FullyShardedProjectedFeedForward<$backend, P> {
            type Gathered = ProjectedFeedForward<$backend, P::Gathered>;
            fn gather_branch<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> { self.$gather(communicator) }
        }
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> GatherMhcResidualBranch<$backend, B> for FullyShardedNativeMoeFeedForward<$backend, P> {
            type Gathered = NativeMoeFeedForward<$backend, P::Gathered>;
            fn gather_branch<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> { self.$gather(communicator) }
        }
    };
}
gather_mhc_branches!(B, [B: Backend], gather_inference);
gather_mhc_branches!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);
