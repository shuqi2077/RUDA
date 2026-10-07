//! Explicit rank communicators for floating tensor collectives.
use crate::{Backend, tensor::{FloatTensor, IntTensor}};
use core::fmt::Debug;

/// Matching rank-ordered collectives used by differentiable sharded tensors.
/// All participating ranks must enter forward and backward in the same order.
pub trait TensorCollective<B: Backend>: Clone + Debug + Send + 'static {
    /// Transport or tensor-contract failure.
    type Error: Debug;
    /// Number of participating ranks, greater than zero.
    fn world_size(&self) -> u32;
    /// Optional typed AD execution context carried by an explicit communicator wrapper.
    /// Native transports and ordinary AD calls retain no context by default.
    fn autodiff_context(&self) -> Option<&(dyn core::any::Any+Send+Sync)> {None}
    /// Gather equal leading-axis shards in rank order, retaining the input dtype.
    fn all_gather_float(&self, value: FloatTensor<B>) -> Result<FloatTensor<B>, Self::Error>;
    /// Sum tensors across ranks and return this rank's equal leading-axis shard.
    fn reduce_scatter_sum(&self, value: FloatTensor<B>) -> Result<FloatTensor<B>, Self::Error>;
}

/// Replicated reductions in addition to leading-axis shard collectives.
pub trait ReplicatedTensorCollective<B: Backend>: TensorCollective<B> {
    /// Sum corresponding tensor elements across ranks, retaining shape and dtype.
    fn all_reduce_sum(&self, value: FloatTensor<B>) -> Result<FloatTensor<B>, Self::Error>;
}

/// Root-owned broadcasts with replicated reductions for their backward pass.
pub trait BroadcastTensorCollective<B: Backend>: ReplicatedTensorCollective<B> {
    /// This communicator's rank.
    fn rank(&self) -> u32;
    /// Broadcast root's tensor while retaining each rank's input snapshot.
    fn broadcast_float(
        &self,
        value: FloatTensor<B>,
        root: u32,
    ) -> Result<FloatTensor<B>, Self::Error>;
}

/// Exact packed-integer storage transport in addition to floating rank collectives.
/// Immutable packed parameters do not acquire a floating surrogate or derivative.
pub trait IntegerTensorCollective<B: Backend>: BroadcastTensorCollective<B> {
    /// Rank-ordered leading-axis gather retaining the original integer dtype and
    /// bit patterns. Implementations must not cast packed words through floating point.
    fn all_gather_int(&self, value: IntTensor<B>) -> Result<IntTensor<B>, Self::Error>;
}
