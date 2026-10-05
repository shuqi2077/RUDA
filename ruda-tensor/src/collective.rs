//! Explicit rank communicators for floating tensor collectives.
use crate::{Backend, tensor::FloatTensor};
use core::fmt::Debug;

/// Matching rank-ordered collectives used by differentiable sharded tensors.
/// All participating ranks must enter forward and backward in the same order.
pub trait TensorCollective<B: Backend>: Clone + Debug + Send + 'static {
    /// Transport or tensor-contract failure.
    type Error: Debug;
    /// Number of participating ranks, greater than zero.
    fn world_size(&self) -> u32;
    /// Gather equal leading-axis shards in rank order, retaining the input dtype.
    fn all_gather_float(&self, value: FloatTensor<B>) -> Result<FloatTensor<B>, Self::Error>;
    /// Sum tensors across ranks and return this rank's equal leading-axis shard.
    fn reduce_scatter_sum(&self, value: FloatTensor<B>) -> Result<FloatTensor<B>, Self::Error>;
}
