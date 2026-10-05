//! Differentiable tensor collectives over an explicit rank communicator.
use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind, unary},
};
use core::marker::PhantomData;
use ruda_tensor::{
    AsIndex, Backend,
    api::Tensor,
    collective::{ReplicatedTensorCollective, TensorCollective},
    primitive::TensorPrimitive,
};

#[derive(Debug)]
struct Collective<C>(PhantomData<C>);

#[derive(Debug)]
struct AllReduce<C>(PhantomData<C>);

impl<B: Backend, C: ReplicatedTensorCollective<B>> Backward<B, 1> for AllReduce<C> {
    type State = C;

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            ops.state
                .all_reduce_sum(grad)
                .unwrap_or_else(|error| panic!("all-reduce backward failed: {error:?}"))
        });
    }
}

impl<B: Backend, C: TensorCollective<B>> Backward<B, 1> for Collective<C> {
    type State = (C, bool);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let (communicator, gathered) = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let result = if gathered {
                communicator.reduce_scatter_sum(grad)
            } else {
                communicator.all_gather_float(grad)
            };
            result.unwrap_or_else(|error| panic!("collective backward failed: {error:?}"))
        });
    }
}

fn apply<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
    gathered: bool,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    let tensor = tensor.into_primitive().tensor();
    let output = if gathered {
        communicator.all_gather_float(tensor.primitive)
    } else {
        communicator.reduce_scatter_sum(tensor.primitive)
    }?;
    let output = match Collective::<C>(PhantomData)
        .prepare::<S>([tensor.node])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((communicator, gathered), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Gather leading-axis shards; backward sums and scatters all ranks' gradients.
/// All ranks must use matching shapes, dtypes, gradient tracking and operation order.
pub fn all_gather<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    apply(tensor, communicator, true)
}

/// Sum and scatter leading-axis shards; backward gathers all ranks' gradients.
/// All ranks must use matching shapes, dtypes, gradient tracking and operation order.
pub fn reduce_scatter_sum<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    apply(tensor, communicator, false)
}

/// Average across ranks and scatter; the original scalar-division backward scales gradients.
pub fn reduce_scatter_mean<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    let world_size = communicator.world_size();
    reduce_scatter_sum(tensor, communicator).map(|tensor| tensor.div_scalar(world_size))
}

/// Gather shards along the selected axis, retaining the original tensor axis order.
/// Uses the original tensor swap-dims operations and shared collective backward.
pub fn all_gather_dim<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
    dim: impl AsIndex,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    let dim = dim.expect_dim_index(D);
    all_gather(tensor.swap_dims(0, dim), communicator).map(|tensor| tensor.swap_dims(0, dim))
}

/// Sum and scatter equal shards along the selected axis, with shared gather backward.
pub fn reduce_scatter_sum_dim<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
    dim: impl AsIndex,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    let dim = dim.expect_dim_index(D);
    reduce_scatter_sum(tensor.swap_dims(0, dim), communicator)
        .map(|tensor| tensor.swap_dims(0, dim))
}

/// Average and scatter equal shards along the selected axis, including gradient scaling.
pub fn reduce_scatter_mean_dim<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
    dim: impl AsIndex,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    let dim = dim.expect_dim_index(D);
    reduce_scatter_mean(tensor.swap_dims(0, dim), communicator)
        .map(|tensor| tensor.swap_dims(0, dim))
}

/// Sum replicated tensor elements; backward sums gradients from all rank-local losses.
/// All ranks use matching shapes, dtypes, gradient tracking and collective order.
pub fn all_reduce_sum<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: ReplicatedTensorCollective<B>,
{
    let tensor = tensor.into_primitive().tensor();
    let output = communicator.all_reduce_sum(tensor.primitive)?;
    let output = match AllReduce::<C>(PhantomData)
        .prepare::<S>([tensor.node])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(communicator, output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Average replicated tensor elements, including original scalar-division backward.
pub fn all_reduce_mean<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: ReplicatedTensorCollective<B>,
{
    let world_size = communicator.world_size();
    all_reduce_sum(tensor, communicator).map(|tensor| tensor.div_scalar(world_size))
}
