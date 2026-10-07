//! Differentiable tensor collectives over an explicit rank communicator.
use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind, unary},
};
use core::marker::PhantomData;
use ruda_tensor::{
    AsIndex, Backend, TensorMetadata,
    api::Tensor,
    collective::{BroadcastTensorCollective, ReplicatedTensorCollective, TensorCollective},
    primitive::TensorPrimitive,
};

mod scope;
pub use scope::{CollectiveScope,ScopedTensorCollective,ScopedCollectiveError};

#[derive(Debug)]
struct Collective<C>(PhantomData<C>);

#[derive(Debug)]
struct AllReduce<C>(PhantomData<C>);

#[derive(Debug)]
struct Broadcast<C>(PhantomData<C>);

impl<B: Backend, C: BroadcastTensorCollective<B>> Backward<B, 1> for Broadcast<C> {
    type State = (C, u32);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let (communicator, root) = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let grad = communicator
                .all_reduce_sum(grad)
                .unwrap_or_else(|error| panic!("broadcast backward failed: {error:?}"));
            if communicator.rank() == root {
                grad
            } else {
                B::float_zeros(grad.shape(), &B::float_device(&grad), grad.dtype().into())
            }
        });
    }
}

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
    type State = (C, bool, bool);

    fn ordered_backward(state:&Self::State) -> bool {state.2}

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let (communicator, gathered, _) = ops.state;
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
    ordered: bool,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: TensorCollective<B>,
{
    let scope=communicator.autodiff_context().and_then(|context|context.downcast_ref::<CollectiveScope<B,S>>()).cloned();
    let ordered=ordered || scope.is_some();
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
        OpsKind::Tracked(prep) => prep.finish((communicator, gathered, ordered), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    let output=Tensor::from_primitive(TensorPrimitive::Float(output));
    if let Some(scope)=scope {scope.capture(output.clone());}
    Ok(output)
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
    apply(tensor, communicator, true, false)
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
    apply(tensor, communicator, false, false)
}

/// Original all-gather/SUM reduce-scatter derivative with reverse forward-creation backward scheduling.
/// Ranks must use matching collective calls and tracking. Unlike loss-tree traversal order, this
/// schedule does not depend on each rank's actual number of supervised rows or projection chunks.
/// Graphs not using an ordered collective retain their original depth schedule and rounding path.
pub fn all_gather_ordered<B,S,C,const D:usize>(tensor:Tensor<Autodiff<B,S>,D>,communicator:C)
    -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where B:Backend,S:CheckpointStrategy,C:TensorCollective<B> {
    apply(tensor,communicator,true,true)
}

/// Original SUM reduce-scatter/all-gather derivative with creation-ordered backward for this graph.
pub fn reduce_scatter_sum_ordered<B,S,C,const D:usize>(tensor:Tensor<Autodiff<B,S>,D>,communicator:C)
    -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where B:Backend,S:CheckpointStrategy,C:TensorCollective<B> {
    apply(tensor,communicator,false,true)
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

/// Broadcast root's tensor; backward sums rank-local gradients into root's input.
/// Non-root inputs are placeholders and receive zeros. All ranks must use matching
/// shapes, dtypes, gradient tracking, root and collective order.
pub fn broadcast<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>,
    communicator: C,
    root: u32,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where
    B: Backend,
    S: CheckpointStrategy,
    C: BroadcastTensorCollective<B>,
{
    let tensor = tensor.into_primitive().tensor();
    let output = communicator.broadcast_float(tensor.primitive, root)?;
    let output = match Broadcast::<C>(PhantomData)
        .prepare::<S>([tensor.node])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((communicator, root), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}
