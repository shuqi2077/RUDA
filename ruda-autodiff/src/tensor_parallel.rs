//! Tensor-parallel regions for one replicated logical loss.
//! These are distinct from data-parallel collectives, which sum independent rank-local losses.
use crate::{
    Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients, ops::{Backward, Ops, OpsKind, unary},
};
use core::marker::PhantomData;
use ruda_tensor::{
    AsIndex, Backend, TensorMetadata, api::Tensor,
    primitive::TensorPrimitive, tensor::FloatTensor,
};
/// Rank-aware transport contract accepted by tensor-parallel regions.
pub use ruda_tensor::collective::BroadcastTensorCollective;

#[derive(Clone, Copy, Debug)]
enum Region { Copy, Reduce, Scatter, Gather }
#[derive(Debug)]
struct Parallel<C, const D: usize>(PhantomData<C>);

fn transform<B, C, const D: usize>(
    value: FloatTensor<B>, communicator: &C, operation: Region, dim: usize,
) -> Result<FloatTensor<B>, C::Error>
where B: Backend, C: BroadcastTensorCollective<B>,
{
    match operation {
        Region::Copy => Ok(value),
        Region::Reduce => communicator.all_reduce_sum(value),
        Region::Gather => {
            let tensor = Tensor::<B, D>::from_primitive(TensorPrimitive::Float(value)).swap_dims(0, dim);
            let gathered = communicator.all_gather_float(tensor.into_primitive().tensor())?;
            Ok(Tensor::<B, D>::from_primitive(TensorPrimitive::Float(gathered))
                .swap_dims(0, dim).into_primitive().tensor())
        }
        Region::Scatter => {
            let tensor = Tensor::<B, D>::from_primitive(TensorPrimitive::Float(value));
            let size = tensor.dims()[dim];
            let world = communicator.world_size() as usize;
            let rank = communicator.rank() as usize;
            assert!(world > 0 && rank < world, "invalid tensor-parallel topology");
            assert!(size > 0 && size % world == 0, "parallel axis must split into nonempty equal shards");
            let width = size / world;
            Ok(tensor.slice_dim(dim, rank * width..(rank + 1) * width).into_primitive().tensor())
        }
    }
}

impl<B: Backend, C: BroadcastTensorCollective<B>, const D: usize> Backward<B, 1> for Parallel<C, D> {
    type State = (C, Region, usize);
    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let (communicator, operation, dim) = ops.state;
        let inverse = match operation {
            Region::Copy => Region::Reduce, Region::Reduce => Region::Copy,
            Region::Scatter => Region::Gather, Region::Gather => Region::Scatter,
        };
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            transform::<B, C, D>(grad, &communicator, inverse, dim)
                .unwrap_or_else(|error| panic!("tensor-parallel backward failed: {error:?}"))
        });
    }
}

fn apply<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>, communicator: C, operation: Region, dim: usize,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where B: Backend, S: CheckpointStrategy, C: BroadcastTensorCollective<B>,
{
    assert!(D > 0 && dim < D, "parallel region requires a valid tensor axis");
    assert!(communicator.world_size() > 0 && communicator.rank() < communicator.world_size(), "invalid tensor-parallel topology");
    let tensor = tensor.into_primitive().tensor();
    let output = transform::<B, C, D>(tensor.primitive, &communicator, operation, dim)?;
    let output = match Parallel::<C, D>(PhantomData).prepare::<S>([tensor.node]).compute_bound().stateful() {
        OpsKind::Tracked(prep) => prep.finish((communicator, operation, dim), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Replicate an input into a model-parallel region; sum shard contributions in backward.
/// All ranks compute the same logical loss and enter backward in the same order.
pub fn copy_to_region<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>, communicator: C,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where B: Backend, S: CheckpointStrategy, C: BroadcastTensorCollective<B>,
{ apply(tensor, communicator, Region::Copy, 0) }

/// Sum output contributions; backward passes the replicated gradient through unchanged.
/// Unlike a data-parallel all-reduce, this does not sum replicated losses again.
pub fn reduce_from_region<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>, communicator: C,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where B: Backend, S: CheckpointStrategy, C: BroadcastTensorCollective<B>,
{ apply(tensor, communicator, Region::Reduce, 0) }

/// Select this rank's equal contiguous axis shard; backward gathers all shards without summing.
pub fn scatter_to_region<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>, communicator: C, dim: impl AsIndex,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where B: Backend, S: CheckpointStrategy, C: BroadcastTensorCollective<B>,
{ apply(tensor, communicator, Region::Scatter, dim.expect_dim_index(D)) }

/// Gather equal rank-ordered shards; backward selects this rank's replicated-output gradient.
pub fn gather_from_region<B, S, C, const D: usize>(
    tensor: Tensor<Autodiff<B, S>, D>, communicator: C, dim: impl AsIndex,
) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
where B: Backend, S: CheckpointStrategy, C: BroadcastTensorCollective<B>,
{ apply(tensor, communicator, Region::Gather, dim.expect_dim_index(D)) }
