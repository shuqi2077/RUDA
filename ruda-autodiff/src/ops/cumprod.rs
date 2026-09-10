use alloc::vec;
use core::ops::Range;
use crate::{
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    graph::NodeId,
    ops::{Backward, Ops, OpsKind, unary},
    tensor::AutodiffTensor,
};
use ruda_tensor::{Backend, Slice, TensorMetadata, tensor::FloatTensor};

fn slice_axis<B: Backend>(tensor: FloatTensor<B>, dim: usize, range: Range<usize>) -> FloatTensor<B> {
    let mut slices = vec![Slice::full(); tensor.shape().num_dims()];
    slices[dim] = Slice::from(range);
    B::float_slice(tensor, &slices)
}

fn cumprod_gradient<B: Backend>(
    input: FloatTensor<B>,
    mut grad: FloatTensor<B>,
    dim: usize,
) -> FloatTensor<B> {
    let shape = input.shape();
    if shape.num_elements() == 0 || shape[dim] == 1 {
        return grad;
    }
    let len = shape[dim];
    let mut edge_shape = shape;
    edge_shape[dim] = 1;
    let ones = B::float_ones(edge_shape, &B::float_device(&input), input.dtype().into());
    let prefix = slice_axis::<B>(B::float_cumprod(input.clone(), dim), dim, 0..len - 1);
    let prefix = B::float_cat(vec![ones.clone(), prefix], dim);
    let next_input = slice_axis::<B>(input, dim, 1..len);
    let mut weights = B::float_cat(vec![next_input, ones], dim);

    // r[i] = grad[i] + input[i + 1] * r[i + 1]; dx[i] = prefix[i] * r[i].
    // Compose adjacent affine maps at doubling distances, without division.
    let mut distance = 1;
    loop {
        let count = len - distance;
        let left = slice_axis::<B>(grad.clone(), dim, 0..count);
        let right = slice_axis::<B>(grad.clone(), dim, distance..len);
        let weight = slice_axis::<B>(weights.clone(), dim, 0..count);
        let updated = B::float_add(left, B::float_mul(weight, right));
        let tail = slice_axis::<B>(grad, dim, count..len);
        grad = B::float_cat(vec![updated, tail], dim);

        if distance >= count {
            break;
        }
        let next_distance = distance * 2;
        let next_count = len - next_distance;
        let left = slice_axis::<B>(weights.clone(), dim, 0..next_count);
        let right = slice_axis::<B>(weights.clone(), dim, distance..distance + next_count);
        let updated = B::float_mul(left, right);
        let tail = slice_axis::<B>(weights, dim, next_count..len);
        weights = B::float_cat(vec![updated, tail], dim);
        distance = next_distance;
    }
    B::float_mul(prefix, grad)
}

#[derive(Debug)]
struct CumProd;

impl<B: Backend> Backward<B, 1> for CumProd {
    type State = (NodeId, usize);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let (input_id, dim) = ops.state;
        let input = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(input_id);
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            cumprod_gradient::<B>(input, grad, dim)
        });
    }
}

pub(super) fn cumprod<B: Backend, C: CheckpointStrategy>(
    tensor: AutodiffTensor<B>,
    dim: usize,
) -> AutodiffTensor<B> {
    match CumProd.prepare::<C>([tensor.node.clone()]).compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let input_id = prep.checkpoint(&tensor);
            prep.finish((input_id, dim), B::float_cumprod(tensor.primitive, dim))
        }
        OpsKind::UnTracked(prep) => prep.finish(B::float_cumprod(tensor.primitive, dim)),
    }
}
