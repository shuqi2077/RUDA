use alloc::vec;
use crate::{
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    graph::NodeId,
    ops::{Backward, Ops, OpsKind, unary},
    tensor::AutodiffTensor,
};
use ruda_tensor::{Backend, Shape, Slice, TensorMetadata, tensor::FloatTensor};

fn product_gradient<B: Backend>(
    input: FloatTensor<B>,
    grad: FloatTensor<B>,
    dim: usize,
) -> FloatTensor<B> {
    let shape = input.shape();
    if shape.num_elements() == 0 {
        return B::float_zeros(shape, &B::float_device(&input), input.dtype().into());
    }
    let len = shape[dim];
    if len == 1 {
        return grad;
    }

    let mut edge_shape = shape.clone();
    edge_shape[dim] = 1;
    let ones = B::float_ones(edge_shape, &B::float_device(&input), input.dtype().into());
    let mut slices = vec![Slice::full(); shape.num_dims()];
    slices[dim] = Slice::from(0..len - 1);
    let prefix = B::float_slice(B::float_cumprod(input.clone(), dim), &slices);
    let prefix = B::float_cat(vec![ones.clone(), prefix], dim);

    let reversed = B::float_flip(input, &[dim]);
    let suffix = B::float_slice(B::float_cumprod(reversed, dim), &slices);
    let suffix = B::float_flip(B::float_cat(vec![ones, suffix], dim), &[dim]);
    B::float_mul(grad, B::float_mul(prefix, suffix))
}

#[derive(Debug)]
struct Product;

impl<B: Backend> Backward<B, 1> for Product {
    type State = (NodeId, Option<usize>);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let (input_id, dim) = ops.state;
        let input = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(input_id);
        unary::<B, _>(ops.parents, ops.node, grads, |grad| match dim {
            Some(dim) => product_gradient::<B>(input, grad, dim),
            None => {
                let shape = input.shape();
                let input = B::float_reshape(input, Shape::new([shape.num_elements()]));
                let grad = B::float_reshape(grad, Shape::new([1]));
                let grad = product_gradient::<B>(input, grad, 0);
                B::float_reshape(grad, shape)
            }
        });
    }
}

pub(super) fn product<B: Backend, C: CheckpointStrategy>(
    tensor: AutodiffTensor<B>,
    dim: Option<usize>,
) -> AutodiffTensor<B> {
    match Product.prepare::<C>([tensor.node.clone()]).compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let input_id = prep.checkpoint(&tensor);
            let output = match dim {
                Some(dim) => B::float_prod_dim(tensor.primitive, dim),
                None => B::float_prod(tensor.primitive),
            };
            prep.finish((input_id, dim), output)
        }
        OpsKind::UnTracked(prep) => {
            let output = match dim {
                Some(dim) => B::float_prod_dim(tensor.primitive, dim),
                None => B::float_prod(tensor.primitive),
            };
            prep.finish(output)
        }
    }
}
