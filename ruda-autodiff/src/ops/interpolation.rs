use super::{Backward, Ops, unary};
use crate::{
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    tensor::AutodiffTensor,
};
use ruda_core::tensor::spatial::InterpolateOptions;
use ruda_tensor::{Backend, FloatDType, TensorMetadata};

#[derive(Debug)]
struct InterpolateBackward {
    output_size: [usize; 2],
    options: InterpolateOptions,
    gradient_dtype: FloatDType,
}

impl<B: Backend> Backward<B, 1> for InterpolateBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let output = B::interpolate(grad, self.output_size, self.options);
            B::float_cast(output, self.gradient_dtype)
        });
    }
}

pub(super) fn backward<B: Backend, C: CheckpointStrategy>(
    x: AutodiffTensor<B>,
    grad: AutodiffTensor<B>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> AutodiffTensor<B> {
    InterpolateBackward { output_size, options: options.clone(), gradient_dtype: grad.primitive.dtype().into() }
        .prepare::<C>([grad.node.clone()])
        .compute_bound()
        .stateless(B::interpolate_backward(x.primitive, grad.primitive, output_size, options))
}
