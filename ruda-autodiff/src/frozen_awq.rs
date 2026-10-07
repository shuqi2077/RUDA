use crate::{
    Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients, ops::{Backward, Ops, OpsKind, unary},
};
use core::marker::PhantomData;
use ruda_tensor::{frozen_awq::{FrozenAwqOps, FrozenAwqError}, tensor::{FloatTensor, IntTensor}};

#[derive(Debug)]
struct PackedAwq<B: FrozenAwqOps>(PhantomData<B>);

impl<B: FrozenAwqOps> Backward<B, 1> for PackedAwq<B> {
    // Original native packed buffers, never a cached dequantized matrix.
    type State = (IntTensor<B>, IntTensor<B>, FloatTensor<B>, usize, bool);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let (qweight, qzeros, scales, group, transposed) = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |gradient| {
            let result = if transposed {
                B::frozen_awq_forward(gradient, qweight, qzeros, scales, None, group)
            } else {
                B::frozen_awq_input_backward(gradient, qweight, qzeros, scales, group)
            };
            result.unwrap_or_else(|error| panic!("native frozen AWQ backward failed: {error:?}"))
        });
    }
}

impl<B: FrozenAwqOps, S: CheckpointStrategy> FrozenAwqOps for Autodiff<B, S> {
    type AwqError = FrozenAwqError<B::AwqError>;

    fn frozen_awq_forward(
        input: FloatTensor<Self>, qweight: IntTensor<Self>, qzeros: IntTensor<Self>,
        scales: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, group_size: usize,
    ) -> Result<FloatTensor<Self>, Self::AwqError> {
        if scales.is_tracked() || bias.as_ref().is_some_and(|value| value.is_tracked()) {
            return Err(FrozenAwqError::TrainableBase);
        }
        let state = (qweight.clone(), qzeros.clone(), scales.primitive.clone(), group_size, false);
        let output = B::frozen_awq_forward(input.primitive, qweight, qzeros, scales.primitive,
            bias.map(|value| value.primitive), group_size).map_err(FrozenAwqError::Native)?;
        Ok(match PackedAwq::<B>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(state, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }

    fn frozen_awq_input_backward(
        gradient: FloatTensor<Self>, qweight: IntTensor<Self>, qzeros: IntTensor<Self>,
        scales: FloatTensor<Self>, group_size: usize,
    ) -> Result<FloatTensor<Self>, Self::AwqError> {
        if scales.is_tracked() { return Err(FrozenAwqError::TrainableBase); }
        let state = (qweight.clone(), qzeros.clone(), scales.primitive.clone(), group_size, true);
        let output = B::frozen_awq_input_backward(gradient.primitive, qweight, qzeros, scales.primitive,
            group_size).map_err(FrozenAwqError::Native)?;
        Ok(match PackedAwq::<B>(PhantomData).prepare::<S>([gradient.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(state, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }
}
