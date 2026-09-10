use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    tensor::AutodiffTensor,
};
use super::{Backward, Ops, OpsKind};
use ruda_tensor::{
    Backend, TensorMetadata, FloatDType, Shape,
    ops::{GridSampleOptions, InterpolateMode, grid_sample::float_grid_sample_2d_ref},
};

#[derive(Debug)]
struct NearestGrid;

impl<B: Backend> Backward<B, 2> for NearestGrid {
    type State = (Shape, B::Device, FloatDType);

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let grad = grads.consume::<B>(&ops.node);
        if let Some(output) = &ops.parents[0] {
            grads.register::<B>(output.id, grad);
        }
        if let Some(grid) = &ops.parents[1] {
            let (shape, device, dtype) = ops.state;
            grads.register::<B>(grid.id, B::float_zeros(shape, &device, dtype));
        }
    }
}

pub(super) fn sample<B: Backend, C: CheckpointStrategy>(
    input: AutodiffTensor<B>,
    grid: AutodiffTensor<B>,
    options: GridSampleOptions,
) -> AutodiffTensor<B> {
    let nearest = matches!(options.mode, InterpolateMode::Nearest);
    let output = float_grid_sample_2d_ref::<Autodiff<B, C>>(input, grid.clone(), options);
    if !nearest { return output; }
    match NearestGrid.prepare::<C>([output.node.clone(), grid.node.clone()])
        .compute_bound().stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            (grid.primitive.shape(), B::float_device(&grid.primitive), grid.primitive.dtype().into()),
            output.primitive,
        ),
        OpsKind::UnTracked(prep) => prep.finish(output.primitive),
    }
}
