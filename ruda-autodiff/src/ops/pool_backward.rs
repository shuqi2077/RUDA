use super::{Backward, Ops, OpsKind, unary};
use crate::{
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    tensor::AutodiffTensor,
};
use ruda_core::tensor::{DType, Shape, element::Scalar};
use ruda_tensor::{Backend, TensorMetadata, get_device_settings};

#[derive(Debug)]
struct Average {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
}

impl<B: Backend> Backward<B, 1> for Average {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            B::avg_pool2d(grad, self.kernel_size, self.stride, self.padding, self.count_include_pad, self.ceil_mode)
        });
    }
}

pub(super) fn average<B: Backend, C: CheckpointStrategy>(
    x: AutodiffTensor<B>,
    grad: AutodiffTensor<B>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> AutodiffTensor<B> {
    Average { kernel_size, stride, padding, count_include_pad, ceil_mode }
        .prepare::<C>([grad.node.clone()])
        .compute_bound()
        .stateless(B::avg_pool2d_backward(x.primitive, grad.primitive, kernel_size, stride, padding, count_include_pad, ceil_mode))
}

#[derive(Debug)]
struct AdaptiveAverage {
    output_size: [usize; 2],
}

impl<B: Backend> Backward<B, 1> for AdaptiveAverage {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            B::adaptive_avg_pool2d(grad, self.output_size)
        });
    }
}

pub(super) fn adaptive_average<B: Backend, C: CheckpointStrategy>(
    x: AutodiffTensor<B>,
    grad: AutodiffTensor<B>,
) -> AutodiffTensor<B> {
    let [_, _, height, width] = grad.primitive.shape().dims::<4>();
    AdaptiveAverage { output_size: [height, width] }
        .prepare::<C>([grad.node.clone()])
        .compute_bound()
        .stateless(B::adaptive_avg_pool2d_backward(x.primitive, grad.primitive))
}

#[derive(Debug)]
struct Maximum;

impl<B: Backend> Backward<B, 1> for Maximum {
    type State = (B::IntTensorPrimitive, Shape);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (indices, output_shape) = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let [batch, channels, height, width] = grad.shape().dims::<4>();
            let device = B::float_device(&grad);
            if height == 0 || width == 0 || output_shape.num_elements() == 0 {
                return B::float_zeros(output_shape, &device, grad.dtype().into());
            }
            let [_, _, out_h, out_w] = output_shape.dims::<4>();
            let input_elements = height.checked_mul(width).expect("Max pool gradient plane size overflow");
            let output_elements = out_h.checked_mul(out_w).expect("Max pool output plane size overflow");
            let sentinel = match indices.dtype() {
                DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::Int(-1),
                DType::U8 => Scalar::UInt(u8::MAX as u64),
                DType::U16 => Scalar::UInt(u16::MAX as u64),
                DType::U32 => Scalar::UInt(u32::MAX as u64),
                DType::U64 => Scalar::UInt(u64::MAX),
                dtype => panic!("Max pool indices require an integer dtype, got {dtype:?}"),
            };
            let bool_dtype = get_device_settings::<B>(&device).bool_dtype;
            let indices = B::int_reshape(indices, Shape::new([batch, channels, output_elements]));
            let empty = B::int_equal_elem(indices.clone(), sentinel, bool_dtype);
            let indices = B::int_mask_fill(indices, empty.clone(), 0i64.into());
            let grad = B::float_reshape(grad, Shape::new([batch, channels, input_elements]));
            let selected = B::float_gather(2, grad, indices);
            let selected = B::float_mask_fill(selected, empty, 0.0.into());
            B::float_reshape(selected, output_shape)
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn maximum<B: Backend, C: CheckpointStrategy>(
    x: AutodiffTensor<B>,
    grad: AutodiffTensor<B>,
    indices: B::IntTensorPrimitive,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> AutodiffTensor<B> {
    let prep = Maximum.prepare::<C>([grad.node.clone()]).compute_bound().stateful();
    let output_shape = grad.primitive.shape();
    match prep {
        OpsKind::Tracked(prep) => {
            let output = B::max_pool2d_with_indices_backward(
                x.primitive, kernel_size, stride, padding, dilation, ceil_mode, grad.primitive, indices.clone(),
            );
            prep.finish((indices, output_shape), output.x_grad)
        }
        OpsKind::UnTracked(prep) => {
            let output = B::max_pool2d_with_indices_backward(
                x.primitive, kernel_size, stride, padding, dilation, ceil_mode, grad.primitive, indices,
            );
            prep.finish(output.x_grad)
        }
    }
}
