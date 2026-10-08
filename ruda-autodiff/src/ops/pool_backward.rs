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

#[derive(Debug)]
struct AverageVolume {
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    include_pad: bool,
    ceil: bool,
    gradient_dtype: DType,
}

impl<B: Backend> Backward<B, 1> for AverageVolume {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let compute = if grad.dtype() == DType::F64 || self.gradient_dtype == DType::F64 {
                DType::F64
            } else { DType::F32 };
            let grad = B::float_cast(grad, compute.into());
            let out = B::avg_pool3d(grad, self.kernel, self.stride, self.padding, self.include_pad, self.ceil);
            B::float_cast(out, self.gradient_dtype.into())
        });
    }
}

pub(super) fn average_volume<B: Backend, C: CheckpointStrategy>(x: AutodiffTensor<B>,
    grad: AutodiffTensor<B>, kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3],
    include_pad: bool, ceil: bool) -> AutodiffTensor<B> {
    let prep = AverageVolume { kernel, stride, padding, include_pad, ceil,
        gradient_dtype: grad.primitive.dtype() }
        .prepare::<C>([grad.node.clone()]).compute_bound();
    prep.stateless(B::avg_pool3d_backward(x.primitive, grad.primitive,
        kernel, stride, padding, include_pad, ceil))
}

#[derive(Debug)]
struct AdaptiveAverageVolume {
    output_size: [usize; 3],
    gradient_dtype: DType,
}

impl<B: Backend> Backward<B, 1> for AdaptiveAverageVolume {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let compute = if grad.dtype() == DType::F64 || self.gradient_dtype == DType::F64 {
                DType::F64
            } else { DType::F32 };
            let grad = B::float_cast(grad, compute.into());
            let out = B::adaptive_avg_pool3d(grad, self.output_size);
            B::float_cast(out, self.gradient_dtype.into())
        });
    }
}

pub(super) fn adaptive_average_volume<B: Backend, C: CheckpointStrategy>(x: AutodiffTensor<B>,
    grad: AutodiffTensor<B>) -> AutodiffTensor<B> {
    let [_, _, depth, height, width] = grad.primitive.shape().dims::<5>();
    AdaptiveAverageVolume { output_size: [depth, height, width], gradient_dtype: grad.primitive.dtype() }
        .prepare::<C>([grad.node.clone()]).compute_bound()
        .stateless(B::adaptive_avg_pool3d_backward(x.primitive, grad.primitive))
}

#[derive(Debug)]
struct MaximumVolume;

impl<B: Backend> Backward<B, 1> for MaximumVolume {
    type State = (B::IntTensorPrimitive, Shape, DType);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (indices, output_shape, dtype) = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            use ruda_tensor::api::{Int, Tensor};
            let [batch, channels, depth, height, width] = grad.shape().dims::<5>();
            let volume = depth.checked_mul(height).and_then(|n| n.checked_mul(width))
                .expect("pool gradient volume overflow");
            if volume == 0 || output_shape.num_elements() == 0 {
                return B::float_zeros(output_shape, &B::float_device(&grad), dtype.into());
            }
            assert!(volume <= i64::MAX as usize, "pool gradient positions exceed I64");
            let [_, _, od, oh, ow] = output_shape.dims::<5>();
            let count = od.checked_mul(oh).and_then(|n| n.checked_mul(ow))
                .expect("pool output volume overflow");
            let indices = Tensor::<B, 5, Int>::new(indices).cast(DType::I64)
                .reshape([batch, channels, count]);
            let invalid = indices.clone().lower_elem(0)
                .bool_or(indices.clone().greater_equal_elem(volume as i64));
            let grad = Tensor::<B, 5>::new(ruda_tensor::TensorPrimitive::Float(grad))
                .reshape([batch, channels, volume]);
            grad.gather(2, indices.clamp(0, volume as i64 - 1)).mask_fill(invalid, 0)
                .cast(dtype).reshape(output_shape.dims::<5>()).into_primitive().tensor()
        });
    }
}

pub(super) fn maximum_volume<B: Backend, C: CheckpointStrategy>(x: AutodiffTensor<B>,
    grad: AutodiffTensor<B>, indices: B::IntTensorPrimitive, kernel: [usize; 3], stride: [usize; 3],
    padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> AutodiffTensor<B> {
    let shape = grad.primitive.shape();
    let dtype = grad.primitive.dtype();
    match MaximumVolume.prepare::<C>([grad.node.clone()]).compute_bound().stateful() {
        OpsKind::Tracked(prep) => {
            let out = B::max_pool3d_with_indices_backward(x.primitive, grad.primitive,
                indices.clone(), kernel, stride, padding, dilation, ceil);
            prep.finish((indices, shape, dtype), out.x_grad)
        }
        OpsKind::UnTracked(prep) => {
            let out = B::max_pool3d_with_indices_backward(x.primitive, grad.primitive,
                indices, kernel, stride, padding, dilation, ceil);
            prep.finish(out.x_grad)
        }
    }
}
