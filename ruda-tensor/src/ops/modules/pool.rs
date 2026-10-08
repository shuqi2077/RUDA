use crate::tensor::{FloatTensor, IntTensor};
use crate::{Backend, TensorMetadata};
use ruda_core::tensor::Shape;

use super::{MaxPool1dBackward, MaxPool1dWithIndices};

pub(crate) fn max_pool3d_from_2d<B: Backend>(input: FloatTensor<B>, kernel: [usize; 3],
    stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> FloatTensor<B> {
    let input = crate::api::Tensor::<B, 5>::new(crate::TensorPrimitive::Float(input));
    crate::api::max_pool3d_composed(input, kernel, stride, padding, dilation, ceil).into_primitive().tensor()
}

pub(crate) fn max_pool3d_with_indices_from_2d<B: Backend>(input: FloatTensor<B>, kernel: [usize; 3],
    stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> super::MaxPool3dWithIndices<B> {
    let input = crate::api::Tensor::<B, 5>::new(crate::TensorPrimitive::Float(input));
    let (output, indices) = crate::api::max_pool3d_with_indices_composed(input, kernel, stride, padding, dilation, ceil);
    super::MaxPool3dWithIndices::new(output.into_primitive().tensor(), indices.into_primitive())
}

pub(crate) fn max_pool3d_backward_from_indices<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>,
    indices: IntTensor<B>) -> FloatTensor<B> {
    use crate::api::{DType, Int, Tensor};
    let [batch, channels, depth, height, width] = input.shape().dims();
    let volume = depth.checked_mul(height).and_then(|size| size.checked_mul(width)).expect("pooling volume overflow");
    let storage = input.dtype();
    let device = B::float_device(&input);
    if volume == 0 { return Tensor::<B, 5>::zeros([batch, channels, depth, height, width],
        (&device, storage)).into_primitive().tensor(); }
    assert!(volume <= i64::MAX as usize, "pooling positions exceed I64");
    assert_eq!(indices.shape(), grad.shape(), "pooling gradient and positions differ");
    let [gb, gc, gd, gh, gw] = grad.shape().dims();
    assert_eq!([gb, gc], [batch, channels], "pooling gradient batch/channels differ");
    let count = gd.checked_mul(gh).and_then(|size| size.checked_mul(gw)).expect("pooling output volume overflow");
    let compute = if storage == DType::F64 || grad.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
    let indices = Tensor::<B, 5, Int>::new(indices).cast(DType::I64).reshape([batch, channels, count]);
    let invalid = indices.clone().lower_elem(0).bool_or(indices.clone().greater_equal_elem(volume as i64));
    let values = Tensor::<B, 5>::new(crate::TensorPrimitive::Float(grad)).cast(compute)
        .reshape([batch, channels, count]).mask_fill(invalid, 0);
    Tensor::<B, 3>::zeros([batch, channels, volume], (&device, compute))
        .scatter(2, indices.clamp(0, volume as i64 - 1), values, crate::IndexingUpdateOp::Add).cast(storage)
        .reshape([batch, channels, depth, height, width]).into_primitive().tensor()
}

fn volume_planes<B: Backend>(input: FloatTensor<B>) -> FloatTensor<B> {
    let [batch, channels, depth, height, width] = input.shape().dims();
    let planes = batch.checked_mul(depth).expect("pooling plane count overflow");
    B::float_reshape(B::float_permute(input, &[0, 2, 1, 3, 4]),
        Shape::new([planes, channels, height, width]))
}

fn plane_depth_lines<B: Backend>(
    planes: FloatTensor<B>, batch: usize, depth: usize,
) -> FloatTensor<B> {
    let [_, channels, height, width] = planes.shape().dims();
    let lines = batch.checked_mul(channels).and_then(|count| count.checked_mul(height))
        .and_then(|count| count.checked_mul(width)).expect("pooling depth line count overflow");
    let volume = B::float_reshape(planes, Shape::new([batch, depth, channels, height, width]));
    B::float_reshape(B::float_permute(volume, &[0, 2, 3, 4, 1]), Shape::new([lines, 1, depth]))
}

fn depth_lines_volume<B: Backend>(
    lines: FloatTensor<B>, batch: usize, channels: usize, height: usize, width: usize,
) -> FloatTensor<B> {
    let [_, _, depth] = lines.shape().dims();
    let volume = B::float_reshape(lines, Shape::new([batch, channels, height, width, depth]));
    B::float_permute(volume, &[0, 1, 4, 2, 3])
}

pub(crate) fn adaptive_avg_pool3d_from_2d<B: Backend>(
    input: FloatTensor<B>, output_size: [usize; 3],
) -> FloatTensor<B> {
    let [batch, channels, depth, _, _] = input.shape().dims();
    let planes = B::adaptive_avg_pool2d(volume_planes::<B>(input), [output_size[1], output_size[2]]);
    let [_, _, height, width] = planes.shape().dims();
    let lines = B::adaptive_avg_pool1d(plane_depth_lines::<B>(planes, batch, depth), output_size[0]);
    depth_lines_volume::<B>(lines, batch, channels, height, width)
}

/// Execute volume average pooling through the backend's spatial and depth primitives.
pub fn avg_pool3d_from_2d<B: Backend>(input: FloatTensor<B>, kernel: [usize; 3],
    stride: [usize; 3], padding: [usize; 3], count_include_pad: bool, ceil_mode: bool) -> FloatTensor<B> {
    let [batch, channels, depth, _, _] = input.shape().dims();
    let planes = B::avg_pool2d(volume_planes::<B>(input), [kernel[1], kernel[2]],
        [stride[1], stride[2]], [padding[1], padding[2]], count_include_pad, ceil_mode);
    let [_, _, height, width] = planes.shape().dims();
    let lines = B::avg_pool1d(plane_depth_lines::<B>(planes, batch, depth), kernel[0], stride[0],
        padding[0], count_include_pad, ceil_mode);
    depth_lines_volume::<B>(lines, batch, channels, height, width)
}

/// Execute volume average gradients through the backend's original composed operations.
pub fn avg_pool3d_backward_from_2d<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>,
    kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3],
    count_include_pad: bool, ceil_mode: bool) -> FloatTensor<B> {
    let [batch, channels, depth, height, width] = input.shape().dims();
    let [grad_batch, grad_channels, grad_depth, grad_height, grad_width] = grad.shape().dims();
    assert_eq!([grad_batch, grad_channels], [batch, channels], "pooling gradient batch/channels differ");
    let planes = volume_planes::<B>(input);
    let spatial = B::avg_pool2d(planes.clone(), [kernel[1], kernel[2]],
        [stride[1], stride[2]], [padding[1], padding[2]], count_include_pad, ceil_mode);
    assert_eq!(spatial.shape().dims::<4>()[2..], [grad_height, grad_width], "pooling spatial gradient differs");
    let lines = plane_depth_lines::<B>(spatial, batch, depth);
    let grad_lines = plane_depth_lines::<B>(volume_planes::<B>(grad), batch, grad_depth);
    let grad_lines = B::avg_pool1d_backward(lines, grad_lines, kernel[0], stride[0], padding[0],
        count_include_pad, ceil_mode);
    let grad_planes = volume_planes::<B>(depth_lines_volume::<B>(grad_lines,
        batch, channels, grad_height, grad_width));
    let grad_planes = B::avg_pool2d_backward(planes, grad_planes, [kernel[1], kernel[2]],
        [stride[1], stride[2]], [padding[1], padding[2]], count_include_pad, ceil_mode);
    let volume = B::float_reshape(grad_planes, Shape::new([batch, depth, channels, height, width]));
    B::float_permute(volume, &[0, 2, 1, 3, 4])
}

pub(crate) fn adaptive_avg_pool3d_backward_from_2d<B: Backend>(
    input: FloatTensor<B>, grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [batch, channels, depth, height, width] = input.shape().dims();
    let [grad_batch, grad_channels, grad_depth, grad_height, grad_width] = grad.shape().dims();
    assert_eq!([grad_batch, grad_channels], [batch, channels], "adaptive pooling gradient batch/channels differ");
    let planes = volume_planes::<B>(input);
    let spatial = B::adaptive_avg_pool2d(planes.clone(), [grad_height, grad_width]);
    let lines = plane_depth_lines::<B>(spatial, batch, depth);
    let grad_lines = plane_depth_lines::<B>(volume_planes::<B>(grad), batch, grad_depth);
    let grad_lines = B::adaptive_avg_pool1d_backward(lines, grad_lines);
    let grad_planes = volume_planes::<B>(depth_lines_volume::<B>(grad_lines,
        batch, channels, grad_height, grad_width));
    let grad_planes = B::adaptive_avg_pool2d_backward(planes, grad_planes);
    let volume = B::float_reshape(grad_planes, Shape::new([batch, depth, channels, height, width]));
    B::float_permute(volume, &[0, 2, 1, 3, 4])
}

pub(crate) fn avg_pool1d_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> FloatTensor<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length, 1]));
    let x = B::avg_pool2d(
        x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        count_include_pad,
        ceil_mode,
    );

    let [batch_size, channels, length, _] = x.shape().dims();

    B::float_reshape(x, Shape::from([batch_size, channels, length]))
}

pub(crate) fn avg_pool1d_backward_from_2d<B: Backend>(
    x: FloatTensor<B>,
    grad: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> FloatTensor<B> {
    let [batch_size, channels, length_in] = x.shape().dims();
    let [_, _, length_out] = grad.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length_in, 1]));
    let grad_x = B::float_reshape(grad, Shape::from([batch_size, channels, length_out, 1]));

    let grad_x = B::avg_pool2d_backward(
        x,
        grad_x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        count_include_pad,
        ceil_mode,
    );

    B::float_reshape(grad_x, Shape::from([batch_size, channels, length_in]))
}

pub(crate) fn adaptive_avg_pool1d_from_2d<B: Backend>(
    x: FloatTensor<B>,
    output_size: usize,
) -> FloatTensor<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length, 1]));
    let x = B::adaptive_avg_pool2d(x, [output_size, 1]);

    let [batch_size, channels, length, _] = x.shape().dims();

    B::float_reshape(x, Shape::from([batch_size, channels, length]))
}

pub(crate) fn adaptive_avg_pool1d_backward_from_2d<B: Backend>(
    x: FloatTensor<B>,
    grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [batch_size, channels, length_in] = x.shape().dims();
    let [_, _, length_out] = grad.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length_in, 1]));
    let grad_x = B::float_reshape(grad, Shape::from([batch_size, channels, length_out, 1]));

    let grad_x = B::adaptive_avg_pool2d_backward(x, grad_x);

    B::float_reshape(grad_x, Shape::from([batch_size, channels, length_in]))
}

pub(crate) fn max_pool1d_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> FloatTensor<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length, 1]));
    let x = B::max_pool2d(
        x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        [dilation, 1],
        ceil_mode,
    );

    let [batch_size, channels, length, _] = x.shape().dims();

    B::float_reshape(x, Shape::from([batch_size, channels, length]))
}

pub(crate) fn max_pool1d_with_indices_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> MaxPool1dWithIndices<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, 1, length]));
    let x = B::max_pool2d_with_indices(
        x,
        [1, kernel_size],
        [1, stride],
        [0, padding],
        [1, dilation],
        ceil_mode,
    );
    let [batch_size, channels, _, length] = x.output.shape().dims();
    let output = B::float_reshape(x.output, Shape::from([batch_size, channels, length]));
    let indices = B::int_reshape(x.indices, Shape::from([batch_size, channels, length]));
    MaxPool1dWithIndices::new(output, indices)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn max_pool1d_with_indices_backward_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
    output_grad: FloatTensor<B>,
    indices: IntTensor<B>,
) -> MaxPool1dBackward<B> {
    let [batch_size, channels, length_in] = x.shape().dims();
    let [_, _, length_out] = output_grad.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length_in, 1]));
    let grad_x = B::float_reshape(
        output_grad,
        Shape::from([batch_size, channels, length_out, 1]),
    );
    let indices = B::int_reshape(indices, Shape::from([batch_size, channels, length_out, 1]));

    let grad_x = B::max_pool2d_with_indices_backward(
        x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        [dilation, 1],
        ceil_mode,
        grad_x,
        indices,
    )
    .x_grad;

    MaxPool1dBackward::new(B::float_reshape(
        grad_x,
        Shape::from([batch_size, channels, length_in]),
    ))
}
