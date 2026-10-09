use crate::tensor::{FloatTensor, IntTensor};
use crate::{Backend, DType, FloatDType, IntDType, TensorMetadata, get_device_settings};
use ruda_core::tensor::Shape;

use super::{MaxPool1dBackward, MaxPool1dWithIndices};

pub(crate) fn max_pool3d_from_2d<B: Backend>(input: FloatTensor<B>, kernel: [usize; 3],
    stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> FloatTensor<B> {
    let [batch, channels, depth, _, _] = input.shape().dims();
    let planes = B::max_pool2d(volume_planes::<B>(input), [kernel[1], kernel[2]],
        [stride[1], stride[2]], [padding[1], padding[2]], [dilation[1], dilation[2]], ceil);
    let [_, _, height, width] = planes.shape().dims();
    let lines = B::max_pool1d(plane_depth_lines::<B>(planes, batch, depth),
        kernel[0], stride[0], padding[0], dilation[0], ceil);
    depth_lines_volume::<B>(lines, batch, channels, height, width)
}

pub(crate) fn max_pool3d_with_indices_from_2d<B: Backend>(input: FloatTensor<B>, kernel: [usize; 3],
    stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> super::MaxPool3dWithIndices<B> {
    let [batch, channels, depth, input_height, input_width] = input.shape().dims();
    assert!(depth > 0 && input_height > 0 && input_width > 0,
        "volume pooling indices require non-empty spatial axes");
    let area = input_height.checked_mul(input_width).expect("pooling plane area overflow");
    let volume = depth.checked_mul(area).expect("pooling volume size overflow");
    assert!(volume <= i64::MAX as usize, "volume pooling indices exceed I64");
    let planes = B::max_pool2d_with_indices(volume_planes::<B>(input), [kernel[1], kernel[2]],
        [stride[1], stride[2]], [padding[1], padding[2]], [dilation[1], dilation[2]], ceil);
    let [_, _, height, width] = planes.output.shape().dims();
    let lines = B::max_pool1d_with_indices(plane_depth_lines::<B>(planes.output, batch, depth),
        kernel[0], stride[0], padding[0], dilation[0], ceil);
    let depth_indices = indices_i64::<B>(lines.indices);
    let safe_depth_indices = B::int_clamp(depth_indices.clone(), 0i64.into(), (depth as i64 - 1).into());
    let spatial_indices = B::int_gather(2,
        int_plane_depth_lines::<B>(indices_i64::<B>(planes.indices), batch, depth),
        safe_depth_indices.clone());
    let bool_dtype = get_device_settings::<B>(&B::int_device(&depth_indices)).bool_dtype;
    let invalid_depth = B::bool_or(B::int_lower_elem(depth_indices.clone(), 0i64.into(), bool_dtype),
        B::int_greater_equal_elem(depth_indices, (depth as i64).into(), bool_dtype));
    let invalid_spatial = B::bool_or(B::int_lower_elem(spatial_indices.clone(), 0i64.into(), bool_dtype),
        B::int_greater_equal_elem(spatial_indices.clone(), (area as i64).into(), bool_dtype));
    let indices = B::int_add(B::int_mul_scalar(safe_depth_indices, (area as i64).into()),
        B::int_clamp(spatial_indices, 0i64.into(), (area as i64 - 1).into()));
    let indices = B::int_mask_fill(indices, B::bool_or(invalid_depth, invalid_spatial), (-1i64).into());
    super::MaxPool3dWithIndices::new(depth_lines_volume::<B>(lines.output, batch, channels, height, width),
        int_depth_lines_volume::<B>(indices, batch, channels, height, width))
}

pub(crate) fn max_pool3d_backward_from_indices<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>,
    indices: IntTensor<B>) -> FloatTensor<B> {
    let [batch, channels, depth, height, width] = input.shape().dims();
    let volume = depth.checked_mul(height).and_then(|size| size.checked_mul(width)).expect("pooling volume overflow");
    let storage = input.dtype();
    let device = B::float_device(&input);
    if volume == 0 { return B::float_zeros(Shape::new([batch, channels, depth, height, width]),
        &device, storage.into()); }
    assert!(volume <= i64::MAX as usize, "pooling positions exceed I64");
    assert_eq!(indices.shape(), grad.shape(), "pooling gradient and positions differ");
    let [gb, gc, gd, gh, gw] = grad.shape().dims();
    assert_eq!([gb, gc], [batch, channels], "pooling gradient batch/channels differ");
    let count = gd.checked_mul(gh).and_then(|size| size.checked_mul(gw)).expect("pooling output volume overflow");
    let compute = if storage == DType::F64 || grad.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let indices = B::int_reshape(indices_i64::<B>(indices), Shape::new([batch, channels, count]));
    let bool_dtype = get_device_settings::<B>(&B::int_device(&indices)).bool_dtype;
    let invalid = B::bool_or(B::int_lower_elem(indices.clone(), 0i64.into(), bool_dtype),
        B::int_greater_equal_elem(indices.clone(), (volume as i64).into(), bool_dtype));
    let grad = if grad.dtype() == compute.into() { grad } else { B::float_cast(grad, compute) };
    let values = B::float_mask_fill(B::float_reshape(grad, Shape::new([batch, channels, count])), invalid, 0f32.into());
    let output = B::float_scatter_add(2, B::float_zeros(Shape::new([batch, channels, volume]), &device, compute),
        B::int_clamp(indices, 0i64.into(), (volume as i64 - 1).into()), values);
    let output = if output.dtype() == storage { output } else { B::float_cast(output, storage.into()) };
    B::float_reshape(output, Shape::new([batch, channels, depth, height, width]))
}

fn indices_i64<B: Backend>(indices: IntTensor<B>) -> IntTensor<B> {
    if indices.dtype() == DType::I64 { indices } else { B::int_cast(indices, IntDType::I64) }
}

fn int_plane_depth_lines<B: Backend>(planes: IntTensor<B>, batch: usize, depth: usize) -> IntTensor<B> {
    let [_, channels, height, width] = planes.shape().dims();
    let lines = batch.checked_mul(channels).and_then(|count| count.checked_mul(height))
        .and_then(|count| count.checked_mul(width)).expect("pooling depth line count overflow");
    let volume = B::int_reshape(planes, Shape::new([batch, depth, channels, height, width]));
    B::int_reshape(B::int_permute(volume, &[0, 2, 3, 4, 1]), Shape::new([lines, 1, depth]))
}

fn int_depth_lines_volume<B: Backend>(lines: IntTensor<B>, batch: usize, channels: usize,
    height: usize, width: usize) -> IntTensor<B> {
    let [_, _, depth] = lines.shape().dims();
    B::int_permute(B::int_reshape(lines, Shape::new([batch, channels, height, width, depth])), &[0, 1, 4, 2, 3])
}

pub(super) fn volume_planes<B: Backend>(input: FloatTensor<B>) -> FloatTensor<B> {
    let [batch, channels, depth, height, width] = input.shape().dims();
    let planes = batch.checked_mul(depth).expect("pooling plane count overflow");
    B::float_reshape(B::float_permute(input, &[0, 2, 1, 3, 4]),
        Shape::new([planes, channels, height, width]))
}

pub(super) fn plane_depth_lines<B: Backend>(
    planes: FloatTensor<B>, batch: usize, depth: usize,
) -> FloatTensor<B> {
    let [_, channels, height, width] = planes.shape().dims();
    let lines = batch.checked_mul(channels).and_then(|count| count.checked_mul(height))
        .and_then(|count| count.checked_mul(width)).expect("pooling depth line count overflow");
    let volume = B::float_reshape(planes, Shape::new([batch, depth, channels, height, width]));
    B::float_reshape(B::float_permute(volume, &[0, 2, 3, 4, 1]), Shape::new([lines, 1, depth]))
}

pub(super) fn depth_lines_volume<B: Backend>(
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
