use crate::{Backend, Shape, TensorMetadata, tensor::FloatTensor};
use super::{InterpolateOptions, pool::{volume_planes, plane_depth_lines, depth_lines_volume}};

/// Resize lines with the backend's original spatial filter and corner alignment.
pub fn interpolate1d_from_2d<B: Backend>(input: FloatTensor<B>, size: usize,
    options: InterpolateOptions) -> FloatTensor<B> {
    let [batch, channels, width] = input.shape().dims();
    let input = B::float_reshape(input, Shape::new([batch, channels, 1, width]));
    let output = B::interpolate(input, [1, size], options);
    B::float_reshape(output, Shape::new([batch, channels, size]))
}

/// Input gradients for the original line filter, retaining the original input storage.
pub fn interpolate1d_backward_from_2d<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>,
    size: usize, options: InterpolateOptions) -> FloatTensor<B> {
    let [batch, channels, width] = input.shape().dims();
    let input = B::float_reshape(input, Shape::new([batch, channels, 1, width]));
    let grad = B::float_reshape(grad, Shape::new([batch, channels, 1, size]));
    let output = B::interpolate_backward(input, grad, [1, size], options);
    B::float_reshape(output, Shape::new([batch, channels, width]))
}

/// Resize volumes with the existing spatial-then-depth filter sequence.
pub fn interpolate3d_from_2d<B: Backend>(input: FloatTensor<B>, size: [usize; 3],
    options: InterpolateOptions) -> FloatTensor<B> {
    let [batch, channels, depth, _, _] = input.shape().dims();
    let spatial = B::interpolate(volume_planes::<B>(input), [size[1], size[2]], options.clone());
    let lines = interpolate1d_from_2d::<B>(plane_depth_lines::<B>(spatial, batch, depth), size[0], options);
    depth_lines_volume::<B>(lines, batch, channels, size[1], size[2])
}

/// Reverse the original spatial/depth passes through an actual recomputed spatial activation.
pub fn interpolate3d_backward_from_2d<B: Backend>(input: FloatTensor<B>, grad: FloatTensor<B>,
    size: [usize; 3], options: InterpolateOptions) -> FloatTensor<B> {
    let [batch, channels, depth, height, width] = input.shape().dims();
    assert_eq!(grad.shape().dims::<5>(), [batch, channels, size[0], size[1], size[2]],
        "volume interpolation gradient shape differs");
    let planes = volume_planes::<B>(input);
    let spatial = B::interpolate(planes.clone(), [size[1], size[2]], options.clone());
    let lines = plane_depth_lines::<B>(spatial, batch, depth);
    let grad_lines = plane_depth_lines::<B>(volume_planes::<B>(grad), batch, size[0]);
    let lines_grad = interpolate1d_backward_from_2d::<B>(lines, grad_lines, size[0], options.clone());
    let spatial_grad = volume_planes::<B>(depth_lines_volume::<B>(lines_grad,
        batch, channels, size[1], size[2]));
    let planes_grad = B::interpolate_backward(planes, spatial_grad, [size[1], size[2]], options);
    let output = B::float_reshape(planes_grad, Shape::new([batch, depth, channels, height, width]));
    B::float_permute(output, &[0, 2, 1, 3, 4])
}
