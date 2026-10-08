use ruda_core::tensor::{Shape, spatial::InterpolateOptions};
use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, permutation::permute, reshape::reshape}};

use super::interpolate;

fn planes<R: Runtime>(input: RudaTensor<R>) -> RudaTensor<R> {
    let [batch, channels, depth, height, width] = input.meta.shape().dims();
    let count = batch.checked_mul(depth).expect("interpolation plane count overflow");
    reshape(permute(input, &[0, 2, 1, 3, 4]), Shape::new([count, channels, height, width]))
}

fn depth_lines<R: Runtime>(input: RudaTensor<R>, batch: usize, depth: usize) -> RudaTensor<R> {
    let [_, channels, height, width] = input.meta.shape().dims();
    let count = batch.checked_mul(channels).and_then(|n| n.checked_mul(height))
        .and_then(|n| n.checked_mul(width)).expect("interpolation line count overflow");
    let input = reshape(input, Shape::new([batch, depth, channels, height, width]));
    reshape(permute(input, &[0, 2, 3, 4, 1]), Shape::new([count, 1, 1, depth]))
}

fn from_lines<R: Runtime>(input: RudaTensor<R>, batch: usize, channels: usize,
    height: usize, width: usize) -> RudaTensor<R> {
    let depth = input.meta.shape()[3];
    let input = reshape(input, Shape::new([batch, channels, height, width, depth]));
    permute(input, &[0, 1, 4, 2, 3])
}

/// Native line interpolation using the existing filter's singleton spatial axis.
pub fn interpolate1d<R: Runtime>(input: RudaTensor<R>, size: usize,
    options: InterpolateOptions) -> RudaTensor<R> {
    let [batch, channels, width] = input.meta.shape().dims();
    let input = reshape(input, Shape::new([batch, channels, 1, width]));
    reshape(interpolate(input, [1, size], options), Shape::new([batch, channels, size]))
}

/// Native line derivatives with independently typed incoming gradient storage.
pub fn interpolate1d_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>,
    size: usize, options: InterpolateOptions) -> RudaTensor<R> {
    let [batch, channels, width] = input.meta.shape().dims();
    let grad = reshape(grad, Shape::new([batch, channels, 1, size]));
    reshape(super::base::backward_in_shape(&input, [batch, channels, 1, width], grad, options),
        Shape::new([batch, channels, width]))
}

/// Native spatial/depth volume resizing, preserving the original intermediate storage and filters.
pub fn interpolate3d<R: Runtime>(input: RudaTensor<R>, size: [usize; 3],
    options: InterpolateOptions) -> RudaTensor<R> {
    let [batch, channels, depth, _, _] = input.meta.shape().dims();
    let spatial = interpolate(planes(input), [size[1], size[2]], options.clone());
    let lines = interpolate(depth_lines(spatial, batch, depth), [1, size[0]], options);
    from_lines(lines, batch, channels, size[1], size[2])
}

/// Native adjoints of both volume passes using original geometry and storage.
pub fn interpolate3d_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>,
    size: [usize; 3], options: InterpolateOptions) -> RudaTensor<R> {
    let [batch, channels, depth, height, width] = input.meta.shape().dims();
    assert_eq!(grad.meta.shape().dims::<5>(), [batch, channels, size[0], size[1], size[2]],
        "volume interpolation gradient shape differs");
    let gradient_lines = depth_lines(planes(grad), batch, size[0]);
    let line_count = batch.checked_mul(channels).and_then(|n| n.checked_mul(size[1]))
        .and_then(|n| n.checked_mul(size[2])).expect("interpolation gradient line count overflow");
    let line_gradient = super::base::backward_in_shape(&input,
        [line_count, 1, 1, depth], gradient_lines, options.clone());
    let spatial_gradient = planes(from_lines(line_gradient, batch, channels, size[1], size[2]));
    let plane_count = batch.checked_mul(depth).expect("interpolation gradient plane count overflow");
    let gradient = super::base::backward_in_shape(&input,
        [plane_count, channels, height, width], spatial_gradient, options);
    let gradient = reshape(gradient, Shape::new([batch, depth, channels, height, width]));
    permute(gradient, &[0, 2, 1, 3, 4])
}
