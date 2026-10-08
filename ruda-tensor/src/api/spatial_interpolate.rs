use super::{Tensor, backend::Backend, ops::InterpolateOptions};
use super::module::interpolate;
use super::spatial_pool::{volume_planes, plane_depth_lines, depth_lines_volume};

/// Resize native `[batch, channels, length]` activations.
///
/// Applies the existing interpolation kernel along the length axis, preserving
/// its coordinate mapping, mode, storage and differentiable backward operation.
pub fn interpolate1d<B: Backend>(
    input: Tensor<B, 3>,
    output_size: usize,
    options: InterpolateOptions,
) -> Tensor<B, 3> {
    interpolate(input.unsqueeze_dim(2), [1, output_size], options).squeeze_dim(2)
}

/// Resize native `[batch, channels, depth, height, width]` activations.
///
/// Factors interpolation into native height/width and depth passes. `Bilinear`
/// therefore applies linear interpolation in all three axes; cubic and Lanczos
/// apply their existing filters separably. Both passes use the supplied corner
/// alignment, without host resampling or detaching the gradient graph.
pub fn interpolate3d<B: Backend>(
    input: Tensor<B, 5>,
    output_size: [usize; 3],
    options: InterpolateOptions,
) -> Tensor<B, 5> {
    let [batch, channels, depth, _, _] = input.dims();
    let planes = interpolate(volume_planes(input), [output_size[1], output_size[2]], options.clone());
    let [_, _, height, width] = planes.dims();
    let lines = interpolate1d(plane_depth_lines(planes, batch, depth), output_size[0], options);
    depth_lines_volume(lines, batch, channels, height, width)
}
