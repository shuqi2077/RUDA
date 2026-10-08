use super::{Tensor, TensorPrimitive, backend::Backend, ops::InterpolateOptions};

/// Resize native `[batch, channels, length]` activations.
///
/// Applies the existing interpolation kernel along the length axis, preserving
/// its coordinate mapping, mode, storage and differentiable backward operation.
pub fn interpolate1d<B: Backend>(
    input: Tensor<B, 3>,
    output_size: usize,
    options: InterpolateOptions,
) -> Tensor<B, 3> {
    Tensor::new(TensorPrimitive::Float(B::interpolate1d(input.into_primitive().tensor(), output_size, options)))
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
    Tensor::new(TensorPrimitive::Float(B::interpolate3d(input.into_primitive().tensor(), output_size, options)))
}
