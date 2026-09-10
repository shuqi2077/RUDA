use alloc::vec::Vec;
use crate::tensor::{MetadataError, Shape};
use super::calculate_pool_output_size;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Calculate the expected output shape `[batch_size, channels_out, spatial_dims, ..]` for a pooling operation.
pub fn calculate_pool_output_shape<const N: usize>(
    in_shape: &Shape,
    kernel_size: &[usize; N],
    stride: &[usize; N],
    padding: &[usize; N],
    dilation: &[usize; N],
    ceil_mode: bool,
) -> Result<Shape, MetadataError> {
    if in_shape.rank() != N + 2 {
        return Err(MetadataError::RankMismatch {
            left: in_shape.rank(),
            right: N + 2,
        });
    }

    let mut out_shape = in_shape.clone();
    // Spatial dims
    for (i, size_i) in out_shape[2..].iter_mut().enumerate() {
        *size_i = calculate_pool_output_size(
            kernel_size[i],
            stride[i],
            padding[i],
            dilation[i],
            *size_i,
            ceil_mode,
        );
    }

    Ok(out_shape)
}

/// Calculate the expected output shape `[batch_size, channels_out, spatial_dims, ..]` for a convolution.
pub fn calculate_conv_output_shape<const N: usize>(
    in_shape: &Shape,
    weight_shape: &Shape,
    stride: &[usize; N],
    padding: &[usize; N],
    dilation: &[usize; N],
) -> Result<Shape, MetadataError> {
    if weight_shape.rank() != N + 2 {
        return Err(MetadataError::RankMismatch {
            left: weight_shape.rank(),
            right: N + 2,
        });
    }

    if in_shape.rank() != N + 2 {
        return Err(MetadataError::RankMismatch {
            left: in_shape.rank(),
            right: N + 2,
        });
    }

    let kernel_size = &weight_shape[2..];

    let mut out_shape = in_shape.clone();
    // Spatial dims
    for (i, size_i) in out_shape[2..].iter_mut().enumerate() {
        *size_i =
            calculate_conv_output_size(kernel_size[i], stride[i], padding[i], dilation[i], *size_i);
    }
    // Output channels
    out_shape[1] = weight_shape[0];

    Ok(out_shape)
}

/// Calculate the expected output shape `[batch_size, channels_out, spatial_dims, ..]` for a transposed convolution.
pub fn calculate_conv_transpose_output_shape<const N: usize>(
    in_shape: &Shape,
    weight_shape: &Shape,
    stride: &[usize; N],
    padding: &[usize; N],
    padding_out: &[usize; N],
    dilation: &[usize; N],
    groups: usize,
) -> Result<Shape, MetadataError> {
    if weight_shape.rank() != N + 2 {
        return Err(MetadataError::RankMismatch {
            left: weight_shape.rank(),
            right: N + 2,
        });
    }

    if in_shape.rank() != N + 2 {
        return Err(MetadataError::RankMismatch {
            left: in_shape.rank(),
            right: N + 2,
        });
    }

    let kernel_size = &weight_shape[2..];

    let mut out_shape = in_shape.clone();
    // Spatial dims
    for (i, size_i) in out_shape[2..].iter_mut().enumerate() {
        *size_i = calculate_conv_transpose_output_size(
            kernel_size[i],
            stride[i],
            padding[i],
            padding_out[i],
            dilation[i],
            *size_i,
        );
    }
    // Output channels
    out_shape[1] = weight_shape[1] * groups;

    Ok(out_shape)
}

/// Calculate the expected padding size required when applying a convolution.
pub fn calculate_conv_padding(
    kernel_size: usize,
    stride: usize,
    size_in: usize,
    size_out: usize,
) -> usize {
    let kernel_size = kernel_size as f32;
    let stride = stride as f32;
    let size_in = size_in as f32;
    let size_out = size_out as f32;

    let padding = stride * (size_out - 1.) - size_in + kernel_size;
    let padding = (padding / 2.).ceil();

    padding as usize
}

/// Calculate the expected output size when doing a convolution operation.
pub fn calculate_conv_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
) -> usize {
    (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
}

/// Calculate the expected output sizes when doing a convolution operation.
pub fn calculate_conv_output_sizes(
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    size_in: &[usize],
) -> Vec<usize> {
    size_in
        .iter()
        .enumerate()
        .map(|(i, size_in)| {
            calculate_conv_output_size(kernel_size[i], stride[i], padding[i], dilation[i], *size_in)
        })
        .collect()
}

/// Calculate the expected output size when doing a transposed convolution operation.
pub fn calculate_conv_transpose_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    padding_out: usize,
    dilation: usize,
    size_in: usize,
) -> usize {
    (size_in - 1) * stride + (dilation * (kernel_size - 1) + 1) + padding_out - 2 * padding
}

/// Compute the `padding_out` for a transpose conv that exactly recovers the
/// original `size_in` from `size_out`, accounting for any input elements the
/// forward conv dropped. Shared by `conv{1,2,3}d_x_backward` and the Ruda
/// dgrad fallback so the two paths can't drift.
pub fn calculate_padding_out(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
    size_out: usize,
) -> usize {
    if stride <= 1 {
        return 0;
    }

    // Invert the transpose conv output formula to recover the exact number of
    // input elements that a forward conv would drop for this (size_in, size_out).
    //
    // Forward: size_out = floor((size_in + 2*padding - dilated_kernel) / stride) + 1
    // Transpose: trans_out = (size_out - 1)*stride + dilated_kernel + padding_out - 2*padding
    // Setting trans_out == size_in and solving for padding_out:
    let dilated_kernel = dilation * (kernel_size - 1) + 1;
    let base = (size_out as i64 - 1) * stride as i64 + dilated_kernel as i64 - 2 * padding as i64;
    i64::max(0, size_in as i64 - base) as usize
}
