use crate::dsl::Runtime;
use super::RudaTensor;
use ruda_core::tensor::{Shape, Metadata, TensorMetadata, strides};
use ruda_core::tensor::spatial::calculate_unfold_shape;

pub fn expand<R: Runtime>(tensor: RudaTensor<R>, target_shape: Shape) -> RudaTensor<R> {
    if tensor.qparams.is_some() {
        return super::expand_quantized::expand(tensor, target_shape);
    }

    let ndims_in = tensor.meta.shape().num_dims();
    let ndims_out = target_shape.num_dims();

    // Initialize new strides with zeros
    let mut new_strides = strides![0usize; ndims_out];

    // Calculate the difference in dimensions
    let dim_diff = ndims_out.saturating_sub(ndims_in);

    // Compare dimensions from the end, setting strides for matching dimensions or broadcasted ones
    let mut tensor_dim_iter = tensor.meta.shape().iter().rev();
    for i in (0..ndims_out).rev() {
        if i >= dim_diff {
            if let Some(&tensor_dim) = tensor_dim_iter.next() {
                if tensor_dim == target_shape[i] || tensor_dim == 1 {
                    // Copy stride for non-broadcast dimensions or set to 0 for broadcast ones
                    new_strides[i] = if tensor_dim == target_shape[i] {
                        tensor.meta.strides()[i - dim_diff]
                    } else {
                        0
                    };
                } else {
                    // Error handling: Dimension mismatch for broadcasting
                    panic!(
                        "Dimension mismatch: cannot broadcast dimension {tensor_dim} of tensor to target shape"
                    );
                }
            } else {
                // If the input tensor has fewer dimensions, treat missing dimensions as 1
                // and set stride to 0 (broadcasting)
                new_strides[i] = 0;
            }
        } else {
            // For extra dimensions in the target shape, set stride to 0 (broadcasting)
            new_strides[i] = 0;
        }
    }

    RudaTensor {
        client: tensor.client.clone(),
        device: tensor.device.clone(),
        meta: Box::new(Metadata::new(target_shape, new_strides)),
        handle: tensor.handle.clone(),
        dtype: tensor.dtype,
        qparams: tensor.qparams.clone(),
    }
}

/// Unfold windows along a dimension.
///
/// Returns a view of the tensor with all complete windows of size `size` in dimension `dim`;
/// where windows are advanced by `step` at each index.
///
/// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
///
/// The new view will have the unfolded dimension replaced by two dimensions;
/// one in the position of the original dimension, with size equal to the number of windows,
/// and one appended to the right-most position, with size equal to `size`.
///
/// # Arguments
///
/// * `tensor` - The input tensor to unfold; of shape ``[pre=..., dim shape, post=...]``
/// * `dim` - the dimension to unfold.
/// * `size` - the size of each unfolded window.
/// * `step` - the step between each window.
///
/// # Returns
///
/// A tensor view with the shape ``[pre=..., windows, post=..., size]``.
pub fn unfold<R: Runtime>(
    tensor: RudaTensor<R>,
    dim: usize,
    size: usize,
    step: usize,
) -> RudaTensor<R> {
    let shape = calculate_unfold_shape(tensor.shape(), dim, size, step);

    let d_stride = tensor.meta.strides()[dim];
    let mut strides = tensor.meta.strides.clone();
    strides[dim] = step * d_stride;
    strides.push(d_stride);

    RudaTensor {
        meta: Box::new(Metadata::new(shape, strides)),
        client: tensor.client.clone(),
        handle: tensor.handle.clone(),
        device: tensor.device.clone(),
        dtype: tensor.dtype,
        qparams: tensor.qparams.clone(),
    }
}
