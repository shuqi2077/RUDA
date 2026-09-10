pub use ruda_core::tensor::{QPARAM_ALIGN, params_shape};
use ruda_core::tensor::{QuantLevel, QuantMode, QuantScheme, Shape, Slice};
use alloc::vec;

use super::{Calibration, QuantizationParametersPrimitive};
use crate::{Backend, TensorMetadata, get_device_settings};

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match scheme.level {
            QuantLevel::Tensor => (B::float_min(tensor.clone()), B::float_max(tensor)),
            QuantLevel::Block(block_size) => {
                let shape = tensor.shape();
                let block_dims = block_size.to_dim_vec(shape.rank());
                assert!(!block_dims.contains(&0), "Quantization block dimensions must be nonzero");
                let params_shape = params_shape(&shape, scheme.level);
                if shape.num_elements() == 0 {
                    let device = B::float_device(&tensor);
                    let dtype = tensor.dtype().into();
                    return (
                        B::float_empty(params_shape.clone(), &device, dtype),
                        B::float_empty(params_shape, &device, dtype),
                    );
                }

                let mut contiguous_blocks = true;
                let mut inner_axes_complete = true;
                for (&dim, &block) in shape.iter().zip(&block_dims).rev() {
                    let block = block as usize;
                    contiguous_blocks &= dim.is_multiple_of(block)
                        && (block == 1 || inner_axes_complete);
                    inner_axes_complete &= dim == block;
                }
                if contiguous_blocks {
                    let num_blocks = params_shape.num_elements();
                    let block_elems = shape.num_elements() / num_blocks;
                    let blocks = B::float_reshape(tensor, Shape::new([num_blocks, block_elems]));
                    return (
                        B::float_reshape(B::float_min_dim(blocks.clone(), 1), params_shape.clone()),
                        B::float_reshape(B::float_max_dim(blocks, 1), params_shape),
                    );
                }

                let mut min = tensor.clone();
                let mut max = tensor;
                for (axis, &block) in block_dims.iter().enumerate().rev() {
                    if block > 1 {
                        min = reduce_block_axis::<B>(min, axis, block as usize, B::float_min_dim);
                        max = reduce_block_axis::<B>(max, axis, block as usize, B::float_max_dim);
                    }
                }
                (min, max)
            }
        },
    }
}

fn reduce_block_axis<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    axis: usize,
    block: usize,
    reduce: fn(B::FloatTensorPrimitive, usize) -> B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let shape = tensor.shape();
    if shape[axis] <= block {
        return reduce(tensor, axis);
    }
    let full_blocks = shape[axis] / block;
    let full_len = full_blocks * block;
    let has_tail = full_len != shape[axis];
    let mut slices = vec![Slice::from(..); shape.rank()];
    let head = if has_tail {
        slices[axis] = Slice::from(0..full_len);
        B::float_slice(tensor.clone(), &slices)
    } else {
        tensor.clone()
    };
    let mut grouped_shape = shape.clone();
    grouped_shape[axis] = full_blocks;
    grouped_shape.insert(axis + 1, block);
    let head = reduce(B::float_reshape(head, grouped_shape), axis + 1);
    let mut output_shape = shape;
    output_shape[axis] = full_blocks;
    let head = B::float_reshape(head, output_shape);
    if has_tail {
        slices[axis] = Slice::from(full_len..);
        let tail = reduce(B::float_slice(tensor, &slices), axis);
        B::float_cat(vec![head, tail], axis)
    } else {
        head
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params<B: Backend>(
    scheme: &QuantScheme,
    min: B::FloatTensorPrimitive,
    max: B::FloatTensorPrimitive,
) -> QuantizationParametersPrimitive<B> {
    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            ..
        } => {
            let bool_dtype = get_device_settings::<B>(&B::float_device(&min)).bool_dtype;
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

            // Compute scale to convert an input value in range `[-alpha, alpha]`
            let min_abs = B::float_abs(min);
            let max_abs = B::float_abs(max);

            // `min_abs.max_pair(max_abs)`
            let mask = B::float_lower(min_abs.clone(), max_abs.clone(), bool_dtype);
            let values_range =
                B::float_mul_scalar(B::float_mask_where(min_abs, mask, max_abs), 2f32.into());

            QuantizationParametersPrimitive {
                scales: B::float_div_scalar(values_range, (b - a).into()),
            }
        }
    }
}
