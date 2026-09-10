use super::*;

/// Gather values from tensor along a dimension using indices.
///
/// For a 2D tensor with dim=1:
/// output[i, j] = tensor[i, indices[i, j]]
///
/// The output has the same shape as indices.
pub fn gather<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();

    let tensor_shape = tensor.layout().shape();
    let indices_shape = indices.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Validate shapes: all dims except `dim` must match between tensor and indices
    for i in 0..ndims {
        if i != dim {
            assert_eq!(
                tensor_shape[i], indices_shape[i],
                "gather: shape mismatch at dim {}: tensor {} vs indices {}",
                i, tensor_shape[i], indices_shape[i]
            );
        }
    }

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);

    // Calculate strides for tensor (row-major)
    let tensor_strides: Vec<usize> = compute_strides(tensor_shape);
    let indices_strides: Vec<usize> = compute_strides(indices_shape);

    let output_size = indices_shape.num_elements();

    // Use specialized 2D implementation for common case
    if ndims == 2 {
        let result = gather_2d::<E>(
            tensor_data,
            &indices_data,
            tensor_shape[0],
            tensor_shape[1],
            indices_shape[0],
            indices_shape[1],
            dim,
        );
        let bytes = Bytes::from_elems(result);
        return HostTensor::new(bytes, Layout::contiguous(indices_shape.clone()), E::dtype());
    }

    // General N-D case with pre-allocated coordinates
    let dim_stride = tensor_strides[dim];

    let gather_dim_size = tensor_shape[dim];

    #[cfg(feature = "rayon")]
    let result: Vec<E> = (0..output_size)
        .into_par_iter()
        .map(|out_idx| {
            let index_val = checked_index(indices_data[out_idx], gather_dim_size);
            let src_idx = compute_gather_index(
                out_idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            tensor_data[src_idx]
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let result: Vec<E> = (0..output_size)
        .map(|out_idx| {
            let index_val = checked_index(indices_data[out_idx], gather_dim_size);
            let src_idx = compute_gather_index(
                out_idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            tensor_data[src_idx]
        })
        .collect();

    let bytes = Bytes::from_elems(result);
    HostTensor::new(bytes, Layout::contiguous(indices_shape.clone()), E::dtype())
}

/// Optimized 2D gather implementation.
#[inline]
fn gather_2d<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor_data: &[E],
    indices_data: &[isize],
    tensor_rows: usize,
    tensor_cols: usize,
    indices_rows: usize,
    indices_cols: usize,
    dim: usize,
) -> Vec<E> {
    let output_size = indices_rows * indices_cols;
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };

    let mut result = vec![E::default(); output_size];

    #[cfg(feature = "rayon")]
    const PARALLEL_THRESHOLD: usize = 256 * 1024;

    #[cfg(feature = "rayon")]
    if output_size >= PARALLEL_THRESHOLD {
        if dim == 0 {
            result
                .par_chunks_mut(indices_cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for j in 0..indices_cols {
                        let src_row = checked_index(indices_data[i * indices_cols + j], dim_size);
                        row[j] = tensor_data[src_row * tensor_cols + j];
                    }
                });
        } else {
            result
                .par_chunks_mut(indices_cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for j in 0..indices_cols {
                        let src_col = checked_index(indices_data[i * indices_cols + j], dim_size);
                        row[j] = tensor_data[i * tensor_cols + src_col];
                    }
                });
        }
    } else if dim == 0 {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let src_row = checked_index(indices_data[i * indices_cols + j], dim_size);
                result[i * indices_cols + j] = tensor_data[src_row * tensor_cols + j];
            }
        }
    } else {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let src_col = checked_index(indices_data[i * indices_cols + j], dim_size);
                result[i * indices_cols + j] = tensor_data[i * tensor_cols + src_col];
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        if dim == 0 {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let src_row = checked_index(indices_data[i * indices_cols + j], dim_size);
                    result[i * indices_cols + j] = tensor_data[src_row * tensor_cols + j];
                }
            }
        } else {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let src_col = checked_index(indices_data[i * indices_cols + j], dim_size);
                    result[i * indices_cols + j] = tensor_data[i * tensor_cols + src_col];
                }
            }
        }
    }

    result
}

/// Compute source index for gather operation (N-D case).
#[inline]
pub(super) fn compute_gather_index(
    out_idx: usize,
    index_val: usize,
    dim: usize,
    dim_stride: usize,
    indices_strides: &[usize],
    tensor_strides: &[usize],
    ndims: usize,
) -> usize {
    let mut src_idx = index_val * dim_stride;
    let mut remaining = out_idx;

    for d in 0..ndims {
        if d != dim {
            let coord = remaining / indices_strides[d];
            remaining %= indices_strides[d];
            src_idx += coord * tensor_strides[d];
        } else {
            remaining %= indices_strides[d];
        }
    }
    src_idx
}

