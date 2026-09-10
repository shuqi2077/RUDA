use super::*;

/// Select slices from tensor along a dimension using 1D indices.
///
/// Unlike gather, indices is 1D and selects entire slices.
/// For a 2D tensor with dim=0 and indices=[2, 0]:
/// output[0, :] = tensor[2, :]
/// output[1, :] = tensor[0, :]
pub fn select<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();

    let tensor_shape = tensor.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices.layout().num_dims(),
        1,
        "select: indices must be 1D, got {} dims",
        indices.layout().num_dims()
    );

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);
    let num_indices = indices_data.len();

    // Build output shape: replace dim with num_indices
    let mut output_dims = tensor_shape.to_vec();
    output_dims[dim] = num_indices;
    let output_shape = Shape::from(output_dims);

    // Use optimized 2D implementation with bulk copies
    if ndims == 2 {
        let result = select_2d::<E>(
            tensor_data,
            &indices_data,
            tensor_shape[0],
            tensor_shape[1],
            num_indices,
            dim,
        );
        let bytes = Bytes::from_elems(result);
        return HostTensor::new(bytes, Layout::contiguous(output_shape), E::dtype());
    }

    // General N-D case
    let tensor_strides: Vec<usize> = compute_strides(tensor_shape);
    let output_strides: Vec<usize> = compute_strides(&output_shape);
    let output_size = output_shape.num_elements();

    // Calculate slice size (elements after dim)
    let slice_size: usize = tensor_strides[dim];

    let select_dim_size = tensor_shape[dim];

    // If dim is the last dimension or we can use bulk copies
    if dim == ndims - 1 || slice_size == 1 {
        // Element-wise with parallelism
        #[cfg(feature = "rayon")]
        let result: Vec<E> = (0..output_size)
            .into_par_iter()
            .map(|out_idx| {
                let mut remaining = out_idx;
                let mut src_idx = 0;
                for d in 0..ndims {
                    let coord = remaining / output_strides[d];
                    remaining %= output_strides[d];
                    if d == dim {
                        let index_val = checked_index(indices_data[coord], select_dim_size);
                        src_idx += index_val * tensor_strides[d];
                    } else {
                        src_idx += coord * tensor_strides[d];
                    }
                }
                tensor_data[src_idx]
            })
            .collect();

        #[cfg(not(feature = "rayon"))]
        #[allow(clippy::needless_range_loop)]
        let result: Vec<E> = {
            let mut result = vec![E::default(); output_size];
            for out_idx in 0..output_size {
                let mut remaining = out_idx;
                let mut src_idx = 0;
                for d in 0..ndims {
                    let coord = remaining / output_strides[d];
                    remaining %= output_strides[d];
                    if d == dim {
                        let index_val = checked_index(indices_data[coord], select_dim_size);
                        src_idx += index_val * tensor_strides[d];
                    } else {
                        src_idx += coord * tensor_strides[d];
                    }
                }
                result[out_idx] = tensor_data[src_idx];
            }
            result
        };

        let bytes = Bytes::from_elems(result);
        return HostTensor::new(bytes, Layout::contiguous(output_shape), E::dtype());
    }

    // Use bulk copies for contiguous slices
    let mut result = vec![E::default(); output_size];

    // For each position in dimensions before `dim`
    let outer_count = if dim == 0 {
        1
    } else {
        tensor_shape[..dim].iter().product()
    };

    for outer in 0..outer_count {
        let outer_offset_tensor = outer * tensor_strides[if dim == 0 { 0 } else { dim - 1 }];
        let outer_offset_output = outer * output_strides[if dim == 0 { 0 } else { dim - 1 }];

        for (i, &idx) in indices_data.iter().enumerate() {
            let index_val = checked_index(idx, select_dim_size);
            let src_start = outer_offset_tensor + index_val * tensor_strides[dim];
            let dst_start = outer_offset_output + i * output_strides[dim];
            result[dst_start..dst_start + slice_size]
                .copy_from_slice(&tensor_data[src_start..src_start + slice_size]);
        }
    }

    let bytes = Bytes::from_elems(result);
    HostTensor::new(bytes, Layout::contiguous(output_shape), E::dtype())
}

/// Optimized 2D select with bulk row copies when dim=0.
#[inline]
fn select_2d<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor_data: &[E],
    indices_data: &[isize],
    tensor_rows: usize,
    tensor_cols: usize,
    num_indices: usize,
    dim: usize,
) -> Vec<E> {
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };
    let (output_rows, output_cols) = if dim == 0 {
        (num_indices, tensor_cols)
    } else {
        (tensor_rows, num_indices)
    };
    let output_size = output_rows * output_cols;

    // Minimum bytes of output before we consider rayon. Below this, a
    // single-threaded loop is faster because there is not enough work to
    // amortize the work-stealing dispatch overhead.
    #[cfg(feature = "rayon")]
    const PARALLEL_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;

    // Minimum elements per rayon task. Without batching, par_chunks_mut
    // creates one task per row (e.g. 512 single-row tasks of 4 KB each)
    // whose dispatch overhead dominates the actual copy.
    #[cfg(feature = "rayon")]
    const MIN_ELEMS_PER_TASK: usize = 64 * 1024;

    if dim == 0 {
        // SAFETY: the output has exactly num_indices * tensor_cols elements.
        // Both the parallel and serial paths below write every element exactly
        // once via non-overlapping row copies, so no element is left uninitialized.
        let mut result = Vec::with_capacity(output_size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            result.set_len(output_size)
        };

        #[cfg(feature = "rayon")]
        if output_size * size_of::<E>() >= PARALLEL_THRESHOLD_BYTES {
            // Batch multiple rows per rayon task so each task copies at
            // least MIN_ELEMS_PER_TASK elements.
            let rows_per_chunk = (MIN_ELEMS_PER_TASK / tensor_cols).max(1);
            let elems_per_chunk = rows_per_chunk * tensor_cols;
            result.par_chunks_mut(elems_per_chunk).enumerate().for_each(
                |(chunk_idx, dst_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    let chunk_rows = dst_chunk.len() / tensor_cols;
                    for i in 0..chunk_rows {
                        let src_row_idx = checked_index(indices_data[start_row + i], dim_size);
                        let src_start = src_row_idx * tensor_cols;
                        let dst_start = i * tensor_cols;
                        dst_chunk[dst_start..dst_start + tensor_cols]
                            .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
                    }
                },
            );
        } else {
            for (i, &idx) in indices_data.iter().enumerate() {
                let src_row_idx = checked_index(idx, dim_size);
                let src_start = src_row_idx * tensor_cols;
                let dst_start = i * tensor_cols;
                result[dst_start..dst_start + tensor_cols]
                    .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, &idx) in indices_data.iter().enumerate() {
                let src_row_idx = checked_index(idx, dim_size);
                let src_start = src_row_idx * tensor_cols;
                let dst_start = i * tensor_cols;
                result[dst_start..dst_start + tensor_cols]
                    .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
            }
        }

        result
    } else {
        // dim == 1: gather individual elements per row (not contiguous).
        // Zero-init is fine here since the inner loop is per-element anyway.
        let mut result = vec![E::default(); output_size];

        #[cfg(feature = "rayon")]
        if output_size * size_of::<E>() >= PARALLEL_THRESHOLD_BYTES {
            let rows_per_chunk = (MIN_ELEMS_PER_TASK / output_cols).max(1);
            let elems_per_chunk = rows_per_chunk * output_cols;
            result.par_chunks_mut(elems_per_chunk).enumerate().for_each(
                |(chunk_idx, dst_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    let chunk_rows = dst_chunk.len() / output_cols;
                    for r in 0..chunk_rows {
                        let row = start_row + r;
                        let dst_base = r * output_cols;
                        for (j, &idx) in indices_data.iter().enumerate() {
                            let src_col = checked_index(idx, dim_size);
                            dst_chunk[dst_base + j] = tensor_data[row * tensor_cols + src_col];
                        }
                    }
                },
            );
        } else {
            for row in 0..output_rows {
                for (j, &idx) in indices_data.iter().enumerate() {
                    let src_col = checked_index(idx, dim_size);
                    result[row * output_cols + j] = tensor_data[row * tensor_cols + src_col];
                }
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for row in 0..output_rows {
                for (j, &idx) in indices_data.iter().enumerate() {
                    let src_col = checked_index(idx, dim_size);
                    result[row * output_cols + j] = tensor_data[row * tensor_cols + src_col];
                }
            }
        }

        result
    }
}

