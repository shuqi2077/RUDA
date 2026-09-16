use super::*;

// Match the existing 2D policy: small copies stay on the calling thread,
// and large copies batch rows to amortize Rayon dispatch.
#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;
#[cfg(feature = "rayon")]
const MIN_ELEMS_PER_TASK: usize = 64 * 1024;

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

    let mut output_dims = tensor_shape.to_vec();
    output_dims[dim] = num_indices;
    let output_shape = Shape::from(output_dims);
    let output_size = output_shape.num_elements();

    // A contiguous tensor is [outer_count, select_dim_size, slice_size].
    // Only the middle coordinate changes; trailing elements can be copied
    // together without decomposing each output offset into N coordinates.
    let slice_size: usize = tensor_shape[dim + 1..].iter().product();
    let select_dim_size = tensor_shape[dim];

    // Preserve the batched 2D row-copy fast path and use it for higher ranks
    // as well: all trailing dimensions form one contiguous selected row.
    if dim == 0 {
        let result = select_rows::<E>(
            tensor_data,
            &indices_data,
            select_dim_size,
            slice_size,
            output_size,
        );
        return HostTensor::new(
            Bytes::from_elems(result),
            Layout::contiguous(output_shape),
            E::dtype(),
        );
    }

    let outer_count: usize = tensor_shape[..dim].iter().product();
    if output_size == 0 {
        // The previous bulk-copy path still checked indices when trailing
        // dimensions were empty and an outer row existed. Its element-wise
        // path (including the last axis) and zero-outer-row path did no work.
        if slice_size == 0 && outer_count != 0 {
            for &idx in indices_data.iter() {
                checked_index(idx, select_dim_size);
            }
        }
        return HostTensor::new(
            Bytes::from_elems(Vec::<E>::new()),
            Layout::contiguous(output_shape),
            E::dtype(),
        );
    }

    let source_offsets: Vec<usize> = indices_data
        .iter()
        .map(|&idx| checked_index(idx, select_dim_size) * slice_size)
        .collect();
    let source_row_size = select_dim_size * slice_size;
    let output_row_size = num_indices * slice_size;
    let mut result = vec![E::default(); output_size];

    let copy_rows = |start_row: usize, rows: &mut [E]| {
        for (row, dst_row) in rows.chunks_mut(output_row_size).enumerate() {
            let src_start = (start_row + row) * source_row_size;
            let src_row = &tensor_data[src_start..src_start + source_row_size];
            if slice_size == 1 {
                for (dst, &offset) in dst_row.iter_mut().zip(&source_offsets) {
                    *dst = src_row[offset];
                }
            } else {
                for (dst, &offset) in dst_row.chunks_mut(slice_size).zip(&source_offsets) {
                    dst.copy_from_slice(&src_row[offset..offset + slice_size]);
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    if output_size.saturating_mul(size_of::<E>()) >= PARALLEL_THRESHOLD_BYTES {
        let rows_per_chunk = (MIN_ELEMS_PER_TASK / output_row_size)
            .max(1)
            .min(outer_count);
        result
            .par_chunks_mut(rows_per_chunk * output_row_size)
            .enumerate()
            .for_each(|(chunk, rows)| copy_rows(chunk * rows_per_chunk, rows));
    } else {
        copy_rows(0, &mut result);
    }

    #[cfg(not(feature = "rayon"))]
    copy_rows(0, &mut result);

    HostTensor::new(
        Bytes::from_elems(result),
        Layout::contiguous(output_shape),
        E::dtype(),
    )
}

/// Select contiguous rows. Serial copies initialize the Vec by extending it;
/// parallel copies use initialized storage before creating mutable slices.
#[inline]
fn select_rows<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor_data: &[E],
    indices_data: &[isize],
    dim_size: usize,
    row_size: usize,
    output_size: usize,
) -> Vec<E> {
    let mut result = Vec::with_capacity(output_size);

    #[cfg(feature = "rayon")]
    if output_size.saturating_mul(size_of::<E>()) >= PARALLEL_THRESHOLD_BYTES {
        result.resize(output_size, E::default());
        let rows_per_chunk = (MIN_ELEMS_PER_TASK / row_size)
            .max(1)
            .min(indices_data.len());
        result
            .par_chunks_mut(rows_per_chunk * row_size)
            .enumerate()
            .for_each(|(chunk, rows)| {
                let start_row = chunk * rows_per_chunk;
                for (i, dst_row) in rows.chunks_mut(row_size).enumerate() {
                    let src_row = checked_index(indices_data[start_row + i], dim_size);
                    let src_start = src_row * row_size;
                    dst_row.copy_from_slice(&tensor_data[src_start..src_start + row_size]);
                }
            });
        return result;
    }

    for &idx in indices_data {
        let src_row = checked_index(idx, dim_size);
        let src_start = src_row * row_size;
        result.extend_from_slice(&tensor_data[src_start..src_start + row_size]);
    }
    result
}

#[cfg(test)]
#[path = "select_tests.rs"]
mod tests;
