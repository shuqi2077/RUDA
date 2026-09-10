use super::*;

/// Extract 2D matrix strides from a tensor layout.
/// Returns (row_stride, col_stride) for the last two dimensions.
pub(super) fn get_2d_strides(layout: &Layout) -> (isize, isize) {
    let strides = layout.strides();
    let ndim = strides.len();
    let row_stride = strides[ndim - 2];
    let col_stride = strides[ndim - 1];
    (row_stride, col_stride)
}

/// Compute broadcast batch dimensions for batched matmul.
/// Returns (broadcast_shape, lhs_strides, rhs_strides) where strides map
/// output batch index to input batch offset (in matrices).
pub(super) fn broadcast_batch_dims(
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    // Pad shorter batch dims with 1s on the left
    let max_len = lhs_batch.len().max(rhs_batch.len());
    let lhs_padded: Vec<usize> = (0..max_len)
        .map(|i| {
            if i < max_len - lhs_batch.len() {
                1
            } else {
                lhs_batch[i - (max_len - lhs_batch.len())]
            }
        })
        .collect();
    let rhs_padded: Vec<usize> = (0..max_len)
        .map(|i| {
            if i < max_len - rhs_batch.len() {
                1
            } else {
                rhs_batch[i - (max_len - rhs_batch.len())]
            }
        })
        .collect();

    // Compute broadcast shape and strides
    let mut broadcast_shape = Vec::with_capacity(max_len);
    let mut lhs_strides = Vec::with_capacity(max_len);
    let mut rhs_strides = Vec::with_capacity(max_len);

    // Compute strides from right to left
    let mut lhs_stride = 1usize;
    let mut rhs_stride = 1usize;
    for i in (0..max_len).rev() {
        let ld = lhs_padded[i];
        let rd = rhs_padded[i];
        debug_assert!(
            ld == rd || ld == 1 || rd == 1,
            "matmul: batch dimensions not broadcastable: {:?} vs {:?}",
            lhs_batch,
            rhs_batch
        );
        broadcast_shape.push(ld.max(rd));
        // Stride is 0 if dimension is 1 (broadcast), otherwise actual stride
        lhs_strides.push(if ld == 1 { 0 } else { lhs_stride });
        rhs_strides.push(if rd == 1 { 0 } else { rhs_stride });
        lhs_stride *= ld;
        rhs_stride *= rd;
    }

    // Reverse to get correct order
    broadcast_shape.reverse();
    lhs_strides.reverse();
    rhs_strides.reverse();

    (broadcast_shape, lhs_strides, rhs_strides)
}

/// Convert a flat batch index to input batch offset using broadcast strides.
#[inline]
pub(super) fn batch_index_to_offset(b: usize, broadcast_shape: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0;
    let mut remaining = b;
    for i in (0..broadcast_shape.len()).rev() {
        let idx = remaining % broadcast_shape[i];
        offset += idx * strides[i];
        remaining /= broadcast_shape[i];
    }
    offset
}

/// Compute element-level batch strides for a tensor in a broadcast context.
/// Uses the actual layout strides so non-contiguous (transposed/sliced) tensors
/// work without a copy. Dimensions that are broadcast (size 1) get stride 0.
#[allow(clippy::needless_range_loop)]
pub(super) fn broadcast_batch_elem_strides(
    batch_shape: &[usize],
    layout_strides: &[isize],
    broadcast_len: usize,
) -> Vec<isize> {
    let batch_ndim = batch_shape.len();
    debug_assert!(broadcast_len >= batch_ndim);
    let mut result = vec![0isize; broadcast_len];

    for i in 0..broadcast_len {
        let batch_idx = i as isize - (broadcast_len as isize - batch_ndim as isize);
        if batch_idx >= 0 {
            let bi = batch_idx as usize;
            if batch_shape[bi] > 1 {
                result[i] = layout_strides[bi];
            }
        }
    }

    result
}

/// Convert a flat batch index to an element offset using element-level strides.
#[inline]
pub(super) fn batch_elem_offset(b: usize, broadcast_shape: &[usize], elem_strides: &[isize]) -> isize {
    let mut offset: isize = 0;
    let mut remaining = b;
    for i in (0..broadcast_shape.len()).rev() {
        let idx = remaining % broadcast_shape[i];
        offset += idx as isize * elem_strides[i];
        remaining /= broadcast_shape[i];
    }
    offset
}

