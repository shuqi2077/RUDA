use super::*;

// ============================================================================
// Dimension reduction helpers
// ============================================================================

#[derive(Clone, Copy)]
pub(super) enum ReduceOp {
    Sum,
    Prod,
}

/// Optimized f32 dimension reduction with SIMD.
pub(super) fn reduce_dim_f32(tensor: &HostTensor, dim: usize, op: ReduceOp) -> HostTensor {
    let ndims = tensor.layout().shape().num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Copy to contiguous only when the flattened stride assumption breaks:
    // non-contiguous tensor with 2+ outer dims or 2+ inner dims.
    let outer_dims = dim;
    let inner_dims = ndims - dim - 1;
    let needs_copy = !tensor.is_contiguous() && (outer_dims > 1 || inner_dims > 1);
    let tensor = if needs_copy {
        tensor.to_contiguous()
    } else {
        tensor.clone()
    };
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: any zero-sized non-reduced dim means the result has no
    // elements. Return early so the SIMD kernels and fallback loops never
    // see `outer_size == 0` or `inner_size == 0`.
    if out_size == 0 {
        return HostTensor::new(
            Bytes::from_elems(Vec::<f32>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            DType::F32,
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[f32] = tensor.storage();
    let start_offset = tensor.layout().start_offset();
    let dim_stride = strides[dim];

    let (init, reduce_fn): (f32, fn(f32, f32) -> f32) = match op {
        ReduceOp::Sum => (0.0, |a, b| a + b),
        ReduceOp::Prod => (1.0, |a, b| a * b),
    };

    // Check for negative strides (from flip operations) - fall back to general case
    let has_negative_strides = strides.iter().any(|&s| s < 0);

    // Check if inner dimension is contiguous (stride = 1) and no negative strides
    let inner_contiguous = !has_negative_strides && (dim + 1 >= ndims || strides[ndims - 1] == 1);

    let result: Vec<f32> = if inner_contiguous && dim == ndims - 1 && dim_stride == 1 {
        // Reducing last dimension with contiguous data: use SIMD.
        // `reduce_last_dim_f32` reads each row as `&data[start..start + dim_size]`,
        // which only matches the logical row when the reduce dim itself has
        // stride 1. Transposed views (e.g. shape [3,2] strides [1,3]) would
        // otherwise read contiguous storage and return wrong sums.
        reduce_last_dim_f32(data, start_offset, outer_size, dim_size, strides, dim, op)
    } else if dim == 0 && inner_contiguous && matches!(op, ReduceOp::Sum) {
        // First-dim reduction with contiguous inner: use cache-friendly accumulation
        reduce_first_dim_f32(data, start_offset, dim_size, inner_size, dim_stride)
    } else if dim > 0 && dim < ndims - 1 && inner_contiguous && matches!(op, ReduceOp::Sum) {
        // Middle-dim reduction (e.g., [B, M, K] reducing dim=1): cache-friendly accumulation
        let outer_stride = strides[dim - 1];
        reduce_middle_dim_f32(
            data,
            start_offset,
            outer_size,
            dim_size,
            inner_size,
            outer_stride,
            dim_stride,
        )
    } else if dim_stride == 1 && matches!(op, ReduceOp::Sum) && outer_size == 1 {
        // Reduction dimension is contiguous, no outer batch (e.g., transposed 2D reducing dim=0)
        // Storage is [inner_size rows of dim_size elements each] - use sum_rows_f32
        #[cfg(feature = "simd")]
        {
            let mut result = vec![0.0f32; inner_size];
            kernels::sum_rows_f32(
                &data[start_offset..],
                &mut result,
                inner_size, // number of rows (output positions)
                dim_size,   // elements per row (to sum)
            );
            result
        }
        #[cfg(not(feature = "simd"))]
        {
            let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };
            let mut result = Vec::with_capacity(out_size);
            for inner in 0..inner_size {
                let base = (start_offset as isize + inner as isize * inner_stride) as usize;
                let slice = &data[base..base + dim_size];
                result.push(slice.iter().copied().sum());
            }
            result
        }
    } else if dim_stride == 1 && matches!(op, ReduceOp::Sum) {
        // Reduction dimension is contiguous but with outer batches
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = (start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride) as usize;
                let slice = &data[base..base + dim_size];
                #[cfg(feature = "simd")]
                let acc = kernels::sum_f32(slice);
                #[cfg(not(feature = "simd"))]
                let acc = slice.iter().copied().sum();
                result.push(acc);
            }
        }
        result
    } else if tensor.is_contiguous() {
        // Contiguous: use flat index arithmetic (safe for any ndims).
        // outer_size and inner_size are guaranteed positive by the out_size == 0
        // early return above.
        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
        result
    } else {
        // Non-contiguous with at most 1 outer + 1 inner dim (e.g., flipped 2D)
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride;
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = (base + d as isize * dim_stride) as usize;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
        result
    };

    let bytes = Bytes::from_elems(result);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::F32,
    )
}

/// Reduce middle dimension (e.g., [B, M, K] reducing dim=1) with cache-friendly iteration.
/// For each batch, iterate over rows (dim to reduce) sequentially and accumulate into columns.
#[inline]
pub(super) fn reduce_middle_dim_f32(
    data: &[f32],
    start_offset: usize,
    outer_size: usize, // batch size
    dim_size: usize,   // rows to sum
    inner_size: usize, // columns (output per batch)
    outer_stride: isize,
    dim_stride: isize,
) -> Vec<f32> {
    let out_size = outer_size * inner_size;

    #[cfg(feature = "simd")]
    {
        // Use aligned allocation for optimal SIMD scatter-add
        let mut result = aligned::alloc_aligned_zeroed::<f32>(out_size);
        kernels::scatter_add_batched(
            &data[start_offset..],
            &mut result,
            outer_size,
            dim_size,
            inner_size,
            outer_stride as usize,
            dim_stride as usize,
        );
        aligned::to_vec(result)
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; out_size];
        let start = start_offset as isize;
        for batch in 0..outer_size {
            let batch_start = (start + batch as isize * outer_stride) as usize;
            let out_batch_start = batch * inner_size;

            for row in 0..dim_size {
                let row_start = (batch_start as isize + row as isize * dim_stride) as usize;
                for c in 0..inner_size {
                    result[out_batch_start + c] += data[row_start + c];
                }
            }
        }
        result
    }
}

/// Reduce first dimension with cache-friendly row iteration.
/// Instead of iterating per-output (col) and gathering from rows (cache-unfriendly),
/// iterate over rows (sequential access) and scatter-accumulate into outputs.
#[inline]
pub(super) fn reduce_first_dim_f32(
    data: &[f32],
    start_offset: usize,
    dim_size: usize,   // number of rows to sum
    inner_size: usize, // number of columns (output positions)
    dim_stride: isize, // stride between rows
) -> Vec<f32> {
    #[cfg(feature = "simd")]
    {
        // Use aligned allocation for optimal SIMD scatter-add
        let mut result = aligned::alloc_aligned_zeroed::<f32>(inner_size);
        kernels::scatter_add_f32(
            &data[start_offset..],
            &mut result,
            dim_size,
            inner_size,
            dim_stride as usize,
        );
        aligned::to_vec(result)
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; inner_size];
        let start = start_offset as isize;
        for row in 0..dim_size {
            let row_start = (start + row as isize * dim_stride) as usize;
            for c in 0..inner_size {
                result[c] += data[row_start + c];
            }
        }
        result
    }
}

/// Reduce last dimension with SIMD.
///
/// For contiguous Sum: batches all rows in a single kernel call using
/// 4-accumulator SIMD to hide add latency.
#[inline]
pub(super) fn reduce_last_dim_f32(
    data: &[f32],
    start_offset: usize,
    outer_size: usize,
    dim_size: usize,
    strides: &[isize],
    dim: usize,
    op: ReduceOp,
) -> Vec<f32> {
    let outer_stride: isize = if dim > 0 {
        strides[dim - 1]
    } else {
        dim_size as isize
    };

    // `outer_size > 0` is guaranteed by the out_size == 0 early return in
    // `reduce_dim_f32`.
    let rows = outer_size;

    // Contiguous Sum: batch all rows in one kernel call to avoid per-row overhead.
    #[cfg(feature = "simd")]
    if matches!(op, ReduceOp::Sum) && outer_stride == dim_size as isize {
        let mut result = vec![0.0f32; rows];
        kernels::sum_rows_f32(&data[start_offset..], &mut result, rows, dim_size);
        return result;
    }

    // Fallback: non-contiguous strides or Prod.
    let mut result = Vec::with_capacity(rows);
    for outer in 0..rows {
        let row_start = (start_offset as isize + outer as isize * outer_stride) as usize;
        let row = &data[row_start..row_start + dim_size];

        let val = match op {
            ReduceOp::Sum => {
                #[cfg(feature = "simd")]
                {
                    kernels::sum_f32(row)
                }
                #[cfg(not(feature = "simd"))]
                {
                    row.iter().copied().sum()
                }
            }
            ReduceOp::Prod => row.iter().copied().product(),
        };
        result.push(val);
    }
    result
}

/// Generic dimension reduction implementation.
pub(super) fn reduce_dim_impl<E, F>(tensor: &HostTensor, dim: usize, init: E, reduce_fn: F) -> HostTensor
where
    E: Element + bytemuck::Pod + Copy,
    F: Fn(E, E) -> E,
{
    let ndims = tensor.layout().shape().num_dims();
    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Copy to contiguous only when the flattened stride assumption breaks:
    // non-contiguous tensor with 2+ outer dims or 2+ inner dims.
    let outer_dims = dim;
    let inner_dims = ndims - dim - 1;
    let needs_copy = !tensor.is_contiguous() && (outer_dims > 1 || inner_dims > 1);
    let tensor = if needs_copy {
        tensor.to_contiguous()
    } else {
        tensor.clone()
    };
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: any zero-sized non-reduced dim means the result has no
    // elements. Returning early keeps the loops below from producing phantom
    // outputs when `outer_size == 0` or `inner_size == 0`.
    if out_size == 0 {
        return HostTensor::new(
            Bytes::from_elems(Vec::<E>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            tensor.dtype(),
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    if tensor.is_contiguous() {
        // Contiguous: use flat index arithmetic (safe for any ndims).
        // outer_size and inner_size are positive by the early return above.
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
    } else {
        // Non-contiguous with at most 1 outer + 1 inner dim
        let dim_stride = strides[dim];
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride;
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = (base + d as isize * dim_stride) as usize;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
    }

    let bytes = Bytes::from_elems(result);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        tensor.dtype(),
    )
}

/// Widening dimension reduction for small integer types: accumulate in i64 to avoid overflow.
pub(super) fn reduce_dim_widening<E, F>(tensor: &HostTensor, dim: usize, init: i64, reduce_fn: F) -> HostTensor
where
    E: Element + bytemuck::Pod,
    i64: From<E>,
    F: Fn(i64, i64) -> i64,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: skip the loop entirely for zero-sized non-reduced dims.
    if out_size == 0 {
        return HostTensor::new(
            Bytes::from_elems(Vec::<E>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            tensor.dtype(),
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                acc = reduce_fn(acc, i64::from(data[idx]));
            }
            // Truncate back to target type (wrapping, matches PyTorch)
            let val: E = truncate_i64_to_pod(acc);
            result.push(val);
        }
    }

    let bytes = Bytes::from_elems(result);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        tensor.dtype(),
    )
}

/// Half-precision dimension reduction with f32 accumulation.
///
/// Works for both f16 and bf16 via the `to_f32`/`from_f32` closures.
pub(super) fn reduce_dim_half<E, F>(
    tensor: &HostTensor,
    dim: usize,
    init: f32,
    reduce_fn: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> HostTensor
where
    E: Element + bytemuck::Pod,
    F: Fn(f32, f32) -> f32,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: skip the loop entirely for zero-sized non-reduced dims.
    if out_size == 0 {
        return HostTensor::new(
            Bytes::from_elems(Vec::<E>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            E::dtype(),
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                acc = reduce_fn(acc, to_f32(data[idx]));
            }
            result.push(from_f32(acc));
        }
    }

    let bytes = Bytes::from_elems(result);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Sum along `dim` for an already-contiguous row-major f32 slice, producing
/// one output per (outer, inner) position.
///
/// The caller owns the contiguity guarantee: `data.len()` must equal
/// `outer_size * dim_size * inner_size` in logical row-major order. Dispatches
/// to the same SIMD kernels `reduce_dim_f32` uses, but without the stride
/// bookkeeping.
pub(super) fn sum_dim_contiguous_f32(
    data: &[f32],
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) -> Vec<f32> {
    // Empty output: if any non-reduced dim has size 0, the result is empty.
    // Returning early avoids indexing past an empty `data` slice in the SIMD
    // kernels and keeps the output length in sync with the caller's out_shape.
    if outer_size == 0 || inner_size == 0 {
        return Vec::new();
    }

    // Last-dim: each output is the sum of a contiguous run of dim_size elements.
    if inner_size == 1 {
        let rows = outer_size;
        #[cfg(feature = "simd")]
        {
            let mut result = vec![0.0f32; rows];
            kernels::sum_rows_f32(data, &mut result, rows, dim_size);
            return result;
        }
        #[cfg(not(feature = "simd"))]
        {
            return (0..rows)
                .map(|i| data[i * dim_size..(i + 1) * dim_size].iter().sum())
                .collect();
        }
    }

    // First-dim (or equivalent: any collapsed-outer case): scatter-add dim_size
    // rows of inner_size cols into a single inner_size accumulator.
    if outer_size == 1 {
        #[cfg(feature = "simd")]
        {
            let mut result = aligned::alloc_aligned_zeroed::<f32>(inner_size);
            kernels::scatter_add_f32(data, &mut result, dim_size, inner_size, inner_size);
            return aligned::to_vec(result);
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut result = vec![0.0f32; inner_size];
            for row in 0..dim_size {
                let row_start = row * inner_size;
                for c in 0..inner_size {
                    result[c] += data[row_start + c];
                }
            }
            return result;
        }
    }

    // Middle-dim: batched scatter-add. For each outer batch, sum dim_size rows
    // of inner_size cols into a per-batch accumulator.
    let out_size = outer_size * inner_size;
    #[cfg(feature = "simd")]
    {
        let mut result = aligned::alloc_aligned_zeroed::<f32>(out_size);
        kernels::scatter_add_batched(
            data,
            &mut result,
            outer_size,
            dim_size,
            inner_size,
            dim_size * inner_size,
            inner_size,
        );
        aligned::to_vec(result)
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; out_size];
        for outer in 0..outer_size {
            let out_base = outer * inner_size;
            for d in 0..dim_size {
                let in_base = outer * dim_size * inner_size + d * inner_size;
                for c in 0..inner_size {
                    result[out_base + c] += data[in_base + c];
                }
            }
        }
        result
    }
}

/// Mean along a dimension for half-precision types, fusing sum and divide in f32.
///
/// A naive `sum_dim` + `scalar_div` implementation can overflow to +inf when the
/// intermediate sum exceeds `f16::MAX` (65504), even if the final mean fits. This
/// function keeps the entire reduction and division in f32 and only narrows to
/// f16/bf16 on store via `E::from_elem`.
pub(super) fn mean_dim_half<E>(tensor: &HostTensor, dim: usize) -> HostTensor
where
    E: Element + bytemuck::Pod,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape[dim];
    assert!(
        dim_size > 0,
        "mean_dim: cannot take mean of empty dimension"
    );
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data = float_storage_as_f32(&tensor);
    let divisor = dim_size as f32;

    let sums = sum_dim_contiguous_f32(&data, outer_size, dim_size, inner_size);
    let result: Vec<E> = sums
        .into_iter()
        .map(|s| E::from_elem(s / divisor))
        .collect();

    let bytes = Bytes::from_elems(result);
    HostTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Scalar mean for half-precision types, fusing sum and divide in f32.
/// Avoids f16 overflow when the total sum exceeds `f16::MAX`. Empty input
/// produces NaN to match the f32/f64 path in `mean()`.
pub(super) fn mean_scalar_half<E>(tensor: &HostTensor) -> HostTensor
where
    E: Element + bytemuck::Pod,
{
    let tensor = tensor.to_contiguous();
    let n = tensor.layout().num_elements();
    let data = float_storage_as_f32(&tensor);
    // Route through `sum_f32_contiguous` to pick up SIMD + rayon for the
    // f32 reduction. The half-precision narrowing happens after the divide.
    let acc = sum_f32_contiguous(&data);

    let mean = acc / (n as f32);
    let bytes = Bytes::from_elems(vec![E::from_elem(mean)]);
    HostTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), E::dtype())
}

