use super::*;

// ============================================================================
// Extremum helpers (SIMD fast paths + generic scalar, parallelized with rayon)
// ============================================================================

// Lower threshold than the global PARALLEL_THRESHOLD (256K) because the per-element
// work (a single comparison + conditional store) is cheap enough that rayon overhead
// is amortized at smaller sizes. 32K elements * ~1.5ns/elem = ~48µs of serial work,
// enough to justify thread-pool dispatch.
#[cfg(feature = "rayon")]
pub(super) const EXTREMUM_PARALLEL_THRESHOLD: usize = 32 * 1024;

/// Minimum row length for the 2-pass SIMD extremum path (reduce + scan).
/// Below this, single-pass scalar is faster since it reads each element once.
#[cfg(feature = "simd")]
pub(super) const EXTREMUM_SIMD_ROW_THRESHOLD: usize = 512;

/// Scalar single-pass f32 last-dim extremum (values only).
/// Avoids the generic closure path by operating directly on contiguous f32 rows.
pub(super) fn extremum_f32_last_scalar<F>(tensor: &HostTensor, dim: usize, is_better: F) -> HostTensor
where
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let reduce_row = |outer: usize| -> f32 {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        // Single left-to-right scan: first NaN wins, otherwise track the
        // extremum. Starting `best = row[0]` means the i=0 `is_better`
        // branch is a no-op for strict comparisons.
        let mut best = row[0];
        for &v in row {
            if v.is_nan() {
                return f32::NAN;
            }
            if is_better(v, best) {
                best = v;
            }
        }
        best
    };

    #[cfg(feature = "rayon")]
    let values: Vec<f32> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(&reduce_row).collect()
    } else {
        (0..outer_size).map(reduce_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<f32> = (0..outer_size).map(reduce_row).collect();

    HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::F32,
    )
}

/// Scalar single-pass f32 last-dim argmax/argmin (indices only).
/// Uses direct pointer access and simple comparisons instead of the generic closure path.
pub(super) fn extremum_indices_f32_last_scalar<F>(tensor: &HostTensor, dim: usize, is_better: F) -> HostTensor
where
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> isize {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        // Single left-to-right scan: first NaN wins, otherwise track the
        // extremum. Starting `best = row[0]` means the i=0 `is_better`
        // branch is a no-op (strict comparisons are false on equal
        // operands), so we don't need a separate row[0] check.
        let mut best = row[0];
        let mut best_idx: isize = 0;
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() {
                return i as isize;
            }
            if is_better(v, best) {
                best = v;
                best_idx = i as isize;
            }
        }
        best_idx
    };

    #[cfg(feature = "rayon")]
    let indices: Vec<isize> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(find_row).collect()
    } else {
        (0..outer_size).map(find_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let indices: Vec<isize> = (0..outer_size).map(find_row).collect();

    HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    )
}

/// Scalar single-pass f32 last-dim extremum with indices (values + indices).
pub(super) fn extremum_with_indices_f32_last_scalar<F>(
    tensor: &HostTensor,
    dim: usize,
    is_better: F,
) -> (HostTensor, HostTensor)
where
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> (f32, isize) {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let mut best = row[0];
        let mut best_idx: isize = 0;
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() {
                return (f32::NAN, i as isize);
            }
            if is_better(v, best) {
                best = v;
                best_idx = i as isize;
            }
        }
        (best, best_idx)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<f32>, Vec<isize>) =
        if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..outer_size).into_par_iter().map(find_row).unzip()
        } else {
            (0..outer_size).map(find_row).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<f32>, Vec<isize>) = (0..outer_size).map(find_row).unzip();

    let val_tensor = HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        DType::F32,
    );
    let idx_tensor = HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

/// Uses macerator SIMD reduction per contiguous row, with NaN propagation.
#[cfg(feature = "simd")]
pub(super) fn extremum_dim_f32_last_simd(
    tensor: &HostTensor,
    dim: usize,
    simd_reduce: fn(&[f32]) -> f32,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let reduce_row = |outer: usize| -> f32 {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let ext = simd_reduce(row);
        // SIMD max/min may silently drop NaN (architecture-dependent).
        // If the result is already NaN, we're done. Otherwise, scan to
        // check for any NaN the SIMD op missed.
        if ext.is_nan() {
            return f32::NAN;
        }
        for &v in row {
            if v.is_nan() {
                return f32::NAN;
            }
        }
        ext
    };

    #[cfg(feature = "rayon")]
    let values: Vec<f32> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(reduce_row).collect()
    } else {
        (0..outer_size).map(reduce_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<f32> = (0..outer_size).map(reduce_row).collect();

    HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::F32,
    )
}

/// SIMD fast path for f32 last-dim extremum with indices.
/// Per row: SIMD reduction finds the extremum value, then a linear scan
/// locates the first NaN or first matching index.
#[cfg(feature = "simd")]
pub(super) fn extremum_dim_with_indices_f32_last_simd(
    tensor: &HostTensor,
    dim: usize,
    simd_reduce: fn(&[f32]) -> f32,
) -> (HostTensor, HostTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> (f32, isize) {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let ext = simd_reduce(row);
        // Single scan: return first NaN (with NaN value) or first match of ext.
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() {
                return (f32::NAN, i as isize);
            }
            if v == ext {
                return (ext, i as isize);
            }
        }
        (ext, 0)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<f32>, Vec<isize>) =
        if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..outer_size).into_par_iter().map(find_row).unzip()
        } else {
            (0..outer_size).map(find_row).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<f32>, Vec<isize>) = (0..outer_size).map(find_row).unzip();

    let val_tensor = HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        DType::F32,
    );
    let idx_tensor = HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

/// SIMD fast path for f32 last-dim argmax/argmin (indices only, no values allocation).
#[cfg(feature = "simd")]
pub(super) fn extremum_indices_f32_last_simd(
    tensor: &HostTensor,
    dim: usize,
    simd_reduce: fn(&[f32]) -> f32,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> isize {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let ext = simd_reduce(row);
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() || v == ext {
                return i as isize;
            }
        }
        0
    };

    #[cfg(feature = "rayon")]
    let indices: Vec<isize> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(find_row).collect()
    } else {
        (0..outer_size).map(find_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let indices: Vec<isize> = (0..outer_size).map(find_row).collect();

    HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    )
}

/// Find extremum value along a dimension. `is_better(new, current) -> bool`.
pub(super) fn extremum_dim<E, F>(tensor: &HostTensor, dim: usize, is_better: F) -> HostTensor
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(E, E) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> E {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = data[base];
        for d in 1..dim_size {
            let val = data[base + d * inner_size];
            if is_better(val, best) {
                best = val;
            }
        }
        best
    };

    #[cfg(feature = "rayon")]
    let values: Vec<E> = if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..out_size).into_par_iter().map(&find).collect()
    } else {
        (0..out_size).map(find).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<E> = (0..out_size).map(find).collect();

    HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Find extremum value and its index along a dimension. `is_better(new, current) -> bool`.
pub(super) fn extremum_dim_with_indices<E, F>(
    tensor: &HostTensor,
    dim: usize,
    is_better: F,
) -> (HostTensor, HostTensor)
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(E, E) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> (E, isize) {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = data[base];
        let mut best_idx: isize = 0;
        for d in 1..dim_size {
            let val = data[base + d * inner_size];
            if is_better(val, best) {
                best = val;
                best_idx = d as isize;
            }
        }
        (best, best_idx)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<E>, Vec<isize>) =
        if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..out_size).into_par_iter().map(&find).unzip()
        } else {
            (0..out_size).map(find).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<E>, Vec<isize>) = (0..out_size).map(find).unzip();

    let val_tensor = HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

/// Find extremum value along a dimension for half-precision types (compared via f32).
pub(super) fn extremum_dim_half<E, F>(
    tensor: &HostTensor,
    dim: usize,
    is_better: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> HostTensor
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> E {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = to_f32(data[base]);
        for d in 1..dim_size {
            let val = to_f32(data[base + d * inner_size]);
            if is_better(val, best) {
                best = val;
            }
        }
        from_f32(best)
    };

    #[cfg(feature = "rayon")]
    let values: Vec<E> = if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..out_size).into_par_iter().map(&find).collect()
    } else {
        (0..out_size).map(find).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<E> = (0..out_size).map(find).collect();

    HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Find extremum value and index along a dimension for half-precision types (compared via f32).
pub(super) fn extremum_dim_with_indices_half<E, F>(
    tensor: &HostTensor,
    dim: usize,
    is_better: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> (HostTensor, HostTensor)
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> (E, isize) {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = to_f32(data[base]);
        let mut best_idx: isize = 0;
        for d in 1..dim_size {
            let val = to_f32(data[base + d * inner_size]);
            if is_better(val, best) {
                best = val;
                best_idx = d as isize;
            }
        }
        (from_f32(best), best_idx)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<E>, Vec<isize>) =
        if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..out_size).into_par_iter().map(&find).unzip()
        } else {
            (0..out_size).map(find).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<E>, Vec<isize>) = (0..out_size).map(find).unzip();

    let val_tensor = HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

