use super::*;

/// Specialized binary operation for f32 with SIMD fast path.
#[cfg(feature = "simd")]
pub(super) fn binary_op_f32<Op>(
    mut lhs: HostTensor,
    rhs: &HostTensor,
    op: Op,
    simd_hint: Option<BinaryOp>,
) -> HostTensor
where
    Op: Fn(f32, f32) -> f32,
{
    // Permuted lhs + broadcast rhs (e.g. `x.permute(...) - mean`):
    // the generic path would walk the permuted lhs with a scalar
    // `StridedIter`, an order of magnitude slower than the SIMD fast
    // path. Pay one memcpy to materialize lhs contiguous so the fast
    // paths below can take over.
    //
    // Gate on `simd_hint.is_some()` so custom ops like `atan2` or
    // `powf` (which have no SIMD fast path and go straight to
    // `binary_op_typed`) don't pay for a memcpy they can't benefit
    // from. Their strided fallback handles non-contig lhs directly.
    if simd_hint.is_some() && !lhs.layout().is_contiguous() && rhs.layout().strides().contains(&0) {
        lhs = lhs.to_contiguous();
    }

    // In-place SIMD fast path: lhs unique contiguous at offset 0, rhs
    // contiguous (no broadcast).
    if let Some(simd_op) = simd_hint
        && lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let r_slice: &[f32] = &rhs.storage()[r_start..r_end];
        let lhs_storage: &mut [f32] = lhs.storage_mut();
        let l_slice = &mut lhs_storage[..l_end];

        match simd_op {
            BinaryOp::Add => simd::add_inplace_f32(l_slice, r_slice),
            BinaryOp::Sub => simd::sub_inplace_f32(l_slice, r_slice),
            BinaryOp::Mul => simd::mul_inplace_f32(l_slice, r_slice),
            BinaryOp::Div => simd::div_inplace_f32(l_slice, r_slice),
        }
        return lhs;
    }

    // Broadcast SIMD fast path: rhs is broadcast via stride-0 dims in
    // one of two hot shapes that dominate layer_norm decomposition --
    // shared-row (`gamma.unsqueeze() * x`) or per-row scalar
    // (`x - x.mean_dim(-1)`).
    if let Some(simd_op) = simd_hint
        && let Some(pattern) = detect_broadcast_pattern(lhs.layout(), rhs)
    {
        return apply_broadcast_pattern_f32(lhs, rhs, simd_op, pattern);
    }

    binary_op_typed(lhs, rhs, op)
}

/// Categorization of the two broadcast patterns we can accelerate.
///
/// Consumers assume `lhs` and `rhs` already share the same logical
/// shape and that `lhs` is row-contiguous at offset 0.
#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy)]
pub(super) enum BroadcastView {
    /// rhs's inner `row_len` elements form a contiguous row that is
    /// shared across `outer_count` outer positions. Starts at
    /// `rhs_row_offset` in rhs's storage.
    SharedRow {
        outer_count: usize,
        row_len: usize,
        rhs_row_offset: usize,
    },
    /// rhs's inner `row_len` elements are all the same scalar,
    /// stepping through `outer_count` scalars along the outer dims
    /// starting at `rhs_scalar_base` in rhs's storage.
    PerRowScalar {
        outer_count: usize,
        row_len: usize,
        rhs_scalar_base: usize,
    },
}

/// Detect whether rhs can be handled as one of the accelerated
/// broadcast patterns, returning `None` if the stride pattern doesn't
/// fit either bucket or the resulting offsets would leave rhs's
/// storage.
#[cfg(feature = "simd")]
pub(super) fn detect_broadcast_pattern(lhs: &Layout, rhs: &HostTensor) -> Option<BroadcastView> {
    let rhs_layout = rhs.layout();
    let rhs_storage_elems = rhs.storage::<f32>().len();
    // Require lhs to be row-contiguous at offset 0. The broadcast kernel
    // below uses linear offsets into lhs's storage; relaxing this would
    // complicate the indexing without helping the hot layer_norm path.
    let (l_start, _) = lhs.contiguous_offsets()?;
    if l_start != 0 {
        return None;
    }
    let ndims = lhs.num_dims();
    if ndims == 0 || rhs_layout.num_dims() != ndims {
        return None;
    }
    let lhs_shape = lhs.shape();
    let rhs_strides = rhs_layout.strides();

    let last_stride = rhs_strides[ndims - 1];

    // Case A: shared row. Innermost rhs stride is 1, and every outer
    // dim either has stride 0 (a broadcast dim) or size 1 (stride
    // doesn't matter since the dim never advances past index 0).
    if last_stride == 1 {
        let outer_ok = (0..ndims - 1).all(|d| rhs_strides[d] == 0 || lhs_shape[d] == 1);
        if outer_ok {
            let outer_count: usize = (0..ndims - 1).map(|d| lhs_shape[d]).product();
            let row_len = lhs_shape[ndims - 1];
            if outer_count == 0 || row_len == 0 {
                return None;
            }
            let rhs_row_offset = rhs_layout.start_offset();
            // Bounds: kernel reads `rhs_storage[off..off+row_len]`.
            if rhs_row_offset.checked_add(row_len)? > rhs_storage_elems {
                return None;
            }
            return Some(BroadcastView::SharedRow {
                outer_count,
                row_len,
                rhs_row_offset,
            });
        }
    }

    // Case B: per-row scalar. Innermost dims all have stride 0 in
    // rhs and outer dims walk rhs contiguously in row-major order.
    if last_stride == 0 {
        // Count the trailing stride-0 dims to find the inner scalar span.
        let mut inner_dims = 0usize;
        let mut row_len: usize = 1;
        for d in (0..ndims).rev() {
            if rhs_strides[d] == 0 {
                inner_dims += 1;
                row_len *= lhs_shape[d];
            } else {
                break;
            }
        }
        if inner_dims == 0 {
            return None;
        }
        // The outer dims must walk rhs's storage contiguously in
        // row-major order.
        let outer_ndims = ndims - inner_dims;
        let mut expected: isize = 1;
        for d in (0..outer_ndims).rev() {
            if rhs_strides[d] != expected {
                return None;
            }
            expected *= lhs_shape[d] as isize;
        }
        let outer_count: usize = (0..outer_ndims).map(|d| lhs_shape[d]).product();
        if outer_count == 0 || row_len == 0 {
            return None;
        }
        let rhs_scalar_base = rhs_layout.start_offset();
        // Bounds: kernel reads `rhs_storage[base..base+outer_count]`.
        if rhs_scalar_base.checked_add(outer_count)? > rhs_storage_elems {
            return None;
        }
        return Some(BroadcastView::PerRowScalar {
            outer_count,
            row_len,
            rhs_scalar_base,
        });
    }

    None
}

/// Execute a detected broadcast pattern for f32. Writes in-place into
/// lhs when unique; otherwise allocates a fresh contiguous output.
#[cfg(feature = "simd")]
pub(super) fn apply_broadcast_pattern_f32(
    mut lhs: HostTensor,
    rhs: &HostTensor,
    simd_op: BinaryOp,
    pattern: BroadcastView,
) -> HostTensor {
    let numel = lhs.layout().num_elements();
    let rhs_storage = rhs.storage::<f32>();

    if lhs.is_unique() {
        let dst = &mut lhs.storage_mut::<f32>()[..numel];
        run_broadcast_pattern_f32(dst, rhs_storage, simd_op, pattern);
        lhs
    } else {
        // Copy lhs once, then apply the broadcast in place. The
        // memcpy is cheaper than the StridedIter fallback it replaces.
        let mut out: Vec<f32> = lhs.storage::<f32>()[..numel].to_vec();
        run_broadcast_pattern_f32(&mut out, rhs_storage, simd_op, pattern);
        make_tensor(out, lhs.layout().shape().clone(), lhs.dtype())
    }
}

/// Shared kernel: run the chosen broadcast pattern against a mutable
/// destination buffer (which already holds lhs's values) and rhs's
/// storage slice.
#[cfg(feature = "simd")]
pub(super) fn run_broadcast_pattern_f32(
    dst: &mut [f32],
    rhs_storage: &[f32],
    simd_op: BinaryOp,
    pattern: BroadcastView,
) {
    match pattern {
        BroadcastView::SharedRow {
            outer_count,
            row_len,
            rhs_row_offset,
        } => {
            let rhs_row: &[f32] = &rhs_storage[rhs_row_offset..rhs_row_offset + row_len];
            let total = outer_count * row_len;
            // One SIMD dispatch covers the whole outer walk. The kernel
            // keeps `rhs_row` in registers across rows for small row
            // lengths, and pays the macerator feature-detection cost
            // exactly once.
            let dst_full = &mut dst[..total];
            match simd_op {
                BinaryOp::Add => simd::add_shared_row_inplace_f32(dst_full, rhs_row),
                BinaryOp::Sub => simd::sub_shared_row_inplace_f32(dst_full, rhs_row),
                BinaryOp::Mul => simd::mul_shared_row_inplace_f32(dst_full, rhs_row),
                BinaryOp::Div => simd::div_shared_row_inplace_f32(dst_full, rhs_row),
            }
        }
        BroadcastView::PerRowScalar {
            outer_count,
            row_len,
            rhs_scalar_base,
        } => {
            let scalars = &rhs_storage[rhs_scalar_base..rhs_scalar_base + outer_count];
            // One monomorphized helper per op. The closure is statically
            // known at each call site so LLVM still autovectorizes the
            // inner scalar loop, and the outer op dispatch happens once.
            match simd_op {
                BinaryOp::Add => per_row_scalar_apply(dst, scalars, row_len, |a, b| a + b),
                BinaryOp::Sub => per_row_scalar_apply(dst, scalars, row_len, |a, b| a - b),
                BinaryOp::Mul => per_row_scalar_apply(dst, scalars, row_len, |a, b| a * b),
                BinaryOp::Div => per_row_scalar_apply(dst, scalars, row_len, |a, b| a / b),
            }
        }
    }
}

/// Apply `dst[r * row_len + j] = op(dst[r * row_len + j], scalars[r])`
/// for `r in 0..scalars.len(), j in 0..row_len`. Generic over `Op` so
/// each call site gets a monomorphized, autovectorizable inner loop.
#[cfg(feature = "simd")]
#[inline]
pub(super) fn per_row_scalar_apply<Op>(dst: &mut [f32], scalars: &[f32], row_len: usize, op: Op)
where
    Op: Fn(f32, f32) -> f32,
{
    for (i, &scalar) in scalars.iter().enumerate() {
        let start = i * row_len;
        for x in dst[start..start + row_len].iter_mut() {
            *x = op(*x, scalar);
        }
    }
}

