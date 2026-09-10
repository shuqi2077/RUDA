use super::*;

// ============================================================================
// f32 comparison ops
// ============================================================================

#[derive(Clone, Copy)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

#[inline]
pub fn cmp_f32(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_f32_par(a, b, out, op);
        return;
    }

    cmp_f32_seq(a, b, out, op);
}

/// Comparison kernel using simple loops that LLVM autovectorizes.
///
/// Autovectorization outperforms explicit SIMD here because comparisons
/// produce u8 output from f32 input (4:1 size ratio). LLVM batches 16+
/// comparisons and packs results into a single wide vector store, while
/// explicit SIMD (store_as_bool) writes only `lanes` bytes per iteration.
#[inline]
fn cmp_f32_seq(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    match op {
        CmpOp::Gt => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a > *b) as u8;
            }
        }
        CmpOp::Ge => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a >= *b) as u8;
            }
        }
        CmpOp::Lt => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a < *b) as u8;
            }
        }
        CmpOp::Le => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a <= *b) as u8;
            }
        }
        CmpOp::Eq => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a == *b) as u8;
            }
        }
        CmpOp::Ne => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a != *b) as u8;
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn cmp_f32_par(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_f32_seq(&a[start..end], &b[start..end], out_chunk, op);
        });
}

#[inline]
pub fn cmp_scalar_f32(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_scalar_f32_par(a, scalar, out, op);
        return;
    }

    cmp_scalar_f32_seq(a, scalar, out, op);
}

/// Scalar comparison kernel using simple loops that LLVM autovectorizes.
/// See `cmp_f32_seq` for rationale.
#[inline]
fn cmp_scalar_f32_seq(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    match op {
        CmpOp::Gt => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a > scalar) as u8;
            }
        }
        CmpOp::Ge => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a >= scalar) as u8;
            }
        }
        CmpOp::Lt => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a < scalar) as u8;
            }
        }
        CmpOp::Le => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a <= scalar) as u8;
            }
        }
        CmpOp::Eq => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a == scalar) as u8;
            }
        }
        CmpOp::Ne => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a != scalar) as u8;
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn cmp_scalar_f32_par(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_scalar_f32_seq(&a[start..end], scalar, out_chunk, op);
        });
}

