use super::*;

// ============================================================================
// f32 in-place binary ops
// ============================================================================

macro_rules! define_inplace_f32_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident, $op:tt) => {
        #[inline]
        pub fn $pub_fn(a: &mut [f32], b: &[f32]) {
            debug_assert_eq!(a.len(), b.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $par_fn(a, b);
                return;
            }

            $seq_fn(a, b);
        }

        #[macerator::with_simd]
        #[allow(clippy::assign_op_pattern)]
        fn $seq_fn<S: Simd>(a: &mut [f32], b: &[f32]) {
            let lanes = S::lanes32();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned(a.as_ptr().add(i));
                    let vb = vload_unaligned(b.as_ptr().add(i));
                    vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), va $op vb);
                }
                i += lanes;
            }

            for j in simd_len..len {
                a[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(a: &mut [f32], b: &[f32]) {
            a.par_chunks_mut(CHUNK_SIZE)
                .zip(b.par_chunks(CHUNK_SIZE))
                .for_each(|(a_chunk, b_chunk)| {
                    $seq_fn(a_chunk, b_chunk);
                });
        }
    };
}

define_inplace_f32_op!(add_inplace_f32, add_inplace_f32_seq, add_inplace_f32_par, +);
define_inplace_f32_op!(sub_inplace_f32, sub_inplace_f32_seq, sub_inplace_f32_par, -);
define_inplace_f32_op!(mul_inplace_f32, mul_inplace_f32_seq, mul_inplace_f32_par, *);
define_inplace_f32_op!(div_inplace_f32, div_inplace_f32_seq, div_inplace_f32_par, /);

// ============================================================================
// f32 shared-row broadcast binary ops
// ============================================================================
//
// Pattern: `dst[r * row_len + j] = dst[r * row_len + j] op row[j]` for
// all `r in 0..num_rows, j in 0..row_len`. That is, a single contiguous
// `row` is broadcast and combined with each of `num_rows` consecutive
// rows in `dst`. This is what `gamma.unsqueeze() * x` and friends
// produce after broadcast expansion.
//
// We call the SIMD dispatch once for the entire walk so that
// `#[macerator::with_simd]`'s feature detection pays once, not once per
// row. Calling the normal `add_inplace_f32` in a `0..num_rows` loop
// gives the right answer but costs ~55k dispatches on the typical
// layer_norm shape; see issue #64 item 2 benchmarks.

macro_rules! define_shared_row_f32_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident, $op:tt) => {
        #[inline]
        pub fn $pub_fn(dst: &mut [f32], row: &[f32]) {
            let row_len = row.len();
            if row_len == 0 || dst.is_empty() {
                return;
            }
            debug_assert_eq!(
                dst.len() % row_len,
                0,
                "shared-row broadcast: dst length must be a multiple of row length"
            );

            #[cfg(feature = "rayon")]
            if dst.len() >= PARALLEL_THRESHOLD {
                $par_fn(dst, row);
                return;
            }

            $seq_fn(dst, row);
        }

        #[macerator::with_simd]
        #[allow(clippy::assign_op_pattern)]
        fn $seq_fn<S: Simd>(dst: &mut [f32], row: &[f32]) {
            let lanes = S::lanes32();
            let row_len = row.len();
            let simd_len = row_len / lanes * lanes;
            let num_rows = dst.len() / row_len;

            for r in 0..num_rows {
                let base = r * row_len;
                let mut i = 0;
                while i < simd_len {
                    unsafe {
                        let va = vload_unaligned::<S, _>(dst.as_ptr().add(base + i));
                        let vb = vload_unaligned::<S, _>(row.as_ptr().add(i));
                        vstore_unaligned::<S, _>(
                            dst.as_mut_ptr().add(base + i),
                            va $op vb,
                        );
                    }
                    i += lanes;
                }
                for j in simd_len..row_len {
                    dst[base + j] = dst[base + j] $op row[j];
                }
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(dst: &mut [f32], row: &[f32]) {
            let row_len = row.len();
            // Chunk on row boundaries so each worker sees whole rows.
            let rows_per_chunk = (CHUNK_SIZE / row_len).max(1);
            let chunk_elems = rows_per_chunk * row_len;
            dst.par_chunks_mut(chunk_elems).for_each(|chunk| {
                $seq_fn(chunk, row);
            });
        }
    };
}

define_shared_row_f32_op!(
    add_shared_row_inplace_f32,
    add_shared_row_inplace_f32_seq,
    add_shared_row_inplace_f32_par,
    +
);
define_shared_row_f32_op!(
    sub_shared_row_inplace_f32,
    sub_shared_row_inplace_f32_seq,
    sub_shared_row_inplace_f32_par,
    -
);
define_shared_row_f32_op!(
    mul_shared_row_inplace_f32,
    mul_shared_row_inplace_f32_seq,
    mul_shared_row_inplace_f32_par,
    *
);
define_shared_row_f32_op!(
    div_shared_row_inplace_f32,
    div_shared_row_inplace_f32_seq,
    div_shared_row_inplace_f32_par,
    /
);

