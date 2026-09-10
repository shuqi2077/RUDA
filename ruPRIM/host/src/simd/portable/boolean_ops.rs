use super::*;

// ============================================================================
// u8 boolean ops
// ============================================================================

macro_rules! define_bool_binary_u8_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident,
     $inplace_pub:ident, $inplace_seq:ident, $inplace_par:ident,
     $trait:ident, $method:ident, $op:tt) => {
        #[inline]
        pub fn $pub_fn(a: &[u8], b: &[u8], out: &mut [u8]) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $par_fn(a, b, out);
                return;
            }

            $seq_fn(a, b, out);
        }

        #[macerator::with_simd]
        fn $seq_fn<S: Simd>(a: &[u8], b: &[u8], out: &mut [u8]) {
            let lanes = S::lanes8();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
                    let vb = vload_unaligned::<S, u8>(b.as_ptr().add(i));
                    vstore_unaligned::<S, u8>(
                        out.as_mut_ptr().add(i),
                        <u8 as $trait>::$method::<S>(va, vb),
                    );
                }
                i += lanes;
            }

            for j in simd_len..len {
                out[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(a: &[u8], b: &[u8], out: &mut [u8]) {
            out.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(a.len());
                    $seq_fn(&a[start..end], &b[start..end], out_chunk);
                });
        }

        #[inline]
        pub fn $inplace_pub(a: &mut [u8], b: &[u8]) {
            debug_assert_eq!(a.len(), b.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $inplace_par(a, b);
                return;
            }

            $inplace_seq(a, b);
        }

        #[allow(clippy::assign_op_pattern)]
        #[macerator::with_simd]
        fn $inplace_seq<S: Simd>(a: &mut [u8], b: &[u8]) {
            let lanes = S::lanes8();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
                    let vb = vload_unaligned::<S, u8>(b.as_ptr().add(i));
                    vstore_unaligned::<S, u8>(
                        a.as_mut_ptr().add(i),
                        <u8 as $trait>::$method::<S>(va, vb),
                    );
                }
                i += lanes;
            }

            for j in simd_len..len {
                a[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $inplace_par(a: &mut [u8], b: &[u8]) {
            a.par_chunks_mut(CHUNK_SIZE)
                .zip(b.par_chunks(CHUNK_SIZE))
                .for_each(|(a_chunk, b_chunk)| {
                    $inplace_seq(a_chunk, b_chunk);
                });
        }
    };
}

define_bool_binary_u8_op!(
    bool_and_u8, bool_and_u8_seq, bool_and_u8_par,
    bool_and_inplace_u8, bool_and_inplace_u8_seq, bool_and_inplace_u8_par,
    VBitAnd, vbitand, &);
define_bool_binary_u8_op!(
    bool_or_u8, bool_or_u8_seq, bool_or_u8_par,
    bool_or_inplace_u8, bool_or_inplace_u8_seq, bool_or_inplace_u8_par,
    VBitOr, vbitor, |);
define_bool_binary_u8_op!(
    bool_xor_u8, bool_xor_u8_seq, bool_xor_u8_par,
    bool_xor_inplace_u8, bool_xor_inplace_u8_seq, bool_xor_inplace_u8_par,
    VBitXor, vbitxor, ^);

// Boolean NOT is special (unary), implemented separately.

#[inline]
pub fn bool_not_u8(a: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_u8_par(a, out);
        return;
    }

    bool_not_u8_seq(a, out);
}

#[macerator::with_simd]
fn bool_not_u8_seq<S: Simd>(a: &[u8], out: &mut [u8]) {
    let lanes = S::lanes8();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
            let mask = va.eq(zeros);
            mask.store_as_bool(out.as_mut_ptr().add(i) as *mut bool);
        }
        i += lanes;
    }

    for j in simd_len..len {
        out[j] = (a[j] == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_u8_par(a: &[u8], out: &mut [u8]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            bool_not_u8_seq(&a[start..end], out_chunk);
        });
}

#[inline]
pub fn bool_not_inplace_u8(a: &mut [u8]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_inplace_u8_par(a);
        return;
    }

    bool_not_inplace_u8_seq(a);
}

#[macerator::with_simd]
fn bool_not_inplace_u8_seq<S: Simd>(a: &mut [u8]) {
    let lanes = S::lanes8();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
            let mask = va.eq(zeros);
            mask.store_as_bool(a.as_mut_ptr().add(i) as *mut bool);
        }
        i += lanes;
    }

    for v in &mut a[simd_len..len] {
        *v = (*v == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_inplace_u8_par(a: &mut [u8]) {
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        bool_not_inplace_u8_seq(chunk);
    });
}

