use super::*;

// ============================================================================
// f32 in-place unary ops (SIMD-accelerated)
// ============================================================================

#[inline]
pub fn abs_inplace_f32(a: &mut [f32]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        abs_inplace_f32_par(a);
        return;
    }

    abs_inplace_f32_seq(a);
}

#[macerator::with_simd]
fn abs_inplace_f32_seq<S: Simd>(a: &mut [f32]) {
    let lanes = S::lanes32();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, _>(a.as_ptr().add(i));
            vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), v.abs());
        }
        i += lanes;
    }

    for v in &mut a[simd_len..len] {
        *v = v.abs();
    }
}

#[cfg(feature = "rayon")]
fn abs_inplace_f32_par(a: &mut [f32]) {
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        abs_inplace_f32_seq(chunk);
    });
}

#[inline]
pub fn recip_inplace_f32(a: &mut [f32]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        recip_inplace_f32_par(a);
        return;
    }

    recip_inplace_f32_seq(a);
}

#[macerator::with_simd]
fn recip_inplace_f32_seq<S: Simd>(a: &mut [f32]) {
    let lanes = S::lanes32();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    // Use exact SIMD division (not VRecip which is approximate on NEON/SSE)
    let ones = 1.0f32.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, _>(a.as_ptr().add(i));
            vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), ones / v);
        }
        i += lanes;
    }

    for v in &mut a[simd_len..len] {
        *v = 1.0 / *v;
    }
}

#[cfg(feature = "rayon")]
fn recip_inplace_f32_par(a: &mut [f32]) {
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        recip_inplace_f32_seq(chunk);
    });
}

