use super::*;

// ============================================================================
// Mask select (mask_where / mask_fill) via bitwise blend
// ============================================================================

/// Conditional select: `out[i] = if mask[i] != 0 { value[i] } else { tensor[i] }`
///
/// Operates on f32 data reinterpreted as u32 for bitwise blend.
/// Uses SIMD bitwise ops: `(mask & value) | (!mask & tensor)`.
#[inline]
pub fn mask_where_f32(tensor: &[f32], mask: &[u8], value: &[f32], out: &mut [f32]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f32, u32>(tensor);
    let v = bytemuck::cast_slice::<f32, u32>(value);
    let o = bytemuck::cast_slice_mut::<f32, u32>(out);

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u32_par(t, mask, v, o);
        return;
    }

    mask_blend_u32(t, mask, v, o);
}

/// Conditional select for f64.
#[inline]
pub fn mask_where_f64(tensor: &[f64], mask: &[u8], value: &[f64], out: &mut [f64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f64, u64>(tensor);
    let v = bytemuck::cast_slice::<f64, u64>(value);
    let o = bytemuck::cast_slice_mut::<f64, u64>(out);

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u64_par(t, mask, v, o);
        return;
    }

    mask_blend_u64(t, mask, v, o);
}

/// Conditional select for i64.
#[inline]
pub fn mask_where_i64(tensor: &[i64], mask: &[u8], value: &[i64], out: &mut [i64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<i64, u64>(tensor);
    let v = bytemuck::cast_slice::<i64, u64>(value);
    let o = bytemuck::cast_slice_mut::<i64, u64>(out);

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u64_par(t, mask, v, o);
        return;
    }

    mask_blend_u64(t, mask, v, o);
}

/// Conditional select for u8 (bool tensors).
#[inline]
pub fn mask_where_u8(tensor: &[u8], mask: &[u8], value: &[u8], out: &mut [u8]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u8_par(tensor, mask, value, out);
        return;
    }

    mask_where_u8_seq(tensor, mask, value, out);
}

/// Conditional fill: `out[i] = if mask[i] != 0 { fill_value } else { tensor[i] }`
#[inline]
pub fn mask_fill_f32(tensor: &[f32], mask: &[u8], fill_value: f32, out: &mut [f32]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f32, u32>(tensor);
    let o = bytemuck::cast_slice_mut::<f32, u32>(out);
    let fill_bits = fill_value.to_bits();

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u32_par(t, mask, fill_bits, o);
        return;
    }

    mask_blend_fill_u32(t, mask, fill_bits, o);
}

/// Conditional fill for f64.
#[inline]
pub fn mask_fill_f64(tensor: &[f64], mask: &[u8], fill_value: f64, out: &mut [f64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f64, u64>(tensor);
    let o = bytemuck::cast_slice_mut::<f64, u64>(out);
    let fill_bits = fill_value.to_bits();

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u64_par(t, mask, fill_bits, o);
        return;
    }

    mask_blend_fill_u64(t, mask, fill_bits, o);
}

/// Conditional fill for i64.
#[inline]
pub fn mask_fill_i64(tensor: &[i64], mask: &[u8], fill_value: i64, out: &mut [i64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<i64, u64>(tensor);
    let o = bytemuck::cast_slice_mut::<i64, u64>(out);
    let fill_bits = fill_value as u64;

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u64_par(t, mask, fill_bits, o);
        return;
    }

    mask_blend_fill_u64(t, mask, fill_bits, o);
}

/// Conditional fill for u8 (bool tensors).
#[inline]
pub fn mask_fill_u8(tensor: &[u8], mask: &[u8], fill_value: u8, out: &mut [u8]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u8_par(tensor, mask, fill_value, out);
        return;
    }

    mask_fill_u8_seq(tensor, mask, fill_value, out);
}

// -- Branchless bitwise blend kernels --
//
// These use a tight branchless loop that LLVM autovectorizes with native
// u8->u32/u64 widening instructions (NEON ushll, AVX2 vpmovzxbd).
//
// Mask values must be exactly 0 or 1 (Ruda bool tensor invariant).
// wrapping_sub(0, 0)=0x00, wrapping_sub(0, 1)=0xFF..FF. Other values
// produce partial masks and corrupt output.

#[inline]
fn mask_blend_u32(tensor: &[u32], mask: &[u8], value: &[u32], out: &mut [u32]) {
    for i in 0..tensor.len() {
        let m = 0u32.wrapping_sub(mask[i] as u32);
        out[i] = (value[i] & m) | (tensor[i] & !m);
    }
}

#[inline]
fn mask_blend_fill_u32(tensor: &[u32], mask: &[u8], fill_bits: u32, out: &mut [u32]) {
    for i in 0..tensor.len() {
        let m = 0u32.wrapping_sub(mask[i] as u32);
        out[i] = (fill_bits & m) | (tensor[i] & !m);
    }
}

#[inline]
fn mask_blend_u64(tensor: &[u64], mask: &[u8], value: &[u64], out: &mut [u64]) {
    for i in 0..tensor.len() {
        let m = 0u64.wrapping_sub(mask[i] as u64);
        out[i] = (value[i] & m) | (tensor[i] & !m);
    }
}

#[inline]
fn mask_blend_fill_u64(tensor: &[u64], mask: &[u8], fill_bits: u64, out: &mut [u64]) {
    for i in 0..tensor.len() {
        let m = 0u64.wrapping_sub(mask[i] as u64);
        out[i] = (fill_bits & m) | (tensor[i] & !m);
    }
}

#[cfg(feature = "rayon")]
fn mask_where_u32_par(tensor: &[u32], mask: &[u8], value: &[u32], out: &mut [u32]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_u32(
                &tensor[start..end],
                &mask[start..end],
                &value[start..end],
                out_chunk,
            );
        });
}

#[cfg(feature = "rayon")]
fn mask_fill_u32_par(tensor: &[u32], mask: &[u8], fill_bits: u32, out: &mut [u32]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_fill_u32(&tensor[start..end], &mask[start..end], fill_bits, out_chunk);
        });
}

#[cfg(feature = "rayon")]
fn mask_where_u64_par(tensor: &[u64], mask: &[u8], value: &[u64], out: &mut [u64]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_u64(
                &tensor[start..end],
                &mask[start..end],
                &value[start..end],
                out_chunk,
            );
        });
}

#[cfg(feature = "rayon")]
fn mask_fill_u64_par(tensor: &[u64], mask: &[u8], fill_bits: u64, out: &mut [u64]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_fill_u64(&tensor[start..end], &mask[start..end], fill_bits, out_chunk);
        });
}

// -- u8 SIMD kernels (for bool tensors) --

#[macerator::with_simd]
fn mask_where_u8_seq<S: Simd>(tensor: &[u8], mask: &[u8], value: &[u8], out: &mut [u8]) {
    let lanes = S::lanes8();
    let len = tensor.len();
    let simd_len = len / lanes * lanes;

    // SIMD subtract wraps: 0-0=0x00, 0-1=0xFF
    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let vm_raw = vload_unaligned::<S, u8>(mask.as_ptr().add(i));
            let vm = zeros - vm_raw; // 0->0x00, 1->0xFF
            let vt = vload_unaligned::<S, u8>(tensor.as_ptr().add(i));
            let vv = vload_unaligned::<S, u8>(value.as_ptr().add(i));
            let selected = (vm & vv) | (!vm & vt);
            vstore_unaligned::<S, u8>(out.as_mut_ptr().add(i), selected);
        }
        i += lanes;
    }

    for j in simd_len..len {
        let m = 0u8.wrapping_sub(mask[j]);
        out[j] = (m & value[j]) | (!m & tensor[j]);
    }
}

#[cfg(feature = "rayon")]
fn mask_where_u8_par(tensor: &[u8], mask: &[u8], value: &[u8], out: &mut [u8]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_where_u8_seq(
                &tensor[start..end],
                &mask[start..end],
                &value[start..end],
                out_chunk,
            );
        });
}

#[macerator::with_simd]
fn mask_fill_u8_seq<S: Simd>(tensor: &[u8], mask: &[u8], fill_value: u8, out: &mut [u8]) {
    let lanes = S::lanes8();
    let len = tensor.len();
    let simd_len = len / lanes * lanes;
    let vfill = fill_value.splat::<S>();
    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let vm_raw = vload_unaligned::<S, u8>(mask.as_ptr().add(i));
            let vm = zeros - vm_raw; // 0->0x00, 1->0xFF
            let vt = vload_unaligned::<S, u8>(tensor.as_ptr().add(i));
            let selected = (vm & vfill) | (!vm & vt);
            vstore_unaligned::<S, u8>(out.as_mut_ptr().add(i), selected);
        }
        i += lanes;
    }

    for j in simd_len..len {
        let m = 0u8.wrapping_sub(mask[j]);
        out[j] = (m & fill_value) | (!m & tensor[j]);
    }
}

#[cfg(feature = "rayon")]
fn mask_fill_u8_par(tensor: &[u8], mask: &[u8], fill_value: u8, out: &mut [u8]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_fill_u8_seq(
                &tensor[start..end],
                &mask[start..end],
                fill_value,
                out_chunk,
            );
        });
}
