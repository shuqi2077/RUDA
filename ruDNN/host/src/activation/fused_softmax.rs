use super::*;

// ============================================================================
// Fused softmax
// ============================================================================
//
// `ActivationOps` does not currently expose a `softmax` hook, so
// `ruda_tensor::activation::softmax` falls back to a 5-op decomposition
// (`max_dim`/`sub`/`exp`/`sum_dim`/`div`). This module provides a fused
// alternative users can opt into directly.

/// Fused softmax along `dim`.
///
/// Three-pass row-wise algorithm (max, exp+sum, normalize) keeping each row
/// cache-hot. Rows are processed in parallel via rayon. For axes other than
/// the last, the tensor is permuted to put `dim` last, the fused kernel runs,
/// and the result is permuted back (both permutes are metadata-only; the
/// fused kernel's internal `to_contiguous` materializes the permuted layout
/// once).
///
/// # Panics
///
/// * If `dim` is out of range for `input`.
/// * If `input`'s dtype is not one of `f32`/`f64`/`f16`/`bf16`.
pub fn softmax(tensor: HostTensor, dim: usize) -> HostTensor {
    let rank = tensor.shape().num_dims();
    assert!(
        dim < rank,
        "softmax dim {} out of range for rank {}",
        dim,
        rank
    );

    if dim != rank - 1 {
        let swapped = tensor.transpose(dim, rank - 1);
        let normed = softmax_last(swapped);
        return normed.transpose(dim, rank - 1);
    }

    softmax_last(tensor)
}

fn softmax_last(tensor: HostTensor) -> HostTensor {
    let tensor = tensor.to_contiguous();
    match tensor.dtype() {
        DType::F32 => softmax_last_f32(tensor),
        DType::F64 => softmax_last_f64(tensor),
        DType::F16 => softmax_last_f16(tensor),
        DType::BF16 => softmax_last_bf16(tensor),
        dtype => panic!("softmax: unsupported dtype {:?}", dtype),
    }
}

fn softmax_last_f32(tensor: HostTensor) -> HostTensor {
    let shape = tensor.layout().shape().clone();
    let last = *shape.last().expect("softmax: empty shape");
    if last == 0 {
        return tensor;
    }
    let input: &[f32] = tensor.storage();
    let n = input.len();

    // Zero-initialize the output. The previous implementation used
    // `Vec::with_capacity` + `spare_capacity_mut` + a raw-pointer cast to
    // `&mut [f32]` to skip the memset, but forming a `&mut [f32]` over
    // uninitialized memory violates Rust's validity invariant (references
    // must point to initialized values of the correct type) even if every
    // element is written before it is read. The sound zero-memset
    // alternative would require threading `&mut [MaybeUninit<f32>]` through
    // the row kernel, which does not compose with macerator's `#[with_simd]`
    // signature. The memset is a streaming write on a bandwidth-bound
    // kernel, so the overhead is small (~10% at the largest bench shape)
    // and the fused path remains well ahead of decomposed and candle.
    let mut output: Vec<f32> = vec![0.0; n];
    let out_slice = output.as_mut_slice();

    // Row-parallel via rayon: one macerator dispatch per chunk of rows,
    // amortized over all rows in the chunk.
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        const ROWS_PER_TASK: usize = 64;
        let chunk_elems = ROWS_PER_TASK * last;
        out_slice
            .par_chunks_mut(chunk_elems)
            .zip(input.par_chunks(chunk_elems))
            .for_each(|(o, i)| softmax_rows_f32(i, o, last));
    }
    #[cfg(not(feature = "rayon"))]
    {
        softmax_rows_f32(input, out_slice, last);
    }

    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::F32,
    )
}

/// Row sweep for f32 softmax. With the `simd` feature, delegates to the
/// `#[macerator::with_simd]` SIMD kernel (one dispatch per chunk of rows,
/// amortized over all rows in the chunk). Without `simd`, uses a scalar
/// row kernel.
#[inline]
fn softmax_rows_f32(input: &[f32], output: &mut [f32], row_len: usize) {
    // Release-mode invariant checks. These run once per chunk of rows
    // (dozens of times per call, not per-element), so the overhead is
    // unmeasurable against the kernel work. A debug-only check would
    // silently pass a short final chunk to the row kernel on release
    // builds if a future refactor broke the row alignment at the call
    // site, yielding wrong softmax output with no panic.
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % row_len, 0);
    #[cfg(feature = "simd")]
    softmax_rows_f32_simd(input, output, row_len);
    #[cfg(not(feature = "simd"))]
    {
        for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
            softmax_row_f32_scalar(in_row, out_row);
        }
    }
}

#[cfg(feature = "simd")]
#[macerator::with_simd]
fn softmax_rows_f32_simd<S: macerator::Simd>(input: &[f32], output: &mut [f32], row_len: usize) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % row_len, 0);
    for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
        softmax_row_f32_simd::<S>(in_row, out_row);
    }
}

/// Scalar fallback row kernel for f32 softmax when the `simd` feature is
/// disabled. Uses the same 3-pass algorithm as the SIMD path; LLVM
/// autovectorizes the max-reduce and normalize loops on most targets.
#[cfg(not(feature = "simd"))]
#[inline]
fn softmax_row_f32_scalar(input: &[f32], output: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }
    let mut sum = 0.0f32;
    for (i, &x) in input.iter().enumerate() {
        let e = (x - max_val).exp();
        output[i] = e;
        sum += e;
    }
    let inv = 1.0f32 / sum;
    for x in output.iter_mut() {
        *x *= inv;
    }
}

/// Inner row kernel for a single softmax row. `#[inline(always)]` so it
/// inlines into `softmax_rows_f32_simd`'s loop body for each monomorphized S,
/// avoiding a per-row call boundary.
#[cfg(feature = "simd")]
#[inline(always)]
fn softmax_row_f32_simd<S: macerator::Simd>(input: &[f32], output: &mut [f32]) {
    use macerator::{Scalar, vload_unaligned, vstore_unaligned};
    let lanes = <f32 as Scalar>::lanes::<S>();
    let len = input.len();
    let simd_len = len / lanes * lanes;

    // Pass 1: row max for numerical stability.
    // SIMD max-reduction across the row, scalar tail.
    let (mut max_val, tail_start) = if simd_len >= lanes {
        let mut max_vec = unsafe { vload_unaligned::<S, _>(input.as_ptr()) };
        let mut j = lanes;
        while j < simd_len {
            let v = unsafe { vload_unaligned::<S, _>(input.as_ptr().add(j)) };
            max_vec = max_vec.max(v);
            j += lanes;
        }
        (max_vec.reduce_max(), simd_len)
    } else {
        (f32::NEG_INFINITY, 0)
    };
    for &x in &input[tail_start..] {
        if x > max_val {
            max_val = x;
        }
    }

    // Pass 2: compute exp(x - max), store in output, accumulate sum.
    // Scalar exp (no SIMD exp in macerator). This pass is the one that
    // actually does memory reads + writes on the whole row, so scalar
    // here still lands us at memory bandwidth.
    let mut sum = 0.0f32;
    for idx in 0..len {
        let e = (input[idx] - max_val).exp();
        output[idx] = e;
        sum += e;
    }

    // Pass 3: normalize.
    // SIMD splat + multiply, scalar tail.
    let inv = 1.0f32 / sum;
    let inv_vec = inv.splat::<S>();
    let mut i = 0;
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, _>(output.as_ptr().add(i));
            vstore_unaligned::<S, _>(output.as_mut_ptr().add(i), v * inv_vec);
        }
        i += lanes;
    }
    for x in &mut output[i..] {
        *x *= inv;
    }
}

// f64, f16, bf16 softmax share the same row-parallel dispatcher shell and
// differ only in their row kernel (native f64 vs via-f32 for half
// precision). Generated via macros to keep the three variants in lockstep.
// Only f32 has a dedicated SIMD fast path above.

macro_rules! softmax_last_dtype {
    ($fn_name:ident, $T:ty, $zero:expr, $dtype:expr, $row_fn:ident) => {
        fn $fn_name(tensor: HostTensor) -> HostTensor {
            let shape = tensor.layout().shape().clone();
            let last = *shape.last().expect("softmax: empty shape");
            if last == 0 {
                return tensor;
            }
            let input: &[$T] = tensor.storage();
            let mut output: Vec<$T> = vec![$zero; input.len()];

            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                output
                    .par_chunks_mut(last)
                    .zip(input.par_chunks(last))
                    .for_each(|(o, i)| $row_fn(i, o));
            }
            #[cfg(not(feature = "rayon"))]
            {
                for (i, o) in input.chunks(last).zip(output.chunks_mut(last)) {
                    $row_fn(i, o);
                }
            }

            HostTensor::new(Bytes::from_elems(output), Layout::contiguous(shape), $dtype)
        }
    };
}

/// Half-precision softmax row kernel. Accumulates in f32 for numerical
/// stability and converts back to the target type at each write. This
/// double-rounds across passes 2 and 3; acceptable for half precision. An
/// f32 scratch buffer would remove the double rounding at the cost of a
/// per-row allocation.
macro_rules! softmax_row_half {
    ($fn_name:ident, $T:ty) => {
        #[inline]
        fn $fn_name(input: &[$T], output: &mut [$T]) {
            let mut max_val = f32::NEG_INFINITY;
            for &x in input {
                let xf = x.to_f32();
                if xf > max_val {
                    max_val = xf;
                }
            }
            let mut sum = 0.0f32;
            for (i, &x) in input.iter().enumerate() {
                let e = (x.to_f32() - max_val).exp();
                output[i] = <$T>::from_f32(e);
                sum += e;
            }
            let inv = 1.0f32 / sum;
            for x in output.iter_mut() {
                *x = <$T>::from_f32(x.to_f32() * inv);
            }
        }
    };
}

#[inline]
fn softmax_row_f64(input: &[f64], output: &mut [f64]) {
    let mut max_val = f64::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }
    let mut sum = 0.0f64;
    for (i, &x) in input.iter().enumerate() {
        let e = (x - max_val).exp();
        output[i] = e;
        sum += e;
    }
    let inv = 1.0f64 / sum;
    for x in output.iter_mut() {
        *x *= inv;
    }
}

softmax_row_half!(softmax_row_f16, f16);
softmax_row_half!(softmax_row_bf16, bf16);

softmax_last_dtype!(softmax_last_f64, f64, 0.0f64, DType::F64, softmax_row_f64);
softmax_last_dtype!(
    softmax_last_f16,
    f16,
    f16::from_f32(0.0),
    DType::F16,
    softmax_row_f16
);
softmax_last_dtype!(
    softmax_last_bf16,
    bf16,
    bf16::from_f32(0.0),
    DType::BF16,
    softmax_row_bf16
);
