use super::*;

// ============================================================================
// Fused layer_norm
// ============================================================================
//
// `ruda_nn::LayerNorm::forward` decomposes into ~6 primitive tensor ops
// with intermediate allocations, and there is no backend trait hook for
// layer_norm. This module provides a fused alternative users can opt into
// directly. Two-pass row kernel (sum+sumsq sweep, then normalize+affine
// sweep), both vectorized via macerator.

/// Fused layer normalization along the last axis.
///
/// Applies `y = ((x - mean) / sqrt(var + eps)) * gamma + beta`, where
/// `mean` and `var` are computed per row along the last axis of `input`.
/// `gamma` and `beta` are 1-D tensors of length `input.shape()[-1]`;
/// `beta` is optional (set to `None` for a bias-free layer norm).
///
/// Two-pass row kernel (mean/variance via a single sum+sum-of-squares
/// sweep, then one normalize+affine sweep). Both passes are SIMD via
/// macerator; each row stays cache-hot across both passes.
///
/// Supports `f32` (SIMD-vectorized), `f64` (scalar + LLVM autovec), and
/// `f16`/`bf16` (via an f32 cast-fuse-cast shell; the f32 row kernel
/// already accumulates in f32, so this matches the precision a
/// half-precision-native kernel would produce).
///
/// # Panics
///
/// * If `input`'s dtype is not one of `f32`/`f64`/`f16`/`bf16`.
/// * If `input` has rank 0.
/// * If `gamma` (or `beta`, when present) is not a 1-D tensor of length
///   equal to the last dim of `input`.
pub fn layer_norm(
    input: HostTensor,
    gamma: HostTensor,
    beta: Option<HostTensor>,
    epsilon: f64,
) -> HostTensor {
    let rank = input.shape().num_dims();
    assert!(rank >= 1, "layer_norm: input must have at least one dim");
    // Keep gamma/beta dtypes aligned with the input. The half-precision path
    // (see `layer_norm_via_f32`) ultimately accesses storage using the input's
    // element type, and a mismatch would panic there; reject it up front with
    // a clearer layer_norm-specific error message.
    assert_eq!(
        gamma.dtype(),
        input.dtype(),
        "layer_norm: gamma dtype {:?} does not match input dtype {:?}",
        gamma.dtype(),
        input.dtype(),
    );
    if let Some(ref b) = beta {
        assert_eq!(
            b.dtype(),
            input.dtype(),
            "layer_norm: beta dtype {:?} does not match input dtype {:?}",
            b.dtype(),
            input.dtype(),
        );
    }
    let input = input.to_contiguous();
    let gamma = gamma.to_contiguous();
    let beta = beta.map(|b| b.to_contiguous());

    let d_model = *input
        .layout()
        .shape()
        .last()
        .expect("layer_norm: empty shape");
    // Validate rank + length explicitly rather than just last-dim == d_model.
    // A gamma shaped like `[2, d_model]` would pass a last-dim check but
    // has 2*d_model elements, which would index the wrong data in the row
    // kernel (caught by an inner assert, but with a confusing message).
    let gamma_shape = gamma.layout().shape();
    assert!(
        gamma_shape.len() == 1 && gamma_shape[0] == d_model,
        "layer_norm: gamma must be a 1-D tensor of length equal to last dim of input \
         (got shape {:?}, expected [{}])",
        gamma_shape,
        d_model,
    );
    if let Some(ref b) = beta {
        let beta_shape = b.layout().shape();
        assert!(
            beta_shape.len() == 1 && beta_shape[0] == d_model,
            "layer_norm: beta must be a 1-D tensor of length equal to last dim of input \
             (got shape {:?}, expected [{}])",
            beta_shape,
            d_model,
        );
    }

    match input.dtype() {
        DType::F32 => layer_norm_f32(input, gamma, beta, epsilon as f32),
        DType::F64 => layer_norm_f64(input, gamma, beta, epsilon),
        DType::F16 => {
            layer_norm_via_f32::<f16>(input, gamma, beta, epsilon, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => {
            layer_norm_via_f32::<bf16>(input, gamma, beta, epsilon, bf16::to_f32, bf16::from_f32)
        }
        dtype => panic!("ruda_tensor_host::layer_norm: unsupported dtype {:?}", dtype),
    }
}

fn layer_norm_via_f32<E: ruda_core::tensor::element::Element + bytemuck::Pod + Copy>(
    input: HostTensor,
    gamma: HostTensor,
    beta: Option<HostTensor>,
    epsilon: f64,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> HostTensor {
    let input_f32 = ruda_core::tensor::host::cast::cast_to_f32::<E>(input, to_f32);
    let gamma_f32 = ruda_core::tensor::host::cast::cast_to_f32::<E>(gamma, to_f32);
    let beta_f32 = beta.map(|b| ruda_core::tensor::host::cast::cast_to_f32::<E>(b, to_f32));
    let out = layer_norm_f32(input_f32, gamma_f32, beta_f32, epsilon as f32);
    ruda_core::tensor::host::cast::cast_from_f32::<E>(out, from_f32)
}

/// Fused f64 layer_norm. The Welford mean/variance pass is serial (the
/// mean update on iteration `k` depends on iteration `k-1`); the
/// normalize+affine pass autovectorizes on targets with f64 SIMD. A
/// macerator f64 path can be added if profiling shows it matters.
fn layer_norm_f64(
    input: HostTensor,
    gamma: HostTensor,
    beta: Option<HostTensor>,
    epsilon: f64,
) -> HostTensor {
    let shape = input.layout().shape().clone();
    let d_model = *shape.last().expect("layer_norm: empty shape");
    if d_model == 0 {
        return input;
    }
    let input_data: &[f64] = input.storage();
    let gamma_data: &[f64] = gamma.storage();
    let beta_data: Option<&[f64]> = beta.as_ref().map(|b| b.storage());
    let mut output: Vec<f64> = vec![0.0; input_data.len()];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        const ROWS_PER_TASK: usize = 64;
        let chunk_elems = ROWS_PER_TASK * d_model;
        match beta_data {
            Some(beta_slice) => {
                output
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f64_with_beta(
                            i, o, gamma_data, beta_slice, d_model, epsilon,
                        );
                    });
            }
            None => {
                output
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f64_no_beta(i, o, gamma_data, d_model, epsilon);
                    });
            }
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        match beta_data {
            Some(beta_slice) => layer_norm_rows_f64_with_beta(
                input_data,
                output.as_mut_slice(),
                gamma_data,
                beta_slice,
                d_model,
                epsilon,
            ),
            None => layer_norm_rows_f64_no_beta(
                input_data,
                output.as_mut_slice(),
                gamma_data,
                d_model,
                epsilon,
            ),
        }
    }

    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::F64,
    )
}

#[inline]
fn layer_norm_rows_f64_with_beta(
    input: &[f64],
    output: &mut [f64],
    gamma: &[f64],
    beta: &[f64],
    d_model: usize,
    epsilon: f64,
) {
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        let (mean, inv_std) = welford_f64(in_row, epsilon);
        for (i, &x) in in_row.iter().enumerate() {
            out_row[i] = (x - mean) * (inv_std * gamma[i]) + beta[i];
        }
    }
}

#[inline]
fn layer_norm_rows_f64_no_beta(
    input: &[f64],
    output: &mut [f64],
    gamma: &[f64],
    d_model: usize,
    epsilon: f64,
) {
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        let (mean, inv_std) = welford_f64(in_row, epsilon);
        for (i, &x) in in_row.iter().enumerate() {
            out_row[i] = (x - mean) * (inv_std * gamma[i]);
        }
    }
}

#[inline]
fn welford_f64(row: &[f64], epsilon: f64) -> (f64, f64) {
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (k, &x) in row.iter().enumerate() {
        let n_k = (k + 1) as f64;
        let delta = x - mean;
        mean += delta / n_k;
        m2 += delta * (x - mean);
    }
    let var = m2 / row.len() as f64;
    (mean, 1.0f64 / (var + epsilon).sqrt())
}

fn layer_norm_f32(
    input: HostTensor,
    gamma: HostTensor,
    beta: Option<HostTensor>,
    epsilon: f32,
) -> HostTensor {
    let shape = input.layout().shape().clone();
    let d_model = *shape.last().expect("layer_norm: empty shape");
    if d_model == 0 {
        return input;
    }

    let input_data: &[f32] = input.storage();
    let gamma_data: &[f32] = gamma.storage();
    let beta_data: Option<&[f32]> = beta.as_ref().map(|b| b.storage());

    let n = input_data.len();
    // See softmax_last_f32 for the rationale on zero-init instead of
    // `spare_capacity_mut` + `&mut [f32]` cast: the latter creates a
    // reference to uninitialized f32 values, which is UB under Rust's
    // aliasing model even with no intervening read.
    let mut output: Vec<f32> = vec![0.0; n];
    let out_slice = output.as_mut_slice();

    // `#[macerator::with_simd]` can't auto-lifetime through
    // `Option<&[T]>`, so we dispatch two separate monomorphized
    // versions, one with beta and one without. Both call into the
    // same shared row kernel.
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        const ROWS_PER_TASK: usize = 64;
        let chunk_elems = ROWS_PER_TASK * d_model;
        match beta_data {
            Some(beta_slice) => {
                out_slice
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f32_with_beta(
                            i, o, gamma_data, beta_slice, d_model, epsilon,
                        );
                    });
            }
            None => {
                out_slice
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f32_no_beta(i, o, gamma_data, d_model, epsilon);
                    });
            }
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        match beta_data {
            Some(beta_slice) => layer_norm_rows_f32_with_beta(
                input_data, out_slice, gamma_data, beta_slice, d_model, epsilon,
            ),
            None => {
                layer_norm_rows_f32_no_beta(input_data, out_slice, gamma_data, d_model, epsilon)
            }
        }
    }

    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::F32,
    )
}

/// Row sweep for f32 layer_norm with bias. Delegates to the SIMD kernel
/// when the `simd` feature is enabled; otherwise uses a scalar row loop.
#[inline]
fn layer_norm_rows_f32_with_beta(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    // Release-mode invariant checks; see softmax_rows_f32 for rationale.
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % d_model, 0);
    assert_eq!(gamma.len(), d_model);
    assert_eq!(beta.len(), d_model);
    #[cfg(feature = "simd")]
    layer_norm_rows_f32_with_beta_simd(input, output, gamma, beta, d_model, epsilon);
    #[cfg(not(feature = "simd"))]
    {
        for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
            layer_norm_row_f32_scalar(in_row, out_row, gamma, Some(beta), epsilon);
        }
    }
}

/// Row sweep for f32 layer_norm without bias.
#[inline]
fn layer_norm_rows_f32_no_beta(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    // Release-mode invariant checks; see softmax_rows_f32 for rationale.
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % d_model, 0);
    assert_eq!(gamma.len(), d_model);
    #[cfg(feature = "simd")]
    layer_norm_rows_f32_no_beta_simd(input, output, gamma, d_model, epsilon);
    #[cfg(not(feature = "simd"))]
    {
        for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
            layer_norm_row_f32_scalar(in_row, out_row, gamma, None, epsilon);
        }
    }
}

/// Scalar fallback row kernel for layer_norm when the `simd` feature is
/// disabled. Two-pass algorithm matching the SIMD version (sum+sumsq,
/// then normalize+affine).
#[cfg(not(feature = "simd"))]
#[inline]
fn layer_norm_row_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    epsilon: f32,
) {
    // Welford's online algorithm for mean and variance, rather than the
    // `sumsq / n - mean * mean` identity the SIMD path uses. The identity
    // is vulnerable to catastrophic cancellation when the two terms are
    // close in magnitude (large mean relative to variance). Welford's
    // single-pass formulation avoids that by tracking the running mean
    // and accumulating squared deviations from it. The scalar path is
    // the contract used when `simd` is disabled, so we prefer numerical
    // stability over bit-for-bit match with the SIMD tree reduction.
    let len = input.len();
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    for (k, &x) in input.iter().enumerate() {
        let n_k = (k + 1) as f32;
        let delta = x - mean;
        mean += delta / n_k;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    let var = m2 / len as f32;
    let inv_std = 1.0f32 / (var + epsilon).sqrt();
    for (i, &x) in input.iter().enumerate() {
        let scale = inv_std * gamma[i];
        let normed = (x - mean) * scale;
        output[i] = match beta {
            Some(b) => normed + b[i],
            None => normed,
        };
    }
}

/// SIMD-dispatched row sweep for f32 layer_norm with bias (beta). One
/// macerator dispatch per chunk of rows, amortized over the whole chunk.
#[cfg(feature = "simd")]
#[macerator::with_simd]
fn layer_norm_rows_f32_with_beta_simd<S: macerator::Simd>(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    debug_assert_eq!(beta.len(), d_model);
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        layer_norm_row_f32_simd::<S>(in_row, out_row, gamma, Some(beta), epsilon);
    }
}

/// SIMD-dispatched row sweep for f32 layer_norm without bias.
#[cfg(feature = "simd")]
#[macerator::with_simd]
fn layer_norm_rows_f32_no_beta_simd<S: macerator::Simd>(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        layer_norm_row_f32_simd::<S>(in_row, out_row, gamma, None, epsilon);
    }
}

/// Single-row layer_norm kernel. Two vectorized passes.
#[cfg(feature = "simd")]
#[inline(always)]
fn layer_norm_row_f32_simd<S: macerator::Simd>(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    epsilon: f32,
) {
    use macerator::{Scalar, vload_unaligned, vstore_unaligned};
    let lanes = <f32 as Scalar>::lanes::<S>();
    let len = input.len();
    let simd_len = len / lanes * lanes;

    // Pass 1: compute sum and sum-of-squares in one sweep, then derive
    // mean and variance. Two independent SIMD accumulators (sum, sumsq)
    // expose ILP to the two FMA ports.
    let (sum, sumsq) = if simd_len >= lanes {
        let mut acc_sum = 0.0f32.splat::<S>();
        let mut acc_sumsq = 0.0f32.splat::<S>();
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let v = vload_unaligned::<S, _>(input.as_ptr().add(i));
                acc_sum += v;
                // acc_sumsq += v * v; Vector::mul_add(self, a, b) = self*a + b,
                // so v.mul_add(v, acc_sumsq) = v*v + acc_sumsq.
                acc_sumsq = v.mul_add(v, acc_sumsq);
            }
            i += lanes;
        }
        let mut s = acc_sum.reduce_add();
        let mut sq = acc_sumsq.reduce_add();
        for &x in &input[simd_len..] {
            s += x;
            sq += x * x;
        }
        (s, sq)
    } else {
        let mut s = 0.0f32;
        let mut sq = 0.0f32;
        for &x in input {
            s += x;
            sq += x * x;
        }
        (s, sq)
    };

    let n = len as f32;
    let mean = sum / n;
    // Biased variance: E[x^2] - E[x]^2. Matches ruda_nn::LayerNorm which
    // uses var_mean_bias (the biased estimator) rather than Bessel's
    // correction.
    let var = (sumsq / n) - mean * mean;
    let inv_std = 1.0f32 / (var + epsilon).sqrt();

    // Pass 2: normalize and affine transform.
    //   out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i]
    // mean_vec and inv_std_vec are hoisted outside the loop (one splat
    // each per row). gamma and beta are read once per element; both
    // fit in L1 and are shared across all rows within a rayon chunk.
    let mean_vec = mean.splat::<S>();
    let inv_std_vec = inv_std.splat::<S>();
    let mut i = 0;
    while i < simd_len {
        unsafe {
            let x = vload_unaligned::<S, _>(input.as_ptr().add(i));
            let g = vload_unaligned::<S, _>(gamma.as_ptr().add(i));
            // scale = inv_std * g
            let scale = inv_std_vec * g;
            // centered = x - mean
            let centered = x - mean_vec;
            // out = centered * scale  (+ beta if present)
            let normed = centered * scale;
            let out = if let Some(b) = beta {
                let b_vec = vload_unaligned::<S, _>(b.as_ptr().add(i));
                normed + b_vec
            } else {
                normed
            };
            vstore_unaligned::<S, _>(output.as_mut_ptr().add(i), out);
        }
        i += lanes;
    }
    // Scalar tail
    while i < len {
        let centered = input[i] - mean;
        let normed = centered * inv_std * gamma[i];
        output[i] = match beta {
            Some(b) => normed + b[i],
            None => normed,
        };
        i += 1;
    }
}
