//! Real FFT (rfft) via Cooley-Tukey with aggressive optimization.
//!
//! Key optimizations:
//! - Real FFT via complex packing: pack N real values as N/2 complex,
//!   do a half-size complex FFT, then unpack using Hermitian symmetry (~2x)
//! - Compile-time twiddle tables via const fn Taylor-series sin/cos
//! - Unrolled small complex FFT kernels for N=2, 4, 8
//! - Mixed radix-4/radix-2 butterfly stages (halves passes over data)
//! - SIMD-vectorized butterflies via macerator
//! - Rayon parallelism across independent fibers

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::{bytes::Bytes, tensor::Shape};

use ruda_core::tensor::host::layout::{contiguous_strides_usize, slice_base_offset};
use ruda_core::tensor::host::{HostTensor, Layout};

// ============================================================================
// Const-evaluable sin/cos via Taylor series (13 terms, ~13 digit accuracy)
// ============================================================================

mod twiddle;
use twiddle::{TwiddleRef, get_twiddles};
#[cfg(test)]
use twiddle::{const_cos, const_sin};

// ============================================================================
// Bit-reversal permutation
// ============================================================================

#[inline]
fn bit_reverse_permute(re: &mut [f32], im: &mut [f32], n: usize) {
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

// ============================================================================
// Unrolled small complex FFT kernels
// ============================================================================

/// Complex FFT of size 2: single butterfly, no twiddles.
#[inline(always)]
fn complex_fft_2(re: &mut [f32], im: &mut [f32]) {
    let (r0, r1) = (re[0], re[1]);
    let (i0, i1) = (im[0], im[1]);
    re[0] = r0 + r1;
    re[1] = r0 - r1;
    im[0] = i0 + i1;
    im[1] = i0 - i1;
}

/// Complex FFT of size 4: 2 stages, fully unrolled.
/// Twiddle for stage 1, k=1 is W_4^1 = -i.
#[inline(always)]
fn complex_fft_4(re: &mut [f32], im: &mut [f32]) {
    // Bit-reversal: swap indices 1 and 2
    re.swap(1, 2);
    im.swap(1, 2);

    // Stage 0: two size-2 butterflies
    let (r0, r1) = (re[0] + re[1], re[0] - re[1]);
    let (i0, i1) = (im[0] + im[1], im[0] - im[1]);
    let (r2, r3) = (re[2] + re[3], re[2] - re[3]);
    let (i2, i3) = (im[2] + im[3], im[2] - im[3]);

    // Stage 1: size-4 butterfly
    // k=0: W=1, butterfly (0,2)
    re[0] = r0 + r2;
    im[0] = i0 + i2;
    re[2] = r0 - r2;
    im[2] = i0 - i2;
    // k=1: W=-i, butterfly (1,3): -i*(r3+i*i3) = (i3, -r3)
    re[1] = r1 + i3;
    im[1] = i1 - r3;
    re[3] = r1 - i3;
    im[3] = i1 + r3;
}

/// Complex FFT of size 8: 3 stages, fully unrolled.
#[inline(always)]
fn complex_fft_8(re: &mut [f32], im: &mut [f32]) {
    // Bit-reversal for n=8: [0,4,2,6,1,5,3,7]
    re.swap(1, 4);
    im.swap(1, 4);
    re.swap(3, 6);
    im.swap(3, 6);

    // Stage 0: four size-2 butterflies
    macro_rules! butterfly2 {
        ($a:expr, $b:expr) => {
            let (ra, rb) = (re[$a] + re[$b], re[$a] - re[$b]);
            let (ia, ib) = (im[$a] + im[$b], im[$a] - im[$b]);
            re[$a] = ra;
            re[$b] = rb;
            im[$a] = ia;
            im[$b] = ib;
        };
    }
    butterfly2!(0, 1);
    butterfly2!(2, 3);
    butterfly2!(4, 5);
    butterfly2!(6, 7);

    // Stage 1: two size-4 butterflies
    // Group [0,1,2,3]: k=0 W=1, k=1 W=-i
    {
        let (r0, r2) = (re[0] + re[2], re[0] - re[2]);
        let (i0, i2) = (im[0] + im[2], im[0] - im[2]);
        re[0] = r0;
        im[0] = i0;
        re[2] = r2;
        im[2] = i2;
        // k=1: W=-i → (im[3], -re[3])
        let (t_re, t_im) = (im[3], -re[3]);
        let (r1a, r1b) = (re[1] + t_re, re[1] - t_re);
        let (i1a, i1b) = (im[1] + t_im, im[1] - t_im);
        re[1] = r1a;
        re[3] = r1b;
        im[1] = i1a;
        im[3] = i1b;
    }
    // Group [4,5,6,7]: same pattern
    {
        let (r4, r6) = (re[4] + re[6], re[4] - re[6]);
        let (i4, i6) = (im[4] + im[6], im[4] - im[6]);
        re[4] = r4;
        im[4] = i4;
        re[6] = r6;
        im[6] = i6;
        let (t_re, t_im) = (im[7], -re[7]);
        let (r5a, r5b) = (re[5] + t_re, re[5] - t_re);
        let (i5a, i5b) = (im[5] + t_im, im[5] - t_im);
        re[5] = r5a;
        re[7] = r5b;
        im[5] = i5a;
        im[7] = i5b;
    }

    // Stage 2: one size-8 butterfly
    // k=0: W=1
    {
        let (a, b) = (re[0] + re[4], re[0] - re[4]);
        let (c, d) = (im[0] + im[4], im[0] - im[4]);
        re[0] = a;
        re[4] = b;
        im[0] = c;
        im[4] = d;
    }
    // k=1: W_8^1 = (sqrt2/2, -sqrt2/2)
    {
        const W: f32 = core::f32::consts::FRAC_1_SQRT_2; // 0.7071...
        let t_re = W * re[5] - (-W) * im[5]; // W*re + W*im
        let t_im = W * im[5] + (-W) * re[5]; // W*im - W*re
        re[5] = re[1] - t_re;
        im[5] = im[1] - t_im;
        re[1] += t_re;
        im[1] += t_im;
    }
    // k=2: W_8^2 = -i
    {
        let (t_re, t_im) = (im[6], -re[6]);
        re[6] = re[2] - t_re;
        im[6] = im[2] - t_im;
        re[2] += t_re;
        im[2] += t_im;
    }
    // k=3: W_8^3 = (-sqrt2/2, -sqrt2/2)
    {
        const W: f32 = core::f32::consts::FRAC_1_SQRT_2;
        let t_re = -W * re[7] - (-W) * im[7]; // -W*re + W*im
        let t_im = -W * im[7] + (-W) * re[7]; // -W*im - W*re
        re[7] = re[3] - t_re;
        im[7] = im[3] - t_im;
        re[3] += t_re;
        im[3] += t_im;
    }
}

// ============================================================================
// General complex FFT: mixed radix-4/radix-2 with SIMD
// ============================================================================

/// Complex FFT of size n (power of 2) using precomputed twiddles.
#[inline]
fn complex_fft(re: &mut [f32], im: &mut [f32], n: usize, tw: &TwiddleRef) {
    match n {
        0 | 1 => return,
        2 => {
            complex_fft_2(re, im);
            return;
        }
        4 => {
            complex_fft_4(re, im);
            return;
        }
        8 => {
            complex_fft_8(re, im);
            return;
        }
        _ => {}
    }

    bit_reverse_permute(re, im, n);

    let tw_re = tw.re();
    let tw_im = tw.im();
    let offsets = tw.offsets();
    let num_stages = offsets.len() - 1;

    // For odd number of stages, do one radix-2 pass first so the
    // remaining stages can be processed in radix-4 pairs.
    // Stage 0 twiddle is always W_2^0 = 1, so just add/sub.
    let start_stage = if num_stages % 2 == 1 {
        let mut start = 0;
        while start < n {
            let (a, b) = (re[start] + re[start + 1], re[start] - re[start + 1]);
            let (c, d) = (im[start] + im[start + 1], im[start] - im[start + 1]);
            re[start] = a;
            re[start + 1] = b;
            im[start] = c;
            im[start + 1] = d;
            start += 2;
        }
        1
    } else {
        0
    };

    #[cfg(feature = "simd")]
    {
        simd_fft::radix4_simd(re, im, n, tw_re, tw_im, offsets, start_stage, num_stages);
    }
    #[cfg(not(feature = "simd"))]
    {
        radix4_scalar(re, im, n, tw_re, tw_im, offsets, start_stage, num_stages);
    }
}

/// Correct DIT radix-4: fuse two radix-2 stages into one pass.
///
/// For each pair of stages (s, s+1) with quarter = 2^s:
/// 1. Apply W_{2q}^k to x[p1] and x[p3] (inner stage twiddle, same for both)
/// 2. Inner butterflies: a=p0+tw1, b=p0-tw1, c=p2+tw3, d=p2-tw3
/// 3. Apply W_{4q}^k to c and d; d also gets -i rotation
/// 4. Outer butterflies: p0=a+tc, p2=a-tc, p1=b+(-i*td), p3=b-(-i*td)
#[cfg(not(feature = "simd"))]
#[allow(clippy::too_many_arguments)]
fn radix4_scalar(
    re: &mut [f32],
    im: &mut [f32],
    n: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    offsets: &[usize],
    start_stage: usize,
    num_stages: usize,
) {
    let mut stage = start_stage;
    while stage + 1 < num_stages {
        let quarter = 1 << stage;
        let group_size = quarter << 2;
        let tw_off_inner = offsets[stage]; // W_{2q}^k
        let tw_off_outer = offsets[stage + 1]; // W_{4q}^k

        let mut group_start = 0;
        while group_start < n {
            for k in 0..quarter {
                let p0 = group_start + k;
                let p1 = p0 + quarter;
                let p2 = p1 + quarter;
                let p3 = p2 + quarter;

                // Inner twiddle: W_{2q}^k applied to p1 and p3
                let wi_re = tw_re[tw_off_inner + k];
                let wi_im = tw_im[tw_off_inner + k];
                let tw1_re = wi_re * re[p1] - wi_im * im[p1];
                let tw1_im = wi_re * im[p1] + wi_im * re[p1];
                let tw3_re = wi_re * re[p3] - wi_im * im[p3];
                let tw3_im = wi_re * im[p3] + wi_im * re[p3];

                // Inner butterflies
                let a_re = re[p0] + tw1_re;
                let a_im = im[p0] + tw1_im;
                let b_re = re[p0] - tw1_re;
                let b_im = im[p0] - tw1_im;
                let c_re = re[p2] + tw3_re;
                let c_im = im[p2] + tw3_im;
                let d_re = re[p2] - tw3_re;
                let d_im = im[p2] - tw3_im;

                // Outer twiddle: W_{4q}^k applied to c and d
                let wo_re = tw_re[tw_off_outer + k];
                let wo_im = tw_im[tw_off_outer + k];
                let tc_re = wo_re * c_re - wo_im * c_im;
                let tc_im = wo_re * c_im + wo_im * c_re;
                let td_re = wo_re * d_re - wo_im * d_im;
                let td_im = wo_re * d_im + wo_im * d_re;

                // Outer butterflies (-i*(td_re+i*td_im) = (td_im, -td_re))
                re[p0] = a_re + tc_re;
                im[p0] = a_im + tc_im;
                re[p2] = a_re - tc_re;
                im[p2] = a_im - tc_im;
                re[p1] = b_re + td_im;
                im[p1] = b_im - td_re;
                re[p3] = b_re - td_im;
                im[p3] = b_im + td_re;
            }
            group_start += group_size;
        }
        stage += 2;
    }
}

#[cfg(feature = "simd")]
mod simd_fft;

// ============================================================================
// Real FFT unpacking
// ============================================================================

/// Unpack N/2-point complex FFT result into N/2+1 real FFT bins.
///
/// Given Z = FFT(pack(x)), recovers X = FFT(x) using:
///   Xe[k] = (Z[k] + conj(Z[N/2-k])) / 2
///   Xo[k] = -i * (Z[k] - conj(Z[N/2-k])) / 2
///   X[k]  = Xe[k] + W_N^k * Xo[k]
fn unpack_rfft(
    z_re: &[f32],
    z_im: &[f32],
    half: usize,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    out_re: &mut [f32],
    out_im: &mut [f32],
) {
    // k=0: X[0] = Z_re[0] + Z_im[0] (real)
    out_re[0] = z_re[0] + z_im[0];
    out_im[0] = 0.0;

    // k=N/2: X[N/2] = Z_re[0] - Z_im[0] (real)
    out_re[half] = z_re[0] - z_im[0];
    out_im[half] = 0.0;

    // k=1..half-1
    for k in 1..half {
        let j = half - k;
        let (zk_re, zk_im) = (z_re[k], z_im[k]);
        let (zj_re, zj_im) = (z_re[j], z_im[j]);

        // Xe = (Z[k] + conj(Z[j])) / 2
        let xe_re = (zk_re + zj_re) * 0.5;
        let xe_im = (zk_im - zj_im) * 0.5;

        // Xo = -i * (Z[k] - conj(Z[j])) / 2
        // diff = Z[k] - conj(Z[j]) = (zk_re - zj_re, zk_im + zj_im)
        // -i * diff = (diff_im, -diff_re)
        let xo_re = (zk_im + zj_im) * 0.5;
        let xo_im = (zj_re - zk_re) * 0.5;

        // X[k] = Xe + W * Xo
        let wr = unpack_tw_re[k];
        let wi = unpack_tw_im[k];

        out_re[k] = xe_re + wr * xo_re - wi * xo_im;
        out_im[k] = xe_im + wr * xo_im + wi * xo_re;
    }
}

// ============================================================================
// Tensor helpers
// ============================================================================

fn make_tensors_typed<E: ruda_core::tensor::element::Element + bytemuck::Pod>(
    re: Vec<E>,
    im: Vec<E>,
    shape: Shape,
) -> (HostTensor, HostTensor) {
    let dtype = E::dtype();
    let re_t = HostTensor::new(
        Bytes::from_elems(re),
        Layout::contiguous(shape.clone()),
        dtype,
    );
    let im_t = HostTensor::new(Bytes::from_elems(im), Layout::contiguous(shape), dtype);
    (re_t, im_t)
}

// ============================================================================
// Top-level rfft: real FFT via complex packing
// ============================================================================

/// Process a single fiber: pack real signal as complex, FFT, unpack.
#[allow(clippy::too_many_arguments)]
#[inline]
fn rfft_fiber(
    signal: &[f32],
    in_stride: usize,
    n: usize,
    sig_len: usize,
    out_re: &mut [f32],
    out_im: &mut [f32],
    tw_half: &TwiddleRef,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
) {
    let half = n / 2;

    if n == 1 {
        out_re[0] = if sig_len >= 1 { signal[0] } else { 0.0 };
        out_im[0] = 0.0;
        return;
    }

    if sig_len >= n {
        if in_stride == 1 {
            for k in 0..half {
                z_re[k] = signal[2 * k];
                z_im[k] = signal[2 * k + 1];
            }
        } else {
            for k in 0..half {
                z_re[k] = signal[(2 * k) * in_stride];
                z_im[k] = signal[(2 * k + 1) * in_stride];
            }
        }
    } else {
        for k in 0..half {
            let even = 2 * k;
            let odd = 2 * k + 1;
            z_re[k] = if even < sig_len {
                signal[even * in_stride]
            } else {
                0.0
            };
            z_im[k] = if odd < sig_len {
                signal[odd * in_stride]
            } else {
                0.0
            };
        }
    }

    complex_fft(z_re, z_im, half, tw_half);
    unpack_rfft(z_re, z_im, half, unpack_tw_re, unpack_tw_im, out_re, out_im);
}

pub fn rfft_f32(tensor: HostTensor, dim: usize, n: Option<usize>) -> (HostTensor, HostTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    assert!(
        dim < shape.num_dims(),
        "rfft: dim {dim} out of bounds for {}-D tensor",
        shape.num_dims()
    );

    let requested_n = n.unwrap_or_else(|| {
        let sig_len = shape[dim];
        assert!(
            sig_len > 0 && sig_len.is_power_of_two(),
            "rfft: dimension size must be a power of 2, got {sig_len}"
        );
        sig_len
    });
    let fft_size = requested_n.next_power_of_two();
    let sig_len = shape[dim].min(requested_n);

    let n = fft_size;
    let out_len = n / 2 + 1;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = out_len;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / shape[dim];

    let data: &[f32] = tensor.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);

    // N=1: each element is its own DFT, no twiddles needed
    if n == 1 {
        let mut re_out = vec![0.0f32; total_out];
        let im_out = vec![0.0f32; total_out];
        if sig_len >= 1 {
            for fiber_idx in 0..num_fibers {
                let base = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
                re_out[out_base] = data[base];
            }
        }
        return make_tensors_typed(re_out, im_out, out_shape);
    }

    let half = n / 2;
    let tw_half = get_twiddles(half);

    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_tw_re = &tw_full.re()[last_stage_off..];
    let unpack_tw_im = &tw_full.im()[last_stage_off..];

    let mut re_out = vec![0.0f32; total_out];
    let mut im_out = vec![0.0f32; total_out];

    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f32>, Vec<f32>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f32; half.max(1)];
                let mut z_im = vec![0.0f32; half.max(1)];
                let mut fiber_re = vec![0.0f32; out_len];
                let mut fiber_im = vec![0.0f32; out_len];

                rfft_fiber(
                    &data[base_offset..],
                    in_stride,
                    n,
                    sig_len,
                    &mut fiber_re,
                    &mut fiber_im,
                    &tw_half,
                    unpack_tw_re,
                    unpack_tw_im,
                    &mut z_re,
                    &mut z_im,
                );
                (fiber_idx, fiber_re, fiber_im)
            })
            .collect();

        for (fiber_idx, fiber_re, fiber_im) in fiber_results {
            let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
            for k in 0..out_len {
                re_out[out_base + k * out_stride] = fiber_re[k];
                im_out[out_base + k * out_stride] = fiber_im[k];
            }
        }

        let (re, im) = make_tensors_typed(re_out, im_out, out_shape);
        return (re, im);
    }

    let mut z_re_buf = vec![0.0f32; half.max(1)];
    let mut z_im_buf = vec![0.0f32; half.max(1)];
    let mut fiber_re = vec![0.0f32; out_len];
    let mut fiber_im = vec![0.0f32; out_len];

    for fiber_idx in 0..num_fibers {
        let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        rfft_fiber(
            &data[base_offset..],
            in_stride,
            n,
            sig_len,
            &mut fiber_re,
            &mut fiber_im,
            &tw_half,
            unpack_tw_re,
            unpack_tw_im,
            &mut z_re_buf,
            &mut z_im_buf,
        );

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = fiber_re[k];
            im_out[out_base + k * out_stride] = fiber_im[k];
        }
    }

    let (re, im) = make_tensors_typed(re_out, im_out, out_shape);
    (re, im)
}

mod double;
pub use double::rfft_f64;

pub fn rfft_f16(tensor: HostTensor, dim: usize, n: Option<usize>) -> (HostTensor, HostTensor) {
    use half::f16;
    let tensor = ruda_core::tensor::host::cast::cast_to_f32(tensor, f16::to_f32);
    let (re, im) = rfft_f32(tensor, dim, n);
    (
        ruda_core::tensor::host::cast::cast_from_f32(re, f16::from_f32),
        ruda_core::tensor::host::cast::cast_from_f32(im, f16::from_f32),
    )
}

pub fn rfft_bf16(tensor: HostTensor, dim: usize, n: Option<usize>) -> (HostTensor, HostTensor) {
    use half::bf16;
    let tensor = ruda_core::tensor::host::cast::cast_to_f32(tensor, bf16::to_f32);
    let (re, im) = rfft_f32(tensor, dim, n);
    (
        ruda_core::tensor::host::cast::cast_from_f32(re, bf16::from_f32),
        ruda_core::tensor::host::cast::cast_from_f32(im, bf16::from_f32),
    )
}

// ============================================================================
// Inverse real FFT (irfft)
// ============================================================================

/// Inverse complex FFT: IFFT(X) = (1/N) * conj(FFT(conj(X))).
///
/// Conjugates input, runs the forward FFT (with SIMD), conjugates
/// output, and scales by 1/N.
#[inline]
fn inverse_complex_fft(re: &mut [f32], im: &mut [f32], n: usize, tw: &TwiddleRef) {
    if n <= 1 {
        return;
    }

    // Conjugate input
    for v in im.iter_mut() {
        *v = -*v;
    }

    // Forward FFT (mixed radix-4/radix-2, SIMD when available)
    complex_fft(re, im, n, tw);

    // Conjugate output and scale by 1/N
    let scale = 1.0 / n as f32;
    for v in re.iter_mut() {
        *v *= scale;
    }
    for v in im.iter_mut() {
        *v = -*v * scale;
    }
}

/// Repack N/2+1 spectrum bins into N/2 complex values Z[k],
/// reversing rfft's unpack step.
///
/// Z[k] = Xe[k] + i*conj(W)*D[k] where:
///   Xe = (X[k] + conj(X[half-k])) / 2
///   D  = (X[k] - conj(X[half-k])) / 2
///   W  = W_N^k (same twiddle used in rfft unpack)
fn repack_irfft(
    x_re: &[f32],
    x_im: &[f32],
    half: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
) {
    // k=0: Z[0] = (X[0] + X[half])/2 + i*(X[0] - X[half])/2
    z_re[0] = (x_re[0] + x_re[half]) * 0.5;
    z_im[0] = (x_re[0] - x_re[half]) * 0.5;

    for k in 1..half {
        let j = half - k;
        let (xk_re, xk_im) = (x_re[k], x_im[k]);
        let (xj_re, xj_im) = (x_re[j], x_im[j]);

        // Xe = (X[k] + conj(X[j])) / 2
        let a_re = (xk_re + xj_re) * 0.5;
        let a_im = (xk_im - xj_im) * 0.5;

        // D = (X[k] - conj(X[j])) / 2
        let d_re = (xk_re - xj_re) * 0.5;
        let d_im = (xk_im + xj_im) * 0.5;

        // i*conj(W)*D where W = (wr, wi), conj(W) = (wr, -wi)
        // conj(W)*D = (wr*d_re + wi*d_im, wr*d_im - wi*d_re)
        // i*(...) = (-(wr*d_im - wi*d_re), wr*d_re + wi*d_im)
        let wr = tw_re[k];
        let wi = tw_im[k];

        z_re[k] = a_re - wr * d_im + wi * d_re;
        z_im[k] = a_im + wr * d_re + wi * d_im;
    }
}

/// Process a single irfft fiber via inverse packing trick.
///
/// Repacks N/2+1 spectrum bins into N/2 complex values, runs N/2-point
/// inverse complex FFT, then de-interleaves to N real output values.
#[allow(clippy::too_many_arguments)]
#[inline]
fn irfft_fiber(
    re_in: &[f32],
    im_in: &[f32],
    in_stride: usize,
    half: usize,
    spec_bins: usize,
    signal_out: &mut [f32],
    out_stride: usize,
    tw_half: &TwiddleRef,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
    spec_re: &mut [f32],
    spec_im: &mut [f32],
) {
    if spec_bins > half {
        for k in 0..=half {
            spec_re[k] = re_in[k * in_stride];
            spec_im[k] = im_in[k * in_stride];
        }
    } else {
        for k in 0..=half {
            spec_re[k] = if k < spec_bins {
                re_in[k * in_stride]
            } else {
                0.0
            };
            spec_im[k] = if k < spec_bins {
                im_in[k * in_stride]
            } else {
                0.0
            };
        }
    }

    repack_irfft(
        spec_re,
        spec_im,
        half,
        unpack_tw_re,
        unpack_tw_im,
        z_re,
        z_im,
    );

    // Inverse complex FFT of size N/2
    inverse_complex_fft(z_re, z_im, half, tw_half);

    // De-interleave: signal[2k] = z_re[k], signal[2k+1] = z_im[k]
    if out_stride == 1 {
        for k in 0..half {
            signal_out[2 * k] = z_re[k];
            signal_out[2 * k + 1] = z_im[k];
        }
    } else {
        for k in 0..half {
            signal_out[(2 * k) * out_stride] = z_re[k];
            signal_out[(2 * k + 1) * out_stride] = z_im[k];
        }
    }
}

pub fn irfft_f32(
    spectrum_re: HostTensor,
    spectrum_im: HostTensor,
    dim: usize,
    n: Option<usize>,
) -> HostTensor {
    let spectrum_re = spectrum_re.to_contiguous();
    let spectrum_im = spectrum_im.to_contiguous();
    let shape = spectrum_re.layout().shape().clone();
    assert!(
        *spectrum_im.layout().shape() == shape,
        "irfft: spectrum_re and spectrum_im shapes must match"
    );
    assert!(
        dim < shape.num_dims(),
        "irfft: dim {dim} out of bounds for {}-D tensor",
        shape.num_dims()
    );
    let half_plus_1 = shape[dim];
    assert!(
        half_plus_1 >= 1,
        "irfft: spectrum dimension cannot be empty"
    );

    let spec_bins = half_plus_1;

    let requested_n = n.unwrap_or_else(|| {
        let sig_len = (half_plus_1 - 1) * 2;
        assert!(
            sig_len.is_power_of_two(),
            "irfft: reconstructed signal length must be a power of 2, got {sig_len}"
        );
        sig_len
    });
    let fft_size = requested_n.next_power_of_two();

    // N=1: single DC bin, output is just the real value along `dim`.
    // If caller's spectrum has more bins, take the DC (bin 0) only.
    if fft_size <= 1 {
        let out = if spectrum_re.layout().shape()[dim] != 1 {
            spectrum_re.narrow(dim, 0, 1)
        } else {
            spectrum_re
        };
        return out;
    }

    let half = fft_size / 2;
    let n = fft_size;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = n;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / half_plus_1;

    let re_data: &[f32] = spectrum_re.storage();
    let im_data: &[f32] = spectrum_im.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);

    // Twiddles for N/2-point inverse complex FFT
    let tw_half = get_twiddles(half);

    // Unpack twiddles: last stage of size-N table (same as rfft)
    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_tw_re = &tw_full.re()[last_stage_off..];
    let unpack_tw_im = &tw_full.im()[last_stage_off..];

    let mut signal_out = vec![0.0f32; total_out];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f32>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let re_base = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f32; half.max(1)];
                let mut z_im = vec![0.0f32; half.max(1)];
                let mut spec_re = vec![0.0f32; half + 1];
                let mut spec_im = vec![0.0f32; half + 1];
                let mut fiber_out = vec![0.0f32; n];

                irfft_fiber(
                    &re_data[re_base..],
                    &im_data[re_base..],
                    in_stride,
                    half,
                    spec_bins,
                    &mut fiber_out,
                    1,
                    &tw_half,
                    unpack_tw_re,
                    unpack_tw_im,
                    &mut z_re,
                    &mut z_im,
                    &mut spec_re,
                    &mut spec_im,
                );
                (fiber_idx, fiber_out)
            })
            .collect();

        for (fiber_idx, fiber_out) in fiber_results {
            let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
            for k in 0..n {
                signal_out[out_base + k * out_stride] = fiber_out[k];
            }
        }

        let result = HostTensor::new(
            Bytes::from_elems(signal_out),
            Layout::contiguous(out_shape),
            ruda_core::tensor::DType::F32,
        );
        return if fft_size > requested_n {
            result.narrow(dim, 0, requested_n)
        } else {
            result
        };
    }

    let mut z_re = vec![0.0f32; half.max(1)];
    let mut z_im = vec![0.0f32; half.max(1)];
    let mut spec_re = vec![0.0f32; half + 1];
    let mut spec_im = vec![0.0f32; half + 1];
    let mut fiber_out = vec![0.0f32; n];

    for fiber_idx in 0..num_fibers {
        let re_base = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        irfft_fiber(
            &re_data[re_base..],
            &im_data[re_base..],
            in_stride,
            half,
            spec_bins,
            &mut fiber_out,
            1,
            &tw_half,
            unpack_tw_re,
            unpack_tw_im,
            &mut z_re,
            &mut z_im,
            &mut spec_re,
            &mut spec_im,
        );

        for k in 0..n {
            signal_out[out_base + k * out_stride] = fiber_out[k];
        }
    }

    let result = HostTensor::new(
        Bytes::from_elems(signal_out),
        Layout::contiguous(out_shape),
        ruda_core::tensor::DType::F32,
    );
    if fft_size > requested_n {
        result.narrow(dim, 0, requested_n)
    } else {
        result
    }
}

pub fn irfft_f64(
    spectrum_re: HostTensor,
    spectrum_im: HostTensor,
    dim: usize,
    n: Option<usize>,
) -> HostTensor {
    use ruda_core::tensor::DType;
    match spectrum_re.dtype() {
        DType::F64 => {
            let re_f32 = ruda_core::tensor::host::cast::cast_to_f32::<f64>(spectrum_re, |v| v as f32);
            let im_f32 = ruda_core::tensor::host::cast::cast_to_f32::<f64>(spectrum_im, |v| v as f32);
            let result = irfft_f32(re_f32, im_f32, dim, n);
            ruda_core::tensor::host::cast::cast_from_f32::<f64>(result, |v| v as f64)
        }
        _ => irfft_f32(spectrum_re, spectrum_im, dim, n),
    }
}

pub fn irfft_f16(
    spectrum_re: HostTensor,
    spectrum_im: HostTensor,
    dim: usize,
    n: Option<usize>,
) -> HostTensor {
    use half::f16;
    let re = ruda_core::tensor::host::cast::cast_to_f32(spectrum_re, f16::to_f32);
    let im = ruda_core::tensor::host::cast::cast_to_f32(spectrum_im, f16::to_f32);
    let result = irfft_f32(re, im, dim, n);
    ruda_core::tensor::host::cast::cast_from_f32(result, f16::from_f32)
}

pub fn irfft_bf16(
    spectrum_re: HostTensor,
    spectrum_im: HostTensor,
    dim: usize,
    n: Option<usize>,
) -> HostTensor {
    use half::bf16;
    let re = ruda_core::tensor::host::cast::cast_to_f32(spectrum_re, bf16::to_f32);
    let im = ruda_core::tensor::host::cast::cast_to_f32(spectrum_im, bf16::to_f32);
    let result = irfft_f32(re, im, dim, n);
    ruda_core::tensor::host::cast::cast_from_f32(result, bf16::from_f32)
}

// Tests kept here exercise flex-specific internals: the FFT kernels
// (`rfft_f32`/`_f64`/`_f16`, `irfft_*`, `complex_fft`, `inverse_complex_fft`)
// across sizes that span the radix-4 and complex packing paths (N=1, 2, 4, 8,
// 256, 1024, 4096), f16/f64 dtype handling, twiddle accuracy, Parseval's
// theorem on synthetic inputs, and a reference cross-check against realfft.
#[cfg(test)]
mod tests;
