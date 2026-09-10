use super::*;

#[allow(clippy::too_many_arguments)]
fn rfft_fiber_f64(
    signal: &[f64],
    in_stride: usize,
    n: usize,
    sig_len: usize,
    half: usize,
    out_re: &mut [f64],
    out_im: &mut [f64],
    tw_re: &[f32],
    tw_im: &[f32],
    tw_offsets: &[usize],
    unpack_re: &[f32],
    unpack_im: &[f32],
    z_re: &mut [f64],
    z_im: &mut [f64],
) {
    if n == 1 {
        out_re[0] = if sig_len >= 1 { signal[0] } else { 0.0 };
        out_im[0] = 0.0;
        return;
    }

    if sig_len >= n {
        for k in 0..half {
            z_re[k] = signal[(2 * k) * in_stride];
            z_im[k] = signal[(2 * k + 1) * in_stride];
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

    fft_f64_inplace(z_re, z_im, half, tw_re, tw_im, tw_offsets);

    out_re[0] = z_re[0] + z_im[0];
    out_im[0] = 0.0;
    out_re[half] = z_re[0] - z_im[0];
    out_im[half] = 0.0;

    for k in 1..half {
        let j = half - k;
        let (zk_re, zk_im) = (z_re[k], z_im[k]);
        let (zj_re, zj_im) = (z_re[j], z_im[j]);

        let xe_re = (zk_re + zj_re) * 0.5;
        let xe_im = (zk_im - zj_im) * 0.5;
        let xo_re = (zk_im + zj_im) * 0.5;
        let xo_im = (zj_re - zk_re) * 0.5;

        let wr = unpack_re[k] as f64;
        let wi = unpack_im[k] as f64;

        out_re[k] = xe_re + wr * xo_re - wi * xo_im;
        out_im[k] = xe_im + wr * xo_im + wi * xo_re;
    }
}

pub fn rfft_f64(tensor: HostTensor, dim: usize, n: Option<usize>) -> (HostTensor, HostTensor) {
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

    let data: &[f64] = tensor.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);
    let half = n / 2;

    let tw_half = get_twiddles(half);
    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_re = &tw_full.re()[last_stage_off..];
    let unpack_im = &tw_full.im()[last_stage_off..];

    let mut re_out = vec![0.0f64; total_out];
    let mut im_out = vec![0.0f64; total_out];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    let tw_half_re = tw_half.re();
    let tw_half_im = tw_half.im();
    let tw_half_offsets = tw_half.offsets();

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f64; half.max(1)];
                let mut z_im = vec![0.0f64; half.max(1)];
                let mut fiber_re = vec![0.0f64; out_len];
                let mut fiber_im = vec![0.0f64; out_len];

                rfft_fiber_f64(
                    &data[base_offset..],
                    in_stride,
                    n,
                    sig_len,
                    half,
                    &mut fiber_re,
                    &mut fiber_im,
                    tw_half_re,
                    tw_half_im,
                    tw_half_offsets,
                    unpack_re,
                    unpack_im,
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

    let mut z_re = vec![0.0f64; half.max(1)];
    let mut z_im = vec![0.0f64; half.max(1)];
    let mut fiber_re = vec![0.0f64; out_len];
    let mut fiber_im = vec![0.0f64; out_len];

    for fiber_idx in 0..num_fibers {
        let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        rfft_fiber_f64(
            &data[base_offset..],
            in_stride,
            n,
            sig_len,
            half,
            &mut fiber_re,
            &mut fiber_im,
            tw_half_re,
            tw_half_im,
            tw_half_offsets,
            unpack_re,
            unpack_im,
            &mut z_re,
            &mut z_im,
        );

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = fiber_re[k];
            im_out[out_base + k * out_stride] = fiber_im[k];
        }
    }

    let (re, im) = make_tensors_typed(re_out, im_out, out_shape);
    (re, im)
}

/// f64 complex FFT using f32 twiddle table (widened in inner loop).
/// Twiddle precision is limited to ~7 digits (f32), so output accuracy
/// is below full f64 precision for large N.
fn fft_f64_inplace(
    re: &mut [f64],
    im: &mut [f64],
    n: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    offsets: &[usize],
) {
    if n <= 1 {
        return;
    }

    // Bit-reversal
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

    // Scalar radix-2 passes
    let num_stages = offsets.len() - 1;
    let mut len = 2;
    for &tw_off in &offsets[..num_stages] {
        let half = len / 2;
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let wr = tw_re[tw_off + k] as f64;
                let wi = tw_im[tw_off + k] as f64;
                let even = start + k;
                let odd = even + half;
                let t_re = wr * re[odd] - wi * im[odd];
                let t_im = wr * im[odd] + wi * re[odd];
                re[odd] = re[even] - t_re;
                im[odd] = im[even] - t_im;
                re[even] += t_re;
                im[even] += t_im;
            }
            start += len;
        }
        len <<= 1;
    }
}

