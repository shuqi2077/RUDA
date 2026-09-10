use super::*;

#[ruda(launch)]
pub(super) fn rfft_fused_kernel<F: Float>(
    signal: &Tensor<F>,
    spectrum_re: &mut Tensor<F>,
    spectrum_im: &mut Tensor<F>,
    num_windows: u32,
    signal_len: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] log2_m: usize,
    #[comptime] threads: usize,
    #[comptime] dim: usize,
) {
    let window = RUDA_POS;
    if window >= num_windows as usize {
        terminate!();
    }
    let input = signal.view(BatchSignalLayout::new(signal, window, dim));
    let mut output_re = spectrum_re.view_mut(BatchSignalLayout::new(spectrum_re, window, dim));
    let mut output_im = spectrum_im.view_mut(BatchSignalLayout::new(spectrum_im, window, dim));
    let mut re = SharedMemory::<F>::new(m);
    let mut im = SharedMemory::<F>::new(m);
    let mut k = UNIT_POS as usize;
    while k < m {
        let even = 2 * k;
        let odd = even + 1;
        let even_active = even < signal_len as usize;
        let odd_active = odd < signal_len as usize;
        let even = select(even_active, even, 0);
        let odd = select(odd_active, odd, 0);
        let dst = bit_reverse(k, log2_m);
        re[dst] = select(even_active, input[even], F::new(0.0));
        im[dst] = select(odd_active, input[odd], F::new(0.0));
        k += threads;
    }
    sync_ruda();
    fft_butterfly_parallel(&mut re, &mut im, m, log2_m, threads, FftMode::Forward);
    let mut k = UNIT_POS as usize;
    while k < m + 1 {
        if k == 0 {
            output_re[k] = re[0] + im[0];
            output_im[k] = F::new(0.0);
        } else if k == m {
            output_re[k] = re[0] - im[0];
            output_im[k] = F::new(0.0);
        } else {
            let a_re = re[k];
            let a_im = im[k];
            let b_re = re[m - k];
            let b_im = -im[m - k];
            let two_pi = F::new(2.0 * PI);
            let theta = -two_pi * F::cast_from(k) / F::cast_from(n_fft);
            let c = theta.cos();
            let s = theta.sin();
            let one_plus_s = F::new(1.0) + s;
            let one_minus_s = F::new(1.0) - s;
            output_re[k] =
                F::new(0.5) * (a_re * one_plus_s + a_im * c + b_re * one_minus_s - b_im * c);
            output_im[k] =
                F::new(0.5) * (a_im * one_plus_s - a_re * c + b_re * c + b_im * one_minus_s);
        }
        k += threads;
    }
}

#[ruda(launch)]
pub(super) fn irfft_fused_kernel<F: Float>(
    spectrum_re: &Tensor<F>,
    spectrum_im: &Tensor<F>,
    signal: &mut Tensor<F>,
    num_windows: u32,
    spec_bins: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] log2_m: usize,
    #[comptime] threads: usize,
    #[comptime] dim: usize,
) {
    let window = RUDA_POS;
    if window >= num_windows as usize {
        terminate!();
    }
    let input_re = spectrum_re.view(BatchSignalLayout::new(spectrum_re, window, dim));
    let input_im = spectrum_im.view(BatchSignalLayout::new(spectrum_im, window, dim));
    let mut output = signal.view_mut(BatchSignalLayout::new(signal, window, dim));
    let mut re = SharedMemory::<F>::new(m);
    let mut im = SharedMemory::<F>::new(m);
    let mut k = UNIT_POS as usize;
    while k < m {
        let dst = bit_reverse(k, log2_m);
        if k == 0 {
            let has_nyquist = m < spec_bins as usize;
            let xm = select(has_nyquist, m, 0);
            let xm_re = select(has_nyquist, input_re[xm], F::new(0.0));
            re[dst] = F::new(0.5) * (input_re[0] + xm_re);
            im[dst] = F::new(0.5) * (input_re[0] - xm_re);
        } else {
            let active = k < spec_bins as usize;
            let src = select(active, k, 0);
            let x_re = select(active, input_re[src], F::new(0.0));
            let x_im = select(active, input_im[src], F::new(0.0));
            let mirror = m - k;
            let mirror_active = mirror < spec_bins as usize;
            let mirror = select(mirror_active, mirror, 0);
            let xm_re = select(mirror_active, input_re[mirror], F::new(0.0));
            let xm_im = -select(mirror_active, input_im[mirror], F::new(0.0));
            let two_pi = F::new(2.0 * PI);
            let theta = two_pi * F::cast_from(k) / F::cast_from(n_fft);
            let c = theta.cos();
            let s = theta.sin();
            let one_plus_s = F::new(1.0) + s;
            let one_minus_s = F::new(1.0) - s;
            re[dst] =
                F::new(0.5) * (x_re * one_minus_s - x_im * c + xm_re * one_plus_s + xm_im * c);
            im[dst] =
                F::new(0.5) * (x_im * one_minus_s + x_re * c - xm_re * c + xm_im * one_plus_s);
        }
        k += threads;
    }
    sync_ruda();
    fft_butterfly_parallel(&mut re, &mut im, m, log2_m, threads, FftMode::Inverse);
    let scale = F::new(1.0) / F::cast_from(m);
    let mut k = UNIT_POS as usize;
    while k < m {
        output[2 * k] = re[k] * scale;
        output[2 * k + 1] = im[k] * scale;
        k += threads;
    }
}

#[ruda(launch)]
pub(super) fn rfft_pack_kernel<F: Float>(
    signal: &Tensor<F>,
    packed_re: &mut Tensor<F>,
    packed_im: &mut Tensor<F>,
    total: u32,
    signal_len: u32,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let signal_view = signal.view(BatchSignalLayout::new(signal, window, dim));
    let mut packed_re_view = packed_re.view_mut(BatchSignalLayout::new(packed_re, window, dim));
    let mut packed_im_view = packed_im.view_mut(BatchSignalLayout::new(packed_im, window, dim));
    let even = 2 * k;
    let odd = even + 1;
    let even_active = even < signal_len as usize;
    let odd_active = odd < signal_len as usize;
    let even = select(even_active, even, 0);
    let odd = select(odd_active, odd, 0);
    packed_re_view[k] = select(even_active, signal_view[even], F::new(0.0));
    packed_im_view[k] = select(odd_active, signal_view[odd], F::new(0.0));
}

#[ruda(launch)]
pub(super) fn rfft_post_kernel<F: Float>(
    packed_re: &Tensor<F>,
    packed_im: &Tensor<F>,
    spectrum_re: &mut Tensor<F>,
    spectrum_im: &mut Tensor<F>,
    total: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let n_freq = comptime![m + 1];
    let k = pos % n_freq;
    let window = pos / n_freq;
    let packed_re_view = packed_re.view(BatchSignalLayout::new(packed_re, window, dim));
    let packed_im_view = packed_im.view(BatchSignalLayout::new(packed_im, window, dim));
    let mut spectrum_re_view =
        spectrum_re.view_mut(BatchSignalLayout::new(spectrum_re, window, dim));
    let mut spectrum_im_view =
        spectrum_im.view_mut(BatchSignalLayout::new(spectrum_im, window, dim));

    if k == 0 {
        let y0_re = packed_re_view[0];
        let y0_im = packed_im_view[0];
        spectrum_re_view[k] = y0_re + y0_im;
        spectrum_im_view[k] = F::new(0.0);
    } else if k == m {
        let y0_re = packed_re_view[0];
        let y0_im = packed_im_view[0];
        spectrum_re_view[k] = y0_re - y0_im;
        spectrum_im_view[k] = F::new(0.0);
    } else {
        let a_re = packed_re_view[k];
        let a_im = packed_im_view[k];
        let b_re = packed_re_view[m - k];
        let b_im_raw = packed_im_view[m - k];
        let b_im = -b_im_raw; // conj(Y[M-k])

        // Forward twiddle W_N^k = cos(-2π k / N) + i sin(-2π k / N).
        let two_pi = F::new(2.0 * PI);
        let theta = -two_pi * F::cast_from(k) / F::cast_from(n_fft);
        let c = theta.cos();
        let s = theta.sin();

        // Precompute reused sums. Derivation:
        //   1 - i*W = (1 + s) - i*c
        //   1 + i*W = (1 - s) + i*c
        //   2 X[k]  = A*(1 - i*W) + B*(1 + i*W)
        let one_plus_s = F::new(1.0) + s;
        let one_minus_s = F::new(1.0) - s;
        let x_re = F::new(0.5) * (a_re * one_plus_s + a_im * c + b_re * one_minus_s - b_im * c);
        let x_im = F::new(0.5) * (a_im * one_plus_s - a_re * c + b_re * c + b_im * one_minus_s);
        spectrum_re_view[k] = x_re;
        spectrum_im_view[k] = x_im;
    }
}

#[ruda(launch)]
pub(super) fn irfft_pre_kernel<F: Float>(
    spectrum_re: &Tensor<F>,
    spectrum_im: &Tensor<F>,
    packed_re: &mut Tensor<F>,
    packed_im: &mut Tensor<F>,
    total: u32,
    spec_bins: u32,
    #[comptime] n_fft: usize,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let spectrum_re_view = spectrum_re.view(BatchSignalLayout::new(spectrum_re, window, dim));
    let spectrum_im_view = spectrum_im.view(BatchSignalLayout::new(spectrum_im, window, dim));
    let mut packed_re_view = packed_re.view_mut(BatchSignalLayout::new(packed_re, window, dim));
    let mut packed_im_view = packed_im.view_mut(BatchSignalLayout::new(packed_im, window, dim));

    if k == 0 {
        let has_nyquist = m < spec_bins as usize;
        let x0_re = spectrum_re_view[0];
        let xm = select(has_nyquist, m, 0);
        let xm_re = select(has_nyquist, spectrum_re_view[xm], F::new(0.0));
        packed_re_view[k] = F::new(0.5) * (x0_re + xm_re);
        packed_im_view[k] = F::new(0.5) * (x0_re - xm_re);
    } else {
        let active = k < spec_bins as usize;
        let src = select(active, k, 0);
        let x_re = select(active, spectrum_re_view[src], F::new(0.0));
        let x_im = select(active, spectrum_im_view[src], F::new(0.0));
        let mirror = m - k;
        let mirror_active = mirror < spec_bins as usize;
        let mirror = select(mirror_active, mirror, 0);
        let xm_re = select(mirror_active, spectrum_re_view[mirror], F::new(0.0));
        let xm_im_raw = select(mirror_active, spectrum_im_view[mirror], F::new(0.0));
        let xm_im = -xm_im_raw; // conj(X[M-k])

        // Inverse twiddle W_N^{-k} = cos(2π k / N) + i sin(2π k / N).
        let two_pi = F::new(2.0 * PI);
        let theta = two_pi * F::cast_from(k) / F::cast_from(n_fft);
        let c = theta.cos();
        let s = theta.sin();

        // Derivation (inverse post). Let W = W_N^{-k} = c + i*s.
        //   1 + i*W = (1 - s) + i*c        → A * (1 + i*W):
        //     Re = x_re*(1-s) - x_im*c
        //     Im = x_re*c    + x_im*(1-s)
        //   1 - i*W = (1 + s) - i*c        → B * (1 - i*W), B = conj(X[M-k]):
        //     Re = xm_re*(1+s) + xm_im*c   (xm_im is already negated here)
        //     Im = -xm_re*c   + xm_im*(1+s)
        //   2 Y[k] = A*(1 + i*W) + B*(1 - i*W).
        let one_plus_s = F::new(1.0) + s;
        let one_minus_s = F::new(1.0) - s;
        let y_re = F::new(0.5) * (x_re * one_minus_s - x_im * c + xm_re * one_plus_s + xm_im * c);
        let y_im = F::new(0.5) * (x_im * one_minus_s + x_re * c - xm_re * c + xm_im * one_plus_s);
        packed_re_view[k] = y_re;
        packed_im_view[k] = y_im;
    }
}

#[ruda(launch)]
pub(super) fn irfft_unpack_kernel<F: Float>(
    packed_re: &Tensor<F>,
    packed_im: &Tensor<F>,
    signal: &mut Tensor<F>,
    total: u32,
    #[comptime] m: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }
    let k = pos % m;
    let window = pos / m;
    let packed_re_view = packed_re.view(BatchSignalLayout::new(packed_re, window, dim));
    let packed_im_view = packed_im.view(BatchSignalLayout::new(packed_im, window, dim));
    let mut signal_view = signal.view_mut(BatchSignalLayout::new(signal, window, dim));
    let scale = F::new(1.0) / F::cast_from(m);
    signal_view[2 * k] = packed_re_view[k] * scale;
    signal_view[2 * k + 1] = packed_im_view[k] * scale;
}
