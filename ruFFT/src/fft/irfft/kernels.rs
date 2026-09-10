use super::*;

#[ruda(launch)]
pub(super) fn irfft_kernel<F: Float>(
    spectrum_re: &Tensor<F>,
    spectrum_im: &Tensor<F>,
    signal: &mut Tensor<F>,
    num_windows: u32,
    spec_bins: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_ruda: usize,
    #[comptime] dim: usize,
) {
    let window_index = RUDA_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let spectrum_re_view = spectrum_re.view(BatchSignalLayout::new(spectrum_re, window_index, dim));
    let spectrum_im_view = spectrum_im.view(BatchSignalLayout::new(spectrum_im, window_index, dim));
    let mut signal_view = signal.view_mut(BatchSignalLayout::new(signal, window_index, dim));

    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);

    let n_freq = comptime![n_fft / 2 + 1];

    let mut k = UNIT_POS as usize;
    while k < n_fft {
        let dst = bit_reverse(k, log2_n);
        let src_bin = select(k < n_freq, k, n_fft - k);
        let active = src_bin < spec_bins as usize;
        let src_bin = select(active, src_bin, 0);
        let im_sign = select(k < n_freq, F::new(1.0), F::new(-1.0));
        shared_re[dst] = select(active, spectrum_re_view[src_bin], F::new(0.0));
        shared_im[dst] = select(active, spectrum_im_view[src_bin] * im_sign, F::new(0.0));
        k += threads_per_ruda;
    }
    sync_ruda();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_ruda,
        FftMode::Inverse,
    );

    let scale = F::new(1.0) / F::cast_from(n_fft);
    let mut i = UNIT_POS as usize;
    while i < n_fft {
        signal_view[i] = shared_re[i] * scale;
        i += threads_per_ruda;
    }
    sync_ruda();
}
