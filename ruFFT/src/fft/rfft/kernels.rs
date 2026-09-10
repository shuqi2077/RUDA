use super::*;

#[ruda(launch)]
pub(super) fn rfft_kernel<F: Float>(
    signal: &Tensor<F>,
    spectrum_re: &mut Tensor<F>,
    spectrum_im: &mut Tensor<F>,
    num_windows: u32,
    signal_len: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_ruda: usize,
    #[comptime] dim: usize,
) {
    let window_index = RUDA_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let signal_view = signal.view(BatchSignalLayout::new(signal, window_index, dim));
    let mut spectrum_re_view =
        spectrum_re.view_mut(BatchSignalLayout::new(spectrum_re, window_index, dim));
    let mut spectrum_im_view =
        spectrum_im.view_mut(BatchSignalLayout::new(spectrum_im, window_index, dim));

    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);

    let mut i = UNIT_POS as usize;
    while i < n_fft {
        let j = bit_reverse(i, log2_n);
        let active = i < signal_len as usize;
        let src = select(active, i, 0);
        shared_re[j] = select(active, signal_view[src], F::new(0.0));
        shared_im[j] = F::new(0.0);
        i += threads_per_ruda;
    }
    sync_ruda();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_ruda,
        FftMode::Forward,
    );

    let n_freq = comptime![n_fft / 2 + 1];
    let mut k = UNIT_POS as usize;
    while k < n_freq {
        spectrum_re_view[k] = shared_re[k];
        spectrum_im_view[k] = shared_im[k];
        k += threads_per_ruda;
    }
    sync_ruda();
}
