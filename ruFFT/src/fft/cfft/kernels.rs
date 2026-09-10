use super::*;

#[ruda(launch)]
pub(super) fn cfft_shared_kernel<F: Float>(
    input_re: &Tensor<F>,
    input_im: &Tensor<F>,
    output_re: &mut Tensor<F>,
    output_im: &mut Tensor<F>,
    num_windows: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_ruda: usize,
    #[comptime] dim: usize,
    #[comptime] fft_mode: FftMode,
) {
    let window_index = RUDA_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let input_re_view = input_re.view(BatchSignalLayout::new(input_re, window_index, dim));
    let input_im_view = input_im.view(BatchSignalLayout::new(input_im, window_index, dim));
    let mut output_re_view =
        output_re.view_mut(BatchSignalLayout::new(output_re, window_index, dim));
    let mut output_im_view =
        output_im.view_mut(BatchSignalLayout::new(output_im, window_index, dim));

    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);

    let mut i = UNIT_POS as usize;
    while i < n_fft {
        let j = bit_reverse(i, log2_n);
        shared_re[j] = input_re_view[i];
        shared_im[j] = input_im_view[i];
        i += threads_per_ruda;
    }
    sync_ruda();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_ruda,
        fft_mode,
    );

    let mut k = UNIT_POS as usize;
    while k < n_fft {
        output_re_view[k] = shared_re[k];
        output_im_view[k] = shared_im[k];
        k += threads_per_ruda;
    }
}

#[ruda(launch)]
pub(super) fn cfft_four_step_radix1_kernel<F: Float>(
    input_re: &Tensor<F>,
    input_im: &Tensor<F>,
    scratch_re: &mut Tensor<F>,
    scratch_im: &mut Tensor<F>,
    num_rudas: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] log2_n1: usize,
    #[comptime] threads_per_ruda: usize,
    #[comptime] dim: usize,
    #[comptime] fft_mode: FftMode,
) {
    let ruda_pos = RUDA_POS;
    if ruda_pos >= num_rudas as usize {
        terminate!();
    }

    let window = ruda_pos / n2;
    let n2_idx = ruda_pos - window * n2;
    let input_re_view = input_re.view(BatchSignalLayout::new(input_re, window, dim));
    let input_im_view = input_im.view(BatchSignalLayout::new(input_im, window, dim));
    let mut scratch_re_view = scratch_re.view_mut(BatchSignalLayout::new(scratch_re, window, dim));
    let mut scratch_im_view = scratch_im.view_mut(BatchSignalLayout::new(scratch_im, window, dim));

    let mut shared_re = SharedMemory::<F>::new(n1);
    let mut shared_im = SharedMemory::<F>::new(n1);

    // Load x[window, n1, n2] at bit-reversed destinations so the butterfly
    // can run directly without a pre-permute pass.
    let mut i = UNIT_POS as usize;
    while i < n1 {
        let j = bit_reverse(i, log2_n1);
        let flat = i * n2 + n2_idx;
        shared_re[j] = input_re_view[flat];
        shared_im[j] = input_im_view[flat];
        i += threads_per_ruda;
    }
    sync_ruda();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n1,
        log2_n1,
        threads_per_ruda,
        fft_mode,
    );

    // Post-twiddle and strided write. W_N^{k1 * n2} with the total-N phase
    // factor is the Cooley-Tukey tying factor between the two sub-FFTs.
    let sign = F::new(fft_mode.sign());
    let n_total = comptime![n1 * n2];
    let two_pi = F::new(2.0 * PI);

    let mut k1 = UNIT_POS as usize;
    while k1 < n1 {
        let theta = sign * two_pi * F::cast_from(k1 * n2_idx) / F::cast_from(n_total);
        let w_re = theta.cos();
        let w_im = theta.sin();
        let ar = shared_re[k1];
        let ai = shared_im[k1];
        let flat = k1 * n2 + n2_idx;
        scratch_re_view[flat] = w_re * ar - w_im * ai;
        scratch_im_view[flat] = w_re * ai + w_im * ar;
        k1 += threads_per_ruda;
    }
}

#[ruda(launch)]
pub(super) fn cfft_four_step_radix2_kernel<F: Float>(
    scratch_re: &mut Tensor<F>,
    scratch_im: &mut Tensor<F>,
    num_rudas: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] log2_n2: usize,
    #[comptime] threads_per_ruda: usize,
    #[comptime] dim: usize,
    #[comptime] fft_mode: FftMode,
) {
    let ruda_pos = RUDA_POS;
    if ruda_pos >= num_rudas as usize {
        terminate!();
    }

    let window = ruda_pos / n1;
    let k1 = ruda_pos - window * n1;
    let row_base = k1 * n2;
    let mut scratch_re_view = scratch_re.view_mut(BatchSignalLayout::new(scratch_re, window, dim));
    let mut scratch_im_view = scratch_im.view_mut(BatchSignalLayout::new(scratch_im, window, dim));

    let mut shared_re = SharedMemory::<F>::new(n2);
    let mut shared_im = SharedMemory::<F>::new(n2);

    let mut i = UNIT_POS as usize;
    while i < n2 {
        let j = bit_reverse(i, log2_n2);
        shared_re[j] = scratch_re_view[row_base + i];
        shared_im[j] = scratch_im_view[row_base + i];
        i += threads_per_ruda;
    }
    sync_ruda();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n2,
        log2_n2,
        threads_per_ruda,
        fft_mode,
    );

    let mut k2 = UNIT_POS as usize;
    while k2 < n2 {
        scratch_re_view[row_base + k2] = shared_re[k2];
        scratch_im_view[row_base + k2] = shared_im[k2];
        k2 += threads_per_ruda;
    }
}

#[ruda(launch)]
pub(super) fn cfft_four_step_transpose_kernel<F: Float>(
    scratch_re: &Tensor<F>,
    scratch_im: &Tensor<F>,
    output_re: &mut Tensor<F>,
    output_im: &mut Tensor<F>,
    total: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }

    let m = comptime![n1 * n2];
    let pos_u = pos;
    let inner = pos_u % m;
    let window = pos_u / m;
    let scratch_re_view = scratch_re.view(BatchSignalLayout::new(scratch_re, window, dim));
    let scratch_im_view = scratch_im.view(BatchSignalLayout::new(scratch_im, window, dim));
    let mut output_re_view = output_re.view_mut(BatchSignalLayout::new(output_re, window, dim));
    let mut output_im_view = output_im.view_mut(BatchSignalLayout::new(output_im, window, dim));
    // pos's inner index is the destination linear index k = k1 + k2 * N1.
    let k2 = inner / n1;
    let k1 = inner - k2 * n1;
    let src = k1 * n2 + k2;

    output_re_view[inner] = scratch_re_view[src];
    output_im_view[inner] = scratch_im_view[src];
}
