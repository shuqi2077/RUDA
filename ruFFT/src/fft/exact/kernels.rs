use super::*;
use crate::layout::BatchSignalLayout;
use ruda_kernel::library::tensor::{AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand};

#[ruda(launch)]
pub(super) fn make_chirp(
    chirp_re: &mut Tensor<f32>, chirp_im: &mut Tensor<f32>,
    b_re: &mut Tensor<f32>, b_im: &mut Tensor<f32>,
    #[comptime] n: usize, #[comptime] m: usize, #[comptime] mode: FftMode,
) {
    let k = ABSOLUTE_POS;
    if k >= m { terminate!(); }
    let sign = comptime![mode.sign()];
    // Reduce the integer square modulo 2N *before* converting to f32. A
    // direct f32(k*k) loses phase accuracy even for moderate transform sizes.
    let j = if k < n { k } else { m - k };
    if j < n {
        let square = (j as u64 * j as u64) % (2u64 * n as u64);
        let phase = sign * core::f32::consts::PI * (square as f32 / n as f32);
        let re = phase.cos();
        let im = phase.sin();
        b_re[k] = re;
        b_im[k] = -im;
        if k < n { chirp_re[k] = re; chirp_im[k] = im; }
    } else { b_re[k] = 0.0; b_im[k] = 0.0; }
}

#[ruda(launch)]
pub(super) fn prepare_forward(
    input: &Tensor<f32>, chirp_re: &Tensor<f32>, chirp_im: &Tensor<f32>,
    work_re: &mut Tensor<f32>, work_im: &mut Tensor<f32>, total: u32, used: u32,
    #[comptime] n: usize, #[comptime] m: usize, #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize { terminate!(); }
    let k = pos % m;
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    if k < n && k < used as usize {
        let view = input.view(BatchSignalLayout::new(input, pos / m, dim));
        let x = view[k];
        re = x * chirp_re[k]; im = x * chirp_im[k];
    }
    work_re[pos] = re; work_im[pos] = im;
}

#[ruda(launch)]
pub(super) fn prepare_inverse(
    input_re: &Tensor<f32>, input_im: &Tensor<f32>,
    chirp_re: &Tensor<f32>, chirp_im: &Tensor<f32>,
    work_re: &mut Tensor<f32>, work_im: &mut Tensor<f32>, total: u32, used: u32,
    #[comptime] n: usize, #[comptime] m: usize, #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize { terminate!(); }
    let k = pos % m;
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    if k < n {
        let bin = if k <= n / 2 { k } else { n - k };
        if bin < used as usize {
            let rv = input_re.view(BatchSignalLayout::new(input_re, pos / m, dim));
            let iv = input_im.view(BatchSignalLayout::new(input_im, pos / m, dim));
            let xr = rv[bin];
            let mut xi = 0.0f32;
            // Never read discarded imaginary DC/Nyquist values, including NaN.
            if bin != 0 && !(n % 2 == 0 && bin == n / 2) {
                xi = if k <= n / 2 { iv[bin] } else { -iv[bin] };
            }
            let cr = chirp_re[k]; let ci = chirp_im[k];
            re = xr * cr - xi * ci; im = xr * ci + xi * cr;
        }
    }
    work_re[pos] = re; work_im[pos] = im;
}

#[ruda(launch)]
pub(super) fn multiply_spectrum(
    re: &mut Tensor<f32>, im: &mut Tensor<f32>,
    b_re: &Tensor<f32>, b_im: &Tensor<f32>, total: u32, #[comptime] m: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize { terminate!(); }
    let k = pos % m;
    let ar = re[pos]; let ai = im[pos]; let br = b_re[k]; let bi = b_im[k];
    re[pos] = ar * br - ai * bi; im[pos] = ar * bi + ai * br;
}

#[ruda(launch)]
pub(super) fn finish_forward(
    re: &Tensor<f32>, im: &Tensor<f32>, cr: &Tensor<f32>, ci: &Tensor<f32>,
    output_re: &mut Tensor<f32>, output_im: &mut Tensor<f32>, total: u32,
    #[comptime] n: usize, #[comptime] m: usize, #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize { terminate!(); }
    let bins = comptime![n / 2 + 1];
    let batch = pos / bins; let k = pos % bins; let src = batch * m + k;
    let mut rv = output_re.view_mut(BatchSignalLayout::new(output_re, batch, dim));
    let mut iv = output_im.view_mut(BatchSignalLayout::new(output_im, batch, dim));
    rv[k] = (re[src] * cr[k] - im[src] * ci[k]) / m as f32;
    if k == 0 || (n % 2 == 0 && k == n / 2) { iv[k] = 0.0; }
    else { iv[k] = (re[src] * ci[k] + im[src] * cr[k]) / m as f32; }
}

#[ruda(launch)]
pub(super) fn finish_inverse(
    re: &Tensor<f32>, im: &Tensor<f32>, cr: &Tensor<f32>, ci: &Tensor<f32>,
    output: &mut Tensor<f32>, total: u32,
    #[comptime] n: usize, #[comptime] m: usize, #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize { terminate!(); }
    let batch = pos / n; let k = pos % n;
    let src = batch * m + k;
    let mut out = output.view_mut(BatchSignalLayout::new(output, batch, dim));
    // Sequential division avoids an overflow-prone integer M*N intermediate.
    out[k] = (re[src] * cr[k] - im[src] * ci[k]) / m as f32 / n as f32;
}

#[ruda(launch)]
pub(super) fn scalar_forward(input: &Tensor<f32>, re: &mut Tensor<f32>, im: &mut Tensor<f32>,
    rows: u32, used: u32, #[comptime] dim: usize) {
    let batch = ABSOLUTE_POS;
    if batch >= rows as usize { terminate!(); }
    let mut rv = re.view_mut(BatchSignalLayout::new(re, batch, dim));
    let mut iv = im.view_mut(BatchSignalLayout::new(im, batch, dim));
    let mut value = 0.0f32;
    if used != 0 { let x = input.view(BatchSignalLayout::new(input, batch, dim)); value = x[0]; }
    rv[0] = value; iv[0] = 0.0;
}

#[ruda(launch)]
pub(super) fn scalar_inverse(re: &Tensor<f32>, output: &mut Tensor<f32>,
    rows: u32, used: u32, #[comptime] dim: usize) {
    let batch = ABSOLUTE_POS;
    if batch >= rows as usize { terminate!(); }
    let mut out = output.view_mut(BatchSignalLayout::new(output, batch, dim));
    let mut value = 0.0f32;
    if used != 0 { let x = re.view(BatchSignalLayout::new(re, batch, dim)); value = x[0]; }
    out[0] = value;
}
