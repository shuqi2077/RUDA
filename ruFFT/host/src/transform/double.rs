use super::*;
use alloc::sync::Arc;

struct Twiddles64 {
    re: Vec<f64>,
    im: Vec<f64>,
    offsets: Vec<usize>,
}

fn get_twiddles64(n: usize) -> Arc<Twiddles64> {
    #[cfg(feature = "std")]
    let mut cache = {
        static CACHE: std::sync::Mutex<Option<Arc<Twiddles64>>> = std::sync::Mutex::new(None);
        let cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(table) = cache.as_ref() {
            if table.re.len() >= n.saturating_sub(1) {
                return Arc::clone(table);
            }
        }
        cache
    };
    let mut table = Twiddles64 {
        re: Vec::with_capacity(n.saturating_sub(1)),
        im: Vec::with_capacity(n.saturating_sub(1)),
        offsets: vec![0],
    };
    for stage in 1..=n.trailing_zeros() {
        let len = 1usize << stage;
        for k in 0..len / 2 {
            let angle = -core::f64::consts::TAU * k as f64 / len as f64;
            let (sin, cos) = libm::sincos(angle);
            table.re.push(cos);
            table.im.push(sin);
        }
        table.offsets.push(table.re.len());
    }
    let table = Arc::new(table);
    #[cfg(feature = "std")]
    {
        *cache = Some(Arc::clone(&table));
    }
    table
}

#[allow(clippy::too_many_arguments)]
fn rfft_fiber_f64(
    signal: &[f64],
    in_stride: usize,
    n: usize,
    sig_len: usize,
    half: usize,
    out_re: &mut [f64],
    out_im: &mut [f64],
    tw_re: &[f64],
    tw_im: &[f64],
    tw_offsets: &[usize],
    unpack_re: &[f64],
    unpack_im: &[f64],
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

        let wr = unpack_re[k];
        let wi = unpack_im[k];

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

    let tw_full = get_twiddles64(n);
    let stages = n.trailing_zeros() as usize;
    let last_stage_off = tw_full.offsets[stages.saturating_sub(1)];
    let unpack_re = &tw_full.re[last_stage_off..];
    let unpack_im = &tw_full.im[last_stage_off..];

    let mut re_out = vec![0.0f64; total_out];
    let mut im_out = vec![0.0f64; total_out];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    let tw_half_re = &tw_full.re;
    let tw_half_im = &tw_full.im;
    let tw_half_offsets = &tw_full.offsets[..stages.max(1)];

    if out_stride == 1 {
        for_each_contiguous_pair(
            &mut re_out,
            &mut im_out,
            out_len,
            num_fibers >= 4 && n >= 64,
            || (vec![0.0; half.max(1)], vec![0.0; half.max(1)]),
            |(z_re, z_im), fiber_idx, re, im| {
                rfft_fiber_f64(
                    &data[fiber_idx * shape[dim]..],
                    1,
                    n,
                    sig_len,
                    half,
                    re,
                    im,
                    tw_half_re,
                    tw_half_im,
                    tw_half_offsets,
                    unpack_re,
                    unpack_im,
                    z_re,
                    z_im,
                );
            },
        );
        return make_tensors_typed(re_out, im_out, out_shape);
    }

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

pub fn irfft_f64(
    spectrum_re: HostTensor,
    spectrum_im: HostTensor,
    dim: usize,
    n: Option<usize>,
) -> HostTensor {
    use ruda_core::tensor::DType;
    if spectrum_re.dtype() != DType::F64 {
        return irfft_f32(spectrum_re, spectrum_im, dim, n);
    }
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
    let spec_bins = shape[dim];
    assert!(spec_bins >= 1, "irfft: spectrum dimension cannot be empty");
    let requested_n = n.unwrap_or_else(|| {
        let sig_len = (spec_bins - 1) * 2;
        assert!(
            sig_len.is_power_of_two(),
            "irfft: reconstructed signal length must be a power of 2, got {sig_len}"
        );
        sig_len
    });
    let n = requested_n.next_power_of_two();
    if n <= 1 {
        return if spec_bins != 1 {
            spectrum_re.narrow(dim, 0, 1)
        } else {
            spectrum_re
        };
    }
    let half = n / 2;
    let mut out_dims = shape.as_slice().to_vec();
    out_dims[dim] = n;
    let out_shape = Shape::from(out_dims);
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];
    let num_fibers = shape.num_elements() / spec_bins;
    let re: &[f64] = spectrum_re.storage();
    let im: &[f64] = spectrum_im.storage();
    let tw = get_twiddles64(n);
    let stages = n.trailing_zeros() as usize;
    let unpack_offset = tw.offsets[stages - 1];
    let init = || (vec![0.0f64; half], vec![0.0f64; half]);
    let run = |(z_re, z_im): &mut (Vec<f64>, Vec<f64>),
               fiber_idx: usize,
               output: &mut [f64],
               stride: usize| {
        let base = if in_stride == 1 {
            fiber_idx * spec_bins
        } else {
            slice_base_offset(fiber_idx, &shape, &in_strides, dim)
        };
        let bin = |k: usize| {
            if k < spec_bins {
                (re[base + k * in_stride], im[base + k * in_stride])
            } else {
                (0.0, 0.0)
            }
        };
        let dc = re[base];
        let nyquist = bin(half).0;
        z_re[0] = (dc + nyquist) * 0.5;
        z_im[0] = (nyquist - dc) * 0.5;
        for k in 1..half {
            let (xkr, xki) = bin(k);
            let (xjr, xji) = bin(half - k);
            let ar = (xkr + xjr) * 0.5;
            let ai = (xki - xji) * 0.5;
            let dr = (xkr - xjr) * 0.5;
            let di = (xki + xji) * 0.5;
            let wr = tw.re[unpack_offset + k];
            let wi = tw.im[unpack_offset + k];
            z_re[k] = ar - wr * di + wi * dr;
            z_im[k] = -(ai + wr * dr + wi * di);
        }
        fft_f64_inplace(z_re, z_im, half, &tw.re, &tw.im, &tw.offsets[..stages]);
        let scale = 1.0 / half as f64;
        for k in 0..half {
            output[2 * k * stride] = z_re[k] * scale;
            output[(2 * k + 1) * stride] = -z_im[k] * scale;
        }
    };
    let mut output = vec![0.0f64; out_shape.num_elements()];
    if out_stride == 1 {
        for_each_contiguous(
            &mut output,
            n,
            num_fibers >= 4 && n >= 64,
            init,
            |scratch, i, out| run(scratch, i, out, 1),
        );
    } else {
        #[cfg(feature = "rayon")]
        let parallel = num_fibers >= 4 && n >= 64;
        #[cfg(not(feature = "rayon"))]
        let parallel = false;
        #[cfg(feature = "rayon")]
        if parallel {
            use rayon::prelude::*;
            let results: Vec<Vec<f64>> = (0..num_fibers)
                .into_par_iter()
                .map_init(init, |scratch, i| {
                    let mut out = vec![0.0; n];
                    run(scratch, i, &mut out, 1);
                    out
                })
                .collect();
            for (i, fiber) in results.iter().enumerate() {
                let base = slice_base_offset(i, &out_shape, &out_strides, dim);
                for k in 0..n {
                    output[base + k * out_stride] = fiber[k];
                }
            }
        }
        if !parallel {
            let mut scratch = init();
            for i in 0..num_fibers {
                let base = slice_base_offset(i, &out_shape, &out_strides, dim);
                run(&mut scratch, i, &mut output[base..], out_stride);
            }
        }
    }
    let result = HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::F64,
    );
    if n > requested_n {
        result.narrow(dim, 0, requested_n)
    } else {
        result
    }
}

fn fft_f64_inplace(
    re: &mut [f64],
    im: &mut [f64],
    n: usize,
    tw_re: &[f64],
    tw_im: &[f64],
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
                let wr = tw_re[tw_off + k];
                let wi = tw_im[tw_off + k];
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
