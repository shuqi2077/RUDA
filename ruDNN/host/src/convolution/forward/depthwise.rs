use super::*;

// ============================================================================
// Depthwise conv fast path (no NHWC, no im2col, no gemm)
// ============================================================================
//
// For canonical depthwise convolutions where every input channel maps to
// exactly one output channel through its own filter (groups == c_in == c_out,
// channels_per_group == 1), the generic im2col + per-group gemm path pays
// enormous overhead:
//
//   - Per (tile, group) heap allocation of a col_tile buffer.
//   - Per (tile, group) gemm dispatch with M=1, K=kernel_spatial, N=tile_size.
//     The gemm kernel's setup cost dominates the tiny inner compute.
//   - Full NHWC conversion of the input even though no channel mixing occurs.
//
// The depthwise path drops all three. It walks NCHW data directly and, for
// each (batch, channel), accumulates the convolution by looping over kernel
// positions on the outside and spatial positions on the inside. The valid
// output range for each kernel position is computed analytically so the
// inner spatial loop has no padding checks and LLVM autovectorizes it in
// the common stride=1, dilation=1 case.

/// Decide whether to use the depthwise fast path.
///
/// Triggers on canonical depthwise 1D or 2D convolutions (restricted to
/// `kd == 1 && in_d == 1` after the 3D expansion used by conv1d/conv2d).
pub(super) fn should_use_depthwise_conv(
    x_shape: &[usize],
    w_shape: &[usize],
    options: &ConvOptions<3>,
) -> bool {
    let channels_in = x_shape[1];
    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let groups = options.groups;

    // Canonical depthwise: one input channel per group, one output channel per group.
    if channels_per_group != 1 || groups != channels_in || channels_out != channels_in {
        return false;
    }

    // Only the 1D/2D-in-3D shapes (kd=1, in_d=1) produced by the conv1d/conv2d
    // expansion. Pure 3D depthwise would need additional loop nesting.
    if w_shape[2] != 1 || x_shape[2] != 1 {
        return false;
    }
    if options.stride[0] != 1 || options.padding[0] != 0 || options.dilation[0] != 1 {
        return false;
    }

    true
}

/// Compute the half-open range `[out_start, out_end)` of output positions `o`
/// for which the corresponding input index `o * stride + k * dilation - pad`
/// lies inside `[0, in_size)`.
#[inline]
pub(super) fn valid_out_range(
    k: usize,
    dilation: usize,
    pad: usize,
    stride: usize,
    in_size: usize,
    out_size: usize,
) -> (usize, usize) {
    debug_assert!(stride >= 1, "stride must be >= 1");
    let offset = k * dilation;

    // Lower: smallest o with o*stride + offset >= pad.
    let out_start = if offset >= pad {
        0
    } else {
        (pad - offset).div_ceil(stride)
    };

    // Upper (exclusive): smallest o with o*stride + offset - pad >= in_size,
    // i.e. smallest o with o*stride >= in_size + pad - offset.
    let threshold = in_size + pad;
    let out_end = if offset >= threshold {
        0
    } else {
        (threshold - offset).div_ceil(stride)
    };

    let out_end = out_end.min(out_size);
    let out_start = out_start.min(out_end);
    (out_start, out_end)
}

/// Output-plane element count above which `conv_plane_accumulate` switches
/// from the `kh, kw, oh, ow` "kh-outer" loop order to `oh, kh, kw, ow`
/// "oh-outer".
///
/// Below the threshold the whole plane fits comfortably in L1 (~32 KB on
/// most modern CPUs), so the hardware prefetcher tracks the regular
/// `oh`-stride access and the kh-outer order amortizes per-kernel-position
/// setup over long inner runs. Above the threshold the plane no longer
/// fits, and the kh-outer order refetches the output from L2/DRAM once
/// per `(kh, kw)` pair: a `kh * kw`-fold amplification of output memory
/// traffic. Flipping to oh-outer pins one output row in L1 across every
/// kernel position and traverses the plane exactly once.
///
/// 8192 f32 elements = 32 KB, i.e. L1 data-cache size on M3/M4 Max and
/// most modern x86 server parts. The gap in the conv benchmarks is clean:
/// every depthwise shape that regressed under pure oh-outer was <= 3136
/// elements, while the Sobel / preproc / mask shapes that win under
/// oh-outer are all > 65000 elements.
pub(super) const CONV_PLANE_OH_OUTER_THRESHOLD: usize = 8192;

/// Accumulate one 2D conv plane: `out_plane += conv2d(in_plane, w_plane)` using
/// the precomputed analytic `oh_ranges`/`ow_ranges` to skip padding checks in
/// the inner loop.
///
/// **Precondition**: `out_plane` must already hold the running accumulator
/// (zero on the first call, or whatever partial sum has been built up across
/// previous `ci` iterations). The function reads each output element before
/// writing, so an uninitialized buffer produces silent garbage.
///
/// Dispatches to one of two loop orders based on `out_plane.len()`; see
/// `CONV_PLANE_OH_OUTER_THRESHOLD` for the tradeoff. Shared by
/// `conv3d_depthwise_impl` and `conv3d_small_channel_impl`. The dispatcher
/// is `#[inline]` (not `inline(always)`) so the runtime length branch lives
/// once at each call site instead of inlining both variants; the variants
/// themselves stay `inline(always)` so LLVM sees the concrete inner loop
/// pattern and emits SIMD fmuladd. Using `num_traits::Float` bounds (rather
/// than fn-pointer arithmetic) is load-bearing for vectorization.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn conv_plane_accumulate<T: num_traits::Float + Copy>(
    out_plane: &mut [T],
    in_plane: &[T],
    w_plane: &[T],
    kernel_h: usize,
    kernel_w: usize,
    in_w: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_ranges: &[(usize, usize)],
    ow_ranges: &[(usize, usize)],
) {
    if out_plane.len() > CONV_PLANE_OH_OUTER_THRESHOLD {
        conv_plane_accumulate_oh_outer(
            out_plane, in_plane, w_plane, kernel_h, kernel_w, in_w, out_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, oh_ranges, ow_ranges,
        );
    } else {
        conv_plane_accumulate_kh_outer(
            out_plane, in_plane, w_plane, kernel_h, kernel_w, in_w, out_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, oh_ranges, ow_ranges,
        );
    }
}

/// `oh`-outermost variant. Pins one output row in L1 across every kernel
/// position that accumulates into it, so each row is touched exactly once
/// regardless of how large the full output plane is. Selected when the
/// plane exceeds L1 (see `CONV_PLANE_OH_OUTER_THRESHOLD`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn conv_plane_accumulate_oh_outer<T: num_traits::Float + Copy>(
    out_plane: &mut [T],
    in_plane: &[T],
    w_plane: &[T],
    kernel_h: usize,
    kernel_w: usize,
    in_w: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_ranges: &[(usize, usize)],
    ow_ranges: &[(usize, usize)],
) {
    // An empty plane or zero-width output is a trivial no-op. This also
    // guards the divide below against `out_w == 0`, which is not reachable
    // from the in-tree callers (ruda's `calculate_conv_output_size` is
    // always >= 1 for valid inputs) but would otherwise panic if some
    // future caller handed us a degenerate slice.
    if out_plane.is_empty() || out_w == 0 {
        return;
    }

    // out_plane is a per-(batch, channel) slice of shape [out_h, out_w]
    // stored contiguously; both call sites produce it by disjoint splitting
    // of a `vec![zero; batch * channels * out_h * out_w]` allocation, so
    // the length is always an exact multiple of `out_w`. The `debug_assert`
    // turns that documentary invariant into an enforceable one.
    debug_assert_eq!(
        out_plane.len() % out_w,
        0,
        "out_plane length must be a whole number of rows"
    );
    let out_h = out_plane.len() / out_w;

    for oh in 0..out_h {
        let out_row = &mut out_plane[oh * out_w..(oh + 1) * out_w];

        for kh in 0..kernel_h {
            let (oh_start, oh_end) = oh_ranges[kh];
            // Skip kernel rows that fall outside the padded image at this oh.
            if oh < oh_start || oh >= oh_end {
                continue;
            }
            let ih = oh * stride_h + kh * dilation_h - pad_h;
            let in_row = &in_plane[ih * in_w..(ih + 1) * in_w];

            for kw in 0..kernel_w {
                let (ow_start, ow_end) = ow_ranges[kw];
                if ow_start >= ow_end {
                    continue;
                }
                let w_val = w_plane[kh * kernel_w + kw];
                // All terms are non-negative because ow_start was chosen so
                // that the corresponding `iw` is in bounds.
                let iw_start = ow_start * stride_w + kw * dilation_w - pad_w;

                if stride_w == 1 {
                    let run_len = ow_end - ow_start;
                    let in_slice = &in_row[iw_start..iw_start + run_len];
                    let out_slice = &mut out_row[ow_start..ow_end];
                    for (o, &xv) in out_slice.iter_mut().zip(in_slice.iter()) {
                        *o = *o + w_val * xv;
                    }
                } else {
                    let mut iw = iw_start;
                    for o in &mut out_row[ow_start..ow_end] {
                        *o = *o + w_val * in_row[iw];
                        iw += stride_w;
                    }
                }
            }
        }
    }
}

/// `kh, kw`-outermost variant. Amortizes per-kernel-position setup over
/// long regular `oh` runs that the hardware prefetcher can track.
/// Selected when the whole output plane already fits in L1 (see
/// `CONV_PLANE_OH_OUTER_THRESHOLD`): the oh-outer trick buys nothing
/// there and the extra per-iteration bookkeeping slightly hurts.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn conv_plane_accumulate_kh_outer<T: num_traits::Float + Copy>(
    out_plane: &mut [T],
    in_plane: &[T],
    w_plane: &[T],
    kernel_h: usize,
    kernel_w: usize,
    in_w: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_ranges: &[(usize, usize)],
    ow_ranges: &[(usize, usize)],
) {
    for kh in 0..kernel_h {
        let (oh_start, oh_end) = oh_ranges[kh];
        if oh_start >= oh_end {
            continue;
        }
        for kw in 0..kernel_w {
            let (ow_start, ow_end) = ow_ranges[kw];
            if ow_start >= ow_end {
                continue;
            }
            let w_val = w_plane[kh * kernel_w + kw];
            let iw_start = ow_start * stride_w + kw * dilation_w - pad_w;
            let run_len = ow_end - ow_start;
            for oh in oh_start..oh_end {
                let ih = oh * stride_h + kh * dilation_h - pad_h;
                let in_row = &in_plane[ih * in_w..(ih + 1) * in_w];
                let out_row = &mut out_plane[oh * out_w..(oh + 1) * out_w];
                if stride_w == 1 {
                    let in_slice = &in_row[iw_start..iw_start + run_len];
                    let out_slice = &mut out_row[ow_start..ow_end];
                    for (o, &xv) in out_slice.iter_mut().zip(in_slice.iter()) {
                        *o = *o + w_val * xv;
                    }
                } else {
                    let mut iw = iw_start;
                    for o in &mut out_row[ow_start..ow_end] {
                        *o = *o + w_val * in_row[iw];
                        iw += stride_w;
                    }
                }
            }
        }
    }
}

macro_rules! conv3d_depthwise_typed {
    ($fn_name:ident, $T:ty, $dtype:expr) => {
        pub(super) fn $fn_name(
            x: HostTensor,
            weight: HostTensor,
            bias: Option<HostTensor>,
            options: &ConvOptions<3>,
        ) -> HostTensor {
            conv3d_depthwise_impl::<$T>(x, weight, bias, options, $dtype)
        }
    };
}

conv3d_depthwise_typed!(conv3d_depthwise_f32, f32, DType::F32);
conv3d_depthwise_typed!(conv3d_depthwise_f64, f64, DType::F64);
conv3d_depthwise_typed!(conv3d_depthwise_f16, f16, DType::F16);

/// Depthwise conv3d: one filter per channel, no channel mixing.
///
/// Preconditions (checked by `should_use_depthwise_conv`):
/// - `channels_per_group == 1`
/// - `groups == channels_in == channels_out`
/// - `kernel_d == 1`, `in_d == 1`, trivial d-axis options.
///
/// Uses `num_traits::Float` bounds so the inner multiply-accumulate compiles
/// down to direct `fmul`/`fadd` instructions that LLVM can autovectorize.
/// Function-pointer arithmetic (`fn(T, T) -> T`) would prevent vectorization
/// because each call is an indirect branch that blocks the loop pattern match.
pub(super) fn conv3d_depthwise_impl<T>(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
) -> HostTensor
where
    T: num_traits::Float + bytemuck::Pod + Clone + Copy + ruda_core::tensor::element::Element + Send + Sync,
{
    let zero = <T as num_traits::Zero>::zero();
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [_, stride_h, stride_w] = options.stride;
    let [_, pad_h, pad_w] = options.padding;
    let [_, dilation_h, dilation_w] = options.dilation;

    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, dilation_h, in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, dilation_w, in_w);

    let total = [batch_size, channels, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv depthwise: output dimensions would overflow");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;
    let k_spatial = kernel_h * kernel_w;

    // Valid oh/ow ranges are identical for every (b, c), so precompute once.
    let oh_ranges: Vec<(usize, usize)> = (0..kernel_h)
        .map(|kh| valid_out_range(kh, dilation_h, pad_h, stride_h, in_h, out_h))
        .collect();
    let ow_ranges: Vec<(usize, usize)> = (0..kernel_w)
        .map(|kw| valid_out_range(kw, dilation_w, pad_w, stride_w, in_w, out_w))
        .collect();

    let mut output = vec![zero; total];

    // Per-plane work for one `(batch, channel)` pair. Output slice is passed
    // in so the rayon and sequential dispatch paths can source it differently
    // (from a raw pointer or a direct borrow) without duplicating this body.
    let plane_work = |bc: usize, out_plane: &mut [T]| {
        let c = bc % channels;
        let in_base = bc * in_spatial;
        let w_base = c * k_spatial;
        conv_plane_accumulate(
            out_plane,
            &x_data[in_base..in_base + in_spatial],
            &w_data[w_base..w_base + k_spatial],
            kernel_h,
            kernel_w,
            in_w,
            out_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            &oh_ranges,
            &ow_ranges,
        );
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let dst_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(output.as_mut_ptr());
        (0..batch_size * channels).into_par_iter().for_each(|bc| {
            // SAFETY: disjoint `[bc * out_spatial, (bc+1) * out_spatial)`
            // ranges tile the output buffer.
            let out_plane: &mut [T] = unsafe {
                core::slice::from_raw_parts_mut(dst_ptr.ptr_add(bc * out_spatial), out_spatial)
            };
            plane_work(bc, out_plane);
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for bc in 0..batch_size * channels {
            let out_base = bc * out_spatial;
            plane_work(bc, &mut output[out_base..out_base + out_spatial]);
        }
    }

    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        assert_eq!(
            bias_data.len(),
            channels,
            "conv depthwise: bias length ({}) must equal channels ({channels})",
            bias_data.len()
        );
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels,
            out_spatial,
            |a, b| a + b,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels, 1, out_h, out_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}
