use super::*;

// ============================================================================
// Small-channel conv fast path (no NHWC, no im2col, no gemm)
// ============================================================================
//
// For `groups=1` convs with very few input channels (e.g. 3-channel Sobel
// filters, single-channel mask networks, early-stage image preprocessors),
// the generic `conv3d_impl` pays the same kind of overhead as the depthwise
// case: gemm is dispatched with `M = channels_out, K = channels_in * k_spatial,
// N = tile_size`, and at small `K` the gemm kernel's pack + select + dispatch
// cost dominates the actual FMAs. On top of that, the full NHWC conversion
// of the input is pure memory traffic that buys nothing when the channel
// dimension is already tiny.
//
// The small-channel path walks NCHW data directly. For each `(batch, out_ch)`
// pair it accumulates contributions from every input channel by calling the
// same `conv_plane_accumulate` helper used by the depthwise path. Compared
// to the depthwise case this adds an outer `for ci in 0..channels_in` loop
// around the accumulate; everything else (analytic output ranges, parallel
// fan-out, bias add) is identical.

/// Threshold on `channels_in` for the small-channel fast path.
///
/// Picked empirically: at `channels_in <= 4`, gemm dispatch overhead exceeds
/// the inner compute by a clear margin. Tuning higher risks regressing shapes
/// where gemm's data reuse across output channels wins.
pub(super) const SMALL_CHANNEL_IN_THRESHOLD: usize = 4;

/// Threshold on `channels_out` for the small-channel fast path.
///
/// The small-channel path loops `for co: for ci:` and re-reads each input
/// channel once per output channel. At large `channels_out`, that redundant
/// memory traffic dominates over gemm's register-tiled data reuse. Picked
/// empirically: classic ImageNet first-layer (`3 -> 64`) runs ~5x slower on
/// the direct path than on gemm, so we cap at 16 to stay safely on the right
/// side of that cliff. Sobel-style filters (`3 -> 3..8`) are well under this
/// and keep their 1.1-1.4x win over ruda-tensor-host.
pub(super) const SMALL_CHANNEL_OUT_THRESHOLD: usize = 16;

/// Decide whether to use the small-channel fast path.
///
/// Triggers on `groups=1` 2D-in-3D convs (or 1D via the conv1d expansion)
/// with both small `channels_in` and small `channels_out`. Non-depthwise
/// grouped convs and the many-channel case both go through the generic
/// `conv3d_impl`.
pub(super) fn should_use_small_channel_conv(
    x_shape: &[usize],
    w_shape: &[usize],
    options: &ConvOptions<3>,
) -> bool {
    if options.groups != 1 {
        return false;
    }

    // Only the 1D/2D-in-3D shapes (kd=1, in_d=1). Pure 3D convs do not benefit
    // and would require adding a d-axis loop.
    if w_shape[2] != 1 || x_shape[2] != 1 {
        return false;
    }
    if options.stride[0] != 1 || options.padding[0] != 0 || options.dilation[0] != 1 {
        return false;
    }

    let channels_in = x_shape[1];
    let channels_out = w_shape[0];
    channels_in > 0
        && channels_in <= SMALL_CHANNEL_IN_THRESHOLD
        && channels_out > 0
        && channels_out <= SMALL_CHANNEL_OUT_THRESHOLD
}

macro_rules! conv3d_small_channel_typed {
    ($fn_name:ident, $T:ty, $dtype:expr) => {
        pub(super) fn $fn_name(
            x: HostTensor,
            weight: HostTensor,
            bias: Option<HostTensor>,
            options: &ConvOptions<3>,
        ) -> HostTensor {
            conv3d_small_channel_impl::<$T>(x, weight, bias, options, $dtype)
        }
    };
}

conv3d_small_channel_typed!(conv3d_small_channel_f32, f32, DType::F32);
conv3d_small_channel_typed!(conv3d_small_channel_f64, f64, DType::F64);
conv3d_small_channel_typed!(conv3d_small_channel_f16, f16, DType::F16);

/// Small-channel conv3d: `groups=1` with small `channels_in`.
///
/// Preconditions (checked by `should_use_small_channel_conv`):
/// - `options.groups == 1`
/// - `channels_in <= SMALL_CHANNEL_IN_THRESHOLD`
/// - `kernel_d == 1`, `in_d == 1`, trivial d-axis options.
pub(super) fn conv3d_small_channel_impl<T>(
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
    let channels_in = x_shape[1];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [_, stride_h, stride_w] = options.stride;
    let [_, pad_h, pad_w] = options.padding;
    let [_, dilation_h, dilation_w] = options.dilation;

    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, dilation_h, in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, dilation_w, in_w);

    let total = [batch_size, channels_out, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv small-channel: output dimensions would overflow");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;
    let k_spatial = kernel_h * kernel_w;
    let w_co_stride = channels_in * k_spatial;
    let x_batch_stride = channels_in * in_spatial;

    let oh_ranges: Vec<(usize, usize)> = (0..kernel_h)
        .map(|kh| valid_out_range(kh, dilation_h, pad_h, stride_h, in_h, out_h))
        .collect();
    let ow_ranges: Vec<(usize, usize)> = (0..kernel_w)
        .map(|kw| valid_out_range(kw, dilation_w, pad_w, stride_w, in_w, out_w))
        .collect();

    let mut output = vec![zero; total];

    // Per-plane work for one `(batch, out_channel)` pair. Accumulates
    // `sum over ci of conv2d(in[b, ci], w[co, ci])` into `out_plane` by
    // calling the shared helper once per input channel.
    let plane_work = |b_co: usize, out_plane: &mut [T]| {
        let b = b_co / channels_out;
        let co = b_co % channels_out;
        for ci in 0..channels_in {
            let in_base = b * x_batch_stride + ci * in_spatial;
            let w_base = co * w_co_stride + ci * k_spatial;
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
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let dst_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(output.as_mut_ptr());
        (0..batch_size * channels_out)
            .into_par_iter()
            .for_each(|b_co| {
                // SAFETY: disjoint `[b_co * out_spatial, (b_co+1) * out_spatial)`
                // ranges tile the output buffer.
                let out_plane: &mut [T] = unsafe {
                    core::slice::from_raw_parts_mut(
                        dst_ptr.ptr_add(b_co * out_spatial),
                        out_spatial,
                    )
                };
                plane_work(b_co, out_plane);
            });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for b_co in 0..batch_size * channels_out {
            let out_base = b_co * out_spatial;
            plane_work(b_co, &mut output[out_base..out_base + out_spatial]);
        }
    }

    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        assert_eq!(
            bias_data.len(),
            channels_out,
            "conv small-channel: bias length ({}) must equal channels_out ({channels_out})",
            bias_data.len()
        );
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            out_spatial,
            |a, b| a + b,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels_out, 1, out_h, out_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}
