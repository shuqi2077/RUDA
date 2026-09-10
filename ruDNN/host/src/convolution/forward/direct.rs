use super::*;

// ============================================================================
// Direct conv path for small spatial sizes (no im2col)
// ============================================================================

/// Decide whether to use the direct conv path instead of tiled im2col + GEMM.
///
/// The direct path decomposes the conv into kw gemm calls with strided pointers
/// into NCHW data, eliminating both the NHWC conversion and im2col buffer.
/// This wins when spatial_out is small enough that im2col allocation and fill
/// overhead is significant relative to compute.
pub(super) fn should_use_direct_conv(x_shape: &[usize], w_shape: &[usize], options: &ConvOptions<3>) -> bool {
    // Only for groups=1, no padding, dilation=1 (the wav2vec2 case).
    if options.groups != 1 || options.padding != [0, 0, 0] || options.dilation != [1, 1, 1] {
        return false;
    }

    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    // Only for "1D-like" convolutions expanded to 3D
    if kernel_d != 1 || kernel_h != 1 {
        return false;
    }
    if x_shape[2] != 1 || x_shape[3] != 1 {
        return false;
    }

    let channels_in = x_shape[1];
    let in_w = x_shape[4];
    let out_w = calculate_conv_output_size(kernel_w, options.stride[2], 0, 1, in_w);

    // The direct path eliminates im2col buffer allocation (kw * c_in * tile_size floats)
    // and the copy into it. This matters when spatial_out is small relative to c_in * kw.
    channels_in >= 32 && kernel_w <= 8 && out_w <= 800
}

macro_rules! conv3d_direct_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $one:expr, $add_fn:expr) => {
        pub(super) fn $fn_name(
            x: HostTensor,
            weight: HostTensor,
            bias: Option<HostTensor>,
            options: &ConvOptions<3>,
        ) -> HostTensor {
            conv3d_direct_impl::<$T>(x, weight, bias, options, $dtype, $zero, $one, $add_fn)
        }
    };
}

conv3d_direct_typed!(
    conv3d_direct_f32,
    f32,
    DType::F32,
    0.0f32,
    1.0f32,
    |a, b| a + b
);
conv3d_direct_typed!(
    conv3d_direct_f64,
    f64,
    DType::F64,
    0.0f64,
    1.0f64,
    |a, b| a + b
);

/// Direct conv3d: decompose into kw gemm calls on NCHW data directly.
///
/// The conv operation is: out[co, o] = sum_k sum_c w[co, c, k] * x[c, o*stride+k]
///
/// This decomposes into kw matrix multiplies, one per kernel position:
///   out += W_k @ X_k  where W_k = w[:, :, k] and X_k = x[:, o*stride+k]
///
/// Each gemm reads directly from NCHW input with strided pointers, eliminating
/// both the NHWC conversion and im2col buffer entirely.
///
/// Constraints: groups=1, padding=0, dilation=1, 1D-like (d=1, h=1).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn conv3d_direct_impl<T: bytemuck::Pod + Clone + Copy + ruda_core::tensor::element::Element + Send + Sync>(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    one: T,
    add_fn: fn(T, T) -> T,
) -> HostTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let kernel_w = w_shape[4];
    let stride_w = options.stride[2];

    let out_w = calculate_conv_output_size(kernel_w, stride_w, 0, 1, in_w);

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    // Weight layout: [c_out, c_in, kw] contiguous (kd=kh=1).
    // For kernel position k: W_k[co, c] = w_data[co * c_in * kw + c * kw + k]
    let lhs_rs = (channels_in * kernel_w) as isize;
    let lhs_cs = kernel_w as isize;

    // For kernel position k: X_k[c_in, o] = x_data[c_in * in_w + o * stride + k]
    let rhs_rs = in_w as isize;
    let rhs_cs = stride_w as isize;

    let m = channels_out;
    let gemm_k = channels_in;
    let n = out_w;

    #[cfg(feature = "rayon")]
    let parallelism = if m.saturating_mul(n).saturating_mul(gemm_k) >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;

    let total_output = batch_size
        .checked_mul(channels_out)
        .and_then(|x| x.checked_mul(out_w))
        .expect("conv direct: output dimensions overflow");
    let mut output = vec![zero; total_output];

    let batch_x_len = channels_in * in_w;

    {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let dst_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(output.as_mut_ptr());

            (0..batch_size).into_par_iter().for_each(|b| {
                let out_offset = b * channels_out * out_w;

                for k in 0..kernel_w {
                    unsafe {
                        let x_base = x_data.as_ptr().add(b * batch_x_len);
                        gemm::gemm(
                            m,
                            n,
                            gemm_k,
                            dst_ptr.ptr_add(out_offset),
                            1,
                            n as isize,
                            k > 0,
                            w_data.as_ptr().add(k),
                            lhs_cs,
                            lhs_rs,
                            x_base.add(k),
                            rhs_cs,
                            rhs_rs,
                            one,
                            one,
                            false,
                            false,
                            false,
                            parallelism,
                        );
                    }
                }
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for b in 0..batch_size {
                let out_offset = b * channels_out * out_w;

                for k in 0..kernel_w {
                    unsafe {
                        let x_base = x_data.as_ptr().add(b * batch_x_len);
                        gemm::gemm(
                            m,
                            n,
                            gemm_k,
                            output.as_mut_ptr().add(out_offset),
                            1,
                            n as isize,
                            k > 0,
                            w_data.as_ptr().add(k),
                            lhs_cs,
                            lhs_rs,
                            x_base.add(k),
                            rhs_cs,
                            rhs_rs,
                            one,
                            one,
                            false,
                            false,
                            false,
                            parallelism,
                        );
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            out_w,
            add_fn,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels_out, 1, 1, out_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

