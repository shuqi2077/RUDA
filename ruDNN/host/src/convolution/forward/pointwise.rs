use super::*;

/// Check if this is a 1x1 convolution that can use the fast path.
pub(super) fn is_1x1_conv(
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    options: &ConvOptions<3>,
) -> bool {
    kernel_d == 1
        && kernel_h == 1
        && kernel_w == 1
        && options.stride == [1, 1, 1]
        && options.padding == [0, 0, 0]
}

/// Optimized 1x1 convolution: skip im2col, call gemm directly on NCHW data.
///
/// For 1x1 conv with stride=1 and padding=0, the input for each (batch, group) is
/// already [channels_per_group, spatial] row-major in NCHW layout. We pass it directly
/// to gemm as the RHS with appropriate strides, avoiding the transpose allocation
/// and the intermediate result buffer.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn conv3d_1x1_impl<T: bytemuck::Pod + Clone + Copy + ruda_core::tensor::element::Element + Send + Sync>(
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
    let spatial = x_shape[2] * x_shape[3] * x_shape[4];

    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let groups = options.groups;
    let out_channels_per_group = channels_out / groups;

    let m = out_channels_per_group;
    let k = channels_per_group;
    let n = spatial;

    // Validate output size won't overflow
    let total_output = [batch_size, channels_out, spatial]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv 1x1: output tensor dimensions would overflow index calculations");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    // Gemm: C[M,N] = W[M,K] * X[K,N]
    // W is [out_channels_per_group, channels_per_group] row-major
    // X is [channels_per_group, spatial] row-major (directly from NCHW layout)
    // C is [out_channels_per_group, spatial] row-major
    #[cfg(feature = "rayon")]
    let parallelism = if m.saturating_mul(n).saturating_mul(k) >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;

    let mut output = vec![zero; total_output];

    {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let dst_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(output.as_mut_ptr());

            (0..batch_size).into_par_iter().for_each(|b| {
                for g in 0..groups {
                    let x_offset = b * channels_in * spatial + g * k * spatial;
                    let w_offset = g * out_channels_per_group * k;
                    let out_offset =
                        b * channels_out * spatial + g * out_channels_per_group * spatial;

                    unsafe {
                        gemm::gemm(
                            m,
                            n,
                            k,
                            dst_ptr.ptr_add(out_offset),
                            1,          // dst_cs
                            n as isize, // dst_rs
                            false,      // read_dst
                            w_data.as_ptr().add(w_offset),
                            1,          // lhs_cs: W row-major
                            k as isize, // lhs_rs
                            x_data.as_ptr().add(x_offset),
                            1,          // rhs_cs: X row-major
                            n as isize, // rhs_rs
                            zero,
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
                for g in 0..groups {
                    let x_offset = b * channels_in * spatial + g * k * spatial;
                    let w_offset = g * out_channels_per_group * k;
                    let out_offset =
                        b * channels_out * spatial + g * out_channels_per_group * spatial;

                    unsafe {
                        gemm::gemm(
                            m,
                            n,
                            k,
                            output.as_mut_ptr().add(out_offset),
                            1,
                            n as isize,
                            false,
                            w_data.as_ptr().add(w_offset),
                            1,
                            k as isize,
                            x_data.as_ptr().add(x_offset),
                            1,
                            n as isize,
                            zero,
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
            spatial,
            add_fn,
        );
    }

    let out_shape = Shape::from(vec![
        batch_size,
        channels_out,
        x_shape[2],
        x_shape[3],
        x_shape[4],
    ]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

conv3d_1x1_typed!(conv3d_1x1_f32, f32, DType::F32, 0.0f32, 1.0f32, |a, b| a
    + b);
conv3d_1x1_typed!(conv3d_1x1_f64, f64, DType::F64, 0.0f64, 1.0f64, |a, b| a
    + b);
conv3d_1x1_typed!(
    conv3d_1x1_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    f16::from_f32(1.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32())
);

