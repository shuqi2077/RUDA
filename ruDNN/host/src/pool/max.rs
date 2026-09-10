use super::*;

// ============================================================================
// Max Pool 3D - core implementation
// ============================================================================

max_pool3d_with_indices_typed!(
    max_pool3d_with_indices_f32,
    f32,
    DType::F32,
    f32::NEG_INFINITY
);
max_pool3d_with_indices_typed!(
    max_pool3d_with_indices_f64,
    f64,
    DType::F64,
    f64::NEG_INFINITY
);
max_pool3d_with_indices_typed!(
    max_pool3d_with_indices_f16,
    f16,
    DType::F16,
    f16::NEG_INFINITY
);

pub fn max_pool3d_with_indices_bf16(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> (HostTensor, HostTensor) {
    let x_f32 = convert_bf16_to_f32(&x);
    let (output_f32, indices) =
        max_pool3d_with_indices_f32(x_f32, kernel_size, stride, padding, dilation, ceil_mode);
    (convert_f32_to_bf16(&output_f32), indices)
}

/// Generic 3D max pooling with indices implementation.
#[allow(clippy::too_many_arguments)]
fn max_pool3d_with_indices_impl<T>(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
    dtype: DType,
    neg_inf: T,
) -> (HostTensor, HostTensor)
where
    T: bytemuck::Pod + Copy + PartialOrd + Send + Sync + Element,
{
    let x = x.to_contiguous();
    let x_shape = x.layout().shape();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let [kernel_d, kernel_h, kernel_w] = kernel_size;
    let [stride_d, stride_h, stride_w] = stride;
    let [pad_d, pad_h, pad_w] = padding;
    let [dilation_d, dilation_h, dilation_w] = dilation;

    let out_d = pool_output_size(in_d, kernel_d, pad_d, stride_d, dilation_d, ceil_mode);
    let out_h = pool_output_size(in_h, kernel_h, pad_h, stride_h, dilation_h, ceil_mode);
    let out_w = pool_output_size(in_w, kernel_w, pad_w, stride_w, dilation_w, ceil_mode);

    let spatial_out = out_d * out_h * out_w;
    let x_data: &[T] = x.storage();

    let (output, indices) = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![neg_inf; batch_size * channels * spatial_out];
            let mut indices = vec![-1i64; batch_size * channels * spatial_out];
            let out_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(output.as_mut_ptr());
            let idx_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(indices.as_mut_ptr());

            // Flatten batch*channels into a single par_iter so rayon chooses
            // the right granularity instead of creating one task per channel.
            let bc_total = batch_size * channels;
            (0..bc_total).into_par_iter().for_each(|bc| {
                let b = bc / channels;
                let c = bc % channels;
                let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                let out_offset = bc * spatial_out;

                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                            let mut max_val = neg_inf;
                            let mut max_idx: i64 = -1;

                            for kd in 0..kernel_d {
                                let id =
                                    (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
                                if id < 0 || id >= in_d as isize {
                                    continue;
                                }
                                let id = id as usize;

                                for kh in 0..kernel_h {
                                    let ih =
                                        (oh * stride_h + kh * dilation_h) as isize - pad_h as isize;
                                    if ih < 0 || ih >= in_h as isize {
                                        continue;
                                    }
                                    let ih = ih as usize;

                                    for kw in 0..kernel_w {
                                        let iw = (ow * stride_w + kw * dilation_w) as isize
                                            - pad_w as isize;
                                        if iw < 0 || iw >= in_w as isize {
                                            continue;
                                        }
                                        let iw = iw as usize;

                                        let x_idx = x_offset + id * in_h * in_w + ih * in_w + iw;
                                        let val = x_data[x_idx];

                                        if max_idx < 0 || val > max_val {
                                            max_val = val;
                                            max_idx = (id * in_h * in_w + ih * in_w + iw) as i64;
                                        }
                                    }
                                }
                            }

                            unsafe {
                                out_ptr.write(out_idx, max_val);
                                idx_ptr.write(out_idx, max_idx);
                            }
                        }
                    }
                }
            });
            (output, indices)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![neg_inf; batch_size * channels * spatial_out];
            let mut indices = vec![-1i64; batch_size * channels * spatial_out];

            for b in 0..batch_size {
                for c in 0..channels {
                    let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                    let out_offset = b * channels * spatial_out + c * spatial_out;

                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut max_val = neg_inf;
                                let mut max_idx: i64 = -1;

                                for kd in 0..kernel_d {
                                    let id =
                                        (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
                                    if id < 0 || id >= in_d as isize {
                                        continue;
                                    }
                                    let id = id as usize;

                                    for kh in 0..kernel_h {
                                        let ih = (oh * stride_h + kh * dilation_h) as isize
                                            - pad_h as isize;
                                        if ih < 0 || ih >= in_h as isize {
                                            continue;
                                        }
                                        let ih = ih as usize;

                                        for kw in 0..kernel_w {
                                            let iw = (ow * stride_w + kw * dilation_w) as isize
                                                - pad_w as isize;
                                            if iw < 0 || iw >= in_w as isize {
                                                continue;
                                            }
                                            let iw = iw as usize;

                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            let val = x_data[x_idx];

                                            if max_idx < 0 || val > max_val {
                                                max_val = val;
                                                max_idx =
                                                    (id * in_h * in_w + ih * in_w + iw) as i64;
                                            }
                                        }
                                    }
                                }

                                output[out_idx] = max_val;
                                indices[out_idx] = max_idx;
                            }
                        }
                    }
                }
            }
            (output, indices)
        }
    };

    let out_shape = Shape::from(vec![batch_size, channels, out_d, out_h, out_w]);
    let output_tensor = HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape.clone()),
        dtype,
    );
    let indices_tensor = HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(out_shape),
        DType::I64,
    );

    (output_tensor, indices_tensor)
}

/// 3D max pooling (without returning indices) for f32.
pub fn max_pool3d_f32(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> HostTensor {
    max_pool3d_with_indices_f32(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 3D max pooling (without returning indices) for f64.
pub fn max_pool3d_f64(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> HostTensor {
    max_pool3d_with_indices_f64(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 3D max pooling (without returning indices) for f16.
pub fn max_pool3d_f16(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> HostTensor {
    max_pool3d_with_indices_f16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 3D max pooling (without returning indices) for bf16.
pub fn max_pool3d_bf16(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> HostTensor {
    max_pool3d_with_indices_bf16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

// ============================================================================
// Max Pool 2D - delegates to 3D
// ============================================================================

/// 2D max pooling with indices for f32.
pub fn max_pool2d_with_indices_f32(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (HostTensor, HostTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f32(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling with indices for f64.
pub fn max_pool2d_with_indices_f64(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (HostTensor, HostTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f64(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling with indices for f16.
pub fn max_pool2d_with_indices_f16(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (HostTensor, HostTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling with indices for bf16.
pub fn max_pool2d_with_indices_bf16(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (HostTensor, HostTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_bf16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling (without indices) for f32.
pub fn max_pool2d_f32(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> HostTensor {
    max_pool2d_with_indices_f32(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 2D max pooling (without indices) for f64.
pub fn max_pool2d_f64(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> HostTensor {
    max_pool2d_with_indices_f64(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 2D max pooling (without indices) for f16.
pub fn max_pool2d_f16(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> HostTensor {
    max_pool2d_with_indices_f16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 2D max pooling (without indices) for bf16.
pub fn max_pool2d_bf16(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> HostTensor {
    max_pool2d_with_indices_bf16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

// ============================================================================
// Max Pool 1D - delegates to 3D
// ============================================================================

/// 1D max pooling with indices for f32.
pub fn max_pool1d_with_indices_f32(
    x: HostTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> (HostTensor, HostTensor) {
    let x_3d = expand_1d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f32(
        x_3d,
        [1, 1, kernel_size],
        [1, 1, stride],
        [0, 0, padding],
        [1, 1, dilation],
        ceil_mode,
    );
    (squeeze_3d_to_1d(output), squeeze_3d_to_1d(indices))
}

/// 1D max pooling (without indices) for f32.
pub fn max_pool1d_f32(
    x: HostTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> HostTensor {
    max_pool1d_with_indices_f32(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

