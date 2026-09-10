use super::*;

// ============================================================================
// Backward passes
// ============================================================================

/// Max pool 2D backward using stored indices.
pub fn max_pool2d_backward_f32(x: HostTensor, grad: HostTensor, indices: HostTensor) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let indices_3d = expand_2d_to_3d(&indices);
    let result = max_pool3d_backward_f32(x_3d, grad_3d, indices_3d);
    squeeze_3d_to_2d(result)
}

/// Max pool 2D backward for f64.
pub fn max_pool2d_backward_f64(x: HostTensor, grad: HostTensor, indices: HostTensor) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let indices_3d = expand_2d_to_3d(&indices);
    let result = max_pool3d_backward_f64(x_3d, grad_3d, indices_3d);
    squeeze_3d_to_2d(result)
}

/// Max pool 2D backward for f16.
pub fn max_pool2d_backward_f16(x: HostTensor, grad: HostTensor, indices: HostTensor) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let indices_3d = expand_2d_to_3d(&indices);
    let result = max_pool3d_backward_f16(x_3d, grad_3d, indices_3d);
    squeeze_3d_to_2d(result)
}

/// Max pool 2D backward for bf16.
pub fn max_pool2d_backward_bf16(
    x: HostTensor,
    grad: HostTensor,
    indices: HostTensor,
) -> HostTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let grad_f32 = convert_bf16_to_f32(&grad);
    let result_f32 = max_pool2d_backward_f32(x_f32, grad_f32, indices);
    convert_f32_to_bf16(&result_f32)
}

max_pool3d_backward_typed!(max_pool3d_backward_f32, f32, DType::F32, 0.0f32, |a, b| a
    + b);
max_pool3d_backward_typed!(max_pool3d_backward_f64, f64, DType::F64, 0.0f64, |a, b| a
    + b);
max_pool3d_backward_typed!(
    max_pool3d_backward_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32())
);

/// Generic max pool 3D backward implementation.
fn max_pool3d_backward_impl<T>(
    x: HostTensor,
    grad: HostTensor,
    indices: HostTensor,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
) -> HostTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x_shape = x.layout().shape();
    let grad = grad.to_contiguous();
    let indices = indices.to_contiguous();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];
    let spatial_in = in_d * in_h * in_w;

    let grad_shape = grad.layout().shape();
    let out_d = grad_shape[2];
    let out_h = grad_shape[3];
    let out_w = grad_shape[4];
    let spatial_out = out_d * out_h * out_w;

    let grad_data: &[T] = grad.storage();
    let indices_data: &[i64] = indices.storage();

    // Accumulate gradients back to input positions
    let mut output = vec![zero; batch_size * channels * spatial_in];

    for b in 0..batch_size {
        for c in 0..channels {
            let grad_offset = b * channels * spatial_out + c * spatial_out;
            let out_offset = b * channels * spatial_in + c * spatial_in;

            for i in 0..spatial_out {
                let idx = indices_data[grad_offset + i];
                if idx >= 0 {
                    let input_idx = out_offset + idx as usize;
                    output[input_idx] = add_fn(output[input_idx], grad_data[grad_offset + i]);
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch_size, channels, in_d, in_h, in_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

/// Avg pool 2D backward.
pub fn avg_pool2d_backward_f32(
    x: HostTensor,
    grad: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = avg_pool3d_backward_f32(
        x_3d,
        grad_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
    );
    squeeze_3d_to_2d(result)
}

/// Avg pool 2D backward for f64.
pub fn avg_pool2d_backward_f64(
    x: HostTensor,
    grad: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = avg_pool3d_backward_f64(
        x_3d,
        grad_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
    );
    squeeze_3d_to_2d(result)
}

/// Avg pool 2D backward for f16.
pub fn avg_pool2d_backward_f16(
    x: HostTensor,
    grad: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = avg_pool3d_backward_f16(
        x_3d,
        grad_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
    );
    squeeze_3d_to_2d(result)
}

/// Avg pool 2D backward for bf16.
pub fn avg_pool2d_backward_bf16(
    x: HostTensor,
    grad: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> HostTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let grad_f32 = convert_bf16_to_f32(&grad);
    let result_f32 = avg_pool2d_backward_f32(
        x_f32,
        grad_f32,
        kernel_size,
        stride,
        padding,
        count_include_pad,
    );
    convert_f32_to_bf16(&result_f32)
}

avg_pool3d_backward_typed!(
    avg_pool3d_backward_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |val, count| val / count as f32
);
avg_pool3d_backward_typed!(
    avg_pool3d_backward_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |val, count| val / count as f64
);
avg_pool3d_backward_typed!(
    avg_pool3d_backward_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |val: f16, count| f16::from_f32(val.to_f32() / count as f32)
);

/// Generic avg pool 3D backward implementation.
#[allow(clippy::too_many_arguments)]
fn avg_pool3d_backward_impl<T>(
    x: HostTensor,
    grad: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
    div_fn: fn(T, usize) -> T,
) -> HostTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x_shape = x.layout().shape();
    let grad = grad.to_contiguous();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];
    let spatial_in = in_d * in_h * in_w;

    let [kernel_d, kernel_h, kernel_w] = kernel_size;
    let [stride_d, stride_h, stride_w] = stride;
    let [pad_d, pad_h, pad_w] = padding;
    let kernel_volume = kernel_d * kernel_h * kernel_w;

    let grad_shape = grad.layout().shape();
    let out_d = grad_shape[2];
    let out_h = grad_shape[3];
    let out_w = grad_shape[4];
    let spatial_out = out_d * out_h * out_w;

    let grad_data: &[T] = grad.storage();

    // Distribute gradient equally across window
    let mut output = vec![zero; batch_size * channels * spatial_in];

    for b in 0..batch_size {
        for c in 0..channels {
            let grad_offset = b * channels * spatial_out + c * spatial_out;
            let out_offset = b * channels * spatial_in + c * spatial_in;

            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grad_idx = grad_offset + od * out_h * out_w + oh * out_w + ow;
                        let grad_val = grad_data[grad_idx];

                        // Count valid positions for this output
                        let mut count = 0usize;
                        for kd in 0..kernel_d {
                            let id = (od * stride_d + kd) as isize - pad_d as isize;
                            if id >= 0 && id < in_d as isize {
                                for kh in 0..kernel_h {
                                    let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                    if ih >= 0 && ih < in_h as isize {
                                        for kw in 0..kernel_w {
                                            let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                            if iw >= 0 && iw < in_w as isize {
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let divisor = if count_include_pad {
                            kernel_volume
                        } else {
                            count.max(1)
                        };

                        // Distribute gradient
                        let distributed = div_fn(grad_val, divisor);
                        for kd in 0..kernel_d {
                            let id = (od * stride_d + kd) as isize - pad_d as isize;
                            if id < 0 || id >= in_d as isize {
                                continue;
                            }
                            let id = id as usize;

                            for kh in 0..kernel_h {
                                let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                if ih < 0 || ih >= in_h as isize {
                                    continue;
                                }
                                let ih = ih as usize;

                                for kw in 0..kernel_w {
                                    let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                    if iw < 0 || iw >= in_w as isize {
                                        continue;
                                    }
                                    let iw = iw as usize;

                                    let input_idx = out_offset + id * in_h * in_w + ih * in_w + iw;
                                    output[input_idx] = add_fn(output[input_idx], distributed);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch_size, channels, in_d, in_h, in_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

/// Adaptive avg pool 2D backward.
pub fn adaptive_avg_pool2d_backward_f32(x: HostTensor, grad: HostTensor) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = adaptive_avg_pool3d_backward_f32(x_3d, grad_3d);
    squeeze_3d_to_2d(result)
}

/// Adaptive avg pool 2D backward for f64.
pub fn adaptive_avg_pool2d_backward_f64(x: HostTensor, grad: HostTensor) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = adaptive_avg_pool3d_backward_f64(x_3d, grad_3d);
    squeeze_3d_to_2d(result)
}

/// Adaptive avg pool 2D backward for f16.
pub fn adaptive_avg_pool2d_backward_f16(x: HostTensor, grad: HostTensor) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = adaptive_avg_pool3d_backward_f16(x_3d, grad_3d);
    squeeze_3d_to_2d(result)
}

/// Adaptive avg pool 2D backward for bf16.
pub fn adaptive_avg_pool2d_backward_bf16(x: HostTensor, grad: HostTensor) -> HostTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let grad_f32 = convert_bf16_to_f32(&grad);
    let result_f32 = adaptive_avg_pool2d_backward_f32(x_f32, grad_f32);
    convert_f32_to_bf16(&result_f32)
}

adaptive_avg_pool3d_backward_typed!(
    adaptive_avg_pool3d_backward_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |val, count| val / count as f32
);
adaptive_avg_pool3d_backward_typed!(
    adaptive_avg_pool3d_backward_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |val, count| val / count as f64
);
adaptive_avg_pool3d_backward_typed!(
    adaptive_avg_pool3d_backward_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |val: f16, count| f16::from_f32(val.to_f32() / count as f32)
);

/// Generic adaptive avg pool 3D backward implementation.
fn adaptive_avg_pool3d_backward_impl<T>(
    x: HostTensor,
    grad: HostTensor,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
    div_fn: fn(T, usize) -> T,
) -> HostTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x_shape = x.layout().shape();
    let grad = grad.to_contiguous();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];
    let spatial_in = in_d * in_h * in_w;

    let grad_shape = grad.layout().shape();
    let out_d = grad_shape[2];
    let out_h = grad_shape[3];
    let out_w = grad_shape[4];
    let spatial_out = out_d * out_h * out_w;

    let grad_data: &[T] = grad.storage();

    let mut output = vec![zero; batch_size * channels * spatial_in];

    for b in 0..batch_size {
        for c in 0..channels {
            let grad_offset = b * channels * spatial_out + c * spatial_out;
            let out_offset = b * channels * spatial_in + c * spatial_in;

            for od in 0..out_d {
                let d_start = (od * in_d) / out_d;
                let d_end = ((od + 1) * in_d).div_ceil(out_d);

                for oh in 0..out_h {
                    let h_start = (oh * in_h) / out_h;
                    let h_end = ((oh + 1) * in_h).div_ceil(out_h);

                    for ow in 0..out_w {
                        let w_start = (ow * in_w) / out_w;
                        let w_end = ((ow + 1) * in_w).div_ceil(out_w);

                        let grad_idx = grad_offset + od * out_h * out_w + oh * out_w + ow;
                        let grad_val = grad_data[grad_idx];

                        let count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
                        let distributed = div_fn(grad_val, count.max(1));

                        for id in d_start..d_end {
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let input_idx = out_offset + id * in_h * in_w + ih * in_w + iw;
                                    output[input_idx] = add_fn(output[input_idx], distributed);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch_size, channels, in_d, in_h, in_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

