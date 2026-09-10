use super::*;

// ============================================================================
// Avg Pool 3D - core implementation
// ============================================================================

avg_pool3d_typed!(
    avg_pool3d_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |sum, count| sum / count as f32
);
avg_pool3d_typed!(
    avg_pool3d_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |sum, count| sum / count as f64
);
avg_pool3d_typed!(
    avg_pool3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |sum: f16, count| f16::from_f32(sum.to_f32() / count as f32)
);

pub fn avg_pool3d_bf16(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
) -> HostTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let result_f32 = avg_pool3d_f32(
        x_f32,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    );
    convert_f32_to_bf16(&result_f32)
}

/// Generic 3D average pooling implementation.
#[allow(clippy::too_many_arguments)]
fn avg_pool3d_impl<T>(
    x: HostTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
    div_fn: fn(T, usize) -> T,
) -> HostTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
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

    // Avg pool doesn't use dilation in typical implementations
    let dilation_d = 1;
    let dilation_h = 1;
    let dilation_w = 1;

    let out_d = pool_output_size(in_d, kernel_d, pad_d, stride_d, dilation_d, ceil_mode);
    let out_h = pool_output_size(in_h, kernel_h, pad_h, stride_h, dilation_h, ceil_mode);
    let out_w = pool_output_size(in_w, kernel_w, pad_w, stride_w, dilation_w, ceil_mode);

    let spatial_out = out_d * out_h * out_w;
    let x_data: &[T] = x.storage();
    let _kernel_volume = kernel_d * kernel_h * kernel_w;

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![zero; batch_size * channels * spatial_out];
            let out_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(output.as_mut_ptr());

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
                            let mut sum = zero;
                            let mut count = 0usize;
                            let mut pad_count = 0usize;

                            for kd in 0..kernel_d {
                                let id = (od * stride_d + kd) as isize - pad_d as isize;
                                let id_in_bounds =
                                    id >= -(pad_d as isize) && id < (in_d + pad_d) as isize;
                                if !id_in_bounds {
                                    continue;
                                }
                                let id_valid = id >= 0 && id < in_d as isize;

                                for kh in 0..kernel_h {
                                    let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                    let ih_in_bounds =
                                        ih >= -(pad_h as isize) && ih < (in_h + pad_h) as isize;
                                    if !ih_in_bounds {
                                        continue;
                                    }
                                    let ih_valid = ih >= 0 && ih < in_h as isize;

                                    for kw in 0..kernel_w {
                                        let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                        let iw_in_bounds =
                                            iw >= -(pad_w as isize) && iw < (in_w + pad_w) as isize;
                                        if !iw_in_bounds {
                                            continue;
                                        }

                                        pad_count += 1;

                                        let iw_valid = iw >= 0 && iw < in_w as isize;
                                        if !id_valid || !ih_valid || !iw_valid {
                                            continue;
                                        }

                                        let id = id as usize;
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let x_idx = x_offset + id * in_h * in_w + ih * in_w + iw;
                                        sum = add_fn(sum, x_data[x_idx]);
                                        count += 1;
                                    }
                                }
                            }

                            let divisor = if count_include_pad {
                                pad_count.max(1)
                            } else {
                                count.max(1)
                            };

                            unsafe {
                                out_ptr.write(out_idx, div_fn(sum, divisor));
                            }
                        }
                    }
                }
            });
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![zero; batch_size * channels * spatial_out];

            for b in 0..batch_size {
                for c in 0..channels {
                    let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                    let out_offset = b * channels * spatial_out + c * spatial_out;

                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut sum = zero;
                                let mut count = 0usize;

                                // Track count for count_include_pad (positions within padded bounds)
                                let mut pad_count = 0usize;

                                for kd in 0..kernel_d {
                                    let id = (od * stride_d + kd) as isize - pad_d as isize;
                                    // Check if within padded bounds (not ceil_mode extension)
                                    let id_in_bounds =
                                        id >= -(pad_d as isize) && id < (in_d + pad_d) as isize;
                                    if !id_in_bounds {
                                        continue; // ceil_mode extension - skip entirely
                                    }
                                    let id_valid = id >= 0 && id < in_d as isize;

                                    for kh in 0..kernel_h {
                                        let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                        let ih_in_bounds =
                                            ih >= -(pad_h as isize) && ih < (in_h + pad_h) as isize;
                                        if !ih_in_bounds {
                                            continue;
                                        }
                                        let ih_valid = ih >= 0 && ih < in_h as isize;

                                        for kw in 0..kernel_w {
                                            let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                            let iw_in_bounds = iw >= -(pad_w as isize)
                                                && iw < (in_w + pad_w) as isize;
                                            if !iw_in_bounds {
                                                continue;
                                            }

                                            // Position is within padded bounds
                                            pad_count += 1;

                                            let iw_valid = iw >= 0 && iw < in_w as isize;
                                            if !id_valid || !ih_valid || !iw_valid {
                                                continue; // In padding zone - count but don't add
                                            }

                                            let id = id as usize;
                                            let ih = ih as usize;
                                            let iw = iw as usize;
                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            sum = add_fn(sum, x_data[x_idx]);
                                            count += 1;
                                        }
                                    }
                                }

                                let divisor = if count_include_pad {
                                    pad_count.max(1) // Positions within padded bounds
                                } else {
                                    count.max(1) // Only actual valid positions
                                };

                                output[out_idx] = div_fn(sum, divisor);
                            }
                        }
                    }
                }
            }
            output
        }
    };

    let out_shape = Shape::from(vec![batch_size, channels, out_d, out_h, out_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

// ============================================================================
// Avg Pool 2D - delegates to 3D
// ============================================================================

/// 2D average pooling for f32.
pub fn avg_pool2d_f32(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_f32(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

/// 2D average pooling for f64.
pub fn avg_pool2d_f64(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_f64(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

/// 2D average pooling for f16.
pub fn avg_pool2d_f16(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_f16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

/// 2D average pooling for bf16.
pub fn avg_pool2d_bf16(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_bf16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

// ============================================================================
// Avg Pool 1D - delegates to 3D
// ============================================================================

/// 1D average pooling for f32.
pub fn avg_pool1d_f32(
    x: HostTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> HostTensor {
    let x_3d = expand_1d_to_3d(&x);
    let result = avg_pool3d_f32(
        x_3d,
        [1, 1, kernel_size],
        [1, 1, stride],
        [0, 0, padding],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_1d(result)
}

