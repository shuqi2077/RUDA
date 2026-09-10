use super::*;

// ============================================================================
// Adaptive Avg Pool 3D - core implementation
// ============================================================================

adaptive_avg_pool3d_typed!(
    adaptive_avg_pool3d_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |sum, count| sum / count as f32
);
adaptive_avg_pool3d_typed!(
    adaptive_avg_pool3d_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |sum, count| sum / count as f64
);
adaptive_avg_pool3d_typed!(
    adaptive_avg_pool3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |sum: f16, count| f16::from_f32(sum.to_f32() / count as f32)
);

pub fn adaptive_avg_pool3d_bf16(x: HostTensor, output_size: [usize; 3]) -> HostTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let result_f32 = adaptive_avg_pool3d_f32(x_f32, output_size);
    convert_f32_to_bf16(&result_f32)
}

/// Generic 3D adaptive average pooling implementation.
fn adaptive_avg_pool3d_impl<T>(
    x: HostTensor,
    output_size: [usize; 3],
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

    let [out_d, out_h, out_w] = output_size;
    let spatial_out = out_d * out_h * out_w;
    let x_data: &[T] = x.storage();

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![zero; batch_size * channels * spatial_out];
            let out_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(output.as_mut_ptr());

            (0..batch_size).into_par_iter().for_each(|b| {
                (0..channels).into_par_iter().for_each(|c| {
                    let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                    let out_offset = b * channels * spatial_out + c * spatial_out;

                    for od in 0..out_d {
                        // Compute input range for this output position
                        // start = floor(out * in / out_size), end = ceil((out+1) * in / out_size)
                        let d_start = (od * in_d) / out_d;
                        let d_end = ((od + 1) * in_d).div_ceil(out_d);

                        for oh in 0..out_h {
                            let h_start = (oh * in_h) / out_h;
                            let h_end = ((oh + 1) * in_h).div_ceil(out_h);

                            for ow in 0..out_w {
                                let w_start = (ow * in_w) / out_w;
                                let w_end = ((ow + 1) * in_w).div_ceil(out_w);

                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut sum = zero;
                                let mut count = 0usize;

                                for id in d_start..d_end {
                                    for ih in h_start..h_end {
                                        for iw in w_start..w_end {
                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            sum = add_fn(sum, x_data[x_idx]);
                                            count += 1;
                                        }
                                    }
                                }

                                unsafe {
                                    out_ptr.write(out_idx, div_fn(sum, count.max(1)));
                                }
                            }
                        }
                    }
                });
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
                        let d_start = (od * in_d) / out_d;
                        let d_end = ((od + 1) * in_d).div_ceil(out_d);

                        for oh in 0..out_h {
                            let h_start = (oh * in_h) / out_h;
                            let h_end = ((oh + 1) * in_h).div_ceil(out_h);

                            for ow in 0..out_w {
                                let w_start = (ow * in_w) / out_w;
                                let w_end = ((ow + 1) * in_w).div_ceil(out_w);

                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut sum = zero;
                                let mut count = 0usize;

                                for id in d_start..d_end {
                                    for ih in h_start..h_end {
                                        for iw in w_start..w_end {
                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            sum = add_fn(sum, x_data[x_idx]);
                                            count += 1;
                                        }
                                    }
                                }

                                output[out_idx] = div_fn(sum, count.max(1));
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
// Adaptive Avg Pool 2D - delegates to 3D
// ============================================================================

/// 2D adaptive average pooling for f32.
pub fn adaptive_avg_pool2d_f32(x: HostTensor, output_size: [usize; 2]) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_f32(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

/// 2D adaptive average pooling for f64.
pub fn adaptive_avg_pool2d_f64(x: HostTensor, output_size: [usize; 2]) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_f64(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

/// 2D adaptive average pooling for f16.
pub fn adaptive_avg_pool2d_f16(x: HostTensor, output_size: [usize; 2]) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_f16(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

/// 2D adaptive average pooling for bf16.
pub fn adaptive_avg_pool2d_bf16(x: HostTensor, output_size: [usize; 2]) -> HostTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_bf16(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

// ============================================================================
// Adaptive Avg Pool 1D - delegates to 3D
// ============================================================================

/// 1D adaptive average pooling for f32.
pub fn adaptive_avg_pool1d_f32(x: HostTensor, output_size: usize) -> HostTensor {
    let x_3d = expand_1d_to_3d(&x);
    let result = adaptive_avg_pool3d_f32(x_3d, [1, 1, output_size]);
    squeeze_3d_to_1d(result)
}

