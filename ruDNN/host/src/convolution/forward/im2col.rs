use super::*;

/// Generic 3D convolution implementation using tiled im2col.
///
/// Tiled approach processes output in TILE_SIZE chunks:
/// - Reduces memory usage (smaller im2col buffer per tile)
/// - Enables tile-level parallelism
/// - Improves cache utilization
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn conv3d_impl<T: bytemuck::Pod + Clone + Copy + ruda_core::tensor::element::Element + Send + Sync>(
    x: HostTensor,
    weight: HostTensor,
    bias: Option<HostTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
    add_fn: fn(T, T) -> T,
) -> HostTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [stride_d, stride_h, stride_w] = options.stride;
    let [pad_d, pad_h, pad_w] = options.padding;
    let groups = options.groups;
    let out_channels_per_group = channels_out / groups;

    let out_d = calculate_conv_output_size(kernel_d, stride_d, pad_d, options.dilation[0], in_d);
    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, options.dilation[1], in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, options.dilation[2], in_w);

    // Validate sizes won't overflow index calculations
    let _total = [batch_size, channels_out, out_d, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv: output tensor dimensions would overflow index calculations");
    let _col_total = [channels_per_group, kernel_d, kernel_h, kernel_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv: kernel dimensions would overflow index calculations");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let col_len = channels_per_group * kernel_d * kernel_h * kernel_w;
    let spatial_out = out_d * out_h * out_w;

    let [dilation_d, dilation_h, dilation_w] = options.dilation;

    // Tile size for processing output pixels. Larger = better GEMM utilization,
    // smaller = more parallelism and better cache usage. 512 is a good balance.
    const TILE_SIZE: usize = 512;
    let num_tiles = spatial_out.div_ceil(TILE_SIZE);

    // Flatten kernel [c_out, c_in, kd, kh, kw] -> [c_out, kd, kh, kw, c_in]
    // for GEMM. Expressed as a 2D transpose of (c_in, k_spatial) per c_out
    // so the compiler can unroll the k_spatial inner loop; the older 5-
    // nested formulation had a 10-term index expression LLVM wouldn't
    // autovectorize.
    let k_spatial = kernel_d * kernel_h * kernel_w;
    let mut w_flat = vec![zero; channels_out * col_len];
    for c_out in 0..channels_out {
        let src_base = c_out * channels_per_group * k_spatial;
        let dst_base = c_out * col_len;
        for c_in in 0..channels_per_group {
            let src_row = src_base + c_in * k_spatial;
            for k in 0..k_spatial {
                w_flat[dst_base + k * channels_per_group + c_in] = w_data[src_row + k];
            }
        }
    }

    // Convert input to NHWC layout for cache-friendly access in im2col.
    // Loop order (c innermost, spatial outer) is intentional: on aarch64
    // M3 Max, strided loads + contiguous stores beat the inverse by
    // 10-35% per layer. Load prefetchers outrun the store write buffer.
    let nhwc_stride = (
        in_d * in_h * in_w * channels_in,
        in_h * in_w * channels_in,
        in_w * channels_in,
        channels_in,
        1,
    );
    let mut x_nhwc = vec![zero; batch_size * in_d * in_h * in_w * channels_in];
    for b in 0..batch_size {
        for d in 0..in_d {
            for h in 0..in_h {
                for w in 0..in_w {
                    for c in 0..channels_in {
                        let src_idx = b * channels_in * in_d * in_h * in_w
                            + c * in_d * in_h * in_w
                            + d * in_h * in_w
                            + h * in_w
                            + w;
                        let dst_idx = b * nhwc_stride.0
                            + d * nhwc_stride.1
                            + h * nhwc_stride.2
                            + w * nhwc_stride.3
                            + c;
                        x_nhwc[dst_idx] = x_data[src_idx];
                    }
                }
            }
        }
    }

    // Use tiled parallel execution
    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut dst = vec![zero; batch_size * channels_out * spatial_out];
            let dst_ptr = ruda_core::tensor::host::parallel::SendMutPtr::new(dst.as_mut_ptr());

            // Process batches and tiles in parallel (nested parallelism)
            (0..batch_size).into_par_iter().for_each(|b| {
                (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                    let tile_start = tile_idx * TILE_SIZE;
                    let tile_end = (tile_start + TILE_SIZE).min(spatial_out);
                    let tile_size = tile_end - tile_start;

                    // Process each group separately
                    for g in 0..groups {
                        let in_c_start = g * channels_per_group;
                        let out_c_start = g * out_channels_per_group;

                        // Build im2col for this tile and group
                        let mut col_tile = vec![zero; col_len * tile_size];

                        im2col_3d_tile(
                            &mut col_tile,
                            &x_nhwc,
                            tile_start,
                            tile_end,
                            out_h,
                            out_w,
                            kernel_d,
                            kernel_h,
                            kernel_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            in_d,
                            in_h,
                            in_w,
                            channels_per_group,
                            col_len,
                            b,
                            in_c_start,
                            nhwc_stride,
                        );

                        // Get weight slice for this group
                        let w_start = out_c_start * col_len;
                        let w_end = w_start + out_channels_per_group * col_len;
                        let w_group = &w_flat[w_start..w_end];

                        // GEMM: w_group[out_c_per_group, col_len] @ col_tile[tile_size, col_len]^T
                        let result = gemm_fn(
                            w_group,
                            &col_tile,
                            out_channels_per_group,
                            col_len,
                            tile_size,
                        );

                        // Write results to output for this group's output channels
                        for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                            for c_out in 0..out_channels_per_group {
                                let dst_idx = b * channels_out * spatial_out
                                    + (out_c_start + c_out) * spatial_out
                                    + global_idx;
                                let res_idx = c_out * tile_size + local_idx;
                                unsafe {
                                    debug_assert!(
                                        dst_idx < batch_size * channels_out * spatial_out
                                    );
                                    dst_ptr.write(dst_idx, result[res_idx]);
                                }
                            }
                        }
                    }
                });
            });
            dst
        }
        #[cfg(not(feature = "rayon"))]
        {
            // Sequential path with tiling
            let mut output = vec![zero; batch_size * channels_out * spatial_out];

            for b in 0..batch_size {
                for tile_idx in 0..num_tiles {
                    let tile_start = tile_idx * TILE_SIZE;
                    let tile_end = (tile_start + TILE_SIZE).min(spatial_out);
                    let tile_size = tile_end - tile_start;

                    // Process each group separately
                    for g in 0..groups {
                        let in_c_start = g * channels_per_group;
                        let out_c_start = g * out_channels_per_group;

                        let mut col_tile = vec![zero; col_len * tile_size];

                        im2col_3d_tile(
                            &mut col_tile,
                            &x_nhwc,
                            tile_start,
                            tile_end,
                            out_h,
                            out_w,
                            kernel_d,
                            kernel_h,
                            kernel_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            in_d,
                            in_h,
                            in_w,
                            channels_per_group,
                            col_len,
                            b,
                            in_c_start,
                            nhwc_stride,
                        );

                        // Get weight slice for this group
                        let w_start = out_c_start * col_len;
                        let w_end = w_start + out_channels_per_group * col_len;
                        let w_group = &w_flat[w_start..w_end];

                        // GEMM for this group
                        let result = gemm_fn(
                            w_group,
                            &col_tile,
                            out_channels_per_group,
                            col_len,
                            tile_size,
                        );

                        // Write results to output for this group's output channels
                        for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                            for c_out in 0..out_channels_per_group {
                                let dst_idx = b * channels_out * spatial_out
                                    + (out_c_start + c_out) * spatial_out
                                    + global_idx;
                                let res_idx = c_out * tile_size + local_idx;
                                output[dst_idx] = result[res_idx];
                            }
                        }
                    }
                }
            }
            output
        }
    };

    let mut output = output;
    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial_out,
            add_fn,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels_out, out_d, out_h, out_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

/// Build im2col tile for a range of output positions.
///
/// Fills `col_tile` with shape [tile_size, col_len] where each row is a flattened
/// patch from the NHWC input for one output position.
#[allow(clippy::too_many_arguments)]
pub(super) fn im2col_3d_tile<T: bytemuck::Pod + Copy>(
    col_tile: &mut [T],
    x_nhwc: &[T],
    tile_start: usize,
    tile_end: usize,
    out_h: usize,
    out_w: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    channels_per_group: usize,
    col_len: usize,
    b: usize,
    in_c_start: usize,
    nhwc_stride: (usize, usize, usize, usize, usize),
) {
    for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
        let out_d_idx = global_idx / (out_h * out_w);
        let rem = global_idx % (out_h * out_w);
        let out_h_idx = rem / out_w;
        let out_w_idx = rem % out_w;

        let mut col_offset = 0;
        for kd in 0..kernel_d {
            let id = (out_d_idx * stride_d + kd * dilation_d) as isize - pad_d as isize;
            for kh in 0..kernel_h {
                let ih = (out_h_idx * stride_h + kh * dilation_h) as isize - pad_h as isize;
                for kw in 0..kernel_w {
                    let iw = (out_w_idx * stride_w + kw * dilation_w) as isize - pad_w as isize;

                    if id >= 0
                        && id < in_d as isize
                        && ih >= 0
                        && ih < in_h as isize
                        && iw >= 0
                        && iw < in_w as isize
                    {
                        let id = id as usize;
                        let ih = ih as usize;
                        let iw = iw as usize;
                        let inp_base = b * nhwc_stride.0
                            + id * nhwc_stride.1
                            + ih * nhwc_stride.2
                            + iw * nhwc_stride.3
                            + in_c_start;
                        for c in 0..channels_per_group {
                            col_tile[local_idx * col_len + col_offset] = x_nhwc[inp_base + c];
                            col_offset += 1;
                        }
                    } else {
                        // Padding: skip channels_per_group positions (already zero)
                        col_offset += channels_per_group;
                    }
                }
            }
        }
    }
}

