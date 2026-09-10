//! Deformable convolution implementation using im2col + GEMM.
//!
//! Deformable convolution applies learned offsets to the sampling grid,
//! allowing the network to adaptively adjust its receptive field.
//!
//! Optimization approach:
//! - Build deformable im2col matrix with bilinear-interpolated samples
//! - Use optimized GEMM for the actual convolution
//! - Rayon parallelism over batch dimension

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::DType;
use ruda_core::{bytes::Bytes, tensor::Shape};

use ruda_core::tensor::host::{HostTensor, Layout};

/// Build deformable im2col matrix for one batch sample and weight group.
///
/// Fills `col` with shape [col_len, spatial_out] where each column holds
/// bilinear-interpolated and optionally masked samples for one output position.
#[allow(clippy::too_many_arguments)]
fn deform_im2col_f32(
    col: &mut [f32],
    x_data: &[f32],
    offset_data: &[f32],
    mask_data: Option<&[f32]>,
    b: usize,
    ic_start: usize,
    channels_per_weight_group: usize,
    channels_per_offset_group: usize,
    channels_in: usize,
    offset_groups: usize,
    offset_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    in_h: usize,
    in_w: usize,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    spatial_out: usize,
) {
    for oh in 0..out_h {
        for ow in 0..out_w {
            let spatial_idx = oh * out_w + ow;

            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let base_h = (oh * stride[0] + kh * dilation[0]) as f32 - padding[0] as f32;
                    let base_w = (ow * stride[1] + kw * dilation[1]) as f32 - padding[1] as f32;

                    for ic in 0..channels_per_weight_group {
                        let global_ic = ic_start + ic;
                        let offset_group = global_ic / channels_per_offset_group;

                        let kernel_idx = kh * kernel_w + kw;
                        let offset_idx_h = offset_group * kernel_h * kernel_w * 2 + kernel_idx * 2;
                        let offset_idx_w = offset_idx_h + 1;

                        let offset_h_flat = b * offset_channels * spatial_out
                            + offset_idx_h * spatial_out
                            + spatial_idx;
                        let offset_w_flat = b * offset_channels * spatial_out
                            + offset_idx_w * spatial_out
                            + spatial_idx;

                        let offset_h = offset_data[offset_h_flat];
                        let offset_w = offset_data[offset_w_flat];

                        let sample_h = base_h + offset_h;
                        let sample_w = base_w + offset_w;

                        let mut val = bilinear_interpolate(
                            x_data,
                            b,
                            global_ic,
                            in_h,
                            in_w,
                            channels_in,
                            sample_h,
                            sample_w,
                        );

                        if let Some(md) = mask_data {
                            let mask_idx_base = offset_group * kernel_h * kernel_w + kernel_idx;
                            let mask_idx = b * (offset_groups * kernel_h * kernel_w) * spatial_out
                                + mask_idx_base * spatial_out
                                + spatial_idx;
                            val *= md[mask_idx];
                        }

                        let col_row = kh * kernel_w * channels_per_weight_group
                            + kw * channels_per_weight_group
                            + ic;
                        col[col_row * spatial_out + spatial_idx] = val;
                    }
                }
            }
        }
    }
}

/// Deformable 2D convolution using im2col + GEMM.
///
/// # Arguments
/// * `x` - Input tensor [batch, channels_in, height, width]
/// * `offset` - Offset tensor \[batch, offset_groups * kernel_h * kernel_w * 2, out_h, out_w\]
/// * `weight` - Weight tensor \[channels_out, channels_in/weight_groups, kernel_h, kernel_w\]
/// * `mask` - Optional mask tensor \[batch, offset_groups * kernel_h * kernel_w, out_h, out_w\]
/// * `bias` - Optional bias tensor \[channels_out\]
/// * `stride` - Stride \[stride_h, stride_w\]
/// * `padding` - Padding \[pad_h, pad_w\]
/// * `dilation` - Dilation \[dil_h, dil_w\]
/// * `weight_groups` - Number of weight groups
/// * `offset_groups` - Number of offset groups
#[allow(clippy::too_many_arguments)]
pub fn deform_conv2d_f32(
    x: HostTensor,
    offset: HostTensor,
    weight: HostTensor,
    mask: Option<HostTensor>,
    bias: Option<HostTensor>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    weight_groups: usize,
    offset_groups: usize,
) -> HostTensor {
    let x = x.to_contiguous();
    let offset = offset.to_contiguous();
    let weight = weight.to_contiguous();
    let mask = mask.map(|m| m.to_contiguous());
    let bias = bias.map(|b| b.to_contiguous());

    let x_shape = x.layout().shape();
    let weight_shape = weight.layout().shape();
    let offset_shape = offset.layout().shape();

    let batch = x_shape[0];
    let channels_in = x_shape[1];
    let in_h = x_shape[2];
    let in_w = x_shape[3];

    let channels_out = weight_shape[0];
    let channels_per_weight_group = weight_shape[1]; // channels_in / weight_groups
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    let out_h = offset_shape[2];
    let out_w = offset_shape[3];

    let x_data: &[f32] = x.storage();
    let offset_data: &[f32] = offset.storage();
    let weight_data: &[f32] = weight.storage();
    let mask_data: Option<&[f32]> = mask.as_ref().map(|m| m.storage());
    let bias_data: Option<&[f32]> = bias.as_ref().map(|b| b.storage());

    let channels_per_offset_group = channels_in / offset_groups;
    let out_channels_per_weight_group = channels_out / weight_groups;
    let spatial_out = out_h * out_w;
    let col_len = channels_per_weight_group * kernel_h * kernel_w;
    let offset_channels = offset_shape[1];

    // Flatten weights to [channels_out, col_len] for GEMM
    // Layout: [oc, ic, kh, kw] -> [oc, kh * kw * ic]
    let mut w_flat = vec![0.0f32; channels_out * col_len];
    for oc in 0..channels_out {
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                for ic in 0..channels_per_weight_group {
                    let w_idx = oc * channels_per_weight_group * kernel_h * kernel_w
                        + ic * kernel_h * kernel_w
                        + kh * kernel_w
                        + kw;
                    let flat_idx = oc * col_len
                        + kh * kernel_w * channels_per_weight_group
                        + kw * channels_per_weight_group
                        + ic;
                    w_flat[flat_idx] = weight_data[w_idx];
                }
            }
        }
    }

    #[cfg(feature = "rayon")]
    let output = {
        use rayon::prelude::*;

        let results: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let mut batch_output = vec![0.0f32; channels_out * spatial_out];

                // Process each weight group
                for g in 0..weight_groups {
                    let ic_start = g * channels_per_weight_group;
                    let oc_start = g * out_channels_per_weight_group;

                    // Build deformable im2col for this batch and group
                    let mut col = vec![0.0f32; col_len * spatial_out];

                    deform_im2col_f32(
                        &mut col,
                        x_data,
                        offset_data,
                        mask_data,
                        b,
                        ic_start,
                        channels_per_weight_group,
                        channels_per_offset_group,
                        channels_in,
                        offset_groups,
                        offset_channels,
                        kernel_h,
                        kernel_w,
                        out_h,
                        out_w,
                        in_h,
                        in_w,
                        stride,
                        padding,
                        dilation,
                        spatial_out,
                    );

                    // GEMM: w_group[out_c_per_wg, col_len] @ col[col_len, spatial_out]
                    // Result: [out_c_per_wg, spatial_out]
                    let w_start = oc_start * col_len;
                    let w_end = w_start + out_channels_per_weight_group * col_len;
                    let w_group = &w_flat[w_start..w_end];

                    let result = gemm_f32(
                        w_group,
                        &col,
                        out_channels_per_weight_group,
                        col_len,
                        spatial_out,
                    );

                    // Copy to output
                    for oc_local in 0..out_channels_per_weight_group {
                        let oc = oc_start + oc_local;
                        for s in 0..spatial_out {
                            batch_output[oc * spatial_out + s] = result[oc_local * spatial_out + s];
                        }
                    }
                }

                // Add bias
                if let Some(bd) = bias_data {
                    for oc in 0..channels_out {
                        for s in 0..spatial_out {
                            batch_output[oc * spatial_out + s] += bd[oc];
                        }
                    }
                }

                batch_output
            })
            .collect();

        // Flatten results
        let mut output = vec![0.0f32; batch * channels_out * spatial_out];
        for (b, batch_out) in results.into_iter().enumerate() {
            let start = b * channels_out * spatial_out;
            output[start..start + channels_out * spatial_out].copy_from_slice(&batch_out);
        }
        output
    };

    #[cfg(not(feature = "rayon"))]
    let output = {
        let mut output = vec![0.0f32; batch * channels_out * spatial_out];

        for b in 0..batch {
            // Process each weight group
            for g in 0..weight_groups {
                let ic_start = g * channels_per_weight_group;
                let oc_start = g * out_channels_per_weight_group;

                // Build deformable im2col for this batch and group
                let mut col = vec![0.0f32; col_len * spatial_out];

                deform_im2col_f32(
                    &mut col,
                    x_data,
                    offset_data,
                    mask_data,
                    b,
                    ic_start,
                    channels_per_weight_group,
                    channels_per_offset_group,
                    channels_in,
                    offset_groups,
                    offset_channels,
                    kernel_h,
                    kernel_w,
                    out_h,
                    out_w,
                    in_h,
                    in_w,
                    stride,
                    padding,
                    dilation,
                    spatial_out,
                );

                // GEMM
                let w_start = oc_start * col_len;
                let w_end = w_start + out_channels_per_weight_group * col_len;
                let w_group = &w_flat[w_start..w_end];

                let result = gemm_f32(
                    w_group,
                    &col,
                    out_channels_per_weight_group,
                    col_len,
                    spatial_out,
                );

                // Copy to output
                for oc_local in 0..out_channels_per_weight_group {
                    let oc = oc_start + oc_local;
                    for s in 0..spatial_out {
                        let out_idx = b * channels_out * spatial_out + oc * spatial_out + s;
                        output[out_idx] = result[oc_local * spatial_out + s];
                    }
                }
            }

            // Add bias
            if let Some(bd) = bias_data {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..channels_out {
                    for s in 0..spatial_out {
                        let idx = b * channels_out * spatial_out + oc * spatial_out + s;
                        output[idx] += bd[oc];
                    }
                }
            }
        }

        output
    };

    let out_shape = Shape::from(vec![batch, channels_out, out_h, out_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::F32,
    )
}

/// GEMM: C = A @ B where A is [m, k], B is [k, n], result C is [m, n]
#[inline]
fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,          // dst_cs: column stride = 1 (row-major)
            n as isize, // dst_rs: row stride = n (row-major)
            false,
            a.as_ptr(),
            1,          // lhs_cs: column stride = 1 (row-major)
            k as isize, // lhs_rs: row stride = k (row-major)
            b.as_ptr(),
            1,          // rhs_cs: column stride = 1 (row-major)
            n as isize, // rhs_rs: row stride = n (row-major)
            0.0,        // alpha: dst = alpha*dst + beta*lhs*rhs
            1.0,        // beta
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
    c
}

/// Bilinear interpolation for sampling at fractional coordinates.
#[inline]
#[allow(clippy::too_many_arguments)]
fn bilinear_interpolate(
    data: &[f32],
    batch: usize,
    channel: usize,
    height: usize,
    width: usize,
    channels: usize,
    h: f32,
    w: f32,
) -> f32 {
    // Out of bounds check
    if h <= -1.0 || h >= height as f32 || w <= -1.0 || w >= width as f32 {
        return 0.0;
    }

    let h_low = h.floor();
    let w_low = w.floor();
    let h_high = (h_low + 1.0) as usize;
    let w_high = (w_low + 1.0) as usize;

    let base = batch * channels * height * width + channel * height * width;

    let v1 = if h_low >= 0.0 && w_low >= 0.0 {
        data[base + (h_low as usize) * width + (w_low as usize)]
    } else {
        0.0
    };
    let v2 = if h_low >= 0.0 && w_high < width {
        data[base + (h_low as usize) * width + w_high]
    } else {
        0.0
    };
    let v3 = if h_high < height && w_low >= 0.0 {
        data[base + h_high * width + (w_low as usize)]
    } else {
        0.0
    };
    let v4 = if h_high < height && w_high < width {
        data[base + h_high * width + w_high]
    } else {
        0.0
    };

    let lh = h - h_low;
    let lw = w - w_low;
    let hh = 1.0 - lh;
    let hw = 1.0 - lw;

    hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4
}

mod backward;
pub use backward::*;

mod double;
pub use double::*;

