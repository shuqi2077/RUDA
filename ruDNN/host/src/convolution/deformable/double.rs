use super::*;

/// f64 version of deform_conv2d using im2col + GEMM.
#[allow(clippy::too_many_arguments)]
pub fn deform_conv2d_f64(
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
    let channels_per_weight_group = weight_shape[1];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    let out_h = offset_shape[2];
    let out_w = offset_shape[3];

    let x_data: &[f64] = x.storage();
    let offset_data: &[f64] = offset.storage();
    let weight_data: &[f64] = weight.storage();
    let mask_data: Option<&[f64]> = mask.as_ref().map(|m| m.storage());
    let bias_data: Option<&[f64]> = bias.as_ref().map(|b| b.storage());

    let channels_per_offset_group = channels_in / offset_groups;
    let out_channels_per_weight_group = channels_out / weight_groups;
    let spatial_out = out_h * out_w;
    let col_len = channels_per_weight_group * kernel_h * kernel_w;

    // Flatten weights
    let mut w_flat = vec![0.0f64; channels_out * col_len];
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

    let mut output = vec![0.0f64; batch * channels_out * spatial_out];

    for b in 0..batch {
        for g in 0..weight_groups {
            let ic_start = g * channels_per_weight_group;
            let oc_start = g * out_channels_per_weight_group;

            let mut col = vec![0.0f64; col_len * spatial_out];

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let spatial_idx = oh * out_w + ow;

                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let base_h =
                                (oh * stride[0] + kh * dilation[0]) as f64 - padding[0] as f64;
                            let base_w =
                                (ow * stride[1] + kw * dilation[1]) as f64 - padding[1] as f64;

                            for ic in 0..channels_per_weight_group {
                                let global_ic = ic_start + ic;
                                let offset_group = global_ic / channels_per_offset_group;

                                let kernel_idx = kh * kernel_w + kw;
                                let offset_idx_h =
                                    offset_group * kernel_h * kernel_w * 2 + kernel_idx * 2;
                                let offset_idx_w = offset_idx_h + 1;

                                let offset_h_flat = b * offset_shape[1] * spatial_out
                                    + offset_idx_h * spatial_out
                                    + spatial_idx;
                                let offset_w_flat = b * offset_shape[1] * spatial_out
                                    + offset_idx_w * spatial_out
                                    + spatial_idx;

                                let offset_h = offset_data[offset_h_flat];
                                let offset_w = offset_data[offset_w_flat];

                                let sample_h = base_h + offset_h;
                                let sample_w = base_w + offset_w;

                                let mut val = bilinear_interpolate_f64(
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
                                    let mask_idx_base =
                                        offset_group * kernel_h * kernel_w + kernel_idx;
                                    let mask_idx =
                                        b * (offset_groups * kernel_h * kernel_w) * spatial_out
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

            // GEMM
            let w_start = oc_start * col_len;
            let w_end = w_start + out_channels_per_weight_group * col_len;
            let w_group = &w_flat[w_start..w_end];

            let result = gemm_f64(
                w_group,
                &col,
                out_channels_per_weight_group,
                col_len,
                spatial_out,
            );

            for oc_local in 0..out_channels_per_weight_group {
                let oc = oc_start + oc_local;
                for s in 0..spatial_out {
                    let out_idx = b * channels_out * spatial_out + oc * spatial_out + s;
                    output[out_idx] = result[oc_local * spatial_out + s];
                }
            }
        }

        if let Some(bd) = bias_data {
            for (oc, &bias_val) in bd.iter().enumerate() {
                for s in 0..spatial_out {
                    let idx = b * channels_out * spatial_out + oc * spatial_out + s;
                    output[idx] += bias_val;
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch, channels_out, out_h, out_w]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::F64,
    )
}

#[inline]
fn gemm_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            1,
            n as isize,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
    c
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn bilinear_interpolate_f64(
    data: &[f64],
    batch: usize,
    channel: usize,
    height: usize,
    width: usize,
    channels: usize,
    h: f64,
    w: f64,
) -> f64 {
    if h <= -1.0 || h >= height as f64 || w <= -1.0 || w >= width as f64 {
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
