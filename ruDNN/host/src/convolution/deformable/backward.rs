use super::*;
use ruda_core::tensor::element::Element;

trait DeformFloat: num_traits::Float + Element + bytemuck::Pod + core::ops::AddAssign {
    fn from_index(index: usize) -> Self;
    fn index(self) -> usize;
    fn interpolate(data: &[Self], position: [usize; 5], coordinates: [Self; 2]) -> Self;
}

macro_rules! deform_float {
    ($elem:ty, $interpolate:path) => {
        impl DeformFloat for $elem {
            #[inline]
            fn from_index(index: usize) -> Self { index as Self }
            #[inline]
            fn index(self) -> usize { self as usize }
            #[inline]
            fn interpolate(data: &[Self], position: [usize; 5], coordinates: [Self; 2]) -> Self {
                let [batch, channel, height, width, channels] = position;
                let [h, w] = coordinates;
                $interpolate(data, batch, channel, height, width, channels, h, w)
            }
        }
    };
}

deform_float!(f32, super::bilinear_interpolate);
deform_float!(f64, super::double::bilinear_interpolate_f64);

macro_rules! backward_entry {
    ($name:ident, $elem:ty) => {
        #[doc = concat!("Backward pass for deformable 2D convolution (", stringify!($elem), ").")]
        /// Returns (x_grad, offset_grad, weight_grad, mask_grad, bias_grad).
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            x: HostTensor, offset: HostTensor, weight: HostTensor,
            mask: Option<HostTensor>, bias: Option<HostTensor>, output_grad: HostTensor,
            stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2],
            weight_groups: usize, offset_groups: usize,
        ) -> (HostTensor, HostTensor, HostTensor, Option<HostTensor>, Option<HostTensor>) {
            deform_conv2d_backward::<$elem>(
                x, offset, weight, mask, bias, output_grad,
                stride, padding, dilation, weight_groups, offset_groups,
            )
        }
    };
}

backward_entry!(deform_conv2d_backward_f32, f32);
backward_entry!(deform_conv2d_backward_f64, f64);

/// Shared backward pass, computed in the input element type.
///
/// Returns (x_grad, offset_grad, weight_grad, mask_grad, bias_grad).
#[allow(clippy::too_many_arguments)]
fn deform_conv2d_backward<E: DeformFloat>(
    x: HostTensor,
    offset: HostTensor,
    weight: HostTensor,
    mask: Option<HostTensor>,
    bias: Option<HostTensor>,
    output_grad: HostTensor,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    weight_groups: usize,
    offset_groups: usize,
) -> (
    HostTensor,
    HostTensor,
    HostTensor,
    Option<HostTensor>,
    Option<HostTensor>,
) {
    let x = x.to_contiguous();
    let offset = offset.to_contiguous();
    let weight = weight.to_contiguous();
    let mask = mask.map(|m| m.to_contiguous());
    let output_grad = output_grad.to_contiguous();

    let x_shape = x.layout().shape();
    let weight_shape = weight.layout().shape();
    let offset_shape = offset.layout().shape();
    let out_grad_shape = output_grad.layout().shape();

    let batch = x_shape[0];
    let channels_in = x_shape[1];
    let in_h = x_shape[2];
    let in_w = x_shape[3];

    let channels_out = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    let out_h = out_grad_shape[2];
    let out_w = out_grad_shape[3];

    let x_data: &[E] = x.storage();
    let offset_data: &[E] = offset.storage();
    let weight_data: &[E] = weight.storage();
    let mask_data: Option<&[E]> = mask.as_ref().map(|m| m.storage());
    let out_grad_data: &[E] = output_grad.storage();

    let channels_per_offset_group = channels_in / offset_groups;
    let channels_per_weight_group = channels_in / weight_groups;
    let out_channels_per_weight_group = channels_out / weight_groups;

    // Initialize gradients
    let mut x_grad = vec![E::zero(); batch * channels_in * in_h * in_w];
    let mut offset_grad = vec![E::zero(); batch * offset_shape[1] * out_h * out_w];
    let mut weight_grad = vec![E::zero(); weight_shape.num_elements()];
    let mut mask_grad = mask
        .as_ref()
        .map(|m| vec![E::zero(); m.layout().shape().num_elements()]);
    let mut bias_grad = bias.as_ref().map(|_| vec![E::zero(); channels_out]);

    // Compute bias gradient (sum over batch and spatial dimensions)
    if let Some(ref mut bg) = bias_grad {
        for b in 0..batch {
            for (oc, bg_oc) in bg.iter_mut().enumerate() {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let idx =
                            b * channels_out * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                        *bg_oc += out_grad_data[idx];
                    }
                }
            }
        }
    }

    // Main backward loop
    for b in 0..batch {
        for oc in 0..channels_out {
            let weight_group = oc / out_channels_per_weight_group;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx =
                        b * channels_out * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                    let grad_out = out_grad_data[out_idx];

                    let ic_start = weight_group * channels_per_weight_group;
                    let ic_end = ic_start + channels_per_weight_group;

                    for ic in ic_start..ic_end {
                        let offset_group = ic / channels_per_offset_group;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let base_h = E::from_index(oh * stride[0]) + E::from_index(kh * dilation[0])
                                    - E::from_index(padding[0]);
                                let base_w = E::from_index(ow * stride[1]) + E::from_index(kw * dilation[1])
                                    - E::from_index(padding[1]);

                                let kernel_idx = kh * kernel_w + kw;
                                let offset_idx_h =
                                    offset_group * kernel_h * kernel_w * 2 + kernel_idx * 2;
                                let offset_idx_w = offset_idx_h + 1;

                                let offset_h_flat = b * offset_shape[1] * out_h * out_w
                                    + offset_idx_h * out_h * out_w
                                    + oh * out_w
                                    + ow;
                                let offset_w_flat = b * offset_shape[1] * out_h * out_w
                                    + offset_idx_w * out_h * out_w
                                    + oh * out_w
                                    + ow;

                                let off_h = offset_data[offset_h_flat];
                                let off_w = offset_data[offset_w_flat];

                                let sample_h = base_h + off_h;
                                let sample_w = base_w + off_w;

                                // Get mask value
                                let (mask_val, mask_flat_idx) = if let Some(md) = mask_data {
                                    let mask_idx_base =
                                        offset_group * kernel_h * kernel_w + kernel_idx;
                                    let mask_idx =
                                        b * (offset_groups * kernel_h * kernel_w) * out_h * out_w
                                            + mask_idx_base * out_h * out_w
                                            + oh * out_w
                                            + ow;
                                    (md[mask_idx], Some(mask_idx))
                                } else {
                                    (E::one(), None)
                                };

                                // Weight index
                                let weight_ic = ic - ic_start;
                                let weight_idx = oc
                                    * (channels_per_weight_group * kernel_h * kernel_w)
                                    + weight_ic * kernel_h * kernel_w
                                    + kh * kernel_w
                                    + kw;

                                let w = weight_data[weight_idx];

                                // Get interpolated value for weight gradient
                                let interp_val = E::interpolate(
                                    x_data,
                                    [b, ic, in_h, in_w, channels_in],
                                    [sample_h, sample_w],
                                );

                                // Weight gradient
                                weight_grad[weight_idx] += grad_out * mask_val * interp_val;

                                // Mask gradient
                                if let (Some(mg), Some(midx)) = (&mut mask_grad, mask_flat_idx) {
                                    mg[midx] += grad_out * w * interp_val;
                                }

                                // Input and offset gradients via bilinear interpolation backward
                                let grad_val = grad_out * mask_val * w;
                                bilinear_interpolate_backward::<E>(
                                    &mut x_grad,
                                    &mut offset_grad,
                                    x_data,
                                    b,
                                    ic,
                                    in_h,
                                    in_w,
                                    channels_in,
                                    sample_h,
                                    sample_w,
                                    grad_val,
                                    offset_h_flat,
                                    offset_w_flat,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // Build output tensors
    let x_grad_tensor = HostTensor::new(
        Bytes::from_elems(x_grad),
        Layout::contiguous(x_shape.clone()),
        E::dtype(),
    );
    let offset_grad_tensor = HostTensor::new(
        Bytes::from_elems(offset_grad),
        Layout::contiguous(offset_shape.clone()),
        E::dtype(),
    );
    let weight_grad_tensor = HostTensor::new(
        Bytes::from_elems(weight_grad),
        Layout::contiguous(weight_shape.clone()),
        E::dtype(),
    );
    let mask_grad_tensor = mask_grad.map(|mg| {
        HostTensor::new(
            Bytes::from_elems(mg),
            Layout::contiguous(mask.as_ref().unwrap().layout().shape().clone()),
            E::dtype(),
        )
    });
    let bias_grad_tensor = bias_grad.map(|bg| {
        HostTensor::new(
            Bytes::from_elems(bg),
            Layout::contiguous(Shape::from(vec![channels_out])),
            E::dtype(),
        )
    });

    (
        x_grad_tensor,
        offset_grad_tensor,
        weight_grad_tensor,
        mask_grad_tensor,
        bias_grad_tensor,
    )
}

/// Backward pass for bilinear interpolation - computes gradients for input and offsets.
#[allow(clippy::too_many_arguments)]
#[inline]
fn bilinear_interpolate_backward<E: DeformFloat>(
    x_grad: &mut [E],
    offset_grad: &mut [E],
    x_data: &[E],
    batch: usize,
    channel: usize,
    height: usize,
    width: usize,
    channels: usize,
    h: E,
    w: E,
    grad_val: E,
    offset_h_flat: usize,
    offset_w_flat: usize,
) {
    // Out of bounds check
    if h <= -E::one() || h >= E::from_index(height) || w <= -E::one() || w >= E::from_index(width) {
        return;
    }

    let h_low = h.floor();
    let w_low = w.floor();
    let h_high = h_low + E::one();
    let w_high = w_low + E::one();

    let lh = h - h_low;
    let lw = w - w_low;
    let hh = E::one() - lh;
    let hw = E::one() - lw;

    let base = batch * channels * height * width + channel * height * width;

    // Get input values for offset gradient computation
    let v1 = if h_low >= E::zero() && w_low >= E::zero() {
        x_data[base + h_low.index() * width + w_low.index()]
    } else {
        E::zero()
    };
    let v2 = if h_low >= E::zero() && w_high.index() < width {
        x_data[base + h_low.index() * width + w_high.index()]
    } else {
        E::zero()
    };
    let v3 = if h_high.index() < height && w_low >= E::zero() {
        x_data[base + h_high.index() * width + w_low.index()]
    } else {
        E::zero()
    };
    let v4 = if h_high.index() < height && w_high.index() < width {
        x_data[base + h_high.index() * width + w_high.index()]
    } else {
        E::zero()
    };

    // Input gradient (distribute grad_val to the 4 corners)
    if h_low >= E::zero() && w_low >= E::zero() {
        let idx = base + h_low.index() * width + w_low.index();
        x_grad[idx] += hh * hw * grad_val;
    }
    if h_low >= E::zero() && w_high.index() < width {
        let idx = base + h_low.index() * width + w_high.index();
        x_grad[idx] += hh * lw * grad_val;
    }
    if h_high.index() < height && w_low >= E::zero() {
        let idx = base + h_high.index() * width + w_low.index();
        x_grad[idx] += lh * hw * grad_val;
    }
    if h_high.index() < height && w_high.index() < width {
        let idx = base + h_high.index() * width + w_high.index();
        x_grad[idx] += lh * lw * grad_val;
    }

    // Offset gradient (derivative of bilinear interpolation w.r.t. coordinates)
    let grad_h = hw * (v3 - v1) + lw * (v4 - v2);
    let grad_w = hh * (v2 - v1) + lh * (v4 - v3);

    offset_grad[offset_h_flat] += grad_val * grad_h;
    offset_grad[offset_w_flat] += grad_val * grad_w;
}

