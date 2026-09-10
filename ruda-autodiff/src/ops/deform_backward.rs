use core::ops::Range;
use ruda_tensor::{
    Backend, FloatDType, TensorPrimitive,
    api::{Bool, IndexingUpdateOp, Int, Tensor},
    ops::{DeformConv2dBackward, DeformConvOptions},
    tensor::FloatTensor,
};

fn wrap<B: Backend, const D: usize>(value: FloatTensor<B>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(value))
}

fn add_slice<B: Backend, const D: usize>(
    target: Tensor<B, D>,
    ranges: [Range<usize>; D],
    value: Tensor<B, D>,
) -> Tensor<B, D> {
    let sum = target.clone().slice(ranges.clone()) + value;
    target.slice_assign(ranges, sum)
}

fn corner<B: Backend>(
    input: &Tensor<B, 3>,
    y: Tensor<B, 3>,
    x: Tensor<B, 3>,
    outside: &Tensor<B, 3, Bool>,
    height: usize,
    width: usize,
) -> (Tensor<B, 3>, Tensor<B, 3, Int>, Tensor<B, 3, Bool>) {
    let [batch, channels, _] = input.dims();
    let positions = y.dims()[2];
    let invalid = outside.clone()
        .bool_or(y.clone().lower_elem(0.0))
        .bool_or(y.clone().greater_equal_elem(height as f64))
        .bool_or(x.clone().lower_elem(0.0))
        .bool_or(x.clone().greater_equal_elem(width as f64))
        .expand([batch, channels, positions]);
    let indices = (y.clamp(0.0, (height - 1) as f64).int() * width as i64
        + x.clamp(0.0, (width - 1) as f64).int())
        .expand([batch, channels, positions]);
    let values = input.clone().gather(2, indices.clone()).mask_fill(invalid.clone(), 0.0);
    (values, indices, invalid)
}

pub(super) fn backward<B: Backend>(
    x: FloatTensor<B>,
    offset: FloatTensor<B>,
    weight: FloatTensor<B>,
    mask: Option<FloatTensor<B>>,
    bias: Option<FloatTensor<B>>,
    output_grad: FloatTensor<B>,
    options: DeformConvOptions<2>,
) -> DeformConv2dBackward<B> {
    let x = wrap::<B, 4>(x);
    let offset = wrap::<B, 4>(offset);
    let weight = wrap::<B, 4>(weight);
    let mask = mask.map(wrap::<B, 4>);
    let output_grad = wrap::<B, 4>(output_grad);
    let [batch, channels, height, width] = x.dims();
    let [outputs, group_channels, kernel_h, kernel_w] = weight.dims();
    let [grad_batch, grad_channels, out_h, out_w] = output_grad.dims();
    assert!(height > 0 && width > 0);
    assert!(options.weight_groups > 0 && options.offset_groups > 0);
    assert_eq!(channels % options.weight_groups, 0);
    assert_eq!(channels % options.offset_groups, 0);
    assert_eq!(outputs % options.weight_groups, 0);
    assert_eq!(group_channels, channels / options.weight_groups);
    assert_eq!([grad_batch, grad_channels], [batch, outputs]);
    assert_eq!(offset.dims(), [batch, 2 * options.offset_groups * kernel_h * kernel_w, out_h, out_w]);
    if let Some(mask) = &mask {
        assert_eq!(mask.dims(), [batch, options.offset_groups * kernel_h * kernel_w, out_h, out_w]);
    }
    let positions = out_h * out_w;
    let offset_channels = channels / options.offset_groups;
    let group_outputs = outputs / options.weight_groups;
    let device = x.device();
    let dtype: FloatDType = x.dtype().into();
    let rows = Tensor::<B, 1, Int>::arange(0..out_h as i64, &device)
        .cast(dtype).reshape([1, 1, out_h, 1]) * options.stride[0] as f64;
    let columns = Tensor::<B, 1, Int>::arange(0..out_w as i64, &device)
        .cast(dtype).reshape([1, 1, 1, out_w]) * options.stride[1] as f64;
    let mut x_grad = x.zeros_like();
    let mut offset_grad = offset.zeros_like();
    let mut weight_grad = weight.zeros_like();
    let mut mask_grad = mask.as_ref().map(|value| value.zeros_like());
    let bias_grad = bias.map(|value| {
        let value = wrap::<B, 1>(value);
        output_grad.clone().sum_dim(0).sum_dim(2).sum_dim(3)
            .reshape(value.dims()).into_primitive().tensor()
    });

    for kh in 0..kernel_h {
        for kw in 0..kernel_w {
            for offset_group in 0..options.offset_groups {
                let mask_channel = offset_group * kernel_h * kernel_w + kh * kernel_w + kw;
                let offset_channel = mask_channel * 2;
                let yr = [0..batch, offset_channel..offset_channel + 1, 0..out_h, 0..out_w];
                let xr = [0..batch, offset_channel + 1..offset_channel + 2, 0..out_h, 0..out_w];
                let mr = [0..batch, mask_channel..mask_channel + 1, 0..out_h, 0..out_w];
                let y = (offset.clone().slice(yr.clone()) + rows.clone()
                    + (kh * options.dilation[0]) as f64 - options.padding[0] as f64)
                    .reshape([batch, 1, positions]);
                let xx = (offset.clone().slice(xr.clone()) + columns.clone()
                    + (kw * options.dilation[1]) as f64 - options.padding[1] as f64)
                    .reshape([batch, 1, positions]);
                let outside = y.clone().lower_equal_elem(-1.0)
                    .bool_or(y.clone().greater_equal_elem(height as f64))
                    .bool_or(xx.clone().lower_equal_elem(-1.0))
                    .bool_or(xx.clone().greater_equal_elem(width as f64));
                let y = y.mask_fill(outside.clone(), 0.0);
                let xx = xx.mask_fill(outside.clone(), 0.0);
                let y0 = y.clone().floor();
                let x0 = xx.clone().floor();
                let ly = y - y0.clone();
                let lx = xx - x0.clone();
                let hy = ly.clone().neg() + 1.0;
                let hx = lx.clone().neg() + 1.0;
                let modulation = mask.as_ref().map(|value| {
                    value.clone().slice(mr.clone()).reshape([batch, 1, positions])
                });

                for group in 0..options.weight_groups {
                    let start = (group * group_channels).max(offset_group * offset_channels);
                    let end = ((group + 1) * group_channels).min((offset_group + 1) * offset_channels);
                    if start >= end { continue; }
                    let count = end - start;
                    let oc = group * group_outputs;
                    let ic = start - group * group_channels;
                    let input_range = [0..batch, start..end, 0..height, 0..width];
                    let weight_range = [oc..oc + group_outputs, ic..ic + count, kh..kh + 1, kw..kw + 1];
                    let input = x.clone().slice(input_range.clone()).reshape([batch, count, height * width]);
                    let grad = output_grad.clone()
                        .slice([0..batch, oc..oc + group_outputs, 0..out_h, 0..out_w])
                        .reshape([batch, group_outputs, positions]);
                    let weights = weight.clone().slice(weight_range.clone()).reshape([1, group_outputs, count]);
                    let column_grad = weights.swap_dims(1, 2).matmul(grad.clone());
                    let sample_grad = match &modulation {
                        Some(mask) => column_grad.clone() * mask.clone(),
                        None => column_grad.clone(),
                    };
                    let (v00, i00, m00) = corner(&input, y0.clone(), x0.clone(), &outside, height, width);
                    let (v01, i01, m01) = corner(&input, y0.clone(), x0.clone() + 1.0, &outside, height, width);
                    let (v10, i10, m10) = corner(&input, y0.clone() + 1.0, x0.clone(), &outside, height, width);
                    let (v11, i11, m11) = corner(&input, y0.clone() + 1.0, x0.clone() + 1.0, &outside, height, width);
                    let c00 = hy.clone() * hx.clone();
                    let c01 = hy.clone() * lx.clone();
                    let c10 = ly.clone() * hx.clone();
                    let c11 = ly.clone() * lx.clone();
                    let sampled = v00.clone() * c00.clone() + v01.clone() * c01.clone()
                        + v10.clone() * c10.clone() + v11.clone() * c11.clone();
                    let dy = (v10.clone() - v00.clone()) * hx.clone()
                        + (v11.clone() - v01.clone()) * lx.clone();
                    let dx = (v01 - v00) * hy.clone() + (v11 - v10) * ly.clone();
                    let invalid = outside.clone().expand([batch, count, positions]);
                    offset_grad = add_slice(offset_grad, yr.clone(),
                        (sample_grad.clone() * dy).mask_fill(invalid.clone(), 0.0)
                            .sum_dim(1).reshape([batch, 1, out_h, out_w]));
                    offset_grad = add_slice(offset_grad, xr.clone(),
                        (sample_grad.clone() * dx).mask_fill(invalid, 0.0)
                            .sum_dim(1).reshape([batch, 1, out_h, out_w]));
                    if let Some(current) = mask_grad.take() {
                        mask_grad = Some(add_slice(current, mr.clone(),
                            (column_grad * sampled.clone()).sum_dim(1).reshape([batch, 1, out_h, out_w])));
                    }
                    let weighted_samples = match &modulation {
                        Some(mask) => sampled * mask.clone(),
                        None => sampled,
                    };
                    weight_grad = add_slice(weight_grad, weight_range,
                        grad.matmul(weighted_samples.swap_dims(1, 2)).sum_dim(0)
                            .reshape([group_outputs, count, 1, 1]));
                    let mut input_grad = input.zeros_like();
                    for (indices, invalid, coefficient) in [
                        (i00, m00, c00), (i01, m01, c01), (i10, m10, c10), (i11, m11, c11),
                    ] {
                        input_grad = input_grad.scatter(2, indices,
                            (sample_grad.clone() * coefficient).mask_fill(invalid, 0.0), IndexingUpdateOp::Add);
                    }
                    x_grad = add_slice(x_grad, input_range, input_grad.reshape([batch, count, height, width]));
                }
            }
        }
    }

    DeformConv2dBackward {
        x_grad: x_grad.into_primitive().tensor(),
        offset_grad: offset_grad.into_primitive().tensor(),
        weight_grad: weight_grad.into_primitive().tensor(),
        mask_grad: mask_grad.map(|value| value.into_primitive().tensor()),
        bias_grad,
    }
}
