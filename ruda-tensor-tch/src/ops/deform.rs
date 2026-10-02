//! Deformable convolution expressed in LibTorch tensor primitives, avoiding a
//! torchvision binary dependency. Sampling and matrix multiplication stay on
//! the input device. This is a correctness path, not a fused deform-conv kernel.
use crate::{LibTorch, TchElement, TchTensor};
use ruda_tensor::ops::{DeformConvOptions, DeformConv2dBackward};
use tch::{Kind, Tensor};

fn parameter(value: usize) -> i64 {
    i64::try_from(value).expect("deform-conv option exceeds i64")
}
fn check_operand(tensor: &Tensor, x: &Tensor, name: &str) {
    assert_eq!(tensor.device(), x.device(), "{name} is on a different device");
    assert_eq!(tensor.kind(), x.kind(), "{name} has a different floating dtype");
}

/// Bilinear zero-padded sampling directly in pixel coordinates. Avoiding a
/// normalized-grid round trip is important: at integer coordinates, tiny
/// rounding errors can select the wrong one-sided offset derivative.
fn bilinear(input: &Tensor, y: &Tensor, x: &Tensor) -> Tensor {
    let (n, c, h, w) = input.size4().unwrap();
    let size = y.size();
    let (oh, ow) = (size[1], size[2]);
    let y0 = y.floor();
    let x0 = x.floor();
    let ly = (y - &y0).unsqueeze(1);
    let lx = (x - &x0).unsqueeze(1);
    let hy = ly.neg() + 1.0;
    let hx = lx.neg() + 1.0;
    let corner = |yy: &Tensor, xx: &Tensor| {
        let valid = yy.ge(0.0).logical_and(&yy.lt(h as f64))
            .logical_and(&xx.ge(0.0)).logical_and(&xx.lt(w as f64))
            .to_kind(input.kind()).unsqueeze(1);
        let index = (yy.clamp(0.0, (h-1) as f64).to_kind(Kind::Int64)*w
            + xx.clamp(0.0, (w-1) as f64).to_kind(Kind::Int64))
            .reshape([n, 1, oh*ow]).expand([n, c, oh*ow], false);
        input.reshape([n, c, h*w]).gather(2, &index, false).reshape([n, c, oh, ow])*valid
    };
    let y1 = &y0 + 1.0;
    let x1 = &x0 + 1.0;
    corner(&y0, &x0)*&hy*&hx + corner(&y0, &x1)*&hy*&lx
        + corner(&y1, &x0)*&ly*&hx + corner(&y1, &x1)*&ly*&lx
}

pub(super) fn forward(x: &Tensor, offset: &Tensor, weight: &Tensor,
    mask: Option<&Tensor>, bias: Option<&Tensor>, options: &DeformConvOptions<2>) -> Tensor
{
    let (n, ci, h, w) = x.size4().expect("deform-conv input must be NCHW");
    let (co, weight_ci, kh, kw) = weight.size4().expect("deform-conv weight must have four dimensions");
    assert!(matches!(x.kind(), Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16),
        "deform-conv requires a floating dtype");
    check_operand(weight, x, "weight");
    check_operand(offset, x, "offset");
    let wg = parameter(options.weight_groups);
    let og = parameter(options.offset_groups);
    assert!(wg > 0 && og > 0 && ci > 0 && co > 0 && h > 0 && w > 0 && kh > 0 && kw > 0,
        "deform-conv groups, channels, spatial dimensions and kernels must be positive");
    assert!(ci % wg == 0 && ci % og == 0 && co % wg == 0 && weight_ci == ci/wg,
        "deform-conv channel/group mismatch");
    let [sh, sw] = options.stride.map(parameter);
    let [dh, dw] = options.dilation.map(parameter);
    let [ph, pw] = options.padding.map(parameter);
    let [peh, pew] = options.padding_end.unwrap_or(options.padding).map(parameter);
    assert!(sh > 0 && sw > 0 && dh > 0 && dw > 0, "stride and dilation must be positive");
    let extent_h = h + ph + peh - dh*(kh-1) - 1;
    let extent_w = w + pw + pew - dw*(kw-1) - 1;
    assert!(extent_h >= 0 && extent_w >= 0, "deform-conv kernel exceeds padded input");
    let (oh, ow) = (extent_h/sh+1, extent_w/sw+1);
    let kernel = kh*kw;
    assert_eq!(offset.size(), vec![n, 2*og*kernel, oh, ow], "invalid offset shape");
    if let Some(mask) = mask {
        check_operand(mask, x, "mask");
        assert_eq!(mask.size(), vec![n, og*kernel, oh, ow], "invalid mask shape");
    }
    if let Some(bias) = bias {
        check_operand(bias, x, "bias");
        assert_eq!(bias.size(), vec![co], "invalid bias shape");
    }

    // Use FP32 accumulation for Half/BFloat16; Double inputs retain precision.
    // All sampling, indexing and matrix products remain on the input device.
    let kind = if x.kind() == Kind::Double { Kind::Double } else { Kind::Float };
    let x_acc = x.to_kind(kind);
    let offsets = offset.to_kind(kind);
    let masks = mask.map(|m| m.to_kind(kind));
    let weights = weight.to_kind(kind);
    let base_y = (Tensor::arange(oh, (kind, x.device())) * sh as f64 - ph as f64).reshape([1, oh, 1]);
    let base_x = (Tensor::arange(ow, (kind, x.device())) * sw as f64 - pw as f64).reshape([1, 1, ow]);
    let mut samples = Vec::with_capacity(kernel as usize);
    for ky in 0..kh {
        for kx in 0..kw {
            let k = ky*kw+kx;
            let mut channels = Vec::with_capacity(og as usize);
            for group in 0..og {
                let channel = group*kernel+k;
                let y = &base_y + (ky*dh) as f64 + offsets.select(1, 2*channel);
                let x_pos = &base_x + (kx*dw) as f64 + offsets.select(1, 2*channel+1);
                let input = x_acc.narrow(1, group*(ci/og), ci/og);
                let mut sampled = bilinear(&input, &y, &x_pos);
                if let Some(mask) = &masks {
                    sampled = sampled * mask.select(1, channel).unsqueeze(1);
                }
                channels.push(sampled);
            }
            samples.push(Tensor::cat(&channels, 1));
        }
    }
    let columns = Tensor::stack(&samples, 2).reshape([n, ci*kernel, oh*ow]);
    let mut output_groups = Vec::with_capacity(wg as usize);
    for group in 0..wg {
        let width = (ci/wg)*kernel;
        let patch = columns.narrow(1, group*width, width);
        let matrix = weights.narrow(0, group*(co/wg), co/wg).reshape([co/wg, width]);
        output_groups.push(matrix.matmul(&patch));
    }
    let mut output = Tensor::cat(&output_groups, 1).reshape([n, co, oh, ow]);
    if let Some(bias) = bias { output = output + bias.to_kind(kind).reshape([1, co, 1, 1]); }
    output.to_kind(x.kind())
}

pub(super) fn backward<E: TchElement>(x: TchTensor, offset: TchTensor, weight: TchTensor,
    mask: Option<TchTensor>, bias: Option<TchTensor>, out_grad: TchTensor,
    options: DeformConvOptions<2>) -> DeformConv2dBackward<LibTorch<E>>
{
    // Local leaves prevent accumulating into any caller-owned LibTorch graph.
    // Explicit with_grad is required even when the outer backend disabled grad.
    tch::with_grad(|| {
        let x = x.tensor.detach().set_requires_grad(true);
        let offset = offset.tensor.detach().set_requires_grad(true);
        let weight = weight.tensor.detach().set_requires_grad(true);
        let mask = mask.map(|m| m.tensor.detach().set_requires_grad(true));
        let bias = bias.map(|b| b.tensor.detach().set_requires_grad(true));
        let output = forward(&x, &offset, &weight, mask.as_ref(), bias.as_ref(), &options);
        check_operand(&out_grad.tensor, &x, "output gradient");
        assert_eq!(out_grad.tensor.size(), output.size(), "invalid output gradient shape");
        let kind = if x.kind() == Kind::Double { Kind::Double } else { Kind::Float };
        let loss = (output * out_grad.tensor.detach()).sum(kind);
        let mut inputs = vec![&x, &offset, &weight];
        inputs.extend(mask.iter());
        inputs.extend(bias.iter());
        let mut gradients = Tensor::run_backward(&[loss], &inputs, false, false).into_iter();
        DeformConv2dBackward {
            x_grad: TchTensor::new(gradients.next().unwrap().detach()),
            offset_grad: TchTensor::new(gradients.next().unwrap().detach()),
            weight_grad: TchTensor::new(gradients.next().unwrap().detach()),
            mask_grad: mask.as_ref().map(|_| TchTensor::new(gradients.next().unwrap().detach())),
            bias_grad: bias.as_ref().map(|_| TchTensor::new(gradients.next().unwrap().detach())),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    fn options(groups: usize, offsets: usize) -> DeformConvOptions<2> {
        DeformConvOptions { stride: [1, 1], padding: [1, 1], dilation: [1, 1],
            weight_groups: groups, offset_groups: offsets, padding_end: None }
    }
    fn close(actual: &Tensor, expected: &Tensor, tolerance: f64) {
        assert_eq!(actual.size(), expected.size());
        assert!(actual.allclose(expected, tolerance, tolerance, false), "deform-conv numerical mismatch");
    }

    #[test]
    fn zero_offsets_match_grouped_convolution_including_asymmetric_padding() {
        let x = Tensor::randn([2, 4, 5, 6], (Kind::Double, Device::Cpu));
        let w = Tensor::randn([6, 2, 3, 2], (Kind::Double, Device::Cpu));
        let b = Tensor::randn([6], (Kind::Double, Device::Cpu));
        let mut opts = options(2, 4);
        opts.padding_end = Some([0, 2]);
        opts.stride = [2, 1];
        opts.dilation = [1, 2];
        let padded = x.constant_pad_nd([1, 2, 1, 0]);
        let reference = padded.conv2d(&w, Some(&b), [2,1], [0,0], [1,2], 2);
        let shape = reference.size();
        let offset = Tensor::zeros([2, 48, shape[2], shape[3]], (Kind::Double, Device::Cpu));
        let mask = Tensor::ones([2, 24, shape[2], shape[3]], (Kind::Double, Device::Cpu));
        close(&forward(&x, &offset, &w, Some(&mask), Some(&b), &opts), &reference, 1e-10);
    }

    #[test]
    fn backward_returns_input_offset_weight_mask_and_bias_gradients() {
        let x = Tensor::randn([1, 2, 3, 3], (Kind::Double, Device::Cpu));
        let weight = Tensor::randn([2, 2, 2, 2], (Kind::Double, Device::Cpu));
        let offset = Tensor::full([1, 16, 2, 2], 0.3, (Kind::Double, Device::Cpu));
        let mask = Tensor::full([1, 8, 2, 2], 0.7, (Kind::Double, Device::Cpu));
        let bias = Tensor::randn([2], (Kind::Double, Device::Cpu));
        let mut opts = options(1, 2);
        opts.padding = [0, 0];
        let gradient = Tensor::ones([1, 2, 2, 2], (Kind::Double, Device::Cpu));
        let result = backward::<f32>(TchTensor::new(x.shallow_clone()), TchTensor::new(offset.shallow_clone()),
            TchTensor::new(weight.shallow_clone()), Some(TchTensor::new(mask.shallow_clone())),
            Some(TchTensor::new(bias.shallow_clone())), TchTensor::new(gradient), opts.clone());
        let args = [&x, &offset, &weight, &mask, &bias];
        let grads = [&result.x_grad.tensor, &result.offset_grad.tensor, &result.weight_grad.tensor,
            &result.mask_grad.as_ref().unwrap().tensor, &result.bias_grad.as_ref().unwrap().tensor];
        for (index, argument) in args.iter().enumerate() {
            let mut delta = vec![0f64; argument.numel()];
            delta[0] = 1e-5;
            let delta = Tensor::from_slice(&delta).reshape(argument.size());
            let evaluate = |sign: f64| {
                let mut values = args.iter().map(|v| v.shallow_clone()).collect::<Vec<_>>();
                values[index] = &values[index] + &delta * sign;
                forward(&values[0], &values[1], &values[2], Some(&values[3]), Some(&values[4]), &opts)
                    .sum(Kind::Double).double_value(&[])
            };
            let numerical = (evaluate(1.0)-evaluate(-1.0))/2e-5;
            let actual = grads[index].reshape([-1]).double_value(&[0]);
            assert!((actual-numerical).abs() < 1e-7, "gradient {index}: {actual} vs {numerical}");
        }
    }

    #[test]
    fn singleton_spatial_dimension_is_not_divided_by_zero() {
        let x = Tensor::from_slice(&[2f32]).reshape([1,1,1,1]);
        let w = Tensor::ones([1,1,1,1], (Kind::Float, Device::Cpu));
        let offset = Tensor::zeros([1,2,1,1], (Kind::Float, Device::Cpu));
        let mut opts = options(1,1);
        opts.padding = [0,0];
        close(&forward(&x,&offset,&w,None,None,&opts), &x, 1e-6);
    }
}
