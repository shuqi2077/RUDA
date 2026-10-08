use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, num_traits::Zero, prelude::*};
use ruda_kernel::library::FastDivmod;
use ruda_kernel::tensor::{
    RudaTensor, allocation::empty_device_dtype, contiguous::into_contiguous_aligned,
    layout::{address_type, decompose_linear, max_vector_size, shape_divmod},
    permutation::{permute_nchw_to_nhwc, permute_nhwc_to_nchw},
};
use ruda_core::{ir::AddressType, tensor::Shape};
use super::adaptive_avg_pool3d::accumulation_dtype;

#[derive(RudaLaunch, RudaType)]
struct AverageVolumeArgs {
    kd: usize, kh: usize, kw: usize,
    sd: usize, sh: usize, sw: usize,
    pd: usize, ph: usize, pw: usize,
}

#[ruda]
fn clipped_end(start: usize, kernel: usize, bound: usize) -> usize {
    if start >= bound || kernel >= bound - start { bound } else { start + kernel }
}

#[ruda]
fn window_count(start: usize, kernel: usize, padding: usize, input: usize,
    #[comptime] include_pad: bool) -> usize {
    let lower = if include_pad { start } else { start.max(padding) };
    let bound = if include_pad { input + padding * 2 } else { input + padding };
    let upper = clipped_end(start, kernel, bound);
    if upper > lower { upper - lower } else { 0usize.into() }
}

#[ruda]
fn first_covering_window(position: usize, kernel: usize, stride: usize) -> usize {
    let extent = position + 1;
    let numerator = if extent > kernel { extent - kernel } else { 0usize.into() };
    let quotient = numerator / stride;
    if numerator % stride == 0 { quotient } else { quotient + 1 }
}

#[ruda(launch, address_type = "dynamic")]
fn average_volume<E: Numeric, A: Float, N: Size>(
    input: &Tensor<Vector<E, N>>, output: &mut Tensor<Vector<E, N>>,
    output_shape: Sequence<FastDivmod<usize>>, working_units: usize,
    args: &AverageVolumeArgs, #[comptime] include_pad: bool,
    #[define(E)] _storage: StorageType, #[define(A)] _compute: StorageType,
) {
    if ABSOLUTE_POS >= working_units { terminate!(); }
    let (_, position) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &output_shape);
    let [batch, od, oh, ow, channel] = *position else { unreachable!() };
    let ds = od * args.sd;
    let hs = oh * args.sh;
    let ws = ow * args.sw;
    let depth = input.shape(1);
    let height = input.shape(2);
    let width = input.shape(3);
    let dl = ds.max(args.pd);
    let hl = hs.max(args.ph);
    let wl = ws.max(args.pw);
    let de = clipped_end(ds, args.kd, depth + args.pd);
    let he = clipped_end(hs, args.kh, height + args.ph);
    let we = clipped_end(ws, args.kw, width + args.pw);
    let base = batch * input.stride(0) + channel * input.stride(4);
    let mut sum = Vector::<A, N>::zero();
    for pd in dl..de {
        for ph in hl..he {
            for pw in wl..we {
                let index = base + (pd - args.pd) * input.stride(1)
                    + (ph - args.ph) * input.stride(2) + (pw - args.pw) * input.stride(3);
                sum += Vector::cast_from(input[index / input.vector_size()]);
            }
        }
    }
    let cd = Vector::<A, N>::cast_from(window_count(ds, args.kd, args.pd, depth, include_pad));
    let ch = Vector::<A, N>::cast_from(window_count(hs, args.kh, args.ph, height, include_pad));
    let cw = Vector::<A, N>::cast_from(window_count(ws, args.kw, args.pw, width, include_pad));
    output[ABSOLUTE_POS] = Vector::cast_from(sum / (cd * ch * cw));
}

#[ruda(launch, address_type = "dynamic")]
fn average_volume_backward<E: Numeric, A: Float, N: Size>(
    grad: &Tensor<Vector<E, N>>, output: &mut Tensor<Vector<E, N>>,
    output_shape: Sequence<FastDivmod<usize>>, working_units: usize,
    args: &AverageVolumeArgs, #[comptime] include_pad: bool,
    #[define(E)] _storage: StorageType, #[define(A)] _compute: StorageType,
) {
    if ABSOLUTE_POS >= working_units { terminate!(); }
    let (_, position) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &output_shape);
    let [batch, id, ih, iw, channel] = *position else { unreachable!() };
    let depth = output.shape(1);
    let height = output.shape(2);
    let width = output.shape(3);
    let pd = id + args.pd;
    let ph = ih + args.ph;
    let pw = iw + args.pw;
    let dl = first_covering_window(pd, args.kd, args.sd);
    let hl = first_covering_window(ph, args.kh, args.sh);
    let wl = first_covering_window(pw, args.kw, args.sw);
    let de = (pd / args.sd + 1).min(grad.shape(1));
    let he = (ph / args.sh + 1).min(grad.shape(2));
    let we = (pw / args.sw + 1).min(grad.shape(3));
    let base = batch * grad.stride(0) + channel * grad.stride(4);
    let mut sum = Vector::<A, N>::zero();
    for od in dl..de {
        let cd = Vector::<A, N>::cast_from(window_count(od * args.sd, args.kd, args.pd, depth, include_pad));
        for oh in hl..he {
            let ch = Vector::<A, N>::cast_from(window_count(oh * args.sh, args.kh, args.ph, height, include_pad));
            for ow in wl..we {
                let cw = Vector::<A, N>::cast_from(window_count(ow * args.sw, args.kw, args.pw, width, include_pad));
                let index = base + od * grad.stride(1) + oh * grad.stride(2) + ow * grad.stride(3);
                let value = Vector::<A, N>::cast_from(grad[index / grad.vector_size()]);
                sum += value / (cd * ch * cw);
            }
        }
    }
    output[ABSOLUTE_POS] = Vector::cast_from(sum);
}

fn output_size(input: [usize; 3], kernel: [usize; 3], stride: [usize; 3],
    padding: [usize; 3], ceil: bool) -> [usize; 3] {
    core::array::from_fn(|axis| {
        assert!(kernel[axis] > 0 && stride[axis] > 0, "pooling kernel and stride must be non-zero");
        let step = stride[axis] as i128;
        let numerator = input[axis] as i128 + 2 * padding[axis] as i128 - kernel[axis] as i128;
        let mut output = (numerator + if ceil { step - 1 } else { 0 }).div_euclid(step) + 1;
        if ceil && (output - 1) * step >= input[axis] as i128 + padding[axis] as i128 { output -= 1; }
        assert!(output > 0, "average pooling output must be positive");
        usize::try_from(output).expect("average pooling output size overflow")
    })
}

fn geometry_address<R: Runtime>(input: &RudaTensor<R>, output: &RudaTensor<R>,
    sizes: [usize; 3], kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3],
    outputs: [usize; 3]) -> AddressType {
    let mut address = address_type!(input, output);
    for axis in 0..3 {
        let padded = sizes[axis].checked_add(padding[axis]).and_then(|size| size.checked_add(padding[axis]))
            .expect("pooling padded extent overflow");
        let last_start = (outputs[axis] - 1).checked_mul(stride[axis]).expect("pooling window position overflow");
        if [padded, kernel[axis], stride[axis], last_start].iter().any(|value| *value > u32::MAX as usize) {
            address = AddressType::U64;
        }
    }
    address
}

/// Native average pooling of `[batch, channels, depth, height, width]` volumes.
///
/// FP32 accumulation is used for half storage, with F64 retained for F64 input.
/// Explicit padding and partial ceil windows use the same denominator rules as
/// native 2D pooling, without materializing a padded volume or depth intermediate.
pub fn avg_pool3d<R: Runtime>(input: RudaTensor<R>, kernel: [usize; 3], stride: [usize; 3],
    padding: [usize; 3], include_pad: bool, ceil: bool) -> RudaTensor<R> {
    let [batch, channels, depth, height, width] = input.meta.shape().dims();
    let sizes = [depth, height, width];
    let outputs = output_size(sizes, kernel, stride, padding, ceil);
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(input));
    let output = empty_device_dtype(input.client.clone(), input.device.clone(),
        Shape::new([batch, outputs[0], outputs[1], outputs[2], channels]), input.dtype);
    let vector_size = max_vector_size(&input);
    let working_units = output.meta.num_elements() / vector_size as usize;
    let address = geometry_address(&input, &output, sizes, kernel, stride, padding, outputs);
    if working_units == 0 { return permute_nhwc_to_nchw(output); }
    let dim = RudaDim::new(input.client.properties(), working_units);
    let count = calculate_ruda_count_elemwise(&input.client, working_units, dim);
    average_volume::launch(&output.client, count, dim, address, vector_size,
        input.into_tensor_arg(), output.clone().into_tensor_arg(), shape_divmod(&output), working_units,
        AverageVolumeArgsLaunch::new(kernel[0], kernel[1], kernel[2], stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2]),
        include_pad, output.dtype.into(), accumulation_dtype(output.dtype).into());
    permute_nhwc_to_nchw(output)
}

/// Native average-pooling input gradients with deterministic per-input gathers.
///
/// Overlapping windows accumulate in the working dtype without floating atomic
/// additions. Padding is never written as a gradient of the original input.
pub fn avg_pool3d_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>, kernel: [usize; 3],
    stride: [usize; 3], padding: [usize; 3], include_pad: bool, ceil: bool) -> RudaTensor<R> {
    let [batch, channels, depth, height, width] = input.meta.shape().dims();
    let sizes = [depth, height, width];
    let outputs = output_size(sizes, kernel, stride, padding, ceil);
    assert_eq!(grad.meta.shape().dims::<5>(), [batch, channels, outputs[0], outputs[1], outputs[2]],
        "average pooling gradient shape differs from the forward output");
    assert_eq!(input.dtype, grad.dtype, "average pooling gradient storage differs from input");
    let grad = into_contiguous_aligned(permute_nchw_to_nhwc(grad));
    let output = empty_device_dtype(input.client.clone(), input.device.clone(),
        Shape::new([batch, depth, height, width, channels]), input.dtype);
    let vector_size = max_vector_size(&grad);
    let working_units = output.meta.num_elements() / vector_size as usize;
    let address = geometry_address(&grad, &output, sizes, kernel, stride, padding, outputs);
    if working_units == 0 { return permute_nhwc_to_nchw(output); }
    let dim = RudaDim::new(input.client.properties(), working_units);
    let count = calculate_ruda_count_elemwise(&input.client, working_units, dim);
    average_volume_backward::launch(&output.client, count, dim, address, vector_size,
        grad.into_tensor_arg(), output.clone().into_tensor_arg(), shape_divmod(&output), working_units,
        AverageVolumeArgsLaunch::new(kernel[0], kernel[1], kernel[2], stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2]),
        include_pad, output.dtype.into(), accumulation_dtype(output.dtype).into());
    permute_nhwc_to_nchw(output)
}
