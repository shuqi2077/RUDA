use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, num_traits::Zero, prelude::*};
use ruda_kernel::library::FastDivmod;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype,
    contiguous::into_contiguous_aligned,
    layout::{address_type, decompose_linear, max_vector_size, shape_divmod},
    permutation::{permute_nchw_to_nhwc, permute_nhwc_to_nchw}};
use ruda_core::{ir::AddressType, tensor::{DType, Shape, spatial::calculate_pool_output_size}};

#[derive(RudaLaunch, RudaType)]
struct MaxVolumeArgs {
    kd: usize, kh: usize, kw: usize,
    sd: usize, sh: usize, sw: usize,
    pd: usize, ph: usize, pw: usize,
    dd: usize, dh: usize, dw: usize,
}

#[ruda]
fn ceiling_division(value: usize, divisor: usize) -> usize {
    let quotient = value / divisor;
    if value % divisor == 0 { quotient } else { quotient + 1 }
}

#[ruda]
fn samples(start: usize, input: usize, padding: usize, kernel: usize, dilation: usize) -> (usize, usize) {
    let first = if start < padding { padding - start } else { 0usize.into() };
    let bound = input + padding;
    let last = if start < bound { bound - start } else { 0usize.into() };
    (ceiling_division(first, dilation), ceiling_division(last, dilation).min(kernel))
}

#[ruda]
fn scan_volume<E: Numeric, I: Int, N: Size>(input: &Tensor<Vector<E, N>>, args: &MaxVolumeArgs,
    batch: usize, od: usize, oh: usize, ow: usize, channel: usize) -> (Vector<E, N>, Vector<I, N>) {
    let ds = od * args.sd;
    let hs = oh * args.sh;
    let ws = ow * args.sw;
    let (dl, de) = samples(ds, input.shape(1), args.pd, args.kd, args.dd);
    let (hl, he) = samples(hs, input.shape(2), args.ph, args.kh, args.dh);
    let (wl, we) = samples(ws, input.shape(3), args.pw, args.kw, args.dw);
    let base = batch * input.stride(0) + channel * input.stride(4);
    let mut maximum = Vector::<E, N>::cast_from(f32::NEG_INFINITY);
    let mut position = Vector::<I, N>::new(I::new(-1));
    let mut has_depth = false;
    for kd in dl..de {
        let id = ds + kd * args.dd - args.pd;
        let mut spatial_maximum = Vector::<E, N>::cast_from(f32::NEG_INFINITY);
        let mut spatial_position = Vector::<I, N>::new(I::new(-1));
        let mut has_spatial = false;
        for kh in hl..he {
            let ih = hs + kh * args.dh - args.ph;
            for kw in wl..we {
                let iw = ws + kw * args.dw - args.pw;
                let offset = base + id * input.stride(1) + ih * input.stride(2) + iw * input.stride(3);
                let value = input[offset / input.vector_size()];
                let flat = Vector::<I, N>::cast_from((id * input.shape(2) + ih) * input.shape(3) + iw);
                if !has_spatial {
                    spatial_maximum = value;
                    spatial_position = flat;
                    has_spatial = true;
                } else {
                    let replace = value.greater_than(spatial_maximum);
                    spatial_maximum = select_many(replace, value, spatial_maximum);
                    spatial_position = select_many(replace, flat, spatial_position);
                }
            }
        }
        if !has_depth {
            maximum = spatial_maximum;
            position = spatial_position;
            has_depth = true;
        } else {
            let replace = spatial_maximum.greater_than(maximum);
            maximum = select_many(replace, spatial_maximum, maximum);
            position = select_many(replace, spatial_position, position);
        }
    }
    (maximum, position)
}

#[ruda(launch, address_type = "dynamic")]
fn maximum_volume<E: Numeric, I: Int, N: Size>(input: &Tensor<Vector<E, N>>,
    output: &mut Tensor<Vector<E, N>>, shape: Sequence<FastDivmod<usize>>, count: usize,
    args: &MaxVolumeArgs, #[define(E, I)] _types: [StorageType; 2]) {
    if ABSOLUTE_POS >= count { terminate!(); }
    let (_, position) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &shape);
    let [b, d, h, w, c] = *position else { unreachable!() };
    let (value, _) = scan_volume::<E, I, N>(input, args, b, d, h, w, c);
    output[ABSOLUTE_POS] = value;
}

#[ruda(launch, address_type = "dynamic")]
fn maximum_volume_indices<E: Numeric, I: Int, N: Size>(input: &Tensor<Vector<E, N>>,
    output: &mut Tensor<Vector<E, N>>, indices: &mut Tensor<Vector<I, N>>,
    shape: Sequence<FastDivmod<usize>>, count: usize, args: &MaxVolumeArgs,
    #[define(E, I)] _types: [StorageType; 2]) {
    if ABSOLUTE_POS >= count { terminate!(); }
    let (_, position) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &shape);
    let [b, d, h, w, c] = *position else { unreachable!() };
    let (value, index) = scan_volume::<E, I, N>(input, args, b, d, h, w, c);
    output[ABSOLUTE_POS] = value;
    indices[ABSOLUTE_POS] = index;
}

#[ruda]
fn first_covering(position: usize, extent: usize, stride: usize) -> usize {
    let last = position + 1;
    let distance = if last > extent { last - extent } else { 0usize.into() };
    ceiling_division(distance, stride)
}

#[ruda(launch, address_type = "dynamic")]
fn maximum_volume_backward<E: Numeric, G: Numeric, I: Int, A: Float, N: Size>(
    grad: &Tensor<Vector<G, N>>, indices: &Tensor<Vector<I, N>>, output: &mut Tensor<Vector<E, N>>,
    shape: Sequence<FastDivmod<usize>>, count: usize, args: &MaxVolumeArgs,
    #[define(E, G, I, A)] _types: [StorageType; 4]) {
    if ABSOLUTE_POS >= count { terminate!(); }
    let (_, position) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &shape);
    let [b, d, h, w, c] = *position else { unreachable!() };
    let pd = d + args.pd;
    let ph = h + args.ph;
    let pw = w + args.pw;
    let dl = first_covering(pd, (args.kd - 1) * args.dd + 1, args.sd);
    let hl = first_covering(ph, (args.kh - 1) * args.dh + 1, args.sh);
    let wl = first_covering(pw, (args.kw - 1) * args.dw + 1, args.sw);
    let de = (pd / args.sd + 1).min(grad.shape(1));
    let he = (ph / args.sh + 1).min(grad.shape(2));
    let we = (pw / args.sw + 1).min(grad.shape(3));
    let current = Vector::<I, N>::cast_from((d * output.shape(2) + h) * output.shape(3) + w);
    let grad_base = b * grad.stride(0) + c * grad.stride(4);
    let index_base = b * indices.stride(0) + c * indices.stride(4);
    let mut sum = Vector::<A, N>::zero();
    for od in dl..de {
        for oh in hl..he {
            for ow in wl..we {
                let gi = grad_base + od * grad.stride(1) + oh * grad.stride(2) + ow * grad.stride(3);
                let ii = index_base + od * indices.stride(1) + oh * indices.stride(2) + ow * indices.stride(3);
                let selected = indices[ii / indices.vector_size()].equal(current);
                let value = Vector::<A, N>::cast_from(grad[gi / grad.vector_size()]);
                sum += select_many(selected, value, Vector::zero());
            }
        }
    }
    output[ABSOLUTE_POS] = Vector::cast_from(sum);
}

/// Native volume maximum-pooling output extents, matching the existing spatial kernels.
pub fn max_pool3d_output_size(input: [usize; 3], kernel: [usize; 3], stride: [usize; 3],
    padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> [usize; 3] {
    core::array::from_fn(|axis| {
        assert!(kernel[axis] > 0 && stride[axis] > 0 && dilation[axis] > 0,
            "max pooling kernel, stride and dilation must be non-zero");
        calculate_pool_output_size(kernel[axis], stride[axis], padding[axis], dilation[axis], input[axis], ceil)
    })
}

fn geometry_address<R: Runtime>(input: &RudaTensor<R>, output: &RudaTensor<R>,
    sizes: [usize; 3], kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3],
    dilation: [usize; 3], outputs: [usize; 3]) -> AddressType {
    let mut address = address_type!(input, output);
    for axis in 0..3 {
        let padded = sizes[axis].checked_add(padding[axis]).and_then(|size| size.checked_add(padding[axis]))
            .expect("max pooling padded extent overflow");
        let extent = (kernel[axis] - 1).checked_mul(dilation[axis]).and_then(|size| size.checked_add(1))
            .expect("max pooling dilated extent overflow");
        let last = outputs[axis].saturating_sub(1).checked_mul(stride[axis])
            .expect("max pooling window position overflow");
        if [padded, extent, last, stride[axis], dilation[axis]].iter().any(|value| *value > u32::MAX as usize) {
            address = AddressType::U64;
        }
    }
    address
}

fn forward<R: Runtime>(input: RudaTensor<R>, kernel: [usize; 3], stride: [usize; 3],
    padding: [usize; 3], dilation: [usize; 3], ceil: bool, indexed: bool) -> (RudaTensor<R>, Option<RudaTensor<R>>) {
    let [batch, channels, depth, height, width] = input.meta.shape().dims();
    let sizes = [depth, height, width];
    if indexed {
        let volume = sizes.iter().try_fold(1usize, |size, axis| size.checked_mul(*axis)).expect("pooling volume overflow");
        assert!(volume <= i64::MAX as usize, "pooling positions exceed I64");
    }
    let outputs = max_pool3d_output_size(sizes, kernel, stride, padding, dilation, ceil);
    let input = into_contiguous_aligned(permute_nchw_to_nhwc(input));
    let shape = Shape::new([batch, outputs[0], outputs[1], outputs[2], channels]);
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), shape.clone(), input.dtype);
    let indices = indexed.then(|| empty_device_dtype(input.client.clone(), input.device.clone(), shape, DType::I64));
    let vector = max_vector_size(&input);
    let work = output.meta.num_elements() / vector as usize;
    let address = geometry_address(&input, &output, sizes, kernel, stride, padding, dilation, outputs);
    if work > 0 {
        let dim = RudaDim::new(input.client.properties(), work);
        let count = calculate_ruda_count_elemwise(&input.client, work, dim);
        let args = MaxVolumeArgsLaunch::new(kernel[0], kernel[1], kernel[2], stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2], dilation[0], dilation[1], dilation[2]);
        let types = [output.dtype.into(), DType::I64.into()];
        if let Some(indices) = indices.as_ref() {
            maximum_volume_indices::launch(&output.client, count, dim, address, vector,
                input.into_tensor_arg(), output.clone().into_tensor_arg(), indices.clone().into_tensor_arg(),
                shape_divmod(&output), work, args, types);
        } else {
            maximum_volume::launch(&output.client, count, dim, address, vector,
                input.into_tensor_arg(), output.clone().into_tensor_arg(), shape_divmod(&output), work, args, types);
        }
    }
    (permute_nhwc_to_nchw(output), indices.map(permute_nhwc_to_nchw))
}

/// Native 3D maximum pooling without allocating an index tensor.
/// Spatial then depth comparison order retains the existing kernels' tie and NaN behavior.
pub fn max_pool3d<R: Runtime>(input: RudaTensor<R>, kernel: [usize; 3], stride: [usize; 3],
    padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> RudaTensor<R> {
    forward(input, kernel, stride, padding, dilation, ceil, false).0
}

/// Native 3D maxima and I64 positions in each unpadded input volume; empty selections are `-1`.
pub fn max_pool3d_with_indices<R: Runtime>(input: RudaTensor<R>, kernel: [usize; 3], stride: [usize; 3],
    padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> (RudaTensor<R>, RudaTensor<R>) {
    let (output, indices) = forward(input, kernel, stride, padding, dilation, ceil, true);
    (output, indices.expect("indexed pooling must return indices"))
}

/// Native input gradients gathered from saved volume positions, without floating atomics.
pub fn max_pool3d_with_indices_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>,
    indices: RudaTensor<R>, kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3],
    dilation: [usize; 3], ceil: bool) -> RudaTensor<R> {
    let [batch, channels, depth, height, width] = input.meta.shape().dims();
    let sizes = [depth, height, width];
    let outputs = max_pool3d_output_size(sizes, kernel, stride, padding, dilation, ceil);
    let expected = [batch, channels, outputs[0], outputs[1], outputs[2]];
    assert_eq!(grad.meta.shape().dims::<5>(), expected, "volume pooling gradient shape differs");
    assert_eq!(indices.meta.shape().dims::<5>(), expected, "volume pooling index shape differs");
    assert_eq!(indices.dtype, DType::I64, "volume pooling positions must use I64");
    let grad = into_contiguous_aligned(permute_nchw_to_nhwc(grad));
    let indices = into_contiguous_aligned(permute_nchw_to_nhwc(indices));
    let output = empty_device_dtype(input.client.clone(), input.device.clone(),
        Shape::new([batch, depth, height, width, channels]), input.dtype);
    let vector = max_vector_size(&grad).min(max_vector_size(&indices));
    let work = output.meta.num_elements() / vector as usize;
    let address = geometry_address(&grad, &output, sizes, kernel, stride, padding, dilation, outputs)
        .max(address_type!(indices));
    if work > 0 {
        let dim = RudaDim::new(input.client.properties(), work);
        let count = calculate_ruda_count_elemwise(&input.client, work, dim);
        let types = [output.dtype.into(), grad.dtype.into(), indices.dtype.into(),
            if output.dtype == DType::F64 || grad.dtype == DType::F64 { DType::F64.into() } else { DType::F32.into() }];
        maximum_volume_backward::launch(&output.client, count, dim, address, vector,
            grad.into_tensor_arg(), indices.into_tensor_arg(), output.clone().into_tensor_arg(), shape_divmod(&output), work,
            MaxVolumeArgsLaunch::new(kernel[0], kernel[1], kernel[2], stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2], dilation[0], dilation[1], dilation[2]), types);
    }
    permute_nhwc_to_nchw(output)
}
