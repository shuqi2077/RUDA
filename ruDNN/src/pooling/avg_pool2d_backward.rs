use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use crate::pooling::pool2d::Position;
use crate::pooling::pool2d::view4d;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::decompose_linear;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::permutation::permute_nchw_to_nhwc;
use ruda_kernel::tensor::permutation::permute_nhwc_to_nchw;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::Shape;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::num_traits::Zero;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::View;

#[derive(RudaLaunch, RudaType)]
pub struct PoolBackwardArgs {
    pub stride_0: i32,
    pub stride_1: i32,
    pub dilation_0: i32,
    pub dilation_1: i32,
    pub padding_0: i32,
    pub padding_1: i32,
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn avg_pool2d_backward_kernel<E: Numeric, N: Size>(
    grad: &Tensor<Vector<E, N>>,
    output: &mut View<Vector<E, N>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[comptime] count_include_pad: bool,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let vector_size = grad.vector_size();

    let (_, pos) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &out_shape);
    let [batch, ih, iw, channel] = *pos else {
        unreachable!()
    };

    let mut grad_acc = Vector::zero();

    let (oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        ih as i32,
        iw as i32,
        grad.shape(1) as u32,
        grad.shape(2) as u32,
        args,
        kernel_size_0,
        kernel_size_1,
    );

    let padding_0 = args.padding_0 as u32;
    let padding_1 = args.padding_1 as u32;
    let stride_0 = args.stride_0 as u32;
    let stride_1 = args.stride_1 as u32;
    let kernel_size_0 = comptime![kernel_size_0 as u32];
    let kernel_size_1 = comptime![kernel_size_1 as u32];

    let index_base = batch * grad.stride(0) + channel * grad.stride(3);
    let border_bottom = output.shape().1 as u32 + padding_0;
    let border_right = output.shape().2 as u32 + padding_1;
    let begin_h = ih as u32 + padding_0;
    let begin_w = iw as u32 + padding_1;

    for oh in oh_start..oh_end {
        let ih_start = oh * stride_0;
        let ih_end = clamp_max(ih_start + kernel_size_0, border_bottom);
        let ih_start = clamp_min(ih_start, padding_0);

        if begin_h >= ih_start && begin_h < ih_end {
            for ow in ow_start..ow_end {
                let index =
                    index_base + oh as usize * grad.stride(1) + ow as usize * grad.stride(2);

                let iw_start = ow * stride_1;
                let iw_end = clamp_max(iw_start + kernel_size_1, border_right);
                let iw_start = clamp_min(iw_start, padding_1);

                if begin_w >= iw_start && begin_w < iw_end {
                    if count_include_pad {
                        let padded_h_end = clamp_max(oh * stride_0 + kernel_size_0, border_bottom + padding_0);
                        let padded_w_end = clamp_max(ow * stride_1 + kernel_size_1, border_right + padding_1);
                        let count = (padded_h_end - oh * stride_0) * (padded_w_end - ow * stride_1);
                        grad_acc += grad[index / vector_size]
                            / Vector::cast_from(count);
                    } else {
                        let ih_diff = ih_end - ih_start;
                        let iw_diff = iw_end - iw_start;
                        let count = Vector::cast_from(ih_diff * iw_diff);
                        grad_acc += grad[index / vector_size] / count;
                    }
                }
            }
        }
    }

    output[(batch, ih, iw, channel)] = grad_acc;
}

#[ruda]
fn loop_ranges(
    ih: i32,
    iw: i32,
    grad_h: u32,
    grad_w: u32,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
) -> (u32, u32, u32, u32) {
    let h = (ih + args.padding_0) as u32;
    let w = (iw + args.padding_1) as u32;
    let kh = comptime![kernel_size_0 as u32];
    let kw = comptime![kernel_size_1 as u32];
    let sh = args.stride_0 as u32;
    let sw = args.stride_1 as u32;
    let oh_start = if h >= kh { (h - kh) / sh + 1 } else { 0u32.runtime() };
    let ow_start = if w >= kw { (w - kw) / sw + 1 } else { 0u32.runtime() };
    let oh_end = clamp_max(h / sh + 1, grad_h);
    let ow_end = clamp_max(w / sw + 1, grad_w);

    (oh_start, oh_end, ow_start, ow_end)
}

pub fn avg_pool2d_backward<R: Runtime>(
    x: RudaTensor<R>,
    grad: RudaTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    _ceil_mode: bool,
) -> RudaTensor<R> {
    let [batches, channels, height, width] = x.meta.shape().dims();

    let grad = permute_nchw_to_nhwc(grad);

    let vector_size = if x.meta.strides()[3] == grad.meta.strides()[3] {
        max_vector_size(&x)
    } else {
        1
    };

    let dilation = 1;

    let out_shape = Shape::new([batches, height, width, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), out_shape, x.dtype);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(x.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&x.client, working_units, ruda_dim);

    unsafe {
        avg_pool2d_backward_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(grad, output),
            vector_size,
            grad.into_tensor_arg(),
            view4d(output.clone(), vector_size),
            shape_divmod(&output),
            working_units,
            PoolBackwardArgsLaunch::new(
                stride[0] as i32,
                stride[1] as i32,
                dilation,
                dilation,
                padding[0] as i32,
                padding[1] as i32,
            ),
            kernel_size[0] as i32,
            kernel_size[1] as i32,
            count_include_pad,
            output.dtype.into(),
        )
    };

    permute_nhwc_to_nchw(output)
}
