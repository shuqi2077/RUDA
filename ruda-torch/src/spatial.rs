use super::{CudaRuntime, View, primitives::{store, tensor}};
use ruda_core::tensor::{Shape, spatial::{ConvOptions, ConvTransposeOptions}};
use ruda_kernel::tensor::{RudaTensor, reshape::reshape};
use rudnn::convolution::tensor::{conv_forward, conv_data_backward, conv_weight_backward,
    conv_transpose2d, conv_transpose3d};
use ruda_kernel::dsl::prelude::{RudaCount, RudaDim};
use super::{client, finish_dispatch, kernels, LAUNCHES, Ordering, bf16, f16};

fn array<const N: usize>(params: &[i64]) -> [usize; N] {
    std::array::from_fn(|i| usize::try_from(params[i]).expect("negative spatial parameter"))
}

fn convolution<const N: usize>(op: u32, a: &View, b: &View, out: &View, p: &[i64]) -> RudaTensor<CudaRuntime> {
    assert_eq!(p.len(), 4 * N + 1);
    assert_eq!(a.shape.len(), N + 2);
    assert_eq!(b.shape.len(), N + 2);
    assert_eq!(out.shape.len(), N + 2);
    let stride = array::<N>(p);
    let padding = array::<N>(&p[N..]);
    let dilation = array::<N>(&p[2 * N..]);
    let options = ConvOptions::new(stride, padding, dilation,
        usize::try_from(p[4 * N]).expect("negative convolution groups"));
    match op {
        0 => conv_forward::<_, N>(tensor(a), tensor(b), None, options, Default::default())
            .expect("RUDA convolution failed"),
        1 => conv_data_backward::<_, N>(tensor(a), tensor(b), Shape::from(out.shape.clone()), options, Default::default())
            .expect("RUDA convolution data gradient failed"),
        2 => conv_weight_backward::<_, N>(tensor(a), tensor(b), Shape::from(out.shape.clone()), options, Default::default())
            .expect("RUDA convolution weight gradient failed"),
        3 => {
            let output_padding = array::<N>(&p[3 * N..]);
            if N == 3 {
                conv_transpose3d(tensor(a), tensor(b), None, ConvTransposeOptions::new(
                    array::<3>(p), array::<3>(&p[N..]), array::<3>(&p[3 * N..]),
                    array::<3>(&p[2 * N..]), options.groups)).expect("RUDA transposed convolution failed")
            } else if N == 2 {
                conv_transpose2d(tensor(a), tensor(b), None, ConvTransposeOptions::new(
                    array::<2>(p), array::<2>(&p[N..]), array::<2>(&p[3 * N..]),
                    array::<2>(&p[2 * N..]), options.groups), Default::default())
                    .expect("RUDA transposed convolution failed")
            } else {
                let mut a_shape = a.shape.clone();
                let mut b_shape = b.shape.clone();
                a_shape.push(1);
                b_shape.push(1);
                let result = conv_transpose2d(reshape(tensor(a), a_shape.into()),
                    reshape(tensor(b), b_shape.into()), None, ConvTransposeOptions::new(
                        [stride[0], 1], [padding[0], 0], [output_padding[0], 0],
                        [dilation[0], 1], options.groups), Default::default())
                    .expect("RUDA transposed convolution failed");
                reshape(result, out.shape.clone().into())
            }
        }
        _ => unreachable!(),
    }
}

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View, p: &[i64]) {
    assert!(a.dtype <= 2);
    assert_eq!(a.dtype, out.dtype);
    if op == 8 {
        assert_eq!(p, &[0]);
        assert!((3..=5).contains(&a.shape.len()));
        assert_eq!(out.shape, vec![a.shape[1]]);
        if out.len == 0 { return; }
        let client = client();
        let count = u32::try_from(out.len.div_ceil(128)).expect("bias gradient grid overflow");
        macro_rules! launch {
            ($dtype:ty) => { kernels::transposed_bias_backward::launch::<$dtype, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), out.arg()) };
        }
        unsafe { match a.dtype {
            0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16), _ => unreachable!(),
        } }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if op == 6 || op == 7 {
        assert_eq!(b.dtype, 4);
        assert_eq!(p.len(), 10);
        assert_eq!(a.shape.len(), 4);
        assert_eq!(b.shape.len(), 4);
        assert_eq!(out.shape.len(), 4);
        assert_eq!(&b.shape, if op == 6 { &out.shape } else { &a.shape });
        if out.len == 0 { return; }
        let kernel = array::<2>(p);
        let stride = array::<2>(&p[2..]);
        let padding = array::<2>(&p[4..]);
        let dilation = array::<2>(&p[6..]);
        if op == 6 {
            let (values, indices) = rudnn::pooling::max_pool2d_with_indices_aten(
                tensor(a), kernel, stride, padding, dilation, p[8] != 0, p[9] != 0);
            store(values, out);
            store(indices, b);
        } else {
            let result = rudnn::pooling::max_pool2d_with_indices_backward(
                tensor(out), tensor(a), tensor(b), kernel, stride, padding, dilation, p[8] != 0);
            store(result, out);
        }
        return;
    }
    assert_eq!(a.dtype, b.dtype);
    if out.len == 0 { return; }
    let result = match op {
        0..=3 => match a.shape.len() {
            3 => convolution::<1>(op, a, b, out, p),
            4 => convolution::<2>(op, a, b, out, p),
            5 => convolution::<3>(op, a, b, out, p),
            _ => panic!("convolution requires one, two or three spatial dimensions"),
        },
        4 | 5 => {
            assert_eq!(p.len(), 9);
            assert_eq!(a.shape.len(), 4);
            assert_eq!(b.shape.len(), 4);
            assert_eq!(out.shape.len(), 4);
            let divisor = (p[8] != 0).then_some(p[8]);
            if op == 4 {
                rudnn::pooling::avg_pool2d_with_divisor(tensor(a), array::<2>(p), array::<2>(&p[2..]),
                    array::<2>(&p[4..]), p[6] != 0, p[7] != 0, divisor)
            } else {
                rudnn::pooling::avg_pool2d_backward_with_divisor(tensor(a), tensor(b), array::<2>(p),
                    array::<2>(&p[2..]), array::<2>(&p[4..]), p[6] != 0, p[7] != 0, divisor)
            }
        }
        _ => panic!("unsupported RUDA spatial operation {op}"),
    };
    store(result, out);
}
