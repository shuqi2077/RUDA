use super::{CudaDevice, CudaRuntime, DOWNLOAD, LAUNCHES, Ordering, View, client, convert, sync};
use ruda_core::tensor::{DType, Metadata};
use ruda_kernel::dsl::prelude::InputScalar;
use ruda_kernel::tensor::RudaTensor;
use ruprim::elementwise::{arithmetic, binary::integer_power, unary::int::unary_basic_int};
use ruprim::reduce::{components::instructions::ReduceOperationConfig, tensor::{KernelReduceStrategy, reduce_dim}};

fn dtype(code: u32) -> DType {
    match code {
        0 => DType::F32, 1 => DType::F16, 2 => DType::BF16,
        3 | 8 => DType::U8, 4 => DType::I64, 5 => DType::I32,
        6 => DType::I16, 7 => DType::I8, _ => panic!("unsupported RUDA dtype"),
    }
}

fn tensor(view: &View) -> RudaTensor<CudaRuntime> {
    RudaTensor::new(client(), view.handle.clone(),
        Metadata::new(view.shape.clone(), view.strides.clone()),
        CudaDevice::default(), dtype(view.dtype))
}

fn check_indices(indices: &View, bound: usize) {
    assert!(indices.dtype == 4 || indices.dtype == 5);
    if indices.len == 0 { return; }
    let widened;
    let indices = if indices.dtype == 5 {
        widened = View::packed(client().empty(indices.len.checked_mul(8).expect("index size overflow")),
            indices.shape.clone(), 4);
        convert(indices, &widened);
        &widened
    } else { indices };
    let negative = ruprim::elementwise::comparison::lower_elem(
        tensor(indices), InputScalar::new(0i64, DType::I64), DType::U32);
    let too_large = ruprim::elementwise::comparison::greater_equal_elem(
        tensor(indices), InputScalar::new(i64::try_from(bound).expect("index bound overflow"), DType::I64), DType::U32);
    let invalid = arithmetic::bitwise_or(negative, too_large);
    let status = ruprim::reduce::tensor::reduce(invalid, None,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::Max)
        .expect("RUDA index validation reduction failed");
    sync(&client());
    LAUNCHES.fetch_add(3 + indices.shape.len() as u64, Ordering::Relaxed);
    let excess = status.handle.size_in_used().checked_sub(4).expect("invalid index status size");
    let bytes = client().read_one(status.handle.offset_end(excess)).expect("RUDA index status readback failed");
    DOWNLOAD.fetch_add(4, Ordering::Relaxed);
    assert_eq!(bytes.len(), 4);
    assert_eq!(u32::from_le_bytes(bytes.as_ref().try_into().unwrap()), 0, "RUDA index out of bounds");
}

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View, scalar: f32) {
    assert_eq!(a.dtype, out.dtype);
    let result = match op {
        89..=95 => {
            assert!((4..=8).contains(&a.dtype));
            assert_eq!(a.dtype, b.dtype);
            match op {
                89 => arithmetic::bitwise_and(tensor(a), tensor(b)),
                90 => arithmetic::bitwise_or(tensor(a), tensor(b)),
                91 => arithmetic::bitwise_xor(tensor(a), tensor(b)),
                92 => unary_basic_int::launch(tensor(a), |_| unary_basic_int::BasicIntUnaryKind::BitwiseNot),
                93 => arithmetic::add(tensor(a), tensor(b)),
                94 => arithmetic::sub(tensor(a), tensor(b)),
                95 => arithmetic::mul(tensor(a), tensor(b)),
                _ => unreachable!(),
            }
        }
        96 => {
            assert!(a.dtype <= 2 && (4..=7).contains(&b.dtype));
            integer_power::tensor(tensor(a), tensor(b))
        }
        97 => {
            assert!(scalar >= 0.0 && scalar.fract() == 0.0 && (scalar as usize) < a.shape.len());
            ruprim::indexing::flip_on_output(tensor(a), tensor(out), &[scalar as usize], DType::U8);
            sync(&client());
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
            return;
        }
        98 | 99 => {
            assert_ne!(a.dtype, 3);
            let config = if op == 98 { ReduceOperationConfig::Prod } else { ReduceOperationConfig::Sum };
            reduce_dim(tensor(a), None, scalar as usize, KernelReduceStrategy::Unspecified, config)
                .expect("RUDA primitive reduction failed")
        }
        100 | 101 => {
            assert_ne!(a.dtype, 3);
            if op == 100 { ruprim::scan::cumsum(tensor(a), scalar as usize) }
            else { ruprim::scan::cumprod(tensor(a), scalar as usize) }
        }
        102..=106 => {
            let axis = scalar as usize;
            let bound = if op <= 103 { a.shape[axis] } else { out.shape[axis] };
            check_indices(b, bound);
            match op {
                102 => ruprim::indexing::gather(axis, tensor(a), tensor(b)),
                103 => ruprim::indexing::select(tensor(a), axis, tensor(b)),
                104 => {
                    if b.len == 0 { return; }
                    let result = ruprim::indexing::scatter(axis, tensor(out), tensor(b), tensor(a), out.dtype == 3);
                    LAUNCHES.fetch_add(1, Ordering::Relaxed);
                    result
                }
                105 => {
                    if b.len == 0 { return; }
                    let result = ruprim::indexing::select_assign(tensor(out), axis, tensor(b), tensor(a), out.dtype == 3);
                    LAUNCHES.fetch_add(1, Ordering::Relaxed);
                    result
                }
                106 => {
                    if b.len == 0 { return; }
                    let result = ruprim::indexing::scatter_assign(axis, tensor(out), tensor(b), tensor(a));
                    LAUNCHES.fetch_add(1, Ordering::Relaxed);
                    result
                }
                _ => unreachable!(),
            }
        }
        _ => panic!("unsupported RUDA primitive {op}"),
    };
    sync(&client());
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    let shape = result.meta.shape.iter().copied().collect::<Vec<_>>();
    let strides = result.meta.strides.iter().copied().collect::<Vec<_>>();
    assert_eq!(shape, out.shape);
    assert_eq!(result.dtype, dtype(out.dtype));
    let len = out.len;
    let view = View { handle: result.handle, shape, strides, len, dtype: out.dtype };
    convert(&view, out);
}
