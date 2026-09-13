use super::{CudaDevice, CudaRuntime, LAUNCHES, Ordering, View, client, convert, sync};
use ruda_core::tensor::{DType, Metadata};
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
