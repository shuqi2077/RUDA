use alloc::vec::Vec;
use half::{bf16, f16};
use num_traits::Float;
use ruda_core::tensor::{DType, element::Element, host::{HostTensor, strided_index::StridedIter}};

trait Exponent: Element + bytemuck::Pod {
    fn parts(self) -> (bool, u64);
}

macro_rules! signed_exponent {
    ($($ty:ty),*) => {$(
        impl Exponent for $ty {
            fn parts(self) -> (bool, u64) {
                let exponent = self as i64;
                (exponent < 0, exponent.unsigned_abs())
            }
        }
    )*};
}

macro_rules! unsigned_exponent {
    ($($ty:ty),*) => {$(
        impl Exponent for $ty {
            fn parts(self) -> (bool, u64) {
                (false, self as u64)
            }
        }
    )*};
}

signed_exponent!(i8, i16, i32, i64);
unsigned_exponent!(u8, u16, u32, u64);

fn powi<F: Float>(mut base: F, (negative, mut exponent): (bool, u64)) -> F {
    let mut result = F::one();
    if negative {
        base = F::one() / base;
    }
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result * base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base * base;
        }
    }
    result
}

pub(super) fn scalar(lhs: HostTensor, negative: bool, exponent: u64) -> HostTensor {
    crate::unary::unary_op(
        lhs,
        |base| powi(base, (negative, exponent)),
        |base| powi(base, (negative, exponent)),
    )
}

fn mixed<E, I, Op>(mut lhs: HostTensor, rhs: &HostTensor, op: Op) -> HostTensor
where
    E: Element + bytemuck::Pod,
    I: Exponent,
    Op: Fn(E, I) -> E,
{
    let rhs_storage: &[I] = rhs.storage();
    if lhs.is_unique()
        && let (Some((0, end)), Some((start_rhs, end_rhs))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let lhs_storage: &mut [E] = lhs.storage_mut();
        for (base, &exponent) in lhs_storage[..end].iter_mut()
            .zip(&rhs_storage[start_rhs..end_rhs])
        {
            *base = op(*base, exponent);
        }
        return lhs;
    }

    let lhs_storage: &[E] = lhs.storage();
    let result: Vec<E> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((start_lhs, end_lhs)), Some((start_rhs, end_rhs))) => {
            lhs_storage[start_lhs..end_lhs].iter()
                .zip(&rhs_storage[start_rhs..end_rhs])
                .map(|(&base, &exponent)| op(base, exponent))
                .collect()
        }
        _ => StridedIter::new(lhs.layout()).zip(StridedIter::new(rhs.layout()))
            .map(|(base, exponent)| op(lhs_storage[base], rhs_storage[exponent]))
            .collect(),
    };
    super::make_tensor(result, lhs.layout().shape().clone(), lhs.dtype())
}

fn with_integer<I: Exponent>(lhs: HostTensor, rhs: &HostTensor) -> HostTensor {
    match lhs.dtype() {
        DType::F32 => mixed(lhs, rhs, |base: f32, exponent: I| powi(base, exponent.parts())),
        DType::F64 => mixed(lhs, rhs, |base: f64, exponent: I| powi(base, exponent.parts())),
        DType::F16 => mixed(lhs, rhs, |base: f16, exponent: I| {
            f16::from_f32(powi(base.to_f32(), exponent.parts()))
        }),
        DType::BF16 => mixed(lhs, rhs, |base: bf16, exponent: I| {
            bf16::from_f32(powi(base.to_f32(), exponent.parts()))
        }),
        dtype => panic!("float_powi: unsupported base dtype {dtype:?}"),
    }
}

pub(super) fn tensor(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);
    match rhs.dtype() {
        DType::I8 => with_integer::<i8>(lhs, &rhs),
        DType::I16 => with_integer::<i16>(lhs, &rhs),
        DType::I32 => with_integer::<i32>(lhs, &rhs),
        DType::I64 => with_integer::<i64>(lhs, &rhs),
        DType::U8 => with_integer::<u8>(lhs, &rhs),
        DType::U16 => with_integer::<u16>(lhs, &rhs),
        DType::U32 => with_integer::<u32>(lhs, &rhs),
        DType::U64 => with_integer::<u64>(lhs, &rhs),
        dtype => panic!("float_powi: unsupported exponent dtype {dtype:?}"),
    }
}
