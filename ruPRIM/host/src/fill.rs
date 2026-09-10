use alloc::vec::Vec;
use ruda_core::{bytes::Bytes, tensor::{DType, FloatDType, IntDType, Shape, host::{HostTensor, Layout}, element::Scalar}};
use half::{bf16, f16};
use num_traits::ToPrimitive;

pub fn float_ones(shape: Shape, dtype: FloatDType) -> HostTensor {
    let dt: ruda_core::tensor::DType = dtype.into();
    match dt {
        DType::F32 => HostTensor::filled_typed(shape, dt, 1.0f32),
        DType::F64 => HostTensor::filled_typed(shape, dt, 1.0f64),
        DType::F16 => HostTensor::filled_typed(shape, dt, f16::ONE),
        DType::BF16 => HostTensor::filled_typed(shape, dt, bf16::ONE),
        _ => unreachable!(),
    }
}

pub fn float_full(
    shape: Shape,
    fill_value: Scalar,
    dtype: FloatDType,
) -> HostTensor {
    let dt: ruda_core::tensor::DType = dtype.into();
    match dt {
        DType::F32 => HostTensor::filled_typed(shape, dt, fill_value.to_f32().unwrap()),
        DType::F64 => HostTensor::filled_typed(shape, dt, fill_value.to_f64().unwrap()),
        DType::F16 => {
            HostTensor::filled_typed(shape, dt, f16::from_f32(fill_value.to_f32().unwrap()))
        }
        DType::BF16 => {
            HostTensor::filled_typed(shape, dt, bf16::from_f32(fill_value.to_f32().unwrap()))
        }
        _ => unreachable!(),
    }
}

pub fn int_ones(shape: Shape, dtype: IntDType) -> HostTensor {
    let dt: DType = dtype.into();
    match dt {
        DType::I64 => HostTensor::filled_typed(shape, dt, 1i64),
        DType::I32 => HostTensor::filled_typed(shape, dt, 1i32),
        DType::I16 => HostTensor::filled_typed(shape, dt, 1i16),
        DType::I8 => HostTensor::filled_typed(shape, dt, 1i8),
        DType::U64 => HostTensor::filled_typed(shape, dt, 1u64),
        DType::U32 => HostTensor::filled_typed(shape, dt, 1u32),
        DType::U16 => HostTensor::filled_typed(shape, dt, 1u16),
        DType::U8 => HostTensor::filled_typed(shape, dt, 1u8),
        _ => unreachable!(),
    }
}

pub fn int_full(
    shape: Shape,
    fill_value: ruda_core::tensor::element::Scalar,
    dtype: IntDType,
) -> HostTensor {
    let dt: DType = dtype.into();
    let v = fill_value.to_i64().unwrap();
    match dt {
        DType::I64 => HostTensor::filled_typed(shape, dt, v),
        DType::I32 => HostTensor::filled_typed(shape, dt, v as i32),
        DType::I16 => HostTensor::filled_typed(shape, dt, v as i16),
        DType::I8 => HostTensor::filled_typed(shape, dt, v as i8),
        DType::U64 => HostTensor::filled_typed(shape, dt, v as u64),
        DType::U32 => HostTensor::filled_typed(shape, dt, v as u32),
        DType::U16 => HostTensor::filled_typed(shape, dt, v as u16),
        DType::U8 => HostTensor::filled_typed(shape, dt, v as u8),
        _ => unreachable!(),
    }
}

pub fn int_arange_step(
    range: core::ops::Range<i64>,
    step: usize,
    dtype: IntDType,
) -> HostTensor {
    let dt: DType = dtype.into();

    macro_rules! arange_typed {
        ($ty:ty) => {{
            let data: Vec<$ty> = range.step_by(step).map(|v| v as $ty).collect();
            let shape = Shape::from(alloc::vec![data.len()]);
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), dt)
        }};
    }

    match dt {
        DType::I64 => arange_typed!(i64),
        DType::I32 => arange_typed!(i32),
        DType::I16 => arange_typed!(i16),
        DType::I8 => arange_typed!(i8),
        DType::U64 => arange_typed!(u64),
        DType::U32 => arange_typed!(u32),
        DType::U16 => arange_typed!(u16),
        DType::U8 => arange_typed!(u8),
        _ => unreachable!(),
    }
}
