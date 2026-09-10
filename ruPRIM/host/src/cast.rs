use alloc::vec::Vec;
use half::{bf16, f16};
use ruda_core::{bytes::Bytes, tensor::{DType, FloatDType, IntDType, host::{HostTensor, Layout}}};

pub fn bool_into_int(tensor: HostTensor, out_dtype: ruda_core::tensor::IntDType) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let out_dt = DType::from(out_dtype);
    let bools = tensor.bytes();

    macro_rules! convert {
        ($int_ty:ty) => {{
            let data: Vec<$int_ty> =
                bools.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }};
    }

    match out_dtype {
        IntDType::I64 => convert!(i64),
        IntDType::I32 => convert!(i32),
        IntDType::I16 => convert!(i16),
        IntDType::I8 => convert!(i8),
        IntDType::U64 => convert!(u64),
        IntDType::U32 => convert!(u32),
        IntDType::U16 => convert!(u16),
        IntDType::U8 => convert!(u8),
    }
}

pub fn bool_into_float(
    tensor: HostTensor,
    out_dtype: ruda_core::tensor::FloatDType,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let out_dt = DType::from(out_dtype);
    let bools = tensor.bytes();

    match out_dtype {
        FloatDType::F64 => {
            let data: Vec<f64> = bools
                .iter()
                .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                .collect();
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
        FloatDType::F32 | FloatDType::Flex32 => {
            let data: Vec<f32> = bools
                .iter()
                .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                .collect();
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
        FloatDType::F16 => {
            let one = f16::from_f32(1.0);
            let zero = f16::from_f32(0.0);
            let data: Vec<f16> = bools
                .iter()
                .map(|&x| if x != 0 { one } else { zero })
                .collect();
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
        FloatDType::BF16 => {
            let one = bf16::from_f32(1.0);
            let zero = bf16::from_f32(0.0);
            let data: Vec<bf16> = bools
                .iter()
                .map(|&x| if x != 0 { one } else { zero })
                .collect();
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
    }
}

// Precision limits: i64/u64 > 2^24 for f32/f16/bf16, > 2^53 for f64.
pub fn int_into_float(
    tensor: HostTensor,
    out_dtype: ruda_core::tensor::FloatDType,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let src = tensor.dtype();
    let out_dt = DType::from(out_dtype);

    // Read source ints, applying conversion per-element.
    // Each arm binds `$x` to the native int value; `$conv` must work for all int types.
    macro_rules! read_ints {
        (|$x:ident| $conv:expr) => {
            match src {
                DType::I64 => tensor.storage::<i64>().iter().map(|&$x| $conv).collect(),
                DType::I32 => tensor.storage::<i32>().iter().map(|&$x| $conv).collect(),
                DType::I16 => tensor.storage::<i16>().iter().map(|&$x| $conv).collect(),
                DType::I8 => tensor.storage::<i8>().iter().map(|&$x| $conv).collect(),
                DType::U64 => tensor.storage::<u64>().iter().map(|&$x| $conv).collect(),
                DType::U32 => tensor.storage::<u32>().iter().map(|&$x| $conv).collect(),
                DType::U16 => tensor.storage::<u16>().iter().map(|&$x| $conv).collect(),
                DType::U8 => tensor.storage::<u8>().iter().map(|&$x| $conv).collect(),
                _ => panic!("int_into_float: unsupported source dtype {:?}", src),
            }
        };
    }

    match out_dtype {
        FloatDType::F64 => {
            let data: Vec<f64> = read_ints!(|x| x as f64);
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
        FloatDType::F32 | FloatDType::Flex32 => {
            let data: Vec<f32> = read_ints!(|x| x as f32);
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
        FloatDType::F16 => {
            let data: Vec<f16> = read_ints!(|x| f16::from_f32(x as f32));
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
        FloatDType::BF16 => {
            let data: Vec<bf16> = read_ints!(|x| bf16::from_f32(x as f32));
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }
    }
}

pub fn int_cast(tensor: HostTensor, dtype: IntDType) -> HostTensor {
    let target_dtype: DType = dtype.into();

    // If already the target dtype, return as-is
    if tensor.dtype() == target_dtype {
        return tensor;
    }

    // Make contiguous for easier iteration
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();

    // Helper macro to convert between types
    macro_rules! cast_impl {
        ($src_type:ty, $dst_type:ty, $dst_dtype:expr) => {{
            let src: &[$src_type] = tensor.storage();
            let dst: Vec<$dst_type> = src.iter().map(|&x| x as $dst_type).collect();
            HostTensor::new(
                Bytes::from_elems(dst),
                Layout::contiguous(shape),
                $dst_dtype,
            )
        }};
    }

    // Match source dtype to target dtype
    match (tensor.dtype(), target_dtype) {
        // From I64
        (DType::I64, DType::I32) => cast_impl!(i64, i32, DType::I32),
        (DType::I64, DType::I16) => cast_impl!(i64, i16, DType::I16),
        (DType::I64, DType::I8) => cast_impl!(i64, i8, DType::I8),
        (DType::I64, DType::U64) => cast_impl!(i64, u64, DType::U64),
        (DType::I64, DType::U32) => cast_impl!(i64, u32, DType::U32),
        (DType::I64, DType::U16) => cast_impl!(i64, u16, DType::U16),
        (DType::I64, DType::U8) => cast_impl!(i64, u8, DType::U8),

        // From I32
        (DType::I32, DType::I64) => cast_impl!(i32, i64, DType::I64),
        (DType::I32, DType::I16) => cast_impl!(i32, i16, DType::I16),
        (DType::I32, DType::I8) => cast_impl!(i32, i8, DType::I8),
        (DType::I32, DType::U64) => cast_impl!(i32, u64, DType::U64),
        (DType::I32, DType::U32) => cast_impl!(i32, u32, DType::U32),
        (DType::I32, DType::U16) => cast_impl!(i32, u16, DType::U16),
        (DType::I32, DType::U8) => cast_impl!(i32, u8, DType::U8),

        // From I16
        (DType::I16, DType::I64) => cast_impl!(i16, i64, DType::I64),
        (DType::I16, DType::I32) => cast_impl!(i16, i32, DType::I32),
        (DType::I16, DType::I8) => cast_impl!(i16, i8, DType::I8),
        (DType::I16, DType::U64) => cast_impl!(i16, u64, DType::U64),
        (DType::I16, DType::U32) => cast_impl!(i16, u32, DType::U32),
        (DType::I16, DType::U16) => cast_impl!(i16, u16, DType::U16),
        (DType::I16, DType::U8) => cast_impl!(i16, u8, DType::U8),

        // From I8
        (DType::I8, DType::I64) => cast_impl!(i8, i64, DType::I64),
        (DType::I8, DType::I32) => cast_impl!(i8, i32, DType::I32),
        (DType::I8, DType::I16) => cast_impl!(i8, i16, DType::I16),
        (DType::I8, DType::U64) => cast_impl!(i8, u64, DType::U64),
        (DType::I8, DType::U32) => cast_impl!(i8, u32, DType::U32),
        (DType::I8, DType::U16) => cast_impl!(i8, u16, DType::U16),
        (DType::I8, DType::U8) => cast_impl!(i8, u8, DType::U8),

        // From U64
        (DType::U64, DType::I64) => cast_impl!(u64, i64, DType::I64),
        (DType::U64, DType::I32) => cast_impl!(u64, i32, DType::I32),
        (DType::U64, DType::I16) => cast_impl!(u64, i16, DType::I16),
        (DType::U64, DType::I8) => cast_impl!(u64, i8, DType::I8),
        (DType::U64, DType::U32) => cast_impl!(u64, u32, DType::U32),
        (DType::U64, DType::U16) => cast_impl!(u64, u16, DType::U16),
        (DType::U64, DType::U8) => cast_impl!(u64, u8, DType::U8),

        // From U32
        (DType::U32, DType::I64) => cast_impl!(u32, i64, DType::I64),
        (DType::U32, DType::I32) => cast_impl!(u32, i32, DType::I32),
        (DType::U32, DType::I16) => cast_impl!(u32, i16, DType::I16),
        (DType::U32, DType::I8) => cast_impl!(u32, i8, DType::I8),
        (DType::U32, DType::U64) => cast_impl!(u32, u64, DType::U64),
        (DType::U32, DType::U16) => cast_impl!(u32, u16, DType::U16),
        (DType::U32, DType::U8) => cast_impl!(u32, u8, DType::U8),

        // From U16
        (DType::U16, DType::I64) => cast_impl!(u16, i64, DType::I64),
        (DType::U16, DType::I32) => cast_impl!(u16, i32, DType::I32),
        (DType::U16, DType::I16) => cast_impl!(u16, i16, DType::I16),
        (DType::U16, DType::I8) => cast_impl!(u16, i8, DType::I8),
        (DType::U16, DType::U64) => cast_impl!(u16, u64, DType::U64),
        (DType::U16, DType::U32) => cast_impl!(u16, u32, DType::U32),
        (DType::U16, DType::U8) => cast_impl!(u16, u8, DType::U8),

        // From U8
        (DType::U8, DType::I64) => cast_impl!(u8, i64, DType::I64),
        (DType::U8, DType::I32) => cast_impl!(u8, i32, DType::I32),
        (DType::U8, DType::I16) => cast_impl!(u8, i16, DType::I16),
        (DType::U8, DType::I8) => cast_impl!(u8, i8, DType::I8),
        (DType::U8, DType::U64) => cast_impl!(u8, u64, DType::U64),
        (DType::U8, DType::U32) => cast_impl!(u8, u32, DType::U32),
        (DType::U8, DType::U16) => cast_impl!(u8, u16, DType::U16),

        _ => panic!(
            "int_cast: unsupported conversion from {:?} to {:?}",
            tensor.dtype(),
            target_dtype
        ),
    }
}

pub fn float_into_int(tensor: HostTensor, out_dtype: ruda_core::tensor::IntDType) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let src = tensor.dtype();
    let out_dt = DType::from(out_dtype);

    // Read source floats as f64 (lossless for f32/f16/bf16).
    macro_rules! read_floats {
        (|$x:ident| $conv:expr) => {
            match src {
                DType::F32 => tensor
                    .storage::<f32>()
                    .iter()
                    .map(|v| {
                        let $x = *v as f64;
                        $conv
                    })
                    .collect(),
                DType::F64 => tensor
                    .storage::<f64>()
                    .iter()
                    .map(|v| {
                        let $x = *v;
                        $conv
                    })
                    .collect(),
                DType::F16 => tensor
                    .storage::<f16>()
                    .iter()
                    .map(|v| {
                        let $x = f32::from(*v) as f64;
                        $conv
                    })
                    .collect(),
                DType::BF16 => tensor
                    .storage::<bf16>()
                    .iter()
                    .map(|v| {
                        let $x = f32::from(*v) as f64;
                        $conv
                    })
                    .collect(),
                _ => panic!("float_into_int: unsupported source dtype {:?}", src),
            }
        };
    }

    macro_rules! convert {
        ($int_ty:ty) => {{
            let data: Vec<$int_ty> = read_floats!(|x| x as $int_ty);
            HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
        }};
    }

    match out_dtype {
        IntDType::I64 => convert!(i64),
        IntDType::I32 => convert!(i32),
        IntDType::I16 => convert!(i16),
        IntDType::I8 => convert!(i8),
        IntDType::U64 => convert!(u64),
        IntDType::U32 => convert!(u32),
        IntDType::U16 => convert!(u16),
        IntDType::U8 => convert!(u8),
    }
}

pub fn float_cast(tensor: HostTensor, dtype: FloatDType) -> HostTensor {
    use ruda_core::tensor::host::Layout;
    use ruda_core::bytes::Bytes;
    use half::{bf16, f16};

    let src_dtype = tensor.dtype();
    let target_dtype = DType::from(dtype);

    // No-op if already the same dtype
    if src_dtype == target_dtype {
        return tensor;
    }

    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();

    // Convert to f64 intermediate, then to target
    let f64_values: Vec<f64> = match src_dtype {
        DType::F32 => {
            let src: &[f32] = tensor.storage();
            src.iter().map(|&v| v as f64).collect()
        }
        DType::F64 => {
            let src: &[f64] = tensor.storage();
            src.to_vec()
        }
        DType::F16 => {
            let src: &[f16] = tensor.storage();
            src.iter().map(|&v| v.to_f32() as f64).collect()
        }
        DType::BF16 => {
            let src: &[bf16] = tensor.storage();
            src.iter().map(|&v| v.to_f32() as f64).collect()
        }
        _ => panic!("float_cast: unsupported source dtype {:?}", src_dtype),
    };

    // Convert from f64 to target dtype
    match target_dtype {
        DType::F32 => {
            let result: Vec<f32> = f64_values.iter().map(|&v| v as f32).collect();
            let bytes = Bytes::from_elems(result);
            HostTensor::new(bytes, Layout::contiguous(shape), DType::F32)
        }
        DType::F64 => {
            let bytes = Bytes::from_elems(f64_values);
            HostTensor::new(bytes, Layout::contiguous(shape), DType::F64)
        }
        DType::F16 => {
            let result: Vec<f16> = f64_values.iter().map(|&v| f16::from_f64(v)).collect();
            let bytes = Bytes::from_elems(result);
            HostTensor::new(bytes, Layout::contiguous(shape), DType::F16)
        }
        DType::BF16 => {
            let result: Vec<bf16> = f64_values.iter().map(|&v| bf16::from_f64(v)).collect();
            let bytes = Bytes::from_elems(result);
            HostTensor::new(bytes, Layout::contiguous(shape), DType::BF16)
        }
        _ => panic!("float_cast: unsupported target dtype {:?}", target_dtype),
    }
}
