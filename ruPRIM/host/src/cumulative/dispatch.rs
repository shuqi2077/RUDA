use ruda_core::tensor::{DType, host::HostTensor};
use half::{bf16, f16};

pub fn float_cumsum(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::cumulative::cumsum_f32(tensor, dim),
        DType::F64 => crate::cumulative::cumsum_f64(tensor, dim),
        DType::F16 => {
            crate::cumulative::cumsum_half(tensor, dim, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => {
            crate::cumulative::cumsum_half(tensor, dim, bf16::to_f32, bf16::from_f32)
        }
        _ => panic!("float_cumsum: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn float_cumprod(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::cumulative::cumprod_f32(tensor, dim),
        DType::F64 => crate::cumulative::cumprod_f64(tensor, dim),
        DType::F16 => {
            crate::cumulative::cumprod_half(tensor, dim, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => {
            crate::cumulative::cumprod_half(tensor, dim, bf16::to_f32, bf16::from_f32)
        }
        _ => panic!("float_cumprod: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn float_cummin(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::cumulative::cummin_f32(tensor, dim),
        DType::F64 => crate::cumulative::cummin_f64(tensor, dim),
        DType::F16 => {
            crate::cumulative::cummin_half(tensor, dim, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => {
            crate::cumulative::cummin_half(tensor, dim, bf16::to_f32, bf16::from_f32)
        }
        _ => panic!("float_cummin: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn float_cummax(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::cumulative::cummax_f32(tensor, dim),
        DType::F64 => crate::cumulative::cummax_f64(tensor, dim),
        DType::F16 => {
            crate::cumulative::cummax_half(tensor, dim, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => {
            crate::cumulative::cummax_half(tensor, dim, bf16::to_f32, bf16::from_f32)
        }
        _ => panic!("float_cummax: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn int_cumsum(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::I64 => crate::cumulative::cumsum::<i64>(tensor, dim),
        DType::I32 => crate::cumulative::cumsum::<i32>(tensor, dim),
        DType::I16 => crate::cumulative::cumsum::<i16>(tensor, dim),
        DType::I8 => crate::cumulative::cumsum::<i8>(tensor, dim),
        DType::U64 => crate::cumulative::cumsum::<u64>(tensor, dim),
        DType::U32 => crate::cumulative::cumsum::<u32>(tensor, dim),
        DType::U16 => crate::cumulative::cumsum::<u16>(tensor, dim),
        DType::U8 => crate::cumulative::cumsum::<u8>(tensor, dim),
        dt => panic!("int_cumsum: unsupported dtype {:?}", dt),
    }
}

pub fn int_cumprod(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::I64 => crate::cumulative::cumprod::<i64>(tensor, dim),
        DType::I32 => crate::cumulative::cumprod::<i32>(tensor, dim),
        DType::I16 => crate::cumulative::cumprod::<i16>(tensor, dim),
        DType::I8 => crate::cumulative::cumprod::<i8>(tensor, dim),
        DType::U64 => crate::cumulative::cumprod::<u64>(tensor, dim),
        DType::U32 => crate::cumulative::cumprod::<u32>(tensor, dim),
        DType::U16 => crate::cumulative::cumprod::<u16>(tensor, dim),
        DType::U8 => crate::cumulative::cumprod::<u8>(tensor, dim),
        dt => panic!("int_cumprod: unsupported dtype {:?}", dt),
    }
}

pub fn int_cummin(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::I64 => crate::cumulative::cummin::<i64>(tensor, dim),
        DType::I32 => crate::cumulative::cummin::<i32>(tensor, dim),
        DType::I16 => crate::cumulative::cummin::<i16>(tensor, dim),
        DType::I8 => crate::cumulative::cummin::<i8>(tensor, dim),
        DType::U64 => crate::cumulative::cummin::<u64>(tensor, dim),
        DType::U32 => crate::cumulative::cummin::<u32>(tensor, dim),
        DType::U16 => crate::cumulative::cummin::<u16>(tensor, dim),
        DType::U8 => crate::cumulative::cummin::<u8>(tensor, dim),
        dt => panic!("int_cummin: unsupported dtype {:?}", dt),
    }
}

pub fn int_cummax(tensor: HostTensor, dim: usize) -> HostTensor {
    match tensor.dtype() {
        DType::I64 => crate::cumulative::cummax::<i64>(tensor, dim),
        DType::I32 => crate::cumulative::cummax::<i32>(tensor, dim),
        DType::I16 => crate::cumulative::cummax::<i16>(tensor, dim),
        DType::I8 => crate::cumulative::cummax::<i8>(tensor, dim),
        DType::U64 => crate::cumulative::cummax::<u64>(tensor, dim),
        DType::U32 => crate::cumulative::cummax::<u32>(tensor, dim),
        DType::U16 => crate::cumulative::cummax::<u16>(tensor, dim),
        DType::U8 => crate::cumulative::cummax::<u8>(tensor, dim),
        dt => panic!("int_cummax: unsupported dtype {:?}", dt),
    }
}

