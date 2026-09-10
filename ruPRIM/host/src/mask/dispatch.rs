use ruda_core::tensor::{DType, host::HostTensor, element::Scalar};
use half::{bf16, f16};
use num_traits::ToPrimitive;

pub fn float_mask_where(
    tensor: HostTensor,
    mask: HostTensor,
    value: HostTensor,
) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::mask::mask_where_f32(tensor, mask, value),
        DType::F64 => crate::mask::mask_where_f64(tensor, mask, value),
        DType::F16 => crate::mask::mask_where_f16(tensor, mask, value),
        DType::BF16 => crate::mask::mask_where_bf16(tensor, mask, value),
        dtype => panic!("float_mask_where: unsupported dtype {:?}", dtype),
    }
}

pub fn float_mask_fill(
    tensor: HostTensor,
    mask: HostTensor,
    value: Scalar,
) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::mask::mask_fill_f32(tensor, mask, value.to_f32().unwrap()),
        DType::F64 => crate::mask::mask_fill_f64(tensor, mask, value.to_f64().unwrap()),
        DType::F16 => crate::mask::mask_fill_f16(
            tensor,
            mask,
            f16::from_f64(value.to_f64().unwrap()),
        ),
        DType::BF16 => crate::mask::mask_fill_bf16(
            tensor,
            mask,
            bf16::from_f64(value.to_f64().unwrap()),
        ),
        dtype => panic!("float_mask_fill: unsupported dtype {:?}", dtype),
    }
}

pub fn int_mask_where(
    tensor: HostTensor,
    mask: HostTensor,
    value: HostTensor,
) -> HostTensor {
    debug_assert_eq!(
        tensor.dtype(),
        value.dtype(),
        "int_mask_where: dtype mismatch"
    );
    match tensor.dtype() {
        DType::I64 => crate::mask::mask_where::<i64>(tensor, mask, value),
        DType::I32 => crate::mask::mask_where::<i32>(tensor, mask, value),
        DType::I16 => crate::mask::mask_where::<i16>(tensor, mask, value),
        DType::I8 => crate::mask::mask_where::<i8>(tensor, mask, value),
        DType::U64 => crate::mask::mask_where::<u64>(tensor, mask, value),
        DType::U32 => crate::mask::mask_where::<u32>(tensor, mask, value),
        DType::U16 => crate::mask::mask_where::<u16>(tensor, mask, value),
        DType::U8 => crate::mask::mask_where::<u8>(tensor, mask, value),
        dt => panic!("int_mask_where: unsupported dtype {:?}", dt),
    }
}

pub fn int_mask_fill(
    tensor: HostTensor,
    mask: HostTensor,
    value: Scalar,
) -> HostTensor {
    match tensor.dtype() {
        DType::I64 => crate::mask::mask_fill(tensor, mask, value.to_i64().unwrap()),
        DType::I32 => crate::mask::mask_fill(tensor, mask, value.to_i64().unwrap() as i32),
        DType::I16 => crate::mask::mask_fill(tensor, mask, value.to_i64().unwrap() as i16),
        DType::I8 => crate::mask::mask_fill(tensor, mask, value.to_i64().unwrap() as i8),
        DType::U64 => crate::mask::mask_fill(tensor, mask, value.to_u64().unwrap()),
        DType::U32 => crate::mask::mask_fill(tensor, mask, value.to_u64().unwrap() as u32),
        DType::U16 => crate::mask::mask_fill(tensor, mask, value.to_u64().unwrap() as u16),
        DType::U8 => crate::mask::mask_fill(tensor, mask, value.to_u64().unwrap() as u8),
        dt => panic!("int_mask_fill: unsupported dtype {:?}", dt),
    }
}

