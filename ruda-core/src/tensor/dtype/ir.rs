use crate::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};
use crate::quant::scheme::QuantScheme;

use crate::tensor::DType;
use crate::quant::scheme::{QuantStore, QuantValue};

impl From<DType> for crate::ir::ElemType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F64 => ElemType::Float(FloatKind::F64),
            DType::F32 => ElemType::Float(FloatKind::F32),
            DType::Flex32 => ElemType::Float(FloatKind::Flex32),
            DType::F16 => ElemType::Float(FloatKind::F16),
            DType::BF16 => ElemType::Float(FloatKind::BF16),
            DType::I64 => ElemType::Int(IntKind::I64),
            DType::I32 => ElemType::Int(IntKind::I32),
            DType::I16 => ElemType::Int(IntKind::I16),
            DType::I8 => ElemType::Int(IntKind::I8),
            DType::U64 => ElemType::UInt(UIntKind::U64),
            DType::U32 => ElemType::UInt(UIntKind::U32),
            DType::U16 => ElemType::UInt(UIntKind::U16),
            DType::U8 => ElemType::UInt(UIntKind::U8),
            DType::Bool(store) => match store {
                super::BoolStore::Native => ElemType::Bool,
                super::BoolStore::U8 => ElemType::UInt(UIntKind::U8),
                super::BoolStore::U32 => ElemType::UInt(UIntKind::U32),
            },
            DType::QFloat(scheme) => match scheme.store {
                QuantStore::Native => match scheme.value {
                    QuantValue::Q8F | QuantValue::Q8S => Self::Int(IntKind::I8),
                    QuantValue::E4M3 => Self::Float(FloatKind::E4M3),
                    QuantValue::E5M2 => Self::Float(FloatKind::E5M2),
                    QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::Q2F
                    | QuantValue::Q2S
                    | QuantValue::E2M1 => {
                        panic!("Can't store native sub-byte values")
                    }
                },
                QuantStore::PackedU32(_) => Self::UInt(UIntKind::U32),
                QuantStore::PackedNative(_) => match scheme.value {
                    QuantValue::E2M1 => panic!("Can't store native sub-byte values"),
                    other => panic!("{other:?} doesn't support native packing"),
                },
            },
        }
    }
}

impl From<DType> for crate::ir::StorageType {
    fn from(dtype: DType) -> crate::ir::StorageType {
        match dtype {
            DType::QFloat(QuantScheme {
                store: QuantStore::PackedNative(_),
                value: QuantValue::E2M1,
                ..
            }) => StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
            _ => {
                let elem: ElemType = dtype.into();
                elem.into()
            }
        }
    }
}

impl From<crate::ir::ElemType> for DType {
    fn from(value: crate::ir::ElemType) -> Self {
        match value {
            crate::ir::ElemType::Float(float_kind) => match float_kind {
                crate::ir::FloatKind::F16 => DType::F16,
                crate::ir::FloatKind::BF16 => DType::BF16,
                crate::ir::FloatKind::Flex32 => DType::Flex32,
                crate::ir::FloatKind::F32 => DType::F32,
                crate::ir::FloatKind::F64 => DType::F64,
                crate::ir::FloatKind::TF32 => panic!("Not a valid DType for tensors."),
                crate::ir::FloatKind::E2M1
                | crate::ir::FloatKind::E2M3
                | crate::ir::FloatKind::E3M2
                | crate::ir::FloatKind::E4M3
                | crate::ir::FloatKind::E5M2
                | crate::ir::FloatKind::UE8M0 => {
                    unimplemented!("Not yet supported, will be used for quantization")
                }
            },
            crate::ir::ElemType::Int(int_kind) => match int_kind {
                crate::ir::IntKind::I8 => DType::I8,
                crate::ir::IntKind::I16 => DType::I16,
                crate::ir::IntKind::I32 => DType::I32,
                crate::ir::IntKind::I64 => DType::I64,
            },
            crate::ir::ElemType::UInt(uint_kind) => match uint_kind {
                crate::ir::UIntKind::U8 => DType::U8,
                crate::ir::UIntKind::U16 => DType::U16,
                crate::ir::UIntKind::U32 => DType::U32,
                crate::ir::UIntKind::U64 => DType::U64,
            },
            _ => panic!("Not a valid DType for tensors."),
        }
    }
}
