use super::{Result, invalid, unsupported};
use ruda_core::ir::{ConstantValue, ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scalar {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    F16,
    BF16,
    E4M3,
    E5M2,
    Pred,
}

impl Scalar {
    pub fn is_tf32(ty: Type) -> bool {
        matches!(ty, Type::Scalar(StorageType::Scalar(ElemType::Float(FloatKind::TF32))) | Type::Vector(StorageType::Scalar(ElemType::Float(FloatKind::TF32)), _))
    }

    pub fn memory_element(ty: Type) -> Result<Self> {
        match ty {
            Type::Scalar(StorageType::Atomic(elem)) => Self::of(Type::scalar(elem)),
            Type::Scalar(_) | Type::Vector(_, _) => Self::of(ty.with_vector_size(1)),
            Type::Semantic(_) => Err(unsupported("semantic type has no memory element")),
        }
    }

    pub fn of(ty: Type) -> Result<Self> {
        match ty {
            Type::Scalar(StorageType::Scalar(elem)) => match elem {
                ElemType::UInt(UIntKind::U8) => Ok(Self::U8),
                ElemType::UInt(UIntKind::U16) => Ok(Self::U16),
                ElemType::Int(IntKind::I8) => Ok(Self::I8),
                ElemType::Int(IntKind::I16) => Ok(Self::I16),
                ElemType::UInt(UIntKind::U32) => Ok(Self::U32),
                ElemType::UInt(UIntKind::U64) => Ok(Self::U64),
                ElemType::Int(IntKind::I32) => Ok(Self::I32),
                ElemType::Int(IntKind::I64) => Ok(Self::I64),
                ElemType::Float(FloatKind::F32 | FloatKind::TF32 | FloatKind::Flex32) => Ok(Self::F32),
                ElemType::Float(FloatKind::F64) => Ok(Self::F64),
                ElemType::Float(FloatKind::F16) => Ok(Self::F16),
                ElemType::Float(FloatKind::BF16) => Ok(Self::BF16),
                ElemType::Float(FloatKind::E4M3) => Ok(Self::E4M3),
                ElemType::Float(FloatKind::E5M2) => Ok(Self::E5M2),
                ElemType::Bool => Ok(Self::Pred),
                _ => Err(unsupported(format!("element type {elem:?}"))),
            },
            _ => Err(unsupported(format!("storage/vector type {ty:?}"))),
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Self::U8 | Self::U16 | Self::U32 => "u32",
            Self::I8 | Self::I16 | Self::I32 => "s32",
            Self::U64 => "u64",
            Self::I64 => "s64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::E4M3 => "e4m3",
            Self::E5M2 => "e5m2",
            Self::Pred => "pred",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::Pred | Self::I8 | Self::U8 | Self::E4M3 | Self::E5M2 => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            _ => 4,
        }
    }

    pub fn float(self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F16 | Self::BF16 | Self::E4M3 | Self::E5M2)
    }
    pub fn integer(self) -> bool {
        self.signed() || matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }
    pub fn signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }
    pub fn narrow(self) -> bool {
        matches!(self, Self::I8 | Self::U8 | Self::I16 | Self::U16)
    }
    pub fn memory_suffix(self) -> &'static str {
        match self {
            Self::I8 => "s8",
            Self::U8 | Self::E4M3 | Self::E5M2 | Self::Pred => "u8",
            Self::I16 => "s16",
            Self::U16 => "u16",
            _ => self.storage(),
        }
    }
    pub fn bits(self) -> &'static str {
        if self.narrow() { return "b32"; }
        match self.bytes() {
            8 => "b64",
            2 => "b16",
            _ => "b32",
        }
    }

    pub fn half(self) -> bool {
        matches!(self, Self::F16 | Self::BF16)
    }

    pub fn storage(self) -> &'static str {
        if self.fp8() { "u32" } else if self.half() { "b16" } else { self.suffix() }
    }

    pub fn fp8(self) -> bool {
        matches!(self, Self::E4M3 | Self::E5M2)
    }

    pub fn constant(self, value: ConstantValue) -> Result<String> {
        match (self, value) {
            (Self::Pred, ConstantValue::Bool(value)) => Ok(u8::from(value).to_string()),
            (Self::F32, ConstantValue::Float(v)) => Ok(format!("0f{:08X}", (v as f32).to_bits())),
            (Self::F64, ConstantValue::Float(v)) => Ok(format!("0d{:016X}", v.to_bits())),
            (Self::E4M3, ConstantValue::Float(v)) => Ok(ruda_core::e4m3::from_f64(v).to_bits().to_string()),
            (Self::E5M2, ConstantValue::Float(v)) => Ok(ruda_core::e5m2::from_f64(v).to_bits().to_string()),
            (Self::F16 | Self::BF16, ConstantValue::Float(v)) => {
                Ok(format!("0x{:04X}", super::half::constant_bits(self, v)))
            }
            (Self::I32, ConstantValue::Int(v)) => Ok((v as i32).to_string()),
            (Self::I8, ConstantValue::Int(v)) => Ok((v as i8).to_string()),
            (Self::I16, ConstantValue::Int(v)) => Ok((v as i16).to_string()),
            (Self::I64, ConstantValue::Int(v)) => Ok(v.to_string()),
            (Self::U32, ConstantValue::UInt(v)) => Ok((v as u32).to_string()),
            (Self::U8, ConstantValue::UInt(v)) => Ok((v as u8).to_string()),
            (Self::U16, ConstantValue::UInt(v)) => Ok((v as u16).to_string()),
            (Self::U64, ConstantValue::UInt(v)) => Ok(v.to_string()),
            _ => Err(invalid(format!("constant {value:?} for {self:?}"))),
        }
    }
}
