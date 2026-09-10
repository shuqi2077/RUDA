use super::TensorData;
use crate::tensor::{BoolStore, DType};
use crate::tensor::element::{Element, ElementConversion};
use alloc::vec::Vec;
use bytemuck::{AnyBitPattern, CheckedBitPattern, Zeroable, cast_mut};
use half::{bf16, f16};

impl TensorData {
    // Unchecked, used to overwrite the dtype
    pub(super) fn into_bool_u8(mut self) -> Self {
        self.dtype = DType::Bool(BoolStore::U8);
        self
    }

    // Unchecked, used to overwrite the dtype
    pub(super) fn into_bool_u32(mut self) -> Self {
        self.dtype = DType::Bool(BoolStore::U32);
        self
    }

    /// Converts the data to a different element type.
    pub fn convert<E: Element>(self) -> Self {
        self.convert_dtype(E::dtype())
    }

    /// Converts the data to a different element type.
    pub fn convert_dtype(self, dtype: DType) -> Self {
        if dtype == self.dtype {
            self
        } else if dtype.size() == self.dtype.size()
            && !matches!(
                self.dtype,
                DType::Bool(BoolStore::Native) | DType::QFloat(_)
            )
            && !matches!(dtype, DType::Bool(BoolStore::Native) | DType::QFloat(_))
        {
            match self.dtype {
                DType::F64 => self.convert_inplace_dtype::<f64>(dtype),
                DType::F32 | DType::Flex32 => self.convert_inplace_dtype::<f32>(dtype),
                DType::F16 => self.convert_inplace_dtype::<f16>(dtype),
                DType::BF16 => self.convert_inplace_dtype::<bf16>(dtype),
                DType::I64 => self.convert_inplace_dtype::<i64>(dtype),
                DType::I32 => self.convert_inplace_dtype::<i32>(dtype),
                DType::I16 => self.convert_inplace_dtype::<i16>(dtype),
                DType::I8 => self.convert_inplace_dtype::<i8>(dtype),
                DType::U64 => self.convert_inplace_dtype::<u64>(dtype),
                DType::U32 => self.convert_inplace_dtype::<u32>(dtype),
                DType::U16 => self.convert_inplace_dtype::<u16>(dtype),
                DType::U8 => self.convert_inplace_dtype::<u8>(dtype),
                DType::Bool(BoolStore::U8) => self.convert_inplace_dtype::<u8>(dtype),
                DType::Bool(BoolStore::U32) => self.convert_inplace_dtype::<u32>(dtype),
                DType::Bool(BoolStore::Native) | DType::QFloat(_) => unreachable!(),
            }
        } else {
            match self.dtype {
                DType::F64 => self.convert_clone_dtype::<f64>(dtype),
                DType::F32 | DType::Flex32 => self.convert_clone_dtype::<f32>(dtype),
                DType::F16 => self.convert_clone_dtype::<f16>(dtype),
                DType::BF16 => self.convert_clone_dtype::<bf16>(dtype),
                DType::I64 => self.convert_clone_dtype::<i64>(dtype),
                DType::I32 => self.convert_clone_dtype::<i32>(dtype),
                DType::I16 => self.convert_clone_dtype::<i16>(dtype),
                DType::I8 => self.convert_clone_dtype::<i8>(dtype),
                DType::U64 => self.convert_clone_dtype::<u64>(dtype),
                DType::U32 => self.convert_clone_dtype::<u32>(dtype),
                DType::U16 => self.convert_clone_dtype::<u16>(dtype),
                DType::U8 => self.convert_clone_dtype::<u8>(dtype),
                DType::Bool(BoolStore::Native) => self.convert_clone_dtype::<bool>(dtype),
                DType::Bool(BoolStore::U8) => self.convert_clone_dtype::<u8>(dtype),
                DType::Bool(BoolStore::U32) => self.convert_clone_dtype::<u32>(dtype),
                DType::QFloat(_) => unreachable!(),
            }
        }
    }

    fn convert_inplace_dtype<Current: Element + AnyBitPattern>(self, dtype: DType) -> Self {
        match dtype {
            DType::F64 => self.convert_inplace::<Current, f64>(),
            DType::F32 | DType::Flex32 => self.convert_inplace::<Current, f32>(),
            DType::F16 => self.convert_inplace::<Current, f16>(),
            DType::BF16 => self.convert_inplace::<Current, bf16>(),
            DType::I64 => self.convert_inplace::<Current, i64>(),
            DType::I32 => self.convert_inplace::<Current, i32>(),
            DType::I16 => self.convert_inplace::<Current, i16>(),
            DType::I8 => self.convert_inplace::<Current, i8>(),
            DType::U64 => self.convert_inplace::<Current, u64>(),
            DType::U32 => self.convert_inplace::<Current, u32>(),
            DType::U16 => self.convert_inplace::<Current, u16>(),
            DType::U8 => self.convert_inplace::<Current, u8>(),
            DType::Bool(BoolStore::U8) => self.convert_inplace::<Current, u8>().into_bool_u8(),
            DType::Bool(BoolStore::U32) => self.convert_inplace::<Current, u32>().into_bool_u32(),
            DType::Bool(BoolStore::Native) | DType::QFloat(_) => unreachable!(),
        }
    }

    fn convert_inplace<Current: Element + AnyBitPattern, Target: Element + AnyBitPattern>(
        mut self,
    ) -> Self {
        for x in bytemuck::cast_slice_mut::<_, Current>(&mut self.bytes) {
            let t: Target = x.elem();
            let x = cast_mut::<_, Target>(x);
            *x = t;
        }

        self.dtype = Target::dtype();

        self
    }

    fn convert_clone_dtype<Current: Element + CheckedBitPattern>(self, dtype: DType) -> Self {
        match dtype {
            DType::F64 => self.convert_clone::<Current, f64>(),
            DType::F32 | DType::Flex32 => self.convert_clone::<Current, f32>(),
            DType::F16 => self.convert_clone::<Current, f16>(),
            DType::BF16 => self.convert_clone::<Current, bf16>(),
            DType::I64 => self.convert_clone::<Current, i64>(),
            DType::I32 => self.convert_clone::<Current, i32>(),
            DType::I16 => self.convert_clone::<Current, i16>(),
            DType::I8 => self.convert_clone::<Current, i8>(),
            DType::U64 => self.convert_clone::<Current, u64>(),
            DType::U32 => self.convert_clone::<Current, u32>(),
            DType::U16 => self.convert_clone::<Current, u16>(),
            DType::U8 => self.convert_clone::<Current, u8>(),
            DType::Bool(BoolStore::Native) => self.convert_clone::<Current, bool>(),
            DType::Bool(BoolStore::U8) => self.convert_clone::<Current, u8>().into_bool_u8(),
            DType::Bool(BoolStore::U32) => self.convert_clone::<Current, u32>().into_bool_u32(),
            DType::QFloat(_) => unreachable!(),
        }
    }

    fn convert_clone<Current: Element + CheckedBitPattern, Target: Element + Zeroable>(
        self,
    ) -> Self {
        let this = bytemuck::checked::cast_slice::<_, Current>(&self.bytes);
        let mut out: Vec<Target> = ::alloc::vec![Zeroable::zeroed(); self.num_elements()];

        for (x, out) in this.iter().zip(&mut out) {
            *out = x.elem();
        }

        Self::new(out, self.shape)
    }

}
