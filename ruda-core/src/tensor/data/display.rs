use super::TensorData;
use crate::tensor::{BoolStore, DType, QuantLevel, QuantMode, QuantScheme, QuantValue};
use half::{bf16, f16};
use alloc::format;
use alloc::vec::Vec;

impl core::fmt::Display for TensorData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let fmt = match self.dtype {
            DType::F64 => format!("{:?}", self.as_slice::<f64>().unwrap()),
            DType::F32 | DType::Flex32 => format!("{:?}", self.as_slice::<f32>().unwrap()),
            DType::F16 => format!("{:?}", self.as_slice::<f16>().unwrap()),
            DType::BF16 => format!("{:?}", self.as_slice::<bf16>().unwrap()),
            DType::I64 => format!("{:?}", self.as_slice::<i64>().unwrap()),
            DType::I32 => format!("{:?}", self.as_slice::<i32>().unwrap()),
            DType::I16 => format!("{:?}", self.as_slice::<i16>().unwrap()),
            DType::I8 => format!("{:?}", self.as_slice::<i8>().unwrap()),
            DType::U64 => format!("{:?}", self.as_slice::<u64>().unwrap()),
            DType::U32 => format!("{:?}", self.as_slice::<u32>().unwrap()),
            DType::U16 => format!("{:?}", self.as_slice::<u16>().unwrap()),
            DType::U8 => format!("{:?}", self.as_slice::<u8>().unwrap()),
            DType::Bool(BoolStore::Native) => format!("{:?}", self.as_slice::<bool>().unwrap()),
            DType::Bool(BoolStore::U8) => format!("{:?}", self.as_slice::<u8>().unwrap()),
            DType::Bool(BoolStore::U32) => format!("{:?}", self.as_slice::<u32>().unwrap()),
            DType::QFloat(scheme) => match scheme {
                QuantScheme {
                    level: QuantLevel::Tensor | QuantLevel::Block(_),
                    mode: QuantMode::Symmetric,
                    value:
                        QuantValue::Q8F
                        | QuantValue::Q8S
                        // Display sub-byte values as i8
                        | QuantValue::Q4F
                        | QuantValue::Q4S
                        | QuantValue::Q2F
                        | QuantValue::Q2S,
                    ..
                } => {
                    format!("{:?} {scheme:?}", self.iter::<i8>().collect::<Vec<_>>())
                },
                QuantScheme {
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
                        mode: QuantMode::Symmetric,
                        value:
                            QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1,
                        ..
                    } => {
                        unimplemented!("Can't format yet");
                    }
            },
        };
        f.write_str(fmt.as_str())
    }
}
