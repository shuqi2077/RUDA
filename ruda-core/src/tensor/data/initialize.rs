use super::TensorData;
use crate::tensor::{BoolStore, DType, Shape};
use crate::tensor::distribution::Distribution;
use crate::tensor::element::{Element, ElementConversion, Scalar};
use alloc::vec::Vec;
use half::{bf16, f16};
use rand::Rng;

impl TensorData {
    /// Populates the data with random values.
    pub fn random<E: Element, R: Rng, S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        rng: &mut R,
    ) -> Self {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(E::random(distribution, rng));
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with zeros.
    pub fn zeros<E: Element, S: Into<Shape>>(shape: S) -> TensorData {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(0.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with ones.
    pub fn ones<E: Element, S: Into<Shape>>(shape: S) -> TensorData {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(1.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with the given value
    pub fn full<E: Element, S: Into<Shape>>(shape: S, fill_value: E) -> TensorData {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(fill_value)
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with the given value
    pub fn full_dtype<E: Into<Scalar>, S: Into<Shape>>(
        shape: S,
        fill_value: E,
        dtype: DType,
    ) -> TensorData {
        let fill_value = fill_value.into();
        match dtype {
            DType::F64 => Self::full::<f64, _>(shape, fill_value.elem()),
            DType::F32 | DType::Flex32 => Self::full::<f32, _>(shape, fill_value.elem()),
            DType::F16 => Self::full::<f16, _>(shape, fill_value.elem()),
            DType::BF16 => Self::full::<bf16, _>(shape, fill_value.elem()),
            DType::I64 => Self::full::<i64, _>(shape, fill_value.elem()),
            DType::I32 => Self::full::<i32, _>(shape, fill_value.elem()),
            DType::I16 => Self::full::<i16, _>(shape, fill_value.elem()),
            DType::I8 => Self::full::<i8, _>(shape, fill_value.elem()),
            DType::U64 => Self::full::<u64, _>(shape, fill_value.elem()),
            DType::U32 => Self::full::<u32, _>(shape, fill_value.elem()),
            DType::U16 => Self::full::<u16, _>(shape, fill_value.elem()),
            DType::U8 => Self::full::<u8, _>(shape, fill_value.elem()),
            DType::Bool(BoolStore::Native) => Self::full::<bool, _>(shape, fill_value.elem()),
            DType::Bool(BoolStore::U8) => {
                Self::full::<u8, _>(shape, fill_value.elem()).into_bool_u8()
            }
            DType::Bool(BoolStore::U32) => {
                Self::full::<u32, _>(shape, fill_value.elem()).into_bool_u32()
            }
            DType::QFloat(_) => unreachable!(),
        }
    }

}
