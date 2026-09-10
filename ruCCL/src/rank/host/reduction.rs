use super::super::{ElementType, ReductionOperation};

#[cfg(test)]
mod tests;

pub trait HostReductionElement: Copy {
    const SIZE: usize;

    fn decode(bytes: &[u8]) -> Result<Vec<Self>, String>;
    fn encode(values: &[Self]) -> Vec<u8>;
    fn combine(self, source: Self, operation: ReductionOperation) -> Result<Self, String>;
}

macro_rules! impl_host_integer_reduction {
    ($type:ty) => {
        impl HostReductionElement for $type {
            const SIZE: usize = std::mem::size_of::<Self>();

            fn decode(bytes: &[u8]) -> Result<Vec<Self>, String> {
                if !bytes.len().is_multiple_of(Self::SIZE) {
                    return Err(format!(
                        "host reduction byte length {} is not divisible by {}",
                        bytes.len(),
                        Self::SIZE
                    ));
                }
                Ok(bytes
                    .chunks_exact(Self::SIZE)
                    .map(|chunk| {
                        Self::from_le_bytes(
                            chunk
                                .try_into()
                                .expect("integer reduction chunk has the type width"),
                        )
                    })
                    .collect())
            }

            fn encode(values: &[Self]) -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            }

            fn combine(self, source: Self, operation: ReductionOperation) -> Result<Self, String> {
                Ok(match operation {
                    ReductionOperation::Sum => self.wrapping_add(source),
                    ReductionOperation::Product => self.wrapping_mul(source),
                    ReductionOperation::Minimum => self.min(source),
                    ReductionOperation::Maximum => self.max(source),
                    ReductionOperation::BitAnd => self & source,
                    ReductionOperation::BitOr => self | source,
                    ReductionOperation::BitXor => self ^ source,
                })
            }
        }
    };
}

impl_host_integer_reduction!(u8);
impl_host_integer_reduction!(i8);
impl_host_integer_reduction!(u16);
impl_host_integer_reduction!(i16);
impl_host_integer_reduction!(u32);
impl_host_integer_reduction!(u64);
impl_host_integer_reduction!(i64);

impl HostReductionElement for bool {
    const SIZE: usize = 1;

    fn decode(bytes: &[u8]) -> Result<Vec<Self>, String> {
        bytes
            .iter()
            .map(|value| match value {
                0 => Ok(false),
                1 => Ok(true),
                value => Err(format!("invalid boolean reduction byte {value}")),
            })
            .collect()
    }

    fn encode(values: &[Self]) -> Vec<u8> {
        values.iter().map(|value| u8::from(*value)).collect()
    }

    fn combine(self, source: Self, operation: ReductionOperation) -> Result<Self, String> {
        Ok(match operation {
            ReductionOperation::Sum | ReductionOperation::Maximum | ReductionOperation::BitOr => {
                self | source
            }
            ReductionOperation::Product
            | ReductionOperation::Minimum
            | ReductionOperation::BitAnd => self & source,
            ReductionOperation::BitXor => self ^ source,
        })
    }
}

impl HostReductionElement for f64 {
    const SIZE: usize = 8;

    fn decode(bytes: &[u8]) -> Result<Vec<Self>, String> {
        if !bytes.len().is_multiple_of(Self::SIZE) {
            return Err(format!(
                "host reduction byte length {} is not divisible by {}",
                bytes.len(),
                Self::SIZE
            ));
        }
        Ok(bytes
            .chunks_exact(Self::SIZE)
            .map(|chunk| {
                Self::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("f64 reduction chunk has the type width"),
                )
            })
            .collect())
    }

    fn encode(values: &[Self]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn combine(self, source: Self, operation: ReductionOperation) -> Result<Self, String> {
        match operation {
            ReductionOperation::Sum => Ok(self + source),
            ReductionOperation::Product => Ok(self * source),
            ReductionOperation::Minimum => Ok(if source < self { source } else { self }),
            ReductionOperation::Maximum => Ok(if source > self { source } else { self }),
            ReductionOperation::BitAnd | ReductionOperation::BitOr | ReductionOperation::BitXor => {
                Err("bitwise reductions require an integer or boolean dtype".into())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex32 {
    pub real: f32,
    pub imaginary: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex64 {
    pub real: f64,
    pub imaginary: f64,
}

macro_rules! impl_host_complex_reduction {
    ($type:ty, $component:ty) => {
        impl HostReductionElement for $type {
            const SIZE: usize = 2 * std::mem::size_of::<$component>();

            fn decode(bytes: &[u8]) -> Result<Vec<Self>, String> {
                let component_size = std::mem::size_of::<$component>();
                if !bytes.len().is_multiple_of(Self::SIZE) {
                    return Err(format!(
                        "host complex reduction byte length {} is not divisible by {}",
                        bytes.len(),
                        Self::SIZE
                    ));
                }
                bytes
                    .chunks_exact(Self::SIZE)
                    .map(|chunk| {
                        let real = <$component>::from_le_bytes(
                            chunk[..component_size]
                                .try_into()
                                .expect("complex real component has the scalar width"),
                        );
                        let imaginary = <$component>::from_le_bytes(
                            chunk[component_size..]
                                .try_into()
                                .expect("complex imaginary component has the scalar width"),
                        );
                        Ok(Self { real, imaginary })
                    })
                    .collect()
            }

            fn encode(values: &[Self]) -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|value| {
                        value
                            .real
                            .to_le_bytes()
                            .into_iter()
                            .chain(value.imaginary.to_le_bytes())
                    })
                    .collect()
            }

            fn combine(self, source: Self, operation: ReductionOperation) -> Result<Self, String> {
                match operation {
                    ReductionOperation::Sum => Ok(Self {
                        real: self.real + source.real,
                        imaginary: self.imaginary + source.imaginary,
                    }),
                    ReductionOperation::Product => Ok(Self {
                        real: self.real * source.real - self.imaginary * source.imaginary,
                        imaginary: self.real * source.imaginary + self.imaginary * source.real,
                    }),
                    ReductionOperation::Minimum | ReductionOperation::Maximum => {
                        Err("MIN and MAX reductions are not defined for complex dtypes".into())
                    }
                    ReductionOperation::BitAnd
                    | ReductionOperation::BitOr
                    | ReductionOperation::BitXor => {
                        Err("bitwise reductions require an integer or boolean dtype".into())
                    }
                }
            }
        }
    };
}

impl_host_complex_reduction!(Complex32, f32);
impl_host_complex_reduction!(Complex64, f64);

pub fn validate_host_reduction_operation(
    element_type: ElementType,
    operation: ReductionOperation,
) -> Result<(), String> {
    if matches!(
        element_type,
        ElementType::F64 | ElementType::Complex64 | ElementType::Complex128
    ) && matches!(
        operation,
        ReductionOperation::BitAnd | ReductionOperation::BitOr | ReductionOperation::BitXor
    ) {
        return Err("bitwise reductions require an integer or boolean dtype".into());
    }
    if matches!(
        element_type,
        ElementType::Complex64 | ElementType::Complex128
    ) && matches!(
        operation,
        ReductionOperation::Minimum | ReductionOperation::Maximum
    ) {
        return Err("MIN and MAX reductions are not defined for complex dtypes".into());
    }
    Ok(())
}

pub fn reduce_host_payload(
    element_type: ElementType,
    payload: &[u8],
    elements_per_rank: usize,
    world_size: usize,
    operation: ReductionOperation,
) -> Result<Vec<u8>, String> {
    match element_type {
        ElementType::U8 => {
            reduce_host_values::<u8>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::U32 => {
            reduce_host_values::<u32>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::Bool => {
            reduce_host_values::<bool>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::I8 => {
            reduce_host_values::<i8>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::I16 => {
            reduce_host_values::<i16>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::I64 => {
            reduce_host_values::<i64>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::F64 => {
            reduce_host_values::<f64>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::U16 => {
            reduce_host_values::<u16>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::U64 => {
            reduce_host_values::<u64>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::Complex64 => {
            reduce_host_values::<Complex32>(payload, elements_per_rank, world_size, operation)
        }
        ElementType::Complex128 => {
            reduce_host_values::<Complex64>(payload, elements_per_rank, world_size, operation)
        }
        _ => Err(format!(
            "element type {element_type:?} does not use host reduction"
        )),
    }
}

pub fn reduce_host_values<T: HostReductionElement>(
    payload: &[u8],
    elements_per_rank: usize,
    world_size: usize,
    operation: ReductionOperation,
) -> Result<Vec<u8>, String> {
    let expected_elements = elements_per_rank
        .checked_mul(world_size)
        .ok_or_else(|| "host reduction element count overflow".to_owned())?;
    let expected_bytes = expected_elements
        .checked_mul(T::SIZE)
        .ok_or_else(|| "host reduction byte count overflow".to_owned())?;
    if payload.len() != expected_bytes {
        return Err(format!(
            "host reduction received {} bytes, expected {expected_bytes}",
            payload.len()
        ));
    }
    if elements_per_rank == 0 {
        return Ok(Vec::new());
    }
    let values = T::decode(payload)?;
    let mut output = values[..elements_per_rank].to_vec();
    for source in 1..world_size {
        let start = source * elements_per_rank;
        for (destination, value) in output
            .iter_mut()
            .zip(&values[start..start + elements_per_rank])
        {
            *destination = destination.combine(*value, operation)?;
        }
    }
    Ok(T::encode(&output))
}
