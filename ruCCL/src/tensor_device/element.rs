use crate::rank::ElementType;
use ruda_tensor::{Element, bf16, f16};

mod sealed {
    pub trait Wire: Sized {
        fn append_le(self, bytes: &mut Vec<u8>);
        fn read_le(bytes: &[u8]) -> Self;
    }
}

/// Storage types supported by the shared collective device adapter.
///
/// The sealed implementations preserve F32/F16/BF16/I32 storage bits on the
/// little-endian rank transport; they do not convert half values through F32.
pub trait TensorElement: Element + sealed::Wire {
    const ELEMENT_TYPE: ElementType;
}

macro_rules! native_element {
    ($ty:ty, $kind:ident) => {
        impl TensorElement for $ty {
            const ELEMENT_TYPE: ElementType = ElementType::$kind;
        }
        impl sealed::Wire for $ty {
            fn append_le(self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_le_bytes());
            }
            fn read_le(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("complete wire element"))
            }
        }
    };
}

macro_rules! half_element {
    ($ty:ty, $kind:ident) => {
        impl TensorElement for $ty {
            const ELEMENT_TYPE: ElementType = ElementType::$kind;
        }
        impl sealed::Wire for $ty {
            fn append_le(self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_bits().to_le_bytes());
            }
            fn read_le(bytes: &[u8]) -> Self {
                Self::from_bits(u16::from_le_bytes(bytes.try_into().expect("complete wire element")))
            }
        }
    };
}

native_element!(f32, F32);
native_element!(i32, I32);
half_element!(f16, F16);
half_element!(bf16, BF16);

pub(super) fn encode<T: TensorElement>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        value.append_le(&mut bytes);
    }
    bytes
}

pub(super) fn decode<T: TensorElement>(bytes: &[u8]) -> Result<Vec<T>, super::TensorDeviceError> {
    let width = std::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(width) {
        return Err(super::TensorDeviceError::InvalidBuffer("incomplete wire element"));
    }
    Ok(bytes.chunks_exact(width).map(T::read_le).collect())
}
