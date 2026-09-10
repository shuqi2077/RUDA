use super::{DataError, TensorData};
use crate::bytes::Bytes;
use crate::tensor::{BoolStore, DType};
use crate::tensor::element::Element;
use alloc::format;
use alloc::vec::Vec;

impl TensorData {
    /// Returns the immutable slice view of the tensor data.
    pub fn as_slice<E: Element>(&self) -> Result<&[E], DataError> {
        if self.matches_target_dtype::<E>() {
            bytemuck::checked::try_cast_slice(&self.bytes).map_err(DataError::CastError)
        } else {
            Err(DataError::TypeMismatch(format!(
                "Invalid target element type (expected {:?}, got {:?})",
                self.dtype,
                E::dtype()
            )))
        }
    }

    /// Returns the mutable slice view of the tensor data.
    ///
    /// # Panics
    /// If the target element type is different from the stored element type.
    pub fn as_mut_slice<E: Element>(&mut self) -> Result<&mut [E], DataError> {
        if self.matches_target_dtype::<E>() {
            bytemuck::checked::try_cast_slice_mut(&mut self.bytes).map_err(DataError::CastError)
        } else {
            Err(DataError::TypeMismatch(format!(
                "Invalid target element type (expected {:?}, got {:?})",
                self.dtype,
                E::dtype()
            )))
        }
    }

    /// Returns the tensor data as a vector of scalar values.
    pub fn to_vec<E: Element>(&self) -> Result<Vec<E>, DataError> {
        Ok(self.as_slice()?.to_vec())
    }

    /// Returns the tensor data as a vector of scalar values.
    pub fn into_vec<E: Element>(self) -> Result<Vec<E>, DataError> {
        // This means we cannot call `into_vec` for QFloat
        if !self.matches_target_dtype::<E>() {
            return Err(DataError::TypeMismatch(format!(
                "Invalid target element type (expected {:?}, got {:?})",
                self.dtype,
                E::dtype()
            )));
        }

        self.into_vec_unchecked()
    }

    /// Returns the tensor data as a vector of scalar values. Does not check dtype.
    fn into_vec_unchecked<E: Element>(self) -> Result<Vec<E>, DataError> {
        let mut me = self;
        me.bytes = match me.bytes.try_into_vec::<E>() {
            Ok(elems) => return Ok(elems),
            Err(bytes) => bytes,
        };

        // The bytes might have been deserialized and allocated with a different align.
        // In that case, we have to memcopy the data into a new vector, more suitably allocated
        Ok(bytemuck::checked::try_cast_slice(me.as_bytes())
            .map_err(DataError::CastError)?
            .to_vec())
    }

    fn matches_target_dtype<E: Element>(&self) -> bool {
        let target_dtype = E::dtype();
        match self.dtype {
            DType::Bool(BoolStore::U8) => {
                matches!(target_dtype, DType::U8 | DType::Bool(BoolStore::U8))
            }
            DType::Bool(BoolStore::U32) => {
                matches!(target_dtype, DType::U32 | DType::Bool(BoolStore::U32))
            }
            dtype => dtype == target_dtype,
        }
    }

    /// Returns the data as a slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the bytes representation of the data.
    pub fn into_bytes(self) -> Bytes {
        self.bytes
    }
}
