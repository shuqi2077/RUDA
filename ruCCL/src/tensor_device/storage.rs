use super::{Primitive, TensorBuffer, TensorDevice, TensorDeviceError, TensorElement};
use ruda_tensor::{Backend, DType, FloatDType, IntDType, Slice, TensorData, read_sync};
use std::ops::Range;

pub(super) fn checked_length<T: TensorElement>(length: usize) -> Result<(), TensorDeviceError> {
    let bytes = length.checked_mul(std::mem::size_of::<T>())
        .ok_or(TensorDeviceError::InvalidBuffer("collective buffer byte length overflow"))?;
    if bytes > isize::MAX as usize {
        return Err(TensorDeviceError::InvalidBuffer("collective buffer exceeds addressable size"));
    }
    Ok(())
}

pub(super) fn checked_range(
    total: usize,
    offset: usize,
    length: usize,
) -> Result<Range<usize>, TensorDeviceError> {
    let end = offset.checked_add(length)
        .ok_or(TensorDeviceError::InvalidBuffer("collective element range overflow"))?;
    if end > total {
        return Err(TensorDeviceError::InvalidBuffer("collective element range out of bounds"));
    }
    Ok(offset..end)
}

impl<B: Backend> Primitive<B> {
    pub(super) fn slice(self, range: Range<usize>) -> Self {
        let slice = Slice::from(range);
        match self {
            Self::Float(value) => Self::Float(B::float_slice(value, &[slice])),
            Self::Int(value) => Self::Int(B::int_slice(value, &[slice])),
        }
    }

    pub(super) fn assign(self, range: Range<usize>, value: Self) -> Self {
        let slice = Slice::from(range);
        match (self, value) {
            (Self::Float(tensor), Self::Float(value)) => {
                Self::Float(B::float_slice_assign(tensor, &[slice], value))
            }
            (Self::Int(tensor), Self::Int(value)) => {
                Self::Int(B::int_slice_assign(tensor, &[slice], value))
            }
            _ => unreachable!("typed collective buffer storage kind"),
        }
    }
}

impl<B: Backend> TensorDevice<B> {
    pub fn allocate<T: TensorElement>(
        &self,
        length: usize,
    ) -> Result<TensorBuffer<B, T>, TensorDeviceError> {
        self.validate_type::<T>()?;
        checked_length::<T>(length)?;
        let shape = [length].into();
        let value = match T::dtype() {
            DType::F32 => Primitive::Float(B::float_empty(shape, &self.device, FloatDType::F32)),
            DType::F16 => Primitive::Float(B::float_empty(shape, &self.device, FloatDType::F16)),
            DType::BF16 => Primitive::Float(B::float_empty(shape, &self.device, FloatDType::BF16)),
            DType::I32 => Primitive::Int(B::int_empty(shape, &self.device, IntDType::I32)),
            _ => unreachable!("sealed collective element type"),
        };
        Ok(self.wrap(value, length))
    }

    pub(super) fn from_values<T: TensorElement>(&self, values: &[T]) -> Primitive<B> {
        let data = TensorData::new(values.to_vec(), [values.len()]);
        if T::dtype() == DType::I32 {
            Primitive::Int(B::int_from_data(data, &self.device))
        } else {
            Primitive::Float(B::float_from_data(data, &self.device))
        }
    }

    pub fn write<T: TensorElement>(
        &self,
        buffer: &TensorBuffer<B, T>,
        values: &[T],
    ) -> Result<(), TensorDeviceError> {
        if values.len() != buffer.length {
            return Err(TensorDeviceError::InvalidBuffer("collective full write length mismatch"));
        }
        self.write_at(buffer, 0, values)
    }

    pub fn write_at<T: TensorElement>(
        &self,
        buffer: &TensorBuffer<B, T>,
        offset: usize,
        values: &[T],
    ) -> Result<(), TensorDeviceError> {
        self.validate_buffer(buffer)?;
        let range = checked_range(buffer.length, offset, values.len())?;
        if range.is_empty() {
            return Ok(());
        }
        let incoming = self.from_values(values);
        buffer.update(|current| {
            if range.start == 0 && range.end == buffer.length {
                incoming
            } else {
                current.assign(range, incoming)
            }
        })
    }

    pub fn download<T: TensorElement>(
        &self,
        buffer: &TensorBuffer<B, T>,
    ) -> Result<Vec<T>, TensorDeviceError> {
        self.download_at(buffer, 0, buffer.length)
    }

    pub fn download_at<T: TensorElement>(
        &self,
        buffer: &TensorBuffer<B, T>,
        offset: usize,
        length: usize,
    ) -> Result<Vec<T>, TensorDeviceError> {
        self.validate_buffer(buffer)?;
        let range = checked_range(buffer.length, offset, length)?;
        if range.is_empty() {
            return Ok(Vec::new());
        }
        let mut value = buffer.snapshot()?;
        if range.start != 0 || range.end != buffer.length {
            value = value.slice(range);
        }
        let data = match value {
            Primitive::Float(value) => read_sync(B::float_into_data(value))?,
            Primitive::Int(value) => read_sync(B::int_into_data(value))?,
        };
        data.into_vec::<T>().map_err(|error| TensorDeviceError::Data(format!("{error:?}")))
    }
}
