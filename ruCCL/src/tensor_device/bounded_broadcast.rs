use super::{Primitive, TensorBuffer, TensorDevice, TensorDeviceError, TensorElement, storage};
use crate::rank::{communicator::RankCommunicator, device_collective::{ChunkedBroadcastPlan, BroadcastProgress}};
use ruda_tensor::{Backend, DType, Shape, TensorMetadata, bf16, f16};

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// Broadcast original F32/F16/BF16 storage using an explicit maximum staged message size.
    /// The original local shape/device and every element's bits are retained.
    /// All ranks must enter with the same source shape/dtype, root and byte budget.
    pub fn broadcast_float_chunked(
        &self, value: B::FloatTensorPrimitive, root: u32, max_chunk_bytes: usize,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.broadcast_float_chunked_with_progress(value, root, max_chunk_bytes, |_| {})
    }

    /// Bounded floating broadcast with actual completed device-write work units.
    /// The callback does not perform checkpoint I/O; the returned tensor is complete.
    pub fn broadcast_float_chunked_with_progress<F: FnMut(BroadcastProgress)>(
        &self, value: B::FloatTensorPrimitive, root: u32, max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        let value = match value.dtype() {
            DType::F32 => self.broadcast_native_chunked::<f32, _>(Primitive::Float(value), root, max_chunk_bytes, &mut progress),
            DType::F16 => self.broadcast_native_chunked::<f16, _>(Primitive::Float(value), root, max_chunk_bytes, &mut progress),
            DType::BF16 => self.broadcast_native_chunked::<bf16, _>(Primitive::Float(value), root, max_chunk_bytes, &mut progress),
            dtype => return Err(TensorDeviceError::UnsupportedDType(dtype)),
        }?;
        match value { Primitive::Float(value) => Ok(value), Primitive::Int(_) => unreachable!("original floating buffer kind") }
    }

    /// Broadcast native U8/U32/I32/I64 storage with bounded host-staged messages.
    /// Suitable for original NF4 byte streams and AWQ packed words, without
    /// widening bytes/words to floating point, dequantizing or reblocking them.
    pub fn broadcast_int_chunked(
        &self, value: B::IntTensorPrimitive, root: u32, max_chunk_bytes: usize,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.broadcast_int_chunked_with_progress(value, root, max_chunk_bytes, |_| {})
    }

    /// Bounded integer broadcast reporting actual completed native element chunks.
    /// The budget is in bytes, rounded down only to retain whole original elements.
    pub fn broadcast_int_chunked_with_progress<F: FnMut(BroadcastProgress)>(
        &self, value: B::IntTensorPrimitive, root: u32, max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        let value = match value.dtype() {
            DType::U8 => self.broadcast_native_chunked::<u8, _>(Primitive::Int(value), root, max_chunk_bytes, &mut progress),
            DType::U32 => self.broadcast_native_chunked::<u32, _>(Primitive::Int(value), root, max_chunk_bytes, &mut progress),
            DType::I32 => self.broadcast_native_chunked::<i32, _>(Primitive::Int(value), root, max_chunk_bytes, &mut progress),
            DType::I64 => self.broadcast_native_chunked::<i64, _>(Primitive::Int(value), root, max_chunk_bytes, &mut progress),
            dtype => return Err(TensorDeviceError::UnsupportedDType(dtype)),
        }?;
        match value { Primitive::Int(value) => Ok(value), Primitive::Float(_) => unreachable!("original integer buffer kind") }
    }

    fn broadcast_native_chunked<T: TensorElement, F: FnMut(BroadcastProgress)>(
        &self, value: Primitive<B>, root: u32, max_chunk_bytes: usize, progress: &mut F,
    ) -> Result<Primitive<B>, TensorDeviceError> {
        let shape = match &value { Primitive::Float(value) => value.shape(), Primitive::Int(value) => value.shape() };
        let elements = shape.iter().try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(TensorDeviceError::InvalidBuffer("native broadcast shape element count overflow"))?;
        storage::checked_length::<T>(elements)?;
        let floating = matches!(&value, Primitive::Float(_));
        let buffer: TensorBuffer<B, T> = match value {
            Primitive::Float(value) => self.execution().import_float::<T>(B::float_reshape(value, Shape::new([elements])))?,
            Primitive::Int(value) => self.execution().import_int::<T>(B::int_reshape(value, Shape::new([elements])))?,
        };
        let plan = ChunkedBroadcastPlan::new(elements, core::mem::size_of::<T>(), max_chunk_bytes)?;
        self.tensor_collective::<T>().broadcast_chunked_with_progress(&buffer, root, plan, 0, progress)?;
        if floating {
            Ok(Primitive::Float(B::float_reshape(buffer.float_tensor()?, shape)))
        } else {
            Ok(Primitive::Int(B::int_reshape(buffer.int_tensor()?, shape)))
        }
    }
}
