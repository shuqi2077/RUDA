use super::{Primitive, TensorDevice, TensorDeviceError};
use super::bounded_reduction::reduction_progress;
use crate::{ReduceOperation, rank::{ElementType, ReductionOperation, communicator::RankCommunicator,
    device_collective::NativeChunkProgress}};
use ruda_core::tensor::collective::CollectiveShape;
use ruda_tensor::{Backend, DType, Shape, TensorMetadata};

#[derive(Clone, Copy)]
enum ScatterOperation { Float(ReduceOperation), Integer(ReductionOperation) }

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// Gather native F32/F16/BF16 tensors in rank/axis-zero order with bounded messages.
    /// The byte limit covers the rank-concatenated reply, not metadata/backend scratch.
    /// An actual complete gathered output is allocated, not a floating stand-in for packed storage.
    pub fn all_gather_float_chunked(&self, value: B::FloatTensorPrimitive, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.all_gather_float_chunked_with_progress(value, max_chunk_bytes, |_| {})
    }

    /// Bounded gather reporting completed source coordinates per rank after all destination writes finish.
    pub fn all_gather_float_chunked_with_progress<F: FnMut(NativeChunkProgress)>(
        &self, value: B::FloatTensorPrimitive, max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        match self.native_gather_chunked(Primitive::Float(value), max_chunk_bytes, &mut progress)? {
            Primitive::Float(value) => Ok(value), Primitive::Int(_) => unreachable!("original floating gather kind"),
        }
    }

    /// Gather native U8/U32/I32/I64 tensors without floating conversion or word/byte widening.
    /// Rank order and original shape axes are restored even when chunks cross row boundaries.
    pub fn all_gather_int_chunked(&self, value: B::IntTensorPrimitive, max_chunk_bytes: usize)
        -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.all_gather_int_chunked_with_progress(value, max_chunk_bytes, |_| {})
    }

    /// Bounded integer gather with actual completed native source-coordinate counts.
    pub fn all_gather_int_chunked_with_progress<F: FnMut(NativeChunkProgress)>(
        &self, value: B::IntTensorPrimitive, max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        match self.native_gather_chunked(Primitive::Int(value), max_chunk_bytes, &mut progress)? {
            Primitive::Int(value) => Ok(value), Primitive::Float(_) => unreachable!("original integer gather kind"),
        }
    }

    /// Reduce original floating values and return the same equal axis-zero rank shard as reduce_scatter_float.
    /// Every chunk contains each destination rank's own original coordinate interval,
    /// not contiguous chunks subsequently assigned to different owners. Native dtype,
    /// SUM/mean choice and rank ownership are retained; algorithms are selected per chunk.
    pub fn reduce_scatter_float_chunked(
        &self, value: B::FloatTensorPrimitive, operation: ReduceOperation, max_chunk_bytes: usize,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.reduce_scatter_float_chunked_with_progress(value, operation, max_chunk_bytes, |_| {})
    }

    /// Bounded floating reduce-scatter with completed coordinates of this rank's actual output shard.
    pub fn reduce_scatter_float_chunked_with_progress<F: FnMut(NativeChunkProgress)>(
        &self, value: B::FloatTensorPrimitive, operation: ReduceOperation,
        max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        match self.native_scatter_chunked(Primitive::Float(value), ScatterOperation::Float(operation), max_chunk_bytes, &mut progress)? {
            Primitive::Float(value) => Ok(value), Primitive::Int(_) => unreachable!("original floating scatter kind"),
        }
    }

    /// Reduce-scatter native I32/I64 values retaining exact integer width and the selected reduction.
    /// Packed U8/U32 buffers are gathered/broadcast, not treated as arithmetic gradients.
    pub fn reduce_scatter_int_chunked(
        &self, value: B::IntTensorPrimitive, operation: ReductionOperation, max_chunk_bytes: usize,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.reduce_scatter_int_chunked_with_progress(value, operation, max_chunk_bytes, |_| {})
    }

    /// Bounded integer reduce-scatter reporting actually completed local-shard coordinates.
    pub fn reduce_scatter_int_chunked_with_progress<F: FnMut(NativeChunkProgress)>(
        &self, value: B::IntTensorPrimitive, operation: ReductionOperation,
        max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        match self.native_scatter_chunked(Primitive::Int(value), ScatterOperation::Integer(operation), max_chunk_bytes, &mut progress)? {
            Primitive::Int(value) => Ok(value), Primitive::Float(_) => unreachable!("original integer scatter kind"),
        }
    }

    fn native_gather_chunked<F: FnMut(NativeChunkProgress)>(
        &self, value: Primitive<B>, max_chunk_bytes: usize, progress: &mut F,
    ) -> Result<Primitive<B>, TensorDeviceError> {
        let (shape, dtype) = self.native_sharded_metadata(&value)?;
        let element_type = native_element_type(dtype)?;
        self.native_chunk_agreement(&shape, element_type, 0x100, max_chunk_bytes)?;
        let layout = CollectiveShape::all_gather(shape, self.world_size() as usize)
            .map_err(|error| TensorDeviceError::Data(format!("{error:?}")))?;
        let plan = self.native_message_plan(layout.input_elements, element_type, max_chunk_bytes)?;
        let value = native_reshape::<B>(value, Shape::new([layout.input_elements]));
        let mut output = self.native_empty_like(&value, layout.output_elements, dtype);
        let mut state = reduction_progress(plan)?;
        progress(state);
        while state.completed_elements < plan.elements() {
            let length = plan.chunk_elements().min(plan.elements() - state.completed_elements);
            let gathered = match value.clone().slice(state.completed_elements..state.completed_elements + length) {
                Primitive::Float(value) => Primitive::Float(self.all_gather_float(value)?),
                Primitive::Int(value) => Primitive::Int(self.all_gather_int(value)?),
            };
            for owner in 0..self.world_size() as usize {
                let start = owner * layout.input_elements + state.completed_elements;
                let piece = gathered.clone().slice(owner * length..(owner + 1) * length);
                output = output.assign(start..start + length, piece);
            }
            B::sync(self.execution().device())?;
            state.completed_elements += length;
            state.completed_chunks += 1;
            progress(state);
        }
        Ok(native_reshape::<B>(output, layout.output))
    }

    fn native_scatter_chunked<F: FnMut(NativeChunkProgress)>(
        &self, value: Primitive<B>, operation: ScatterOperation, max_chunk_bytes: usize, progress: &mut F,
    ) -> Result<Primitive<B>, TensorDeviceError> {
        let (shape, dtype) = self.native_sharded_metadata(&value)?;
        let element_type = native_element_type(dtype)?;
        let tag = match operation {
            ScatterOperation::Float(operation) => 0x200 + operation as u64,
            ScatterOperation::Integer(operation) => 0x200 + operation as u64,
        };
        self.native_chunk_agreement(&shape, element_type, tag, max_chunk_bytes)?;
        let layout = CollectiveShape::reduce_scatter(shape, self.world_size() as usize)
            .map_err(|error| TensorDeviceError::Data(format!("{error:?}")))?;
        if matches!(dtype, DType::U8 | DType::U32) { return Err(TensorDeviceError::UnsupportedDType(dtype)); }
        let plan = self.native_message_plan(layout.output_elements, element_type, max_chunk_bytes)?;
        let value = native_reshape::<B>(value, Shape::new([layout.input_elements]));
        let mut output = self.native_empty_like(&value, layout.output_elements, dtype);
        let mut state = reduction_progress(plan)?;
        progress(state);
        while state.completed_elements < plan.elements() {
            let length = plan.chunk_elements().min(plan.elements() - state.completed_elements);
            let chunks = (0..self.world_size() as usize).map(|owner| {
                let start = owner * layout.output_elements + state.completed_elements;
                value.clone().slice(start..start + length)
            }).collect::<Vec<_>>();
            let reduced = match operation {
                ScatterOperation::Float(operation) => {
                    let chunks = chunks.into_iter().map(|piece| match piece {
                        Primitive::Float(value) => value, Primitive::Int(_) => unreachable!("original floating scatter input"),
                    }).collect();
                    Primitive::Float(self.reduce_scatter_float(B::float_cat(chunks, 0), operation)?)
                }
                ScatterOperation::Integer(operation) => {
                    let chunks = chunks.into_iter().map(|piece| match piece {
                        Primitive::Int(value) => value, Primitive::Float(_) => unreachable!("original integer scatter input"),
                    }).collect();
                    Primitive::Int(self.reduce_scatter_int(B::int_cat(chunks, 0), operation)?)
                }
            };
            output = output.assign(state.completed_elements..state.completed_elements + length, reduced);
            B::sync(self.execution().device())?;
            state.completed_elements += length;
            state.completed_chunks += 1;
            progress(state);
        }
        Ok(native_reshape::<B>(output, layout.output))
    }

    pub(super) fn native_sharded_metadata(&self, value: &Primitive<B>) -> Result<(Shape, DType), TensorDeviceError> {
        let (shape, dtype, device) = match value {
            Primitive::Float(value) => (value.shape(), value.dtype(), B::float_device(value)),
            Primitive::Int(value) => (value.shape(), value.dtype(), B::int_device(value)),
        };
        if device != *self.execution().device() { return Err(TensorDeviceError::DeviceMismatch); }
        Ok((shape, dtype))
    }

    pub(super) fn native_empty_like(&self, source: &Primitive<B>, elements: usize, dtype: DType) -> Primitive<B> {
        let shape = Shape::new([elements]);
        match source {
            Primitive::Float(_) => Primitive::Float(B::float_empty(shape, self.execution().device(), dtype.into())),
            Primitive::Int(_) => Primitive::Int(B::int_empty(shape, self.execution().device(), dtype.into())),
        }
    }
}

pub(super) fn native_reshape<B: Backend>(value: Primitive<B>, shape: Shape) -> Primitive<B> {
    match value { Primitive::Float(value) => Primitive::Float(B::float_reshape(value, shape)),
        Primitive::Int(value) => Primitive::Int(B::int_reshape(value, shape)) }
}

pub(super) fn native_element_type(dtype: DType) -> Result<ElementType, TensorDeviceError> {
    Ok(match dtype {
        DType::F32 => ElementType::F32, DType::F16 => ElementType::F16, DType::BF16 => ElementType::BF16,
        DType::U8 => ElementType::U8, DType::U32 => ElementType::U32, DType::I32 => ElementType::I32, DType::I64 => ElementType::I64,
        dtype => return Err(TensorDeviceError::UnsupportedDType(dtype)),
    })
}
