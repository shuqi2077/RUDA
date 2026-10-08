use super::{TensorDevice, TensorDeviceError};
use crate::{ReduceOperation, rank::{ElementType, ReductionOperation, communicator::RankCommunicator,
    device_collective::{NativeChunkPlan, NativeChunkProgress}}};
use ruda_tensor::{Backend, DType, Shape, Slice, TensorMetadata};

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// All-reduce actual F32/F16/BF16 coordinates in bounded native payload chunks.
    ///
    /// Shape/dtype/device and the caller's SUM/mean choice are retained. Backend
    /// arithmetic and configured collective selection are reused on each actual
    /// chunk, so floating reduction trees need not bit-match unchunked reductions.
    /// This does not promote precision, scale losses, cast gradients or select an
    /// optimizer. The byte budget bounds tensor messages (including rank-concatenated
    /// replies), not metadata or transport/backend scratch. Nonempty inputs require
    /// enough bytes for one native element from each participant.
    pub fn all_reduce_float_chunked(
        &self, value: B::FloatTensorPrimitive, operation: ReduceOperation, max_chunk_bytes: usize,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.all_reduce_float_chunked_with_progress(value, operation, max_chunk_bytes, |_| {})
    }

    /// Bounded floating all-reduce reporting actual completed coordinate writes.
    /// No tensor value is downloaded until the ranks' complete geometry/type/operation/budget agrees.
    pub fn all_reduce_float_chunked_with_progress<F: FnMut(NativeChunkProgress)>(
        &self, value: B::FloatTensorPrimitive, operation: ReduceOperation,
        max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        let element_type = match value.dtype() {
            DType::F32 => ElementType::F32,
            DType::F16 => ElementType::F16,
            DType::BF16 => ElementType::BF16,
            dtype => return Err(TensorDeviceError::UnsupportedDType(dtype)),
        };
        if B::float_device(&value) != *self.execution().device() { return Err(TensorDeviceError::DeviceMismatch); }
        let shape = value.shape();
        let plan = self.reduction_chunk_agreement(&shape, element_type, operation as u64, max_chunk_bytes)?;
        let mut state = reduction_progress(plan)?;
        progress(state);
        let mut value = B::float_reshape(value, Shape::new([plan.elements()]));
        while state.completed_elements < plan.elements() {
            let end = state.completed_elements + plan.chunk_elements().min(plan.elements() - state.completed_elements);
            let range = Slice::from(state.completed_elements..end);
            let chunk = B::float_slice(value.clone(), core::slice::from_ref(&range));
            let reduced = self.all_reduce_float(chunk, operation)?;
            value = B::float_slice_assign(value, core::slice::from_ref(&range), reduced);
            B::sync(self.execution().device())?;
            state.completed_elements = end;
            state.completed_chunks += 1;
            progress(state);
        }
        Ok(B::float_reshape(value, shape))
    }

    /// All-reduce actual I32/I64 coordinates in bounded native payload chunks.
    /// Preserves integer width and the existing sum/product/min/max/bitwise semantics;
    /// no floating conversion is used for wide counters, IDs or source values.
    pub fn all_reduce_int_chunked(
        &self, value: B::IntTensorPrimitive, operation: ReductionOperation, max_chunk_bytes: usize,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.all_reduce_int_chunked_with_progress(value, operation, max_chunk_bytes, |_| {})
    }

    /// Bounded integer all-reduce with actual completed native coordinate counts.
    pub fn all_reduce_int_chunked_with_progress<F: FnMut(NativeChunkProgress)>(
        &self, value: B::IntTensorPrimitive, operation: ReductionOperation,
        max_chunk_bytes: usize, mut progress: F,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        let element_type = match value.dtype() {
            DType::I32 => ElementType::I32,
            DType::I64 => ElementType::I64,
            dtype => return Err(TensorDeviceError::UnsupportedDType(dtype)),
        };
        if B::int_device(&value) != *self.execution().device() { return Err(TensorDeviceError::DeviceMismatch); }
        let shape = value.shape();
        let plan = self.reduction_chunk_agreement(&shape, element_type, operation as u64, max_chunk_bytes)?;
        let mut state = reduction_progress(plan)?;
        progress(state);
        let mut value = B::int_reshape(value, Shape::new([plan.elements()]));
        while state.completed_elements < plan.elements() {
            let end = state.completed_elements + plan.chunk_elements().min(plan.elements() - state.completed_elements);
            let range = Slice::from(state.completed_elements..end);
            let chunk = B::int_slice(value.clone(), core::slice::from_ref(&range));
            let reduced = self.all_reduce_int(chunk, operation)?;
            value = B::int_slice_assign(value, core::slice::from_ref(&range), reduced);
            B::sync(self.execution().device())?;
            state.completed_elements = end;
            state.completed_chunks += 1;
            progress(state);
        }
        Ok(B::int_reshape(value, shape))
    }

    fn reduction_chunk_agreement(
        &self, shape: &Shape, element_type: ElementType, operation: u64, max_chunk_bytes: usize,
    ) -> Result<NativeChunkPlan, TensorDeviceError> {
        self.native_chunk_agreement(shape, element_type, operation, max_chunk_bytes)?;
        let elements = if shape.contains(&0) { 0 } else { shape.iter().try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(TensorDeviceError::InvalidBuffer("chunked reduction shape element count overflow"))?
        };
        self.native_message_plan(elements, element_type, max_chunk_bytes)
    }

    pub(super) fn native_chunk_agreement(
        &self, shape: &Shape, element_type: ElementType, operation: u64, max_chunk_bytes: usize,
    ) -> Result<(), TensorDeviceError> {
        let mut fields = vec![element_type as u64, operation, max_chunk_bytes as u64, self.world_size() as u64, shape.num_dims() as u64];
        fields.extend(shape.iter().map(|dimension| *dimension as u64));
        let payload = fields.iter().flat_map(|field| field.to_le_bytes()).collect::<Vec<_>>();
        let (requests, _) = self.host().all_gather_host_staged(ElementType::U64, fields.len(), payload.clone())?;
        if requests.chunks_exact(payload.len()).any(|other| other != payload.as_slice()) {
            return Err(TensorDeviceError::InvalidOperation("chunked reduction geometry/type/operation/budget differs across ranks"));
        }
        Ok(())
    }

    pub(super) fn native_message_plan(
        &self, elements: usize, element_type: ElementType, max_chunk_bytes: usize,
    ) -> Result<NativeChunkPlan, TensorDeviceError> {
        let world = self.world_size() as usize;
        if world == 0 { return Err(TensorDeviceError::InvalidOperation("native chunk communicator world must be positive")); }
        let per_rank_bytes = if elements == 0 { element_type.byte_width() } else { max_chunk_bytes / world };
        Ok(NativeChunkPlan::new(elements, element_type.byte_width(), per_rank_bytes)?)
    }
}

pub(super) fn reduction_progress(plan: NativeChunkPlan) -> Result<NativeChunkProgress, TensorDeviceError> {
    Ok(NativeChunkProgress { completed_elements: 0, total_elements: plan.elements(),
        completed_chunks: 0, total_chunks: plan.chunk_count()?, restored_elements: 0 })
}
