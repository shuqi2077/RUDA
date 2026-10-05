use super::{TensorDevice, TensorDeviceError, TensorElement, storage};
use crate::ReduceOperation;
use crate::rank::{ReductionOperation, communicator::RankCommunicator};
use ruda_tensor::{Backend, DType, Shape, TensorMetadata, bf16, f16};

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// Broadcast an I32 or I64 tensor without converting integer storage to floating point.
    /// Shape, dtype and each rank's local device are retained; transfers are host-staged.
    pub fn broadcast_int(
        &self,
        value: B::IntTensorPrimitive,
        root: u32,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::I32 => self.integer_collective::<i32>(value, None, root),
            DType::I64 => self.integer_collective::<i64>(value, None, root),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    /// Reduce an I32 or I64 tensor using the selected backend's integer arithmetic.
    /// Supports sum, product, minimum, maximum and integer bitwise operations.
    /// All ranks must use matching shapes, dtypes and operations in the same order.
    pub fn all_reduce_int(
        &self,
        value: B::IntTensorPrimitive,
        operation: ReductionOperation,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::I32 => self.integer_collective::<i32>(value, Some(operation), 0),
            DType::I64 => self.integer_collective::<i64>(value, Some(operation), 0),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    fn integer_collective<T: TensorElement>(
        &self,
        value: B::IntTensorPrimitive,
        operation: Option<ReductionOperation>,
        root: u32,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        let execution = self.execution();
        execution.validate_type::<T>()?;
        if &B::int_device(&value) != execution.device() {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        let shape = value.shape();
        let length = shape.iter().try_fold(1_usize, |length, dim| {
            length
                .checked_mul(*dim)
                .ok_or(TensorDeviceError::InvalidBuffer(
                    "collective tensor element count overflow",
                ))
        })?;
        storage::checked_length::<T>(length)?;
        let buffer = execution.import_int::<T>(B::int_reshape(value, Shape::new([length])))?;
        if let Some(operation) = operation {
            let kernel = execution.reduction_kernel::<T>(operation)?;
            self.tensor_collective::<T>()
                .all_reduce(&buffer, operation, &kernel)?;
        } else {
            self.tensor_collective::<T>().broadcast(&buffer, root)?;
        }
        Ok(B::int_reshape(buffer.int_tensor()?, shape))
    }

    /// Broadcast a floating tensor, retaining its shape, dtype and local device.
    pub fn broadcast_float(
        &self,
        value: B::FloatTensorPrimitive,
        root: u32,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::F32 => self.broadcast_typed::<f32>(value, root),
            DType::F16 => self.broadcast_typed::<f16>(value, root),
            DType::BF16 => self.broadcast_typed::<bf16>(value, root),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    fn broadcast_typed<T: TensorElement>(
        &self,
        value: B::FloatTensorPrimitive,
        root: u32,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        let execution = self.execution();
        execution.validate_type::<T>()?;
        if &B::float_device(&value) != execution.device() {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        let shape = value.shape();
        let length = shape.iter().try_fold(1_usize, |length, dim| {
            length.checked_mul(*dim).ok_or(TensorDeviceError::InvalidBuffer(
                "collective tensor element count overflow",
            ))
        })?;
        storage::checked_length::<T>(length)?;
        let buffer = execution.import_float::<T>(B::float_reshape(value, Shape::new([length])))?;
        self.tensor_collective::<T>().broadcast(&buffer, root)?;
        Ok(B::float_reshape(buffer.float_tensor()?, shape))
    }

    /// Reduce a floating tensor through this rank's configured transport.
    ///
    /// All ranks must submit the same tensor shapes, dtypes and operations in
    /// the same order. The backend reshapes non-contiguous inputs according to
    /// its normal tensor semantics; output shape, dtype and device are retained.
    /// Transfers are host-staged, while arithmetic uses the tensor backend.
    pub fn all_reduce_float(
        &self,
        value: B::FloatTensorPrimitive,
        operation: ReduceOperation,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::F32 => self.reduce_float::<f32>(value, operation),
            DType::F16 => self.reduce_float::<f16>(value, operation),
            DType::BF16 => self.reduce_float::<bf16>(value, operation),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    fn reduce_float<T: TensorElement>(
        &self,
        value: B::FloatTensorPrimitive,
        operation: ReduceOperation,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        let execution = self.execution();
        execution.validate_type::<T>()?;
        if &B::float_device(&value) != execution.device() {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        let shape = value.shape();
        let length = shape.iter().try_fold(1_usize, |length, dim| {
            length.checked_mul(*dim).ok_or(TensorDeviceError::InvalidBuffer(
                "collective tensor element count overflow",
            ))
        })?;
        storage::checked_length::<T>(length)?;
        let value = B::float_reshape(value, Shape::new([length]));
        let buffer = execution.import_float::<T>(value)?;
        self.tensor_collective::<T>().all_reduce(
            &buffer,
            ReductionOperation::Sum,
            &ReductionOperation::Sum,
        )?;
        let mut value = buffer.float_tensor()?;
        if operation == ReduceOperation::Mean {
            value = B::float_div_scalar(value, (self.world_size() as f32).into());
        }
        Ok(B::float_reshape(value, shape))
    }
}
