use super::{TensorDevice, TensorDeviceError, TensorElement, storage};
use crate::ReduceOperation;
use crate::rank::{ReductionOperation, communicator::RankCommunicator};
use ruda_tensor::{Backend, DType, Shape, TensorMetadata, bf16, f16};

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
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
