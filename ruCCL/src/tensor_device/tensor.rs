use super::{TensorDevice, TensorDeviceError, TensorElement, storage};
use crate::ReduceOperation;
use crate::rank::{ReductionOperation, communicator::RankCommunicator};
use ruda_core::tensor::collective::CollectiveShape;
use ruda_tensor::{Backend, DType, Shape, TensorMetadata, bf16, f16};

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// Gather equal-size floating tensors along axis zero in rank order.
    /// Retains dtype, remaining axes and input snapshots; transfers are host-staged.
    pub fn all_gather_float(
        &self,
        value: B::FloatTensorPrimitive,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::F32 => self.sharded_float::<f32>(value, None),
            DType::F16 => self.sharded_float::<f16>(value, None),
            DType::BF16 => self.sharded_float::<bf16>(value, None),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    /// Reduce floating tensors and return this rank's equal axis-zero shard.
    /// The leading axis must divide by world size. All ranks use matching contracts.
    pub fn reduce_scatter_float(
        &self,
        value: B::FloatTensorPrimitive,
        operation: ReduceOperation,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::F32 => self.sharded_float::<f32>(value, Some(operation)),
            DType::F16 => self.sharded_float::<f16>(value, Some(operation)),
            DType::BF16 => self.sharded_float::<bf16>(value, Some(operation)),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    /// Gather I32/I64 tensors in rank order without floating-point conversion.
    pub fn all_gather_int(
        &self,
        value: B::IntTensorPrimitive,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::I32 => self.sharded_int::<i32>(value, None),
            DType::I64 => self.sharded_int::<i64>(value, None),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    /// Reduce I32/I64 tensors using backend arithmetic and return an axis-zero shard.
    pub fn reduce_scatter_int(
        &self,
        value: B::IntTensorPrimitive,
        operation: ReductionOperation,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        match value.dtype() {
            DType::I32 => self.sharded_int::<i32>(value, Some(operation)),
            DType::I64 => self.sharded_int::<i64>(value, Some(operation)),
            dtype => Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }

    fn sharded_float<T: TensorElement>(
        &self,
        value: B::FloatTensorPrimitive,
        operation: Option<ReduceOperation>,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        let execution = self.execution();
        execution.validate_type::<T>()?;
        if &B::float_device(&value) != execution.device() {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        let plan = sharded_shape(value.shape(), self.world_size(), operation.is_some())?;
        storage::checked_length::<T>(plan.input_elements)?;
        storage::checked_length::<T>(plan.output_elements)?;
        let buffer = execution
            .import_float::<T>(B::float_reshape(value, Shape::new([plan.input_elements])))?;
        let collective = self.tensor_collective::<T>();
        let (output, _) = if operation.is_some() {
            collective.reduce_scatter(&buffer, ReductionOperation::Sum, &ReductionOperation::Sum)?
        } else {
            collective.all_gather(&buffer)?
        };
        let mut output = output.float_tensor()?;
        if operation == Some(ReduceOperation::Mean) {
            output = B::float_div_scalar(output, (self.world_size() as f32).into());
        }
        Ok(B::float_reshape(output, plan.output))
    }

    fn sharded_int<T: TensorElement>(
        &self,
        value: B::IntTensorPrimitive,
        operation: Option<ReductionOperation>,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        let execution = self.execution();
        execution.validate_type::<T>()?;
        if &B::int_device(&value) != execution.device() {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        let plan = sharded_shape(value.shape(), self.world_size(), operation.is_some())?;
        storage::checked_length::<T>(plan.input_elements)?;
        storage::checked_length::<T>(plan.output_elements)?;
        let buffer =
            execution.import_int::<T>(B::int_reshape(value, Shape::new([plan.input_elements])))?;
        let collective = self.tensor_collective::<T>();
        let (output, _) = if let Some(operation) = operation {
            let kernel = execution.reduction_kernel::<T>(operation)?;
            collective.reduce_scatter(&buffer, operation, &kernel)?
        } else {
            collective.all_gather(&buffer)?
        };
        Ok(B::int_reshape(output.int_tensor()?, plan.output))
    }

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

fn sharded_shape(
    shape: Shape,
    world_size: u32,
    scatter: bool,
) -> Result<CollectiveShape, TensorDeviceError> {
    let result = if scatter {
        CollectiveShape::reduce_scatter(shape, world_size as usize)
    } else {
        CollectiveShape::all_gather(shape, world_size as usize)
    };
    result.map_err(|error| TensorDeviceError::Data(format!("{error:?}")))
}
