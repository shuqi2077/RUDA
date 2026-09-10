use super::{Primitive, TensorBuffer, TensorDevice, TensorDeviceError, TensorElement, TensorReductionLaunch};
use super::storage::{checked_length, checked_range};
use crate::rank::ReductionOperation;
use ruda_tensor::{Backend, DType, get_device_settings};
use std::sync::Arc;

impl<B: Backend> TensorDevice<B> {
    pub fn reduction_kernel<T: TensorElement>(
        &self,
        operation: ReductionOperation,
    ) -> Result<ReductionOperation, TensorDeviceError> {
        self.validate_type::<T>()?;
        if T::dtype() != DType::I32 && matches!(operation,
            ReductionOperation::BitAnd | ReductionOperation::BitOr | ReductionOperation::BitXor)
        {
            return Err(TensorDeviceError::InvalidOperation("bitwise reductions require I32 elements"));
        }
        Ok(operation)
    }

    pub fn prepare<T: TensorElement>(
        &self,
        length: usize,
        destination_offset: usize,
    ) -> Result<TensorReductionLaunch, TensorDeviceError> {
        let end = destination_offset.checked_add(length)
            .ok_or(TensorDeviceError::InvalidBuffer("collective reduction range overflow"))?;
        checked_length::<T>(end)?;
        Ok(TensorReductionLaunch { length, destination_offset })
    }

    pub fn reduce<T: TensorElement>(
        &self,
        operation: ReductionOperation,
        launch: &TensorReductionLaunch,
        source: &TensorBuffer<B, T>,
        destination: &TensorBuffer<B, T>,
    ) -> Result<(), TensorDeviceError> {
        self.validate_buffer(source)?;
        self.validate_buffer(destination)?;
        self.reduction_kernel::<T>(operation)?;
        let source_range = checked_range(source.length, 0, launch.length)?;
        let target_range = checked_range(destination.length, launch.destination_offset, launch.length)?;
        if launch.length == 0 {
            return Ok(());
        }
        let apply = |current: Primitive<B>, incoming: Primitive<B>| {
            if target_range.start == 0 && target_range.end == destination.length {
                self.reduce_values(operation, current, incoming)
            } else {
                let target = current.clone().slice(target_range.clone());
                let reduced = self.reduce_values(operation, target, incoming);
                current.assign(target_range.clone(), reduced)
            }
        };
        if Arc::ptr_eq(&source.value, &destination.value) {
            destination.update(|current| {
                let incoming = current.clone().slice(source_range);
                apply(current, incoming)
            })
        } else {
            // Drop the source lock before acquiring the destination lock.
            let mut incoming = source.snapshot()?;
            if launch.length != source.length {
                incoming = incoming.slice(source_range);
            }
            destination.update(|current| apply(current, incoming))
        }
    }

    fn reduce_values(&self, operation: ReductionOperation, destination: Primitive<B>, source: Primitive<B>) -> Primitive<B> {
        use ReductionOperation::*;
        let bool_dtype = get_device_settings::<B>(&self.device).bool_dtype;
        match (destination, source) {
            (Primitive::Float(destination), Primitive::Float(source)) => Primitive::Float(match operation {
                Sum => B::float_add(destination, source),
                Product => B::float_mul(destination, source),
                // Preserve the destination for unordered/equal comparisons,
                // including NaNs and opposite signed zero, like native kernels.
                Minimum | Maximum => {
                    let mask = if operation == Minimum {
                        B::float_lower(source.clone(), destination.clone(), bool_dtype)
                    } else {
                        B::float_lower(destination.clone(), source.clone(), bool_dtype)
                    };
                    B::float_mask_where(destination, mask, source)
                }
                BitAnd | BitOr | BitXor => unreachable!("validated floating reduction"),
            }),
            (Primitive::Int(destination), Primitive::Int(source)) => Primitive::Int(match operation {
                Sum => B::int_add(destination, source),
                Product => B::int_mul(destination, source),
                Minimum | Maximum => {
                    let mask = if operation == Minimum {
                        B::int_lower(source.clone(), destination.clone(), bool_dtype)
                    } else {
                        B::int_lower(destination.clone(), source.clone(), bool_dtype)
                    };
                    B::int_mask_where(destination, mask, source)
                }
                BitAnd => B::bitwise_and(destination, source),
                BitOr => B::bitwise_or(destination, source),
                BitXor => B::bitwise_xor(destination, source),
            }),
            _ => unreachable!("typed collective buffer storage kind"),
        }
    }
}
