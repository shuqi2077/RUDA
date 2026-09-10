//! Shared tensor-backend adapters for host-staged rank and in-process collectives.
//!
//! Transfers use the existing collective transports. Reductions execute through
//! the selected backend, including its normal runtime, fusion and copy-on-write
//! rules. This module does not provide a peer-memory or NCCL ABI implementation.

mod device;
mod element;
mod error;
mod reduction;
mod storage;
mod communicator;
mod tensor;

pub use element::TensorElement;
pub use error::TensorDeviceError;

use ruda_tensor::{Backend, DType, TensorMetadata};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Execution context shared by both collective device interfaces.
#[derive(Clone, Debug)]
pub struct TensorDevice<B: Backend> {
    device: B::Device,
}

/// A typed, one-dimensional collective buffer on a tensor backend.
///
/// Clones refer to the same collective state. Exported tensor primitives are
/// snapshots: later collective writes use the backend's copy-on-write semantics
/// and do not modify previously exported tensors.
#[derive(Clone, Debug)]
pub struct TensorBuffer<B: Backend, T: TensorElement> {
    value: Arc<Mutex<Option<Primitive<B>>>>,
    device: B::Device,
    length: usize,
    element: PhantomData<T>,
}

#[derive(Clone, Debug)]
enum Primitive<B: Backend> {
    Float(B::FloatTensorPrimitive),
    Int(B::IntTensorPrimitive),
}

/// Checked launch coordinates, in elements rather than bytes.
#[derive(Clone, Copy, Debug)]
pub struct TensorReductionLaunch {
    length: usize,
    destination_offset: usize,
}

impl<B: Backend> TensorDevice<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Import a rank-one floating tensor without downloading its values.
    pub fn import_float<T: TensorElement>(
        &self,
        value: B::FloatTensorPrimitive,
    ) -> Result<TensorBuffer<B, T>, TensorDeviceError> {
        if T::dtype() == DType::I32 {
            return Err(TensorDeviceError::InvalidBuffer("integer elements require import_int"));
        }
        self.validate_type::<T>()?;
        self.validate_import::<T>(&value, &B::float_device(&value))?;
        let length = value.shape()[0];
        Ok(self.wrap(Primitive::Float(value), length))
    }

    /// Import a rank-one integer tensor without downloading its values.
    pub fn import_int<T: TensorElement>(
        &self,
        value: B::IntTensorPrimitive,
    ) -> Result<TensorBuffer<B, T>, TensorDeviceError> {
        if T::dtype() != DType::I32 {
            return Err(TensorDeviceError::InvalidBuffer("floating elements require import_float"));
        }
        self.validate_type::<T>()?;
        self.validate_import::<T>(&value, &B::int_device(&value))?;
        let length = value.shape()[0];
        Ok(self.wrap(Primitive::Int(value), length))
    }

    pub fn upload<T: TensorElement>(
        &self,
        values: &[T],
    ) -> Result<TensorBuffer<B, T>, TensorDeviceError> {
        self.validate_type::<T>()?;
        storage::checked_length::<T>(values.len())?;
        Ok(self.wrap(self.from_values(values), values.len()))
    }

    /// Wait for work already submitted to the selected tensor backend.
    pub fn synchronize(&self) -> Result<(), TensorDeviceError> {
        B::sync(&self.device).map_err(Into::into)
    }

    fn wrap<T: TensorElement>(&self, value: Primitive<B>, length: usize) -> TensorBuffer<B, T> {
        TensorBuffer {
            value: Arc::new(Mutex::new(Some(value))),
            device: self.device.clone(),
            length,
            element: PhantomData,
        }
    }

    fn validate_type<T: TensorElement>(&self) -> Result<(), TensorDeviceError> {
        if !B::supports_dtype(&self.device, T::dtype()) {
            return Err(TensorDeviceError::UnsupportedDType(T::dtype()));
        }
        Ok(())
    }

    fn validate_import<T: TensorElement>(
        &self,
        value: &impl TensorMetadata,
        device: &B::Device,
    ) -> Result<(), TensorDeviceError> {
        if value.dtype() != T::dtype() {
            return Err(TensorDeviceError::DTypeMismatch {
                expected: T::dtype(),
                actual: value.dtype(),
            });
        }
        if device != &self.device {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        let shape = value.shape();
        if shape.num_dims() != 1 {
            return Err(TensorDeviceError::InvalidBuffer("collective tensors must have rank one"));
        }
        storage::checked_length::<T>(shape[0])
    }

    fn validate_buffer<T: TensorElement>(
        &self,
        buffer: &TensorBuffer<B, T>,
    ) -> Result<(), TensorDeviceError> {
        if buffer.device != self.device {
            return Err(TensorDeviceError::DeviceMismatch);
        }
        Ok(())
    }
}

impl<B: Backend, T: TensorElement> TensorBuffer<B, T> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Export the current floating tensor without a host transfer.
    pub fn float_tensor(&self) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        match self.snapshot()? {
            Primitive::Float(value) => Ok(value),
            Primitive::Int(_) => Err(TensorDeviceError::InvalidBuffer("expected a floating tensor")),
        }
    }

    /// Export the current integer tensor without a host transfer.
    pub fn int_tensor(&self) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        match self.snapshot()? {
            Primitive::Int(value) => Ok(value),
            Primitive::Float(_) => Err(TensorDeviceError::InvalidBuffer("expected an integer tensor")),
        }
    }

    fn snapshot(&self) -> Result<Primitive<B>, TensorDeviceError> {
        self.value.lock().map_err(|_| TensorDeviceError::Poisoned)?
            .as_ref().cloned().ok_or(TensorDeviceError::Poisoned)
    }

    fn update(&self, update: impl FnOnce(Primitive<B>) -> Primitive<B>) -> Result<(), TensorDeviceError> {
        let mut value = self.value.lock().map_err(|_| TensorDeviceError::Poisoned)?;
        let previous = value.take().ok_or(TensorDeviceError::Poisoned)?;
        *value = Some(update(previous));
        Ok(())
    }
}
