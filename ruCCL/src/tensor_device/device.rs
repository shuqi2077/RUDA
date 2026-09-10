use super::{TensorBuffer, TensorDevice, TensorDeviceError, TensorElement, TensorReductionLaunch, element};
use crate::in_process::device::InProcessDevice;
use crate::rank::{ReductionOperation, device::RankDevice};
use ruda_tensor::Backend;

impl<B: Backend, T: TensorElement> RankDevice<T> for TensorDevice<B> {
    type Buffer = TensorBuffer<B, T>;
    type Kernel = ReductionOperation;
    type ReductionLaunch = TensorReductionLaunch;
    type Error = TensorDeviceError;

    const ELEMENT_SIZE: usize = std::mem::size_of::<T>();

    fn buffer_len(&self, buffer: &Self::Buffer) -> usize { buffer.len() }
    fn buffer_bytes(&self, buffer: &Self::Buffer) -> usize { buffer.len() * std::mem::size_of::<T>() }
    fn encode(values: &[T]) -> Vec<u8> { element::encode(values) }
    fn decode(bytes: &[u8]) -> Result<Vec<T>, Self::Error> { element::decode(bytes) }

    fn alloc(&self, length: usize) -> Result<Self::Buffer, Self::Error> { self.allocate(length) }

    fn copy_to_device(&self, buffer: &Self::Buffer, values: &[T]) -> Result<(), Self::Error> {
        self.write(buffer, values)
    }

    fn copy_from_device(&self, buffer: &Self::Buffer) -> Result<Vec<T>, Self::Error> {
        self.download(buffer)
    }

    fn copy_from_device_at(&self, buffer: &Self::Buffer, element_offset: usize, length: usize) -> Result<Vec<T>, Self::Error> {
        self.download_at(buffer, element_offset, length)
    }

    fn copy_bytes_to_device_at(&self, buffer: &Self::Buffer, element_offset: usize, values: &[u8]) -> Result<(), Self::Error> {
        self.write_at(buffer, element_offset, &element::decode::<T>(values)?)
    }

    fn copy_bytes_from_device_at(&self, buffer: &Self::Buffer, element_offset: usize, length: usize) -> Result<Vec<u8>, Self::Error> {
        Ok(element::encode(&self.download_at(buffer, element_offset, length)?))
    }

    fn prepare_reduction(&self, length: usize, destination_offset: usize) -> Result<Self::ReductionLaunch, Self::Error> {
        self.prepare::<T>(length, destination_offset)
    }

    fn launch_reduction(&self, kernel: &Self::Kernel, launch: &Self::ReductionLaunch, source: &Self::Buffer, destination: &Self::Buffer) -> Result<(), Self::Error> {
        self.reduce(*kernel, launch, source, destination)
    }
}

impl<B: Backend, T: TensorElement> InProcessDevice<T> for TensorDevice<B> {
    type Context = Self;
    type Buffer = TensorBuffer<B, T>;
    type Kernel = ReductionOperation;
    type ReductionLaunch = TensorReductionLaunch;
    type Error = TensorDeviceError;

    const ELEMENT_SIZE: usize = std::mem::size_of::<T>();

    fn buffer_len(buffer: &Self::Buffer) -> usize { buffer.len() }
    fn buffer_is_empty(buffer: &Self::Buffer) -> bool { buffer.is_empty() }

    fn alloc(context: &Self::Context, length: usize) -> Result<Self::Buffer, Self::Error> {
        context.allocate(length)
    }

    fn copy_to_device(context: &Self::Context, buffer: &Self::Buffer, values: &[T]) -> Result<(), Self::Error> {
        context.write(buffer, values)
    }

    fn copy_to_device_at(context: &Self::Context, buffer: &Self::Buffer, offset: usize, values: &[T]) -> Result<(), Self::Error> {
        context.write_at(buffer, offset, values)
    }

    fn copy_from_device(context: &Self::Context, buffer: &Self::Buffer) -> Result<Vec<T>, Self::Error> {
        context.download(buffer)
    }

    fn copy_from_device_at(context: &Self::Context, buffer: &Self::Buffer, offset: usize, length: usize) -> Result<Vec<T>, Self::Error> {
        context.download_at(buffer, offset, length)
    }

    fn prepare_reduction(length: u32, destination_offset: u32) -> Self::ReductionLaunch {
        TensorReductionLaunch { length: length as usize, destination_offset: destination_offset as usize }
    }

    fn launch_reduction(context: &Self::Context, kernel: &Self::Kernel, launch: &Self::ReductionLaunch, source: &Self::Buffer, destination: &Self::Buffer) -> Result<(), Self::Error> {
        context.reduce(*kernel, launch, source, destination)
    }
}
