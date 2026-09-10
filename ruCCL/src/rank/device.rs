use std::error::Error;

pub trait RankDevice<T>: Clone + Send + Sync + 'static {
    type Buffer: Clone + Send + Sync + 'static;
    type Kernel: Clone + Send + Sync + 'static;
    type ReductionLaunch;
    type Error: Error + Send + Sync + 'static;

    const ELEMENT_SIZE: usize;

    fn buffer_len(&self, buffer: &Self::Buffer) -> usize;

    fn buffer_bytes(&self, buffer: &Self::Buffer) -> usize;

    fn encode(values: &[T]) -> Vec<u8>;

    fn decode(bytes: &[u8]) -> Result<Vec<T>, Self::Error>;

    fn alloc(&self, length: usize) -> Result<Self::Buffer, Self::Error>;

    fn copy_to_device(&self, buffer: &Self::Buffer, values: &[T]) -> Result<(), Self::Error>;

    fn copy_from_device(&self, buffer: &Self::Buffer) -> Result<Vec<T>, Self::Error>;

    fn copy_from_device_at(
        &self,
        buffer: &Self::Buffer,
        element_offset: usize,
        length: usize,
    ) -> Result<Vec<T>, Self::Error>;

    fn copy_bytes_to_device_at(
        &self,
        buffer: &Self::Buffer,
        element_offset: usize,
        values: &[u8],
    ) -> Result<(), Self::Error>;

    fn copy_bytes_from_device_at(
        &self,
        buffer: &Self::Buffer,
        element_offset: usize,
        length: usize,
    ) -> Result<Vec<u8>, Self::Error>;

    fn prepare_reduction(
        &self,
        length: usize,
        destination_offset: usize,
    ) -> Result<Self::ReductionLaunch, Self::Error>;

    fn launch_reduction(
        &self,
        kernel: &Self::Kernel,
        launch: &Self::ReductionLaunch,
        source: &Self::Buffer,
        destination: &Self::Buffer,
    ) -> Result<(), Self::Error>;
}
