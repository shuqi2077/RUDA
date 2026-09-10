use std::error::Error;

pub trait InProcessDevice<T> {
    type Context: Clone + Send + Sync + 'static;
    type Buffer: Clone + Send + Sync + 'static;
    type Kernel: Clone + Send + Sync + 'static;
    type ReductionLaunch;
    type Error: Error + Send + Sync + 'static;

    const ELEMENT_SIZE: usize;
    fn buffer_len(buffer: &Self::Buffer) -> usize;
    fn buffer_is_empty(buffer: &Self::Buffer) -> bool;

    fn alloc(context: &Self::Context, length: usize) -> Result<Self::Buffer, Self::Error>;
    fn copy_to_device(
        context: &Self::Context,
        buffer: &Self::Buffer,
        values: &[T],
    ) -> Result<(), Self::Error>;
    fn copy_to_device_at(
        context: &Self::Context,
        buffer: &Self::Buffer,
        offset: usize,
        values: &[T],
    ) -> Result<(), Self::Error>;
    fn copy_from_device(
        context: &Self::Context,
        buffer: &Self::Buffer,
    ) -> Result<Vec<T>, Self::Error>;
    fn copy_from_device_at(
        context: &Self::Context,
        buffer: &Self::Buffer,
        offset: usize,
        length: usize,
    ) -> Result<Vec<T>, Self::Error>;
    fn prepare_reduction(length: u32, destination_offset: u32) -> Self::ReductionLaunch;
    fn launch_reduction(
        context: &Self::Context,
        kernel: &Self::Kernel,
        launch: &Self::ReductionLaunch,
        source: &Self::Buffer,
        destination: &Self::Buffer,
    ) -> Result<(), Self::Error>;
}
