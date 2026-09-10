use super::device::InProcessDevice;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

pub struct DistributedBuffer<T, D: InProcessDevice<T>> {
    pub(super) communicator_id: u64,
    pub(super) buffers: Vec<D::Buffer>,
    pub(super) length_per_rank: usize,
    pub(super) marker: PhantomData<fn() -> (T, D)>,
}

impl<T, D: InProcessDevice<T>> Clone for DistributedBuffer<T, D> {
    fn clone(&self) -> Self {
        Self {
            communicator_id: self.communicator_id,
            buffers: self.buffers.clone(),
            length_per_rank: self.length_per_rank,
            marker: PhantomData,
        }
    }
}

impl<T, D> Debug for DistributedBuffer<T, D>
where
    D: InProcessDevice<T>,
    D::Buffer: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DistributedBuffer")
            .field("communicator_id", &self.communicator_id)
            .field("buffers", &self.buffers)
            .field("length_per_rank", &self.length_per_rank)
            .finish()
    }
}

pub struct VariableDistributedBuffer<T, D: InProcessDevice<T>> {
    pub(super) communicator_id: u64,
    pub(super) buffers: Vec<Option<D::Buffer>>,
    pub(super) lengths: Vec<usize>,
    pub(super) marker: PhantomData<fn() -> (T, D)>,
}

impl<T, D: InProcessDevice<T>> Clone for VariableDistributedBuffer<T, D> {
    fn clone(&self) -> Self {
        Self {
            communicator_id: self.communicator_id,
            buffers: self.buffers.clone(),
            lengths: self.lengths.clone(),
            marker: PhantomData,
        }
    }
}

impl<T, D> Debug for VariableDistributedBuffer<T, D>
where
    D: InProcessDevice<T>,
    D::Buffer: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VariableDistributedBuffer")
            .field("communicator_id", &self.communicator_id)
            .field("buffers", &self.buffers)
            .field("lengths", &self.lengths)
            .finish()
    }
}

pub struct RootedBuffer<T, D: InProcessDevice<T>> {
    pub(super) communicator_id: u64,
    pub(super) root: usize,
    pub(super) buffer: D::Buffer,
    pub(super) marker: PhantomData<fn() -> (T, D)>,
}

impl<T, D: InProcessDevice<T>> Clone for RootedBuffer<T, D> {
    fn clone(&self) -> Self {
        Self {
            communicator_id: self.communicator_id,
            root: self.root,
            buffer: self.buffer.clone(),
            marker: PhantomData,
        }
    }
}

impl<T, D> Debug for RootedBuffer<T, D>
where
    D: InProcessDevice<T>,
    D::Buffer: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RootedBuffer")
            .field("communicator_id", &self.communicator_id)
            .field("root", &self.root)
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl<T, D: InProcessDevice<T>> DistributedBuffer<T, D> {
    pub fn world_size(&self) -> usize {
        self.buffers.len()
    }

    pub fn length_per_rank(&self) -> usize {
        self.length_per_rank
    }

    pub fn rank_buffer(&self, rank: usize) -> Option<&D::Buffer> {
        self.buffers.get(rank)
    }
}

impl<T, D: InProcessDevice<T>> VariableDistributedBuffer<T, D> {
    pub fn world_size(&self) -> usize {
        self.buffers.len()
    }

    pub fn rank_length(&self, rank: usize) -> Option<usize> {
        self.lengths.get(rank).copied()
    }

    pub fn rank_buffer(&self, rank: usize) -> Option<&D::Buffer> {
        self.buffers.get(rank).and_then(Option::as_ref)
    }
}

impl<T, D: InProcessDevice<T>> RootedBuffer<T, D> {
    pub const fn root(&self) -> usize {
        self.root
    }

    pub fn len(&self) -> usize {
        D::buffer_len(&self.buffer)
    }

    pub fn is_empty(&self) -> bool {
        D::buffer_is_empty(&self.buffer)
    }

    pub const fn buffer(&self) -> &D::Buffer {
        &self.buffer
    }
}
