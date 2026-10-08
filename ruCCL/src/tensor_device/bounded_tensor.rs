use super::{TensorDevice, TensorDeviceError};
use crate::{ReduceOperation, rank::communicator::RankCommunicator};
use ruda_tensor::{Backend, collective::{TensorCollective, ReplicatedTensorCollective,
    BroadcastTensorCollective, IntegerTensorCollective, VariableTensorCollective, VariableTensorExchange}};

/// Explicit bounded native tensor transport for original TP/EP forward and backward graphs.
///
/// Wraps an existing rank/device session, including its actual collective tuning.
/// Equal-axis shards and variable routed rows retain native storage, full original
/// shapes/counts and rank ordering. Immutable NF4/AWQ buffers do not gain a floating
/// shadow. Chunked arithmetic uses the original backend and selects its configured
/// algorithm per chunk, so floating reduction trees may differ from unchunked ones.
/// The tensor-message budget excludes metadata, complete result tensors and scratch.
/// Nonempty messages must fit one complete native element/row per participant.
/// No device, expert assignment, capacity dropping, optimizer or loss scaling is inferred.
#[derive(Clone, Debug)]
pub struct BoundedTensorCommunicator<B: Backend> {
    inner: RankCommunicator<TensorDevice<B>>,
    max_chunk_bytes: usize,
}

impl<B: Backend> BoundedTensorCommunicator<B> {
    /// Apply one explicit byte budget to an already connected original rank/device session.
    /// All participants use the same budget and forward/backward collective order.
    pub fn new(inner: RankCommunicator<TensorDevice<B>>, max_chunk_bytes: usize) -> Self {
        Self { inner, max_chunk_bytes }
    }
    /// Caller-selected tensor-payload byte limit, without metadata/scratch allocations.
    pub fn max_chunk_bytes(&self) -> usize { self.max_chunk_bytes }
    /// Actual underlying session/device/tuning, not a new simulated communicator.
    pub fn inner(&self) -> &RankCommunicator<TensorDevice<B>> { &self.inner }
    /// Recover the same original session without changing ownership or issuing communication.
    pub fn into_inner(self) -> RankCommunicator<TensorDevice<B>> { self.inner }
    /// Broadcast native integer source buffers without widening their dtype.
    pub fn broadcast_int(&self, value: B::IntTensorPrimitive, root: u32)
        -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.inner.broadcast_int_chunked(value, root, self.max_chunk_bytes)
    }
}

impl<B: Backend> TensorCollective<B> for BoundedTensorCommunicator<B> {
    type Error = TensorDeviceError;
    fn world_size(&self) -> u32 { self.inner.world_size() }
    fn all_gather_float(&self, value: B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive, Self::Error> {
        self.inner.all_gather_float_chunked(value, self.max_chunk_bytes)
    }
    fn reduce_scatter_sum(&self, value: B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive, Self::Error> {
        self.inner.reduce_scatter_float_chunked(value, ReduceOperation::Sum, self.max_chunk_bytes)
    }
}

impl<B: Backend> ReplicatedTensorCollective<B> for BoundedTensorCommunicator<B> {
    fn all_reduce_sum(&self, value: B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive, Self::Error> {
        self.inner.all_reduce_float_chunked(value, ReduceOperation::Sum, self.max_chunk_bytes)
    }
}

impl<B: Backend> BroadcastTensorCollective<B> for BoundedTensorCommunicator<B> {
    fn rank(&self) -> u32 { self.inner.rank() }
    fn broadcast_float(&self, value: B::FloatTensorPrimitive, root: u32) -> Result<B::FloatTensorPrimitive, Self::Error> {
        self.inner.broadcast_float_chunked(value, root, self.max_chunk_bytes)
    }
}

impl<B: Backend> IntegerTensorCollective<B> for BoundedTensorCommunicator<B> {
    fn all_gather_int(&self, value: B::IntTensorPrimitive) -> Result<B::IntTensorPrimitive, Self::Error> {
        self.inner.all_gather_int_chunked(value, self.max_chunk_bytes)
    }
}

impl<B: Backend> VariableTensorCollective<B> for BoundedTensorCommunicator<B> {
    fn all_to_all_v_float(&self, value: B::FloatTensorPrimitive, send_counts: &[usize])
        -> Result<VariableTensorExchange<B::FloatTensorPrimitive>, Self::Error> {
        self.inner.all_to_all_v_float_chunked(value, send_counts, self.max_chunk_bytes)
    }
    fn all_to_all_v_int(&self, value: B::IntTensorPrimitive, send_counts: &[usize])
        -> Result<VariableTensorExchange<B::IntTensorPrimitive>, Self::Error> {
        self.inner.all_to_all_v_int_chunked(value, send_counts, self.max_chunk_bytes)
    }
}
