use super::*;

/// An original communicator supporting bounded floating all-reduce messages too.
pub trait ChunkedAllReduceCommunicator<B: Backend>: ChunkedBroadcastCommunicator<B> {
    /// All-reduce the actual floating tensor using explicit native chunk sizing.
    fn all_reduce_float_chunked(&self, value: B::FloatTensorPrimitive,
        operation: ReduceOperation, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError>;
}

impl<B: Backend> ChunkedAllReduceCommunicator<B> for RankCommunicator<TensorDevice<B>> {
    fn all_reduce_float_chunked(&self, value: B::FloatTensorPrimitive,
        operation: ReduceOperation, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        RankCommunicator::all_reduce_float_chunked(self, value, operation, max_chunk_bytes)
    }
}

/// Explicit bounded broadcasts and all-reduces on the caller's actual replica communicator.
///
/// Reuses full/selected `DataParallel` and ZeRO-1 ownership, weighted mean/SUM
/// and FP32-master gradient contracts without rebuilding a model or optimizer.
/// The original native collective algorithm is selected per payload chunk;
/// floating reduction trees can differ from unchunked communication. Backend
/// arithmetic/storage dtypes are unchanged. The limit does not bound all scratch
/// allocations, and reduce-scatter/all-gather remain the original operations.
#[derive(Clone, Debug)]
pub struct BoundedAllReduceCommunicator<B: Backend, C: ChunkedAllReduceCommunicator<B>> {
    broadcasts: BoundedBroadcastCommunicator<B, C>,
}

impl<B: Backend, C: ChunkedAllReduceCommunicator<B>> BoundedAllReduceCommunicator<B, C> {
    /// Apply one explicit native payload byte limit before replica initialization.
    /// All ranks must select the same limit fitting every selected storage element.
    pub fn new(inner: C, max_chunk_bytes: usize) -> Self {
        Self { broadcasts: BoundedBroadcastCommunicator::new(inner, max_chunk_bytes) }
    }
    /// Explicit caller-selected native payload limit.
    pub fn max_chunk_bytes(&self) -> usize { self.broadcasts.max_chunk_bytes() }
    /// Original caller-owned communicator, without changing group membership or devices.
    pub fn inner(&self) -> &C { self.broadcasts.inner() }
    /// Recover the original communicator/session without executing any collective.
    pub fn into_inner(self) -> C { self.broadcasts.into_inner() }
}

impl<B: Backend, C: ChunkedAllReduceCommunicator<B>> DataParallelCommunicator<B>
    for BoundedAllReduceCommunicator<B, C>
{
    fn rank(&self) -> u32 { self.broadcasts.rank() }
    fn world_size(&self) -> u32 { self.broadcasts.world_size() }
    fn device(&self) -> &B::Device { self.broadcasts.device() }
    fn all_gather_bytes(&self, payload: Vec<u8>) -> Result<Vec<Vec<u8>>, DataParallelError> {
        self.broadcasts.all_gather_bytes(payload)
    }
    fn broadcast_float(&self, value: B::FloatTensorPrimitive, root: u32)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.broadcasts.broadcast_float(value, root)
    }
    fn broadcast_int(&self, value: B::IntTensorPrimitive, root: u32)
        -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.broadcasts.broadcast_int(value, root)
    }
    fn all_reduce_float(&self, value: B::FloatTensorPrimitive, operation: ReduceOperation)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner().all_reduce_float_chunked(value, operation, self.max_chunk_bytes())
    }
}

impl<B: Backend, C: ChunkedAllReduceCommunicator<B> + zero2::ShardedCommunicator<B>>
    zero2::ShardedCommunicator<B> for BoundedAllReduceCommunicator<B, C>
{
    fn reduce_scatter_float(&self, value: B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner().reduce_scatter_float(value)
    }
    fn all_gather_float(&self, value: B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner().all_gather_float(value)
    }
}
