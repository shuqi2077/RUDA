use super::*;

/// Original sharded communicator with explicit bounded rank-slice tensor messages.
pub trait ChunkedShardedCommunicator<B: Backend>:
    ChunkedAllReduceCommunicator<B> + zero2::ShardedCommunicator<B>
{
    /// Sum original coordinates and return this rank's original equal axis-zero slice.
    fn reduce_scatter_float_chunked(&self, value: B::FloatTensorPrimitive, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError>;
    /// Gather original equal slices in rank order without changing native dtype or axes.
    fn all_gather_float_chunked(&self, value: B::FloatTensorPrimitive, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError>;
}

impl<B: Backend> ChunkedShardedCommunicator<B> for RankCommunicator<TensorDevice<B>> {
    fn reduce_scatter_float_chunked(&self, value: B::FloatTensorPrimitive, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        RankCommunicator::reduce_scatter_float_chunked(self, value, ReduceOperation::Sum, max_chunk_bytes)
    }
    fn all_gather_float_chunked(&self, value: B::FloatTensorPrimitive, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        RankCommunicator::all_gather_float_chunked(self, value, max_chunk_bytes)
    }
}

/// Bounded native tensor messages for existing full/selected ZeRO-2 replica sessions.
///
/// Broadcast/all-reduce/reduce-scatter/all-gather use the same explicit byte limit,
/// retaining original group ranks, equal element-slice ownership, native parameter
/// storage and caller-selected optimizers. The ZeRO session still performs its
/// own single global-weight normalization, FP32 gradient communication, transport
/// padding and actual local-state update. No optimizer/model proxy is constructed.
/// Floating collective algorithms are selected per chunk; reduction trees need
/// not bit-match unchunked execution. Metadata and backend/transport scratch are
/// outside the tensor-message limit. Nonempty collective chunks must fit one
/// native element per participant. Restore the same communication settings explicitly.
#[derive(Clone, Debug)]
pub struct BoundedShardedCommunicator<B: Backend, C: ChunkedShardedCommunicator<B>> {
    reductions: BoundedAllReduceCommunicator<B, C>,
}

impl<B: Backend, C: ChunkedShardedCommunicator<B>> BoundedShardedCommunicator<B, C> {
    /// Wrap the original group with one explicit native tensor-message byte limit.
    /// All ranks select the same bound before initializing the actual model replicas.
    pub fn new(inner: C, max_chunk_bytes: usize) -> Self {
        Self { reductions: BoundedAllReduceCommunicator::new(inner, max_chunk_bytes) }
    }
    /// Original caller-selected tensor-message byte bound.
    pub fn max_chunk_bytes(&self) -> usize { self.reductions.max_chunk_bytes() }
    /// Actual original communicator and its owned device/topology.
    pub fn inner(&self) -> &C { self.reductions.inner() }
    /// Recover the original communicator without issuing a collective or moving a device.
    pub fn into_inner(self) -> C { self.reductions.into_inner() }
}

impl<B: Backend, C: ChunkedShardedCommunicator<B>> DataParallelCommunicator<B>
    for BoundedShardedCommunicator<B, C>
{
    fn rank(&self) -> u32 { self.reductions.rank() }
    fn world_size(&self) -> u32 { self.reductions.world_size() }
    fn device(&self) -> &B::Device { self.reductions.device() }
    fn all_gather_bytes(&self, payload: Vec<u8>) -> Result<Vec<Vec<u8>>, DataParallelError> {
        self.reductions.all_gather_bytes(payload)
    }
    fn broadcast_float(&self, value: B::FloatTensorPrimitive, root: u32)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.reductions.broadcast_float(value, root)
    }
    fn broadcast_int(&self, value: B::IntTensorPrimitive, root: u32)
        -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.reductions.broadcast_int(value, root)
    }
    fn all_reduce_float(&self, value: B::FloatTensorPrimitive, operation: ReduceOperation)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.reductions.all_reduce_float(value, operation)
    }
}

impl<B: Backend, C: ChunkedShardedCommunicator<B>> zero2::ShardedCommunicator<B>
    for BoundedShardedCommunicator<B, C>
{
    fn reduce_scatter_float(&self, value: B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner().reduce_scatter_float_chunked(value, self.max_chunk_bytes())
    }
    fn all_gather_float(&self, value: B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner().all_gather_float_chunked(value, self.max_chunk_bytes())
    }
}
