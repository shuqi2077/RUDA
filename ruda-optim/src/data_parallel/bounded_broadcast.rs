use super::*;

/// A communicator implementing explicit bounded-message native broadcasts.
/// Gradient reductions retain the original `DataParallelCommunicator` contract.
pub trait ChunkedBroadcastCommunicator<B: Backend>: DataParallelCommunicator<B> {
    /// Broadcast original floating storage without exceeding this tensor payload's byte budget.
    fn broadcast_float_chunked(&self, value: B::FloatTensorPrimitive, root: u32, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError>;
    /// Broadcast native integer storage without floating conversion or whole-buffer readback.
    fn broadcast_int_chunked(&self, value: B::IntTensorPrimitive, root: u32, max_chunk_bytes: usize)
        -> Result<B::IntTensorPrimitive, TensorDeviceError>;
}

impl<B: Backend> ChunkedBroadcastCommunicator<B> for RankCommunicator<TensorDevice<B>> {
    fn broadcast_float_chunked(&self, value: B::FloatTensorPrimitive, root: u32, max_chunk_bytes: usize)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        RankCommunicator::broadcast_float_chunked(self, value, root, max_chunk_bytes)
    }
    fn broadcast_int_chunked(&self, value: B::IntTensorPrimitive, root: u32, max_chunk_bytes: usize)
        -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        RankCommunicator::broadcast_int_chunked(self, value, root, max_chunk_bytes)
    }
}

/// Use bounded native broadcasts in existing full/selected data-parallel and ZeRO sessions.
///
/// Wrap the caller's actual communicator before initialization. Only broadcasts
/// change: original gradient reduction/normalization and optimizer ownership
/// remain unchanged. U8 NF4 bytes, packed AWQ words and original floating scales
/// retain their native storage. The limit bounds each staged payload, not backend
/// copy-on-write, metadata or transport scratch memory. No peer-memory/NCCL
/// transport, overlap, retry, device choice or checkpoint action is inferred.
#[derive(Clone, Debug)]
pub struct BoundedBroadcastCommunicator<B: Backend, C: ChunkedBroadcastCommunicator<B>> {
    inner: C,
    max_chunk_bytes: usize,
    backend: PhantomData<B>,
}

impl<B: Backend, C: ChunkedBroadcastCommunicator<B>> BoundedBroadcastCommunicator<B, C> {
    /// Apply an explicit byte limit to native broadcasts on the original communicator.
    /// All ranks must provide the same limit, fitting every selected native element.
    pub fn new(inner: C, max_chunk_bytes: usize) -> Self {
        Self { inner, max_chunk_bytes, backend: PhantomData }
    }
    /// Caller-selected maximum native tensor message size, without metadata/scratch.
    pub fn max_chunk_bytes(&self) -> usize { self.max_chunk_bytes }
    /// Inspect the caller's actual underlying communicator/device ownership.
    pub fn inner(&self) -> &C { &self.inner }
    /// Recover the original communicator without changing its session or device.
    pub fn into_inner(self) -> C { self.inner }
}

impl<B: Backend, C: ChunkedBroadcastCommunicator<B>> DataParallelCommunicator<B>
    for BoundedBroadcastCommunicator<B, C>
{
    fn rank(&self) -> u32 { self.inner.rank() }
    fn world_size(&self) -> u32 { self.inner.world_size() }
    fn device(&self) -> &B::Device { self.inner.device() }
    fn all_gather_bytes(&self, payload: Vec<u8>) -> Result<Vec<Vec<u8>>, DataParallelError> {
        let budget = u64::try_from(self.max_chunk_bytes)
            .map_err(|_| contract("broadcast byte budget exceeds wire range"))?.to_le_bytes();
        let mut framed = Vec::with_capacity(budget.len() + payload.len());
        framed.extend_from_slice(&budget);
        framed.extend_from_slice(&payload);
        self.inner.all_gather_bytes(framed)?.into_iter().map(|message| {
            if message.len() < budget.len() || message[..budget.len()] != budget {
                return Err(contract("replica ranks disagree on native broadcast byte budget"));
            }
            Ok(message[budget.len()..].to_vec())
        }).collect()
    }
    fn broadcast_float(&self, value: B::FloatTensorPrimitive, root: u32)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner.broadcast_float_chunked(value, root, self.max_chunk_bytes)
    }
    fn broadcast_int(&self, value: B::IntTensorPrimitive, root: u32)
        -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        self.inner.broadcast_int_chunked(value, root, self.max_chunk_bytes)
    }
    fn all_reduce_float(&self, value: B::FloatTensorPrimitive, operation: ReduceOperation)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner.all_reduce_float(value, operation)
    }
}

impl<B: Backend, C: ChunkedBroadcastCommunicator<B> + zero2::ShardedCommunicator<B>>
    zero2::ShardedCommunicator<B> for BoundedBroadcastCommunicator<B, C>
{
    fn reduce_scatter_float(&self, value: B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner.reduce_scatter_float(value)
    }
    fn all_gather_float(&self, value: B::FloatTensorPrimitive)
        -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        self.inner.all_gather_float(value)
    }
}
