use super::*;
use serde::{Deserialize, Serialize};

/// Actual native buffer workload and an explicit maximum host-staged broadcast message size.
/// This bounds payload chunks, not backend copy-on-write or transport scratch allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkedBroadcastPlan {
    elements: usize,
    element_bytes: usize,
    chunk_elements: usize,
}

impl ChunkedBroadcastPlan {
    /// Plan from actual element count/width, retaining complete original storage elements.
    /// The byte budget must fit at least one element; trailing bytes do not split an element.
    pub fn new(elements: usize, element_bytes: usize, max_chunk_bytes: usize)
        -> Result<Self, RankError>
    {
        if element_bytes == 0 || max_chunk_bytes < element_bytes {
            return Err(RankError::InvalidLength("broadcast chunk must fit one native element"));
        }
        let plan = Self { elements, element_bytes, chunk_elements: max_chunk_bytes / element_bytes };
        plan.validate()?;
        Ok(plan)
    }

    /// Validate a constructed or deserialized plan without downloading any tensor.
    pub fn validate(&self) -> Result<(), RankError> {
        if self.element_bytes == 0 || self.chunk_elements == 0 {
            return Err(RankError::InvalidLength("native broadcast width/chunk must be nonzero"));
        }
        let bytes = self.elements.checked_mul(self.element_bytes)
            .ok_or(RankError::Overflow("native broadcast buffer bytes"))?;
        if bytes > isize::MAX as usize {
            return Err(RankError::InvalidLength("native broadcast buffer exceeds addressable size"));
        }
        Ok(())
    }

    /// Actual full native element count, including empty buffers.
    pub fn elements(&self) -> usize { self.elements }
    /// Native source element width; no serialization work precision is substituted.
    pub fn element_bytes(&self) -> usize { self.element_bytes }
    /// Number of whole native elements per non-tail chunk.
    pub fn chunk_elements(&self) -> usize { self.chunk_elements }
    /// Exact original payload size, without metadata or scratch overhead.
    pub fn payload_bytes(&self) -> Result<usize, RankError> {
        self.validate()?;
        Ok(self.elements * self.element_bytes)
    }
    /// Actual tensor-payload work units; an empty tensor has zero payload chunks.
    pub fn chunk_count(&self) -> Result<usize, RankError> {
        self.validate()?;
        Ok(self.elements.div_ceil(self.chunk_elements))
    }
    /// Largest actual chunk payload, not a bound on total process memory.
    pub fn maximum_chunk_bytes(&self) -> Result<usize, RankError> {
        self.validate()?;
        Ok(self.elements.min(self.chunk_elements) * self.element_bytes)
    }
}

/// Actual completed native buffer prefix, reported only after its device write succeeds.
/// Persist this with the actual corresponding device-buffer checkpoint, not by itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BroadcastProgress {
    /// Complete source elements already present in the destination prefix.
    pub completed_elements: usize,
    /// Actual original full native element count.
    pub total_elements: usize,
    /// Payload chunks already complete, including an explicitly restored prefix.
    pub completed_chunks: usize,
    /// Actual full payload-chunk count, not an elapsed-time estimate.
    pub total_chunks: usize,
    /// Prefix declared restored by the caller before this invocation.
    pub restored_elements: usize,
}

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    /// Broadcast one exact native element range into the same range on every rank.
    ///
    /// Prefix/suffix values are unchanged. Only the root's selected range is
    /// downloaded; incoming bytes retain their original element width/bits.
    /// All ranks must supply the same root, range and wire element type. The
    /// offset is included in the exchange tag rather than silently using rank-local offsets.
    pub fn broadcast_at(&self, buffer: &D::Buffer, offset: usize, length: usize, root: u32)
        -> Result<CollectiveStats, D::Error>
    {
        self.validate_root(root)?;
        let end = offset.checked_add(length).ok_or(RankError::Overflow("broadcast range end"))?;
        if end > self.execution.buffer_len(buffer) {
            return Err(RankError::InvalidLength("broadcast range exceeds actual native buffer").into());
        }
        let expected = length.checked_mul(D::ELEMENT_SIZE)
            .ok_or(RankError::Overflow("broadcast range bytes"))?;
        let payload = if self.rank() == root {
            RankDevice::<T>::copy_bytes_from_device_at(self.execution, buffer, offset, length)?
        } else { Vec::new() };
        if self.rank() == root && payload.len() != expected {
            return Err(RankError::InvalidLength("native range readback changed byte count").into());
        }
        let response = self.session.exchange_with_options(
            Opcode::Broadcast, self.element_type,
            ExchangeOptions { root_rank: root, element_count: length as u64, flags: 0, tag: offset as u64 },
            payload,
        )?;
        if response.payload.len() != expected {
            return Err(NetworkError::InvalidConfiguration("broadcast range response byte count differs".into()).into());
        }
        RankDevice::<T>::copy_bytes_to_device_at(self.execution, buffer, offset, &response.payload)?;
        self.stats(CollectiveAlgorithm::Direct, u32::from(self.world_size() > 1), expected, 0)
    }

    /// Broadcast the actual full buffer using the caller's explicit bounded-message plan.
    /// Empty buffers still enter metadata agreement; they do not download tensor payloads.
    pub fn broadcast_chunked(&self, buffer: &D::Buffer, root: u32, plan: ChunkedBroadcastPlan)
        -> Result<CollectiveStats, D::Error>
    {
        self.broadcast_chunked_with_progress(buffer, root, plan, 0, |_| {})
    }

    /// Broadcast a remaining native suffix with actual work-unit progress.
    ///
    /// For a fresh transfer use `restored_elements=0`. A nonzero prefix must have
    /// been restored from the same actual buffer checkpoint on every rank; this
    /// method does not restore or verify prefix values from a progress counter.
    /// Use a live, ordered communicator appropriate to the restored rank/world.
    /// Root, original length/width/type, chunk size and resume position are
    /// agreed before tensor readback. Resume only at a chunk boundary or the end.
    /// A transport error may leave a written prefix, which the callback reports;
    /// no cancellation, retry, optimizer update or checkpoint I/O is implicit.
    pub fn broadcast_chunked_with_progress<F: FnMut(BroadcastProgress)>(
        &self, buffer: &D::Buffer, root: u32, plan: ChunkedBroadcastPlan,
        restored_elements: usize, mut progress: F,
    ) -> Result<CollectiveStats, D::Error> {
        let fields = [root as u64, self.execution.buffer_len(buffer) as u64,
            plan.elements as u64, plan.element_bytes as u64, plan.chunk_elements as u64,
            restored_elements as u64, self.element_type as u64];
        let payload = fields.into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>();
        let (requests, mut stats) = HostStagedExchange::new(self.session)
            .all_gather_host_staged(ElementType::U64, fields.len(), payload.clone())?;
        if requests.chunks_exact(payload.len()).any(|other| other != payload.as_slice()) {
            return Err(NetworkError::InvalidConfiguration("native broadcast plans/ranges differ across ranks".into()).into());
        }
        plan.validate()?;
        self.validate_root(root)?;
        if plan.elements != self.execution.buffer_len(buffer) || plan.element_bytes != D::ELEMENT_SIZE
            || plan.element_bytes != self.element_type.byte_width()
        {
            return Err(RankError::InvalidLength("broadcast plan does not match actual native buffer/type").into());
        }
        if restored_elements > plan.elements
            || (restored_elements != plan.elements && !restored_elements.is_multiple_of(plan.chunk_elements))
        {
            return Err(RankError::InvalidLength("restored broadcast prefix is not an actual chunk boundary").into());
        }
        let total_chunks = plan.chunk_count()?;
        let mut state = BroadcastProgress {
            completed_elements: restored_elements, total_elements: plan.elements,
            completed_chunks: if restored_elements == plan.elements { total_chunks }
                else { restored_elements / plan.chunk_elements },
            total_chunks, restored_elements,
        };
        progress(state);
        while state.completed_elements < plan.elements {
            let length = plan.chunk_elements.min(plan.elements - state.completed_elements);
            let chunk = self.broadcast_at(buffer, state.completed_elements, length, root)?;
            state.completed_elements += length;
            state.completed_chunks += 1;
            progress(state);
            stats.steps = stats.steps.checked_add(chunk.steps)
                .ok_or(RankError::Overflow("native broadcast step statistics"))?;
            stats.transferred_bytes = stats.transferred_bytes.checked_add(chunk.transferred_bytes)
                .ok_or(RankError::Overflow("native broadcast transfer statistics"))?;
        }
        Ok(stats)
    }
}
