//! Explicitly owned CUDA IPC groups for native expert/candidate kernels.
use super::InferenceError;
use ruda_core::tensor::{DType, Shape};
use ruda_driver_cuda::{
    CudaRuntime,
    peer::{MemoryTicket, PeerBuffer},
};
use ruda_kernel::{
    dsl::{Runtime, client::ComputeClient},
    tensor::RudaTensor,
};
/// Allocation/export and peer-ticket exchange are separate: an application can
/// use its existing ruCCL, MPI, socket or model-server control-plane transport.
pub struct SymmetricAllocation {
    pub local: PeerBuffer,
    pub rank: usize,
    pub world: usize,
}
impl SymmetricAllocation {
    pub fn allocate(
        client: &ComputeClient<CudaRuntime>,
        bytes: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self, InferenceError> {
        if world == 0 || rank >= world {
            return Err(InferenceError("invalid symmetric rank/world"));
        }
        let local = PeerBuffer::allocate(client, bytes)
            .map_err(|_| InferenceError("CUDA IPC allocation failed"))?;
        Ok(Self { local, rank, world })
    }
    pub fn ticket(&self) -> Result<MemoryTicket, InferenceError> {
        self.local
            .export()
            .map_err(|_| InferenceError("CUDA IPC export failed"))
    }
    /// All exporters and this mapped group must remain alive until every rank
    /// has completed all kernels referencing the exported allocations.
    pub unsafe fn map(
        self,
        tickets: &[MemoryTicket],
        like: &RudaTensor<CudaRuntime>,
    ) -> Result<MappedGroup, InferenceError> {
        if tickets.len() != self.world
            || !self.local.client().same_execution_queue(&like.client)
            || tickets.iter().any(|t| t.bytes < self.local.bytes())
        {
            return Err(InferenceError("peer ticket count/capacity/queue mismatch"));
        }
        let mut imports = Vec::with_capacity(self.world - 1);
        let mut pointers = Vec::with_capacity(self.world);
        for (rank, ticket) in tickets.iter().enumerate() {
            if rank == self.rank {
                pointers.push(self.local.address());
            } else {
                let peer = unsafe { PeerBuffer::import(&like.client, *ticket) }
                    .map_err(|_| InferenceError("CUDA IPC import failed"))?;
                pointers.push(peer.address());
                imports.push(peer);
            }
        }
        let addresses = RudaTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            Shape::from([self.world]),
            like.client
                .create_from_slice(bytemuck::cast_slice(&pointers)),
            DType::U64,
        );
        Ok(MappedGroup {
            addresses,
            imports,
            local: self.local,
            rank: self.rank,
            world: self.world,
        })
    }
}
pub struct MappedGroup {
    pub addresses: RudaTensor<CudaRuntime>,
    imports: Vec<PeerBuffer>,
    local: PeerBuffer,
    pub rank: usize,
    pub world: usize,
}
impl MappedGroup {
    pub fn mapped_bytes(&self) -> usize {
        self.local.bytes() + self.imports.iter().map(PeerBuffer::bytes).sum::<usize>()
    }
}
