use crate::rank::device::RankDevice;
use crate::rank::device_collective::DeviceCollective;
use crate::rank::error::RankError;
use crate::rank::host::HostStagedExchange;
use crate::rank::work::{CollectiveWork, OrderedWorkQueue, WorkError};
use crate::rank::{
    CollectiveKind, CollectivePlan, CollectiveTransport, CollectiveTuning, ElementType,
    NetworkError, RankTransport, TopologyError, UniqueId,
};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

mod connect;

#[derive(Debug, Clone)]
pub struct RankCommunicator<D> {
    execution: D,
    session: Arc<dyn RankTransport>,
    async_queue: OrderedWorkQueue,
    tuning: Arc<CollectiveTuning>,
    collective_rail_order: Arc<[usize]>,
    internal_sequence: Arc<AtomicU64>,
}

impl<D> RankCommunicator<D> {
    pub fn rank(&self) -> u32 {
        self.session.rank()
    }

    pub fn world_size(&self) -> u32 {
        self.session.world_size()
    }

    pub fn p2p_rails(&self) -> usize {
        self.session.p2p_rails()
    }

    pub fn collective_rail_order(&self) -> &[usize] {
        &self.collective_rail_order
    }

    pub fn plan_collective(&self, kind: CollectiveKind, payload_bytes: usize) -> CollectivePlan {
        self.tuning.plan(kind, payload_bytes)
    }

    pub fn transport(&self) -> CollectiveTransport {
        self.session.transport()
    }

    pub fn tuning(&self) -> &CollectiveTuning {
        &self.tuning
    }

    pub fn execution(&self) -> &D {
        &self.execution
    }

    pub fn host(&self) -> HostStagedExchange<'_> {
        HostStagedExchange::new(self.session.as_ref())
    }

    pub fn device_collective<T>(&self, element_type: ElementType) -> DeviceCollective<'_, T, D>
    where
        D: RankDevice<T>,
        D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
    {
        DeviceCollective::new(
            &self.execution,
            self.session.as_ref(),
            &self.tuning,
            &self.collective_rail_order,
            &self.internal_sequence,
            element_type,
        )
    }

    pub fn submit<T: Send + 'static, E: From<WorkError> + Send + 'static>(
        &self,
        task: impl FnOnce() -> Result<T, E> + Send + 'static,
    ) -> Result<CollectiveWork<T, E>, E> {
        self.async_queue.submit(task)
    }
}
