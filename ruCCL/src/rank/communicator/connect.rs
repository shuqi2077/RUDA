use super::*;
use crate::rank::TcpRankSession;
use crate::rank::transport::rank_transport_from_environment;
use std::net::ToSocketAddrs;

impl<D> RankCommunicator<D> {
    pub fn connect<E>(
        initialize: impl FnOnce() -> Result<D, E>,
        address: impl ToSocketAddrs,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        timeout: Duration,
        queue_name: &'static str,
    ) -> Result<Self, E>
    where
        E: From<TopologyError> + From<NetworkError> + From<WorkError>,
    {
        let session = TcpRankSession::connect(address, unique_id, rank, world_size, timeout)?;
        Self::from_session(initialize, session, queue_name)
    }

    pub fn from_session<E>(
        initialize: impl FnOnce() -> Result<D, E>,
        session: TcpRankSession,
        queue_name: &'static str,
    ) -> Result<Self, E>
    where
        E: From<TopologyError> + From<NetworkError> + From<WorkError>,
    {
        let tuning = if CollectiveTuning::topology_probe_requested()? {
            let topology = session.probe_topology_from_environment()?;
            CollectiveTuning::from_environment_with_topology(topology)?
        } else {
            CollectiveTuning::from_environment(session.world_size())?
        };
        Self::from_session_with_tuning(initialize, session, tuning, queue_name)
    }

    pub fn from_session_with_tuning<E>(
        initialize: impl FnOnce() -> Result<D, E>,
        session: TcpRankSession,
        tuning: CollectiveTuning,
        queue_name: &'static str,
    ) -> Result<Self, E>
    where
        E: From<TopologyError> + From<NetworkError> + From<WorkError>,
    {
        let transport = rank_transport_from_environment(session)?;
        Self::from_transport_arc_with_tuning(initialize, transport, tuning, queue_name)
    }

    pub fn from_transport_with_tuning<T, E>(
        initialize: impl FnOnce() -> Result<D, E>,
        transport: T,
        tuning: CollectiveTuning,
        queue_name: &'static str,
    ) -> Result<Self, E>
    where
        T: RankTransport + 'static,
        E: From<TopologyError> + From<NetworkError> + From<WorkError>,
    {
        let transport = rank_transport_from_environment(transport)?;
        Self::from_transport_arc_with_tuning(initialize, transport, tuning, queue_name)
    }

    fn from_transport_arc_with_tuning<E>(
        initialize: impl FnOnce() -> Result<D, E>,
        session: Arc<dyn RankTransport>,
        tuning: CollectiveTuning,
        queue_name: &'static str,
    ) -> Result<Self, E>
    where
        E: From<TopologyError> + From<NetworkError> + From<WorkError>,
    {
        let transport = session.transport();
        if tuning.transport() != transport {
            return Err(TopologyError::TransportMismatch {
                tuning: tuning.transport(),
                communicator: transport,
            }
            .into());
        }
        if tuning.ring_order.len() != session.world_size() as usize {
            return Err(TopologyError::InvalidRing(format!(
                "contains {} ranks, expected {}",
                tuning.ring_order.len(),
                session.world_size()
            ))
            .into());
        }
        if tuning.p2p_rails() != session.p2p_rails() {
            return Err(TopologyError::P2pRailMismatch {
                tuning: tuning.p2p_rails(),
                session: session.p2p_rails(),
            }
            .into());
        }
        let execution = initialize()?;
        let async_queue = OrderedWorkQueue::new(queue_name)?;
        let collective_rail_order = Arc::<[usize]>::from(tuning.rail_order());
        Ok(Self {
            execution,
            session,
            async_queue,
            tuning: Arc::new(tuning),
            collective_rail_order,
            internal_sequence: Arc::new(AtomicU64::new(0)),
        })
    }
}
