pub mod communicator;
pub mod device;
pub mod device_collective;
pub mod error;
pub mod host;
pub mod layout;
pub mod network;
pub mod operation;
pub mod payload;
mod peer;
pub mod protocol;
pub mod stats;
pub mod topology;
pub mod tags;
pub mod transport;
pub mod work;

pub use network::{
    ExchangeOptions, NetworkError, TcpRankSession, TcpRendezvousServer, TopologyProbeOptions,
};
pub use operation::ReductionOperation;
pub use stats::CollectiveStats;
pub use protocol::{
    ANY_RANK, CollectiveAgreement, ElementType, FLAG_COUNTS_PREFIX, FLAG_P2P_CHANNEL, Frame,
    FrameHeader, Opcode, OperationDescriptor, ProtocolError, UniqueId,
};
pub use topology::{
    AlgorithmPolicy, COLLECTIVE_AUTOTUNE_PROFILE_VERSION, CollectiveAutotuneEntry,
    CollectiveAutotuneProfile, CollectiveKind, CollectivePlan, CollectivePlanSource,
    CollectiveTopology, CollectiveTuning, TopologyAggregateLink, TopologyError, TopologyLink,
    TopologyRailLink,
};
pub use transport::{
    JsonlTransportTraceSink, RankTransport, TracingRankTransport, TransportTraceEvent,
    TransportTraceOperation, TransportTraceSink,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveTransport {
    HostStaged,
    TcpHostStaged,
    TcpPeer,
    PciePeer,
    Rdma,
    GxLink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CollectiveAlgorithm {
    Direct,
    Ring,
    Pairwise,
    Hierarchical,
}
