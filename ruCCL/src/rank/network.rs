//! TCP rendezvous and host-staged collective exchange using the versioned GX
//! collective wire protocol. Payload reduction remains a rank-side GX kernel
//! responsibility; the coordinator only validates ordering and routes bytes.

use super::protocol::{
    ANY_RANK, CollectiveAgreement, ElementType, FLAG_COUNTS_PREFIX, FLAG_P2P_CHANNEL, Frame,
    FrameHeader, Opcode, OperationDescriptor, ProtocolError, UniqueId,
};
use super::{
    CollectiveTopology, CollectiveTransport, TopologyAggregateLink, TopologyLink, TopologyRailLink,
    peer::DirectPeerMesh,
};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{self, IoSlice, Read, Write};
use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs,
};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const DEFAULT_COLLECTIVE_TIMEOUT: Duration = Duration::from_secs(300);
const COLLECTIVE_RESPONSE_GRACE: Duration = Duration::from_millis(250);
const MAX_P2P_RAILS: usize = 64;
const TOPOLOGY_PROBE_TAG_PREFIX: u64 = 0x475a_0000_0000_0000;
const SERVER_FAILURE_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyProbeOptions {
    pub payload_bytes: usize,
    pub latency_iterations: usize,
    pub bandwidth_iterations: usize,
    pub warmup_iterations: usize,
}

impl Default for TopologyProbeOptions {
    fn default() -> Self {
        Self {
            payload_bytes: 1024 * 1024,
            latency_iterations: 16,
            bandwidth_iterations: 3,
            warmup_iterations: 1,
        }
    }
}

impl TopologyProbeOptions {
    pub fn from_environment() -> Result<Self, NetworkError> {
        let defaults = Self::default();
        let options = Self {
            payload_bytes: parse_probe_usize("GX1_TOPOLOGY_PROBE_BYTES", defaults.payload_bytes)?,
            latency_iterations: parse_probe_usize(
                "GX1_TOPOLOGY_PROBE_LATENCY_ITERATIONS",
                defaults.latency_iterations,
            )?,
            bandwidth_iterations: parse_probe_usize(
                "GX1_TOPOLOGY_PROBE_BANDWIDTH_ITERATIONS",
                defaults.bandwidth_iterations,
            )?,
            warmup_iterations: parse_probe_usize(
                "GX1_TOPOLOGY_PROBE_WARMUP_ITERATIONS",
                defaults.warmup_iterations,
            )?,
        };
        options.validate()?;
        Ok(options)
    }

    fn validate(self) -> Result<(), NetworkError> {
        if self.payload_bytes == 0 || self.payload_bytes > super::protocol::MAX_FRAME_PAYLOAD_BYTES
        {
            return Err(NetworkError::InvalidConfiguration(format!(
                "topology probe payload {} is outside 1..={}",
                self.payload_bytes,
                super::protocol::MAX_FRAME_PAYLOAD_BYTES
            )));
        }
        if self.latency_iterations == 0 || self.bandwidth_iterations == 0 {
            return Err(NetworkError::InvalidConfiguration(
                "topology probe latency and bandwidth iterations must be greater than zero".into(),
            ));
        }
        if self.latency_iterations > 10_000
            || self.bandwidth_iterations > 10_000
            || self.warmup_iterations > 10_000
        {
            return Err(NetworkError::InvalidConfiguration(
                "topology probe iteration counts must not exceed 10000".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LinkProbe {
    bandwidth_mbps: u64,
    latency_ns: u64,
}

type SharedServerFailure = Arc<(Mutex<Option<String>>, Condvar)>;

fn publish_server_failure(failure: &SharedServerFailure, message: &str) {
    let (state, ready) = &**failure;
    if let Ok(mut state) = state.lock() {
        state.get_or_insert_with(|| message.to_owned());
        ready.notify_all();
    }
}

fn server_failure_text(error: &NetworkError) -> String {
    match error {
        NetworkError::RemoteAbort(message) => message.clone(),
        _ => error.to_string(),
    }
}

fn server_failure_message(failure: &SharedServerFailure) -> Result<Option<String>, NetworkError> {
    let (state, _) = &**failure;
    Ok(state.lock().map_err(|_| NetworkError::Poisoned)?.clone())
}

fn wait_for_server_failure(
    failure: &SharedServerFailure,
    timeout: Duration,
) -> Result<Option<String>, NetworkError> {
    let (state, ready) = &**failure;
    let state = state.lock().map_err(|_| NetworkError::Poisoned)?;
    if state.is_some() {
        return Ok(state.clone());
    }
    let (state, _) = ready
        .wait_timeout(state, timeout)
        .map_err(|_| NetworkError::Poisoned)?;
    Ok(state.clone())
}

#[derive(Debug)]
pub enum NetworkError {
    Io(io::Error),
    Protocol(ProtocolError),
    InvalidConfiguration(String),
    RankAlreadyJoined(u32),
    WrongSession,
    WrongDestination { expected: u32, actual: u32 },
    RemoteAbort(String),
    Timeout(String),
    ChannelClosed,
    Poisoned,
}

impl Display for NetworkError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => Display::fmt(error, formatter),
            Self::Protocol(error) => Display::fmt(error, formatter),
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::RankAlreadyJoined(rank) => write!(formatter, "rank {rank} joined twice"),
            Self::WrongSession => write!(formatter, "collective frame belongs to another session"),
            Self::WrongDestination { expected, actual } => write!(
                formatter,
                "collective response destination {actual} does not match rank {expected}"
            ),
            Self::RemoteAbort(message) => {
                write!(formatter, "collective coordinator aborted: {message}")
            }
            Self::Timeout(operation) => write!(formatter, "{operation} timed out"),
            Self::ChannelClosed => write!(formatter, "collective response channel closed"),
            Self::Poisoned => write!(formatter, "TCP collective session is poisoned"),
        }
    }
}

impl Error for NetworkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Protocol(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for NetworkError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ProtocolError> for NetworkError {
    fn from(error: ProtocolError) -> Self {
        Self::Protocol(error)
    }
}

#[derive(Debug)]
pub struct TcpRendezvousServer {
    listener: TcpListener,
    unique_id: UniqueId,
    world_size: usize,
    p2p_rails: usize,
    collective_timeout: Duration,
    heartbeat_timeout: Option<Duration>,
    transport: CollectiveTransport,
}

impl TcpRendezvousServer {
    pub fn bind(
        address: impl ToSocketAddrs,
        unique_id: UniqueId,
        world_size: usize,
    ) -> Result<Self, NetworkError> {
        if world_size == 0 || world_size > u32::MAX as usize {
            return Err(NetworkError::InvalidConfiguration(format!(
                "TCP collective world size {world_size} is outside 1..={} ",
                u32::MAX
            )));
        }
        let listener = TcpListener::bind(address)?;
        let p2p_rails = p2p_rails_from_environment()?;
        let transport = tcp_transport_from_environment()?;
        Ok(Self {
            listener,
            unique_id,
            world_size,
            p2p_rails,
            collective_timeout: DEFAULT_COLLECTIVE_TIMEOUT,
            heartbeat_timeout: None,
            transport,
        })
    }

    pub fn with_collective_timeout(mut self, timeout: Duration) -> Result<Self, NetworkError> {
        if timeout.is_zero() {
            return Err(NetworkError::InvalidConfiguration(
                "TCP collective timeout must be greater than zero".into(),
            ));
        }
        self.collective_timeout = timeout;
        Ok(self)
    }

    pub fn with_heartbeat_timeout(mut self, timeout: Duration) -> Result<Self, NetworkError> {
        if timeout.is_zero() {
            return Err(NetworkError::InvalidConfiguration(
                "TCP heartbeat timeout must be greater than zero".into(),
            ));
        }
        self.heartbeat_timeout = Some(timeout);
        Ok(self)
    }

    pub fn with_p2p_rails(mut self, rails: usize) -> Result<Self, NetworkError> {
        validate_p2p_rails(rails)?;
        self.p2p_rails = rails;
        Ok(self)
    }

    pub fn with_transport(mut self, transport: CollectiveTransport) -> Result<Self, NetworkError> {
        validate_tcp_transport(transport)?;
        self.transport = transport;
        Ok(self)
    }

    pub fn local_addr(&self) -> Result<SocketAddr, NetworkError> {
        Ok(self.listener.local_addr()?)
    }

    /// Accept one ordered collective channel and one or more independently
    /// multiplexed point-to-point rails per rank, then serve until ranks disconnect.
    pub fn run(self) -> Result<(), NetworkError> {
        let mut collective_streams =
            accept_rank_channels(&self.listener, self.unique_id, self.world_size, false, 0)?;
        let unique_id = self.unique_id;
        let world_size = self.world_size;
        let server_failure = Arc::new((Mutex::new(None), Condvar::new()));
        let mut p2p_workers = Vec::with_capacity(self.p2p_rails);
        match self.transport {
            CollectiveTransport::TcpHostStaged => {
                for rail in 0..self.p2p_rails {
                    let p2p_streams = accept_rank_channels(
                        &self.listener,
                        self.unique_id,
                        self.world_size,
                        true,
                        rail,
                    )?;
                    let heartbeat_timeout = (rail == 0).then_some(self.heartbeat_timeout).flatten();
                    p2p_workers.push(spawn_p2p_router(
                        p2p_streams,
                        unique_id,
                        world_size,
                        heartbeat_timeout,
                        rail,
                        Arc::clone(&server_failure),
                    )?);
                }
            }
            CollectiveTransport::TcpPeer => {
                for stream in &collective_streams {
                    stream.set_read_timeout(Some(self.collective_timeout))?;
                }
                exchange_peer_endpoints_server(
                    &mut collective_streams,
                    unique_id,
                    world_size,
                    self.p2p_rails,
                )?;
                for stream in &collective_streams {
                    stream.set_read_timeout(None)?;
                }
                let control_streams =
                    accept_rank_channels(&self.listener, self.unique_id, self.world_size, true, 0)?;
                p2p_workers.push(spawn_p2p_router(
                    control_streams,
                    unique_id,
                    world_size,
                    self.heartbeat_timeout,
                    0,
                    Arc::clone(&server_failure),
                )?);
            }
            CollectiveTransport::HostStaged
            | CollectiveTransport::PciePeer
            | CollectiveTransport::Rdma
            | CollectiveTransport::GxLink => {
                return Err(NetworkError::InvalidConfiguration(format!(
                    "{:?} is not a TCP transport",
                    self.transport
                )));
            }
        }
        let collective_result = run_collective_loop(
            collective_streams,
            self.unique_id,
            self.world_size,
            self.collective_timeout,
            server_failure,
        );
        let mut p2p_result = Ok(());
        for (rail, worker) in p2p_workers.into_iter().enumerate() {
            let result = match worker.join() {
                Ok(result) => result,
                Err(_) => Err(NetworkError::InvalidConfiguration(format!(
                    "point-to-point rail {rail} router panicked"
                ))),
            };
            if p2p_result.is_ok() {
                p2p_result = result;
            }
        }
        collective_result.and(p2p_result)
    }
}

fn spawn_p2p_router(
    streams: Vec<TcpStream>,
    unique_id: UniqueId,
    world_size: usize,
    heartbeat_timeout: Option<Duration>,
    rail: usize,
    server_failure: SharedServerFailure,
) -> Result<JoinHandle<Result<(), NetworkError>>, NetworkError> {
    thread::Builder::new()
        .name(format!("gx1-p2p-router-{rail}"))
        .spawn(move || {
            run_p2p_loop(
                streams,
                unique_id,
                world_size,
                heartbeat_timeout,
                server_failure,
            )
        })
        .map_err(|error| {
            NetworkError::InvalidConfiguration(format!(
                "cannot start point-to-point rail {rail} router: {error}"
            ))
        })
}

fn accept_rank_channels(
    listener: &TcpListener,
    unique_id: UniqueId,
    world_size: usize,
    p2p: bool,
    rail: usize,
) -> Result<Vec<TcpStream>, NetworkError> {
    let mut ranks = (0..world_size)
        .map(|_| None)
        .collect::<Vec<Option<TcpStream>>>();
    for _ in 0..world_size {
        let (mut stream, _) = listener.accept()?;
        stream.set_nodelay(true)?;
        let frame = read_frame(&mut stream)?
            .ok_or_else(|| NetworkError::InvalidConfiguration("rank closed before JOIN".into()))?;
        validate_join(&frame, unique_id, world_size, p2p, rail)?;
        let rank = frame.header.source_rank as usize;
        if ranks[rank].is_some() {
            return Err(NetworkError::RankAlreadyJoined(rank as u32));
        }
        ranks[rank] = Some(stream);
    }
    let mut streams = ranks
        .into_iter()
        .map(|stream| stream.expect("every rank joined"))
        .collect::<Vec<_>>();
    for (rank, stream) in streams.iter_mut().enumerate() {
        let mut ready = control_frame(unique_id, Opcode::Ready, 0, rank as u32, world_size as u32)?;
        if p2p {
            ready.header.flags |= FLAG_P2P_CHANNEL;
            ready.header.tag = rail as u64;
        }
        write_frame(stream, &ready)?;
    }
    Ok(streams)
}

fn exchange_peer_endpoints_server(
    streams: &mut [TcpStream],
    unique_id: UniqueId,
    world_size: usize,
    rails: usize,
) -> Result<(), NetworkError> {
    let mut endpoints = Vec::with_capacity(world_size);
    let mut all_ranks_use_shared_endpoint = true;
    for (rank, stream) in streams.iter_mut().enumerate() {
        let frame = read_frame(stream)?.ok_or_else(|| {
            NetworkError::InvalidConfiguration(format!(
                "rank {rank} closed before publishing its direct peer endpoint"
            ))
        })?;
        if frame.header.opcode != Opcode::PeerEndpoint
            || frame.header.element_type != ElementType::U8
            || frame.header.unique_id != unique_id
            || frame.header.source_rank as usize != rank
            || frame.header.destination_rank != ANY_RANK
            || frame.header.root_rank != ANY_RANK
            || frame.header.world_size as usize != world_size
            || frame.header.sequence != 0
            || frame.header.tag != rails as u64
            || frame.header.flags != super::protocol::FLAG_PAYLOAD
            || frame.header.element_count != frame.payload.len() as u64
            || frame.payload.is_empty()
        {
            return Err(NetworkError::InvalidConfiguration(format!(
                "rank {rank} published an invalid direct peer endpoint frame: {:?}, payload bytes {}",
                frame.header,
                frame.payload.len()
            )));
        }
        let text = std::str::from_utf8(&frame.payload).map_err(|error| {
            NetworkError::InvalidConfiguration(format!(
                "rank {rank} direct peer endpoints are not UTF-8: {error}"
            ))
        })?;
        let entries = text.lines().collect::<Vec<_>>();
        if entries.len() != 1 && entries.len() != rails {
            return Err(NetworkError::InvalidConfiguration(format!(
                "rank {rank} published {} direct peer endpoints, expected one shared endpoint or {rails} rail endpoints",
                entries.len()
            )));
        }
        all_ranks_use_shared_endpoint &= entries.len() == 1;
        let peer_ip = stream.peer_addr()?.ip();
        let mut rank_endpoints = Vec::with_capacity(rails);
        for (rail, entry) in entries.iter().enumerate() {
            let mut endpoint = entry.parse::<SocketAddr>().map_err(|error| {
                NetworkError::InvalidConfiguration(format!(
                    "rank {rank} direct peer endpoint {entry:?} for rail {rail} is invalid: {error}"
                ))
            })?;
            if endpoint.port() == 0 {
                return Err(NetworkError::InvalidConfiguration(format!(
                    "rank {rank} direct peer endpoint for rail {rail} uses port zero"
                )));
            }
            if endpoint.ip().is_unspecified() {
                endpoint.set_ip(peer_ip);
            }
            rank_endpoints.push(endpoint);
        }
        if rank_endpoints.len() == 1 {
            rank_endpoints.resize(rails, rank_endpoints[0]);
        }
        endpoints.push(rank_endpoints);
    }

    // Keep the original one-line-per-rank table when every rank uses one
    // shared listener. New clients accept both layouts, so existing scalar
    // GX1_P2P_LISTEN_ADDR deployments retain their wire representation.
    let table = endpoints
        .iter()
        .flat_map(|rank_endpoints| {
            if all_ranks_use_shared_endpoint {
                &rank_endpoints[..1]
            } else {
                rank_endpoints.as_slice()
            }
        })
        .map(SocketAddr::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes();
    for (rank, stream) in streams.iter_mut().enumerate() {
        let mut header = FrameHeader::collective(
            unique_id,
            Opcode::PeerEndpoint,
            ElementType::U8,
            0,
            ANY_RANK,
            world_size as u32,
            0,
            table.len() as u64,
        );
        header.destination_rank = rank as u32;
        header.tag = rails as u64;
        write_frame(stream, &Frame::new(header, table.clone())?)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PeerEndpointConfiguration {
    listen_addresses: Vec<SocketAddr>,
    advertise_addresses: Vec<IpAddr>,
}

impl PeerEndpointConfiguration {
    fn from_environment(control_stream: &TcpStream, rails: usize) -> Result<Self, NetworkError> {
        let default_listen = match control_stream.local_addr()?.ip() {
            IpAddr::V4(_) => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            IpAddr::V6(_) => SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0),
        };
        let listen_addresses = match optional_environment("GX1_P2P_RAIL_LISTEN_ADDRS")? {
            Some(value) => parse_rail_socket_addresses("GX1_P2P_RAIL_LISTEN_ADDRS", &value, rails)?,
            None => match optional_environment("GX1_P2P_LISTEN_ADDR")? {
                Some(value) => vec![parse_socket_address("GX1_P2P_LISTEN_ADDR", &value)?],
                None => vec![default_listen],
            },
        };
        let advertise_addresses = match optional_environment("GX1_P2P_RAIL_ADVERTISE_ADDRS")? {
            Some(value) => parse_rail_ip_addresses("GX1_P2P_RAIL_ADVERTISE_ADDRS", &value, rails)?,
            None => match optional_environment("GX1_P2P_ADVERTISE_ADDR")? {
                Some(value) => vec![parse_ip_address("GX1_P2P_ADVERTISE_ADDR", &value)?],
                None => Vec::new(),
            },
        };
        let configuration = Self {
            listen_addresses,
            advertise_addresses,
        };
        configuration.validate(rails)?;
        Ok(configuration)
    }

    fn validate(&self, rails: usize) -> Result<(), NetworkError> {
        if self.listen_addresses.len() != 1 && self.listen_addresses.len() != rails {
            return Err(NetworkError::InvalidConfiguration(format!(
                "direct peer listen configuration contains {} addresses, expected one shared address or {rails} rail addresses",
                self.listen_addresses.len()
            )));
        }
        if !self.advertise_addresses.is_empty()
            && self.advertise_addresses.len() != 1
            && self.advertise_addresses.len() != rails
        {
            return Err(NetworkError::InvalidConfiguration(format!(
                "direct peer advertise configuration contains {} addresses, expected zero, one shared address, or {rails} rail addresses",
                self.advertise_addresses.len()
            )));
        }
        Ok(())
    }
}

fn optional_environment(name: &'static str) -> Result<Option<String>, NetworkError> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(value)) => Err(NetworkError::InvalidConfiguration(format!(
            "{name} is not Unicode: {:?}",
            value.to_string_lossy()
        ))),
    }
}

fn parse_socket_address(name: &'static str, value: &str) -> Result<SocketAddr, NetworkError> {
    value.trim().parse::<SocketAddr>().map_err(|error| {
        NetworkError::InvalidConfiguration(format!(
            "{name} must be a socket address, got {value:?}: {error}"
        ))
    })
}

fn parse_ip_address(name: &'static str, value: &str) -> Result<IpAddr, NetworkError> {
    value.trim().parse::<IpAddr>().map_err(|error| {
        NetworkError::InvalidConfiguration(format!(
            "{name} must be an IP address, got {value:?}: {error}"
        ))
    })
}

fn parse_rail_socket_addresses(
    name: &'static str,
    value: &str,
    rails: usize,
) -> Result<Vec<SocketAddr>, NetworkError> {
    let entries = value.split(',').collect::<Vec<_>>();
    if entries.len() != rails {
        return Err(NetworkError::InvalidConfiguration(format!(
            "{name} contains {} addresses, expected {rails}",
            entries.len()
        )));
    }
    entries
        .into_iter()
        .enumerate()
        .map(|(rail, entry)| {
            parse_socket_address(name, entry).map_err(|error| {
                NetworkError::InvalidConfiguration(format!("{error} (rail {rail})"))
            })
        })
        .collect()
}

fn parse_rail_ip_addresses(
    name: &'static str,
    value: &str,
    rails: usize,
) -> Result<Vec<IpAddr>, NetworkError> {
    let entries = value.split(',').collect::<Vec<_>>();
    if entries.len() != rails {
        return Err(NetworkError::InvalidConfiguration(format!(
            "{name} contains {} addresses, expected {rails}",
            entries.len()
        )));
    }
    entries
        .into_iter()
        .enumerate()
        .map(|(rail, entry)| {
            parse_ip_address(name, entry).map_err(|error| {
                NetworkError::InvalidConfiguration(format!("{error} (rail {rail})"))
            })
        })
        .collect()
}

fn bind_peer_listeners(
    control_stream: &TcpStream,
    rails: usize,
) -> Result<(Vec<TcpListener>, Vec<SocketAddr>), NetworkError> {
    let configuration = PeerEndpointConfiguration::from_environment(control_stream, rails)?;
    bind_peer_listeners_with_configuration(configuration, rails)
}

fn bind_peer_listeners_with_configuration(
    configuration: PeerEndpointConfiguration,
    rails: usize,
) -> Result<(Vec<TcpListener>, Vec<SocketAddr>), NetworkError> {
    configuration.validate(rails)?;
    let mut listeners = Vec::with_capacity(configuration.listen_addresses.len());
    let mut local_addresses = Vec::with_capacity(configuration.listen_addresses.len());
    for (listener_index, bind_address) in configuration.listen_addresses.iter().enumerate() {
        let listener = TcpListener::bind(bind_address).map_err(|error| {
            NetworkError::InvalidConfiguration(format!(
                "cannot bind direct peer listener {listener_index} to {bind_address}: {error}"
            ))
        })?;
        local_addresses.push(listener.local_addr()?);
        listeners.push(listener);
    }

    let mut endpoints = Vec::with_capacity(rails);
    for rail in 0..rails {
        let listener_index = if listeners.len() == 1 { 0 } else { rail };
        let local = local_addresses[listener_index];
        let advertised_ip = match configuration.advertise_addresses.len() {
            0 => local.ip(),
            1 => configuration.advertise_addresses[0],
            _ => configuration.advertise_addresses[rail],
        };
        endpoints.push(SocketAddr::new(advertised_ip, local.port()));
    }
    Ok((listeners, endpoints))
}

fn exchange_peer_endpoints_client(
    stream: &mut TcpStream,
    rank_endpoints: &[SocketAddr],
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    rails: usize,
) -> Result<Vec<Vec<SocketAddr>>, NetworkError> {
    if rank_endpoints.len() != rails {
        return Err(NetworkError::InvalidConfiguration(format!(
            "rank {rank} has {} direct peer endpoints, expected {rails}",
            rank_endpoints.len()
        )));
    }
    let shared_endpoint = rank_endpoints
        .windows(2)
        .all(|endpoints| endpoints[0] == endpoints[1]);
    let payload = rank_endpoints[..if shared_endpoint { 1 } else { rails }]
        .iter()
        .map(SocketAddr::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes();
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::PeerEndpoint,
        ElementType::U8,
        rank,
        ANY_RANK,
        world_size,
        0,
        payload.len() as u64,
    );
    header.tag = rails as u64;
    write_frame(stream, &Frame::new(header, payload)?)?;
    let response = read_frame(stream)?.ok_or_else(|| {
        NetworkError::InvalidConfiguration(
            "coordinator closed before returning direct peer endpoints".into(),
        )
    })?;
    validate_response(&response, unique_id, rank, world_size, 0)?;
    if response.header.opcode != Opcode::PeerEndpoint
        || response.header.element_type != ElementType::U8
        || response.header.tag != rails as u64
        || response.header.element_count != response.payload.len() as u64
    {
        return Err(NetworkError::InvalidConfiguration(
            "coordinator returned an invalid direct peer endpoint table".into(),
        ));
    }
    let table = std::str::from_utf8(&response.payload).map_err(|error| {
        NetworkError::InvalidConfiguration(format!(
            "direct peer endpoint table is not UTF-8: {error}"
        ))
    })?;
    let flat_endpoints = table
        .lines()
        .map(|entry| {
            entry.parse::<SocketAddr>().map_err(|error| {
                NetworkError::InvalidConfiguration(format!(
                    "direct peer endpoint {entry:?} is invalid: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if flat_endpoints.len() == world_size as usize {
        return Ok(flat_endpoints
            .into_iter()
            .map(|endpoint| vec![endpoint; rails])
            .collect());
    }
    let expected = world_size as usize * rails;
    if flat_endpoints.len() != expected {
        return Err(NetworkError::InvalidConfiguration(format!(
            "coordinator returned {} direct peer endpoints, expected {world_size} shared endpoints or {expected} rail endpoints",
            flat_endpoints.len()
        )));
    }
    Ok(flat_endpoints
        .chunks_exact(rails)
        .map(<[SocketAddr]>::to_vec)
        .collect())
}

#[derive(Debug)]
enum CollectiveEvent {
    Frame { rank: usize, frame: Frame },
    Closed { rank: usize },
    Failed { rank: usize, message: String },
}

fn run_collective_loop(
    mut streams: Vec<TcpStream>,
    unique_id: UniqueId,
    world_size: usize,
    mut timeout: Duration,
    server_failure: SharedServerFailure,
) -> Result<(), NetworkError> {
    let (sender, receiver) = mpsc::channel::<CollectiveEvent>();
    let mut readers = Vec::with_capacity(world_size);
    for (rank, stream) in streams.iter().enumerate() {
        let mut stream = stream.try_clone()?;
        stream.set_read_timeout(None)?;
        let sender = sender.clone();
        readers.push(
            thread::Builder::new()
                .name(format!("gx1-collective-server-rank-{rank}"))
                .spawn(move || {
                    loop {
                        let event = match read_frame(&mut stream) {
                            Ok(Some(frame)) => CollectiveEvent::Frame { rank, frame },
                            Ok(None) => CollectiveEvent::Closed { rank },
                            Err(error) => CollectiveEvent::Failed {
                                rank,
                                message: error.to_string(),
                            },
                        };
                        let terminal = !matches!(event, CollectiveEvent::Frame { .. });
                        if sender.send(event).is_err() || terminal {
                            return;
                        }
                    }
                })
                .map_err(|error| {
                    NetworkError::InvalidConfiguration(format!(
                        "cannot start collective rank reader: {error}"
                    ))
                })?,
        );
    }
    drop(sender);
    let mut agreement = CollectiveAgreement::new(world_size)?;
    let mut active_sequence = agreement.next_sequence();
    let result = 'session: loop {
        if let Some(message) = server_failure_message(&server_failure)? {
            break Err(NetworkError::InvalidConfiguration(message));
        }
        let sequence = agreement.next_sequence();
        let deadline = Instant::now() + timeout;
        let mut requests = (0..world_size)
            .map(|_| None)
            .collect::<Vec<Option<Frame>>>();
        while requests.iter().any(Option::is_none) {
            if let Some(message) = server_failure_message(&server_failure)? {
                break 'session Err(NetworkError::InvalidConfiguration(message));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break 'session Err(collective_timeout_error(sequence, &requests));
            }
            match receiver.recv_timeout(remaining.min(SERVER_FAILURE_POLL_INTERVAL)) {
                Ok(CollectiveEvent::Frame { rank, frame }) => {
                    if requests[rank].replace(frame).is_some() {
                        break 'session Err(NetworkError::InvalidConfiguration(format!(
                            "rank {rank} submitted collective sequence {sequence} twice"
                        )));
                    }
                }
                Ok(CollectiveEvent::Closed { rank }) => {
                    if requests.iter().all(Option::is_none) {
                        break 'session Ok(());
                    }
                    if let Some(message) =
                        wait_for_server_failure(&server_failure, SERVER_FAILURE_POLL_INTERVAL)?
                    {
                        break 'session Err(NetworkError::InvalidConfiguration(message));
                    }
                    break 'session Err(NetworkError::InvalidConfiguration(format!(
                        "rank {rank} disconnected during collective sequence {sequence}"
                    )));
                }
                Ok(CollectiveEvent::Failed { rank, message }) => {
                    if let Some(message) =
                        wait_for_server_failure(&server_failure, SERVER_FAILURE_POLL_INTERVAL)?
                    {
                        break 'session Err(NetworkError::InvalidConfiguration(message));
                    }
                    break 'session Err(NetworkError::InvalidConfiguration(format!(
                        "rank {rank} failed during collective sequence {sequence}: {message}"
                    )));
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    break 'session Err(NetworkError::ChannelClosed);
                }
            }
        }
        let requests = requests
            .into_iter()
            .map(|request| request.expect("every collective rank submitted"))
            .collect::<Vec<_>>();
        let responses = match validate_and_route(&mut agreement, unique_id, sequence, &requests) {
            Ok(responses) => responses,
            Err(error) => break Err(error),
        };
        for (stream, response) in streams.iter_mut().zip(&responses) {
            if let Err(error) = write_frame(stream, response) {
                break 'session Err(error);
            }
        }
        if requests[0].header.opcode == Opcode::SetTimeout {
            timeout = Duration::from_millis(requests[0].header.tag);
        }
        active_sequence = agreement.next_sequence();
    };
    if let Err(error) = &result {
        broadcast_collective_abort(
            &mut streams,
            unique_id,
            active_sequence,
            world_size,
            &error.to_string(),
        );
    }
    for stream in &streams {
        let _ = stream.shutdown(Shutdown::Both);
    }
    for reader in readers {
        let _ = reader.join();
    }
    result
}

fn collective_timeout_error(sequence: u64, requests: &[Option<Frame>]) -> NetworkError {
    let missing = requests
        .iter()
        .enumerate()
        .filter_map(|(rank, request)| request.is_none().then_some(rank.to_string()))
        .collect::<Vec<_>>()
        .join(",");
    NetworkError::Timeout(format!(
        "collective sequence {sequence}; missing ranks [{missing}]"
    ))
}

fn broadcast_collective_abort(
    streams: &mut [TcpStream],
    unique_id: UniqueId,
    sequence: u64,
    world_size: usize,
    message: &str,
) {
    for (rank, stream) in streams.iter_mut().enumerate() {
        if let Ok(frame) = abort_frame(unique_id, sequence, rank as u32, world_size as u32, message)
        {
            let _ = write_frame(stream, &frame);
        }
    }
}

#[derive(Debug)]
struct TcpSessionInner {
    stream: TcpStream,
    next_sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct P2pResponseKey {
    opcode: Opcode,
    sequence: u64,
}

type PendingP2pResponse = Sender<Result<Frame, String>>;

#[derive(Debug)]
struct P2pSubmission {
    stream: TcpStream,
    next_send_sequence: u64,
    next_receive_sequence: u64,
    next_heartbeat_sequence: u64,
}

#[derive(Debug)]
struct P2pClient {
    submission: Mutex<P2pSubmission>,
    pending: Arc<Mutex<HashMap<P2pResponseKey, PendingP2pResponse>>>,
    failure: Arc<Mutex<Option<String>>>,
    reader: Mutex<Option<JoinHandle<()>>>,
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    timeout: Mutex<Duration>,
}

#[derive(Debug)]
enum P2pDataPlane {
    Coordinator(Vec<P2pClient>),
    Peer {
        mesh: Box<DirectPeerMesh>,
        control: P2pClient,
    },
}

#[derive(Debug)]
pub struct TcpRankSession {
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    transport: CollectiveTransport,
    inner: Mutex<TcpSessionInner>,
    p2p: P2pDataPlane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExchangeOptions {
    pub root_rank: u32,
    pub element_count: u64,
    pub flags: u16,
    pub tag: u64,
}

impl ExchangeOptions {
    pub const fn new(root_rank: u32, element_count: u64) -> Self {
        Self {
            root_rank,
            element_count,
            flags: 0,
            tag: 0,
        }
    }
}

impl P2pClient {
    fn new(
        stream: TcpStream,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        timeout: Duration,
        rail: usize,
        failure: Arc<Mutex<Option<String>>>,
    ) -> Result<Self, NetworkError> {
        stream.set_write_timeout(Some(timeout))?;
        let mut read_stream = stream.try_clone()?;
        read_stream.set_read_timeout(None)?;
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let reader_pending = Arc::clone(&pending);
        let reader_failure = Arc::clone(&failure);
        let reader = thread::Builder::new()
            .name(format!("gx1-p2p-rank-{rank}-rail-{rail}"))
            .spawn(move || {
                run_p2p_client_reader(
                    &mut read_stream,
                    unique_id,
                    rank,
                    world_size,
                    &reader_pending,
                    &reader_failure,
                );
            })
            .map_err(|error| {
                NetworkError::InvalidConfiguration(format!(
                    "cannot start point-to-point response reader: {error}"
                ))
            })?;
        Ok(Self {
            submission: Mutex::new(P2pSubmission {
                stream,
                next_send_sequence: 0,
                next_receive_sequence: 0,
                next_heartbeat_sequence: 0,
            }),
            pending,
            failure,
            reader: Mutex::new(Some(reader)),
            unique_id,
            rank,
            world_size,
            timeout: Mutex::new(timeout),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn request(
        &self,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        opcode: Opcode,
        peer: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<Frame, NetworkError> {
        let timeout = *self.timeout.lock().map_err(|_| NetworkError::Poisoned)?;
        self.request_with_timeout(
            unique_id,
            rank,
            world_size,
            opcode,
            peer,
            tag,
            element_type,
            element_count,
            payload,
            timeout,
        )
    }

    fn set_timeout(&self, timeout: Duration) -> Result<(), NetworkError> {
        if timeout.is_zero() {
            return Err(NetworkError::InvalidConfiguration(
                "point-to-point timeout must be greater than zero".into(),
            ));
        }
        self.submission
            .lock()
            .map_err(|_| NetworkError::Poisoned)?
            .stream
            .set_write_timeout(Some(timeout))?;
        *self.timeout.lock().map_err(|_| NetworkError::Poisoned)? = timeout;
        Ok(())
    }

    fn abort(&self, message: &str) -> Result<(), NetworkError> {
        let mut submission = self.submission.lock().map_err(|_| NetworkError::Poisoned)?;
        let sequence = submission.next_send_sequence;
        submission.next_send_sequence = sequence
            .checked_add(1)
            .ok_or(ProtocolError::SequenceOverflow)?;
        let payload = message.as_bytes().to_vec();
        let mut header = FrameHeader::collective(
            self.unique_id,
            Opcode::Abort,
            ElementType::U8,
            self.rank,
            ANY_RANK,
            self.world_size,
            sequence,
            payload.len() as u64,
        );
        header.destination_rank = ANY_RANK;
        write_frame(&mut submission.stream, &Frame::new(header, payload)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn request_with_timeout(
        &self,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        opcode: Opcode,
        peer: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
        response_timeout: Duration,
    ) -> Result<Frame, NetworkError> {
        let (response_sender, response_receiver) = mpsc::channel();
        let key;
        {
            let mut submission = self.submission.lock().map_err(|_| NetworkError::Poisoned)?;
            let failure = self.failure.lock().map_err(|_| NetworkError::Poisoned)?;
            if let Some(message) = failure.as_deref() {
                return Err(NetworkError::RemoteAbort(message.to_owned()));
            }
            let sequence = match opcode {
                Opcode::Send => &mut submission.next_send_sequence,
                Opcode::Receive => &mut submission.next_receive_sequence,
                Opcode::Heartbeat => &mut submission.next_heartbeat_sequence,
                _ => {
                    return Err(NetworkError::InvalidConfiguration(format!(
                        "{opcode:?} is not a point-to-point request"
                    )));
                }
            };
            let current_sequence = *sequence;
            *sequence = sequence
                .checked_add(1)
                .ok_or(ProtocolError::SequenceOverflow)?;
            key = P2pResponseKey {
                opcode,
                sequence: current_sequence,
            };
            drop(failure);
            let mut header = FrameHeader::collective(
                unique_id,
                opcode,
                element_type,
                rank,
                ANY_RANK,
                world_size,
                current_sequence,
                element_count,
            );
            header.destination_rank = peer;
            header.tag = tag;
            let frame = Frame::new(header, payload)?;
            self.pending
                .lock()
                .map_err(|_| NetworkError::Poisoned)?
                .insert(key, response_sender);
            if let Err(error) = write_frame(&mut submission.stream, &frame) {
                fail_pending(&self.pending, &self.failure, &error.to_string());
                return Err(error);
            }
        }
        match response_receiver.recv_timeout(response_timeout) {
            Ok(Ok(frame)) => Ok(frame),
            Ok(Err(message)) => Err(NetworkError::RemoteAbort(message)),
            Err(RecvTimeoutError::Timeout) => {
                if let Ok(mut pending) = self.pending.lock() {
                    pending.remove(&key);
                }
                if let Ok(submission) = self.submission.lock() {
                    let _ = submission.stream.shutdown(Shutdown::Both);
                }
                Err(NetworkError::Timeout(format!(
                    "point-to-point {opcode:?} with tag {tag}"
                )))
            }
            Err(RecvTimeoutError::Disconnected) => Err(NetworkError::ChannelClosed),
        }
    }
}

impl Drop for P2pClient {
    fn drop(&mut self) {
        if let Ok(mut submission) = self.submission.lock() {
            let header = FrameHeader::collective(
                self.unique_id,
                Opcode::Leave,
                ElementType::None,
                self.rank,
                ANY_RANK,
                self.world_size,
                0,
                0,
            );
            if let Ok(frame) = Frame::new(header, Vec::new()) {
                let _ = write_frame(&mut submission.stream, &frame);
            }
            let _ = submission.stream.shutdown(Shutdown::Both);
        }
        if let Ok(reader) = self.reader.get_mut()
            && let Some(reader) = reader.take()
        {
            let _ = reader.join();
        }
    }
}

impl TcpRankSession {
    pub fn connect(
        address: impl ToSocketAddrs,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        timeout: Duration,
    ) -> Result<Self, NetworkError> {
        Self::connect_with_transport(
            address,
            unique_id,
            rank,
            world_size,
            timeout,
            p2p_rails_from_environment()?,
            tcp_transport_from_environment()?,
        )
    }

    pub fn connect_with_p2p_rails(
        address: impl ToSocketAddrs,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        timeout: Duration,
        p2p_rails: usize,
    ) -> Result<Self, NetworkError> {
        Self::connect_with_transport(
            address,
            unique_id,
            rank,
            world_size,
            timeout,
            p2p_rails,
            tcp_transport_from_environment()?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn connect_with_transport(
        address: impl ToSocketAddrs,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        timeout: Duration,
        p2p_rails: usize,
        transport: CollectiveTransport,
    ) -> Result<Self, NetworkError> {
        Self::connect_with_transport_configuration(
            address, unique_id, rank, world_size, timeout, p2p_rails, transport, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn connect_with_transport_configuration(
        address: impl ToSocketAddrs,
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        timeout: Duration,
        p2p_rails: usize,
        transport: CollectiveTransport,
        peer_endpoint_configuration: Option<PeerEndpointConfiguration>,
    ) -> Result<Self, NetworkError> {
        if world_size == 0 || rank >= world_size {
            return Err(NetworkError::InvalidConfiguration(format!(
                "rank {rank} is outside TCP collective world size {world_size}"
            )));
        }
        validate_p2p_rails(p2p_rails)?;
        validate_tcp_transport(transport)?;
        let addresses = address.to_socket_addrs()?.collect::<Vec<_>>();
        let mut stream =
            connect_rank_channel(&addresses, unique_id, rank, world_size, timeout, false, 0)?;
        let failure = Arc::new(Mutex::new(None));
        let p2p = match transport {
            CollectiveTransport::TcpHostStaged => {
                let mut clients = Vec::with_capacity(p2p_rails);
                for rail in 0..p2p_rails {
                    let p2p_stream = connect_rank_channel(
                        &addresses, unique_id, rank, world_size, timeout, true, rail,
                    )?;
                    clients.push(P2pClient::new(
                        p2p_stream,
                        unique_id,
                        rank,
                        world_size,
                        timeout,
                        rail,
                        Arc::clone(&failure),
                    )?);
                }
                P2pDataPlane::Coordinator(clients)
            }
            CollectiveTransport::TcpPeer => {
                let (listeners, rank_endpoints) = match peer_endpoint_configuration {
                    Some(configuration) => {
                        bind_peer_listeners_with_configuration(configuration, p2p_rails)?
                    }
                    None => bind_peer_listeners(&stream, p2p_rails)?,
                };
                let endpoints = exchange_peer_endpoints_client(
                    &mut stream,
                    &rank_endpoints,
                    unique_id,
                    rank,
                    world_size,
                    p2p_rails,
                )?;
                let control_stream = connect_rank_channel(
                    &addresses, unique_id, rank, world_size, timeout, true, 0,
                )?;
                let control = P2pClient::new(
                    control_stream,
                    unique_id,
                    rank,
                    world_size,
                    timeout,
                    0,
                    Arc::clone(&failure),
                )?;
                let mesh = match DirectPeerMesh::connect(
                    listeners, &endpoints, unique_id, rank, world_size, p2p_rails, timeout, failure,
                ) {
                    Ok(mesh) => Box::new(mesh),
                    Err(error) => {
                        let _ = control.abort(&format!(
                            "rank {rank} direct peer mesh setup failed: {error}"
                        ));
                        return Err(error);
                    }
                };
                P2pDataPlane::Peer { control, mesh }
            }
            CollectiveTransport::HostStaged
            | CollectiveTransport::PciePeer
            | CollectiveTransport::Rdma
            | CollectiveTransport::GxLink => unreachable!("validated TCP transport"),
        };
        stream.set_read_timeout(Some(timeout.saturating_add(COLLECTIVE_RESPONSE_GRACE)))?;
        stream.set_write_timeout(Some(timeout))?;
        Ok(Self {
            unique_id,
            rank,
            world_size,
            transport,
            inner: Mutex::new(TcpSessionInner {
                stream,
                next_sequence: 0,
            }),
            p2p,
        })
    }

    pub const fn rank(&self) -> u32 {
        self.rank
    }

    pub const fn world_size(&self) -> u32 {
        self.world_size
    }

    pub const fn transport(&self) -> CollectiveTransport {
        self.transport
    }

    pub fn p2p_rails(&self) -> usize {
        match &self.p2p {
            P2pDataPlane::Coordinator(rails) => rails.len(),
            P2pDataPlane::Peer { mesh, .. } => mesh.rails(),
        }
    }

    fn p2p_rail(&self, rail: usize) -> Result<&P2pClient, NetworkError> {
        let P2pDataPlane::Coordinator(rails) = &self.p2p else {
            return Err(NetworkError::InvalidConfiguration(
                "coordinator P2P rail requested for a direct peer transport".into(),
            ));
        };
        rails.get(rail).ok_or_else(|| {
            NetworkError::InvalidConfiguration(format!(
                "point-to-point rail {rail} is outside 0..{}",
                rails.len()
            ))
        })
    }

    fn control(&self) -> Result<&P2pClient, NetworkError> {
        match &self.p2p {
            P2pDataPlane::Coordinator(rails) => rails.first().ok_or_else(|| {
                NetworkError::InvalidConfiguration("point-to-point control rail is absent".into())
            }),
            P2pDataPlane::Peer { control, .. } => Ok(control),
        }
    }

    pub fn probe_topology_from_environment(&self) -> Result<CollectiveTopology, NetworkError> {
        self.probe_topology(TopologyProbeOptions::from_environment()?)
    }

    pub fn probe_topology(
        &self,
        options: TopologyProbeOptions,
    ) -> Result<CollectiveTopology, NetworkError> {
        options.validate()?;
        if self.transport != CollectiveTransport::TcpPeer {
            return Err(NetworkError::InvalidConfiguration(
                "automatic link probing requires GX1_COLLECTIVE_TRANSPORT=tcp_peer".into(),
            ));
        }
        let mut topology = CollectiveTopology::empty(self.world_size).map_err(|error| {
            NetworkError::InvalidConfiguration(format!(
                "cannot create probed collective topology: {error}"
            ))
        })?;
        self.barrier()?;
        for first_rank in 0..self.world_size {
            for second_rank in first_rank + 1..self.world_size {
                let mut pair_bandwidth_mbps = u64::MAX;
                let mut pair_latency_ns = 0_u64;
                let mut pair_rail_bandwidth_sum_mbps = 0_u64;
                for rail in 0..self.p2p_rails() {
                    let forward = self.probe_direction(
                        first_rank,
                        second_rank,
                        first_rank,
                        second_rank,
                        rail,
                        0,
                        options,
                    )?;
                    let reverse = self.probe_direction(
                        first_rank,
                        second_rank,
                        second_rank,
                        first_rank,
                        rail,
                        1,
                        options,
                    )?;
                    let mut local = [0_u64; 4];
                    if self.rank == first_rank {
                        let probe = forward.expect("forward sender owns its link measurement");
                        local[0] = probe.bandwidth_mbps;
                        local[1] = probe.latency_ns;
                    }
                    if self.rank == second_rank {
                        let probe = reverse.expect("reverse sender owns its link measurement");
                        local[2] = probe.bandwidth_mbps;
                        local[3] = probe.latency_ns;
                    }
                    let payload = local
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                        .collect::<Vec<_>>();
                    let gathered =
                        self.exchange(Opcode::AllGather, ElementType::U64, ANY_RANK, 4, payload)?;
                    let values = gathered
                        .payload
                        .chunks_exact(8)
                        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let expected_values = self.world_size as usize * 4;
                    if values.len() != expected_values {
                        return Err(NetworkError::InvalidConfiguration(format!(
                            "topology probe gathered {} metrics, expected {expected_values}",
                            values.len()
                        )));
                    }
                    let forward_offset = first_rank as usize * 4;
                    let reverse_offset = second_rank as usize * 4;
                    let forward = LinkProbe {
                        bandwidth_mbps: values[forward_offset],
                        latency_ns: values[forward_offset + 1],
                    };
                    let reverse = LinkProbe {
                        bandwidth_mbps: values[reverse_offset + 2],
                        latency_ns: values[reverse_offset + 3],
                    };
                    if forward.bandwidth_mbps == 0 || reverse.bandwidth_mbps == 0 {
                        return Err(NetworkError::InvalidConfiguration(format!(
                            "topology probe for ranks {first_rank}-{second_rank} rail {rail} returned zero bandwidth"
                        )));
                    }
                    let bandwidth_mbps = conservative_probe_bandwidth_bucket(
                        forward.bandwidth_mbps.min(reverse.bandwidth_mbps),
                    );
                    let latency_ns = conservative_probe_latency_bucket(
                        forward.latency_ns.max(reverse.latency_ns),
                    );
                    topology
                        .add_rail_link(TopologyRailLink {
                            rail,
                            first_rank,
                            second_rank,
                            bandwidth_mbps,
                            latency_ns,
                        })
                        .map_err(|error| {
                            NetworkError::InvalidConfiguration(format!(
                                "cannot add probed topology rail link {rail}@{first_rank}-{second_rank}: {error}"
                            ))
                        })?;
                    pair_bandwidth_mbps = pair_bandwidth_mbps.min(bandwidth_mbps);
                    pair_rail_bandwidth_sum_mbps =
                        pair_rail_bandwidth_sum_mbps.saturating_add(bandwidth_mbps);
                    pair_latency_ns = pair_latency_ns.max(latency_ns);
                }
                if self.p2p_rails() > 1 {
                    let forward = self.probe_concurrent_rails_direction(
                        first_rank,
                        second_rank,
                        first_rank,
                        second_rank,
                        0,
                        options,
                    )?;
                    let reverse = self.probe_concurrent_rails_direction(
                        first_rank,
                        second_rank,
                        second_rank,
                        first_rank,
                        1,
                        options,
                    )?;
                    let mut local = [0_u64; 2];
                    if self.rank == first_rank {
                        local[0] =
                            forward.expect("forward sender owns its aggregate link measurement");
                    }
                    if self.rank == second_rank {
                        local[1] =
                            reverse.expect("reverse sender owns its aggregate link measurement");
                    }
                    let payload = local
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                        .collect::<Vec<_>>();
                    let gathered =
                        self.exchange(Opcode::AllGather, ElementType::U64, ANY_RANK, 2, payload)?;
                    let values = gathered
                        .payload
                        .chunks_exact(8)
                        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let expected_values = self.world_size as usize * 2;
                    if values.len() != expected_values {
                        return Err(NetworkError::InvalidConfiguration(format!(
                            "topology aggregate probe gathered {} metrics, expected {expected_values}",
                            values.len()
                        )));
                    }
                    let forward = values[first_rank as usize * 2];
                    let reverse = values[second_rank as usize * 2 + 1];
                    if forward == 0 || reverse == 0 {
                        return Err(NetworkError::InvalidConfiguration(format!(
                            "topology aggregate probe for ranks {first_rank}-{second_rank} returned zero bandwidth"
                        )));
                    }
                    let bandwidth_mbps = conservative_probe_bandwidth_bucket(forward.min(reverse))
                        .min(pair_rail_bandwidth_sum_mbps)
                        .max(1);
                    topology
                        .add_aggregate_link(TopologyAggregateLink {
                            first_rank,
                            second_rank,
                            bandwidth_mbps,
                        })
                        .map_err(|error| {
                            NetworkError::InvalidConfiguration(format!(
                                "cannot add probed topology aggregate link {first_rank}-{second_rank}: {error}"
                            ))
                        })?;
                }
                topology
                    .add_link(TopologyLink {
                        first_rank,
                        second_rank,
                        bandwidth_mbps: pair_bandwidth_mbps,
                        latency_ns: pair_latency_ns,
                    })
                    .map_err(|error| {
                        NetworkError::InvalidConfiguration(format!(
                            "cannot add probed topology link {first_rank}-{second_rank}: {error}"
                        ))
                    })?;
            }
        }
        Ok(topology)
    }

    #[allow(clippy::too_many_arguments)]
    fn probe_direction(
        &self,
        first_rank: u32,
        second_rank: u32,
        sender_rank: u32,
        receiver_rank: u32,
        rail: usize,
        direction: u64,
        options: TopologyProbeOptions,
    ) -> Result<Option<LinkProbe>, NetworkError> {
        let latency_tag = topology_probe_tag(first_rank, second_rank, rail, direction, 0);
        let bandwidth_tag = topology_probe_tag(first_rank, second_rank, rail, direction, 1);
        let mut latency_samples = Vec::with_capacity(options.latency_iterations);
        let mut bandwidth_samples = Vec::with_capacity(options.bandwidth_iterations);
        if self.rank == sender_rank {
            for _ in 0..options.warmup_iterations {
                self.send_on_rail(
                    rail,
                    receiver_rank,
                    latency_tag,
                    ElementType::U8,
                    1,
                    vec![0],
                )?;
            }
            for _ in 0..options.latency_iterations {
                let started = Instant::now();
                self.send_on_rail(
                    rail,
                    receiver_rank,
                    latency_tag,
                    ElementType::U8,
                    1,
                    vec![0],
                )?;
                latency_samples.push(duration_ns(started.elapsed()));
            }
            let probe_payload = vec![0xa5; options.payload_bytes];
            for _ in 0..options.warmup_iterations {
                self.send_on_rail(
                    rail,
                    receiver_rank,
                    bandwidth_tag,
                    ElementType::U8,
                    options.payload_bytes as u64,
                    probe_payload.clone(),
                )?;
            }
            for _ in 0..options.bandwidth_iterations {
                let payload = probe_payload.clone();
                let started = Instant::now();
                self.send_on_rail(
                    rail,
                    receiver_rank,
                    bandwidth_tag,
                    ElementType::U8,
                    options.payload_bytes as u64,
                    payload,
                )?;
                bandwidth_samples.push(duration_ns(started.elapsed()));
            }
        } else if self.rank == receiver_rank {
            for _ in 0..options.warmup_iterations + options.latency_iterations {
                self.receive_on_rail(rail, Some(sender_rank), latency_tag, ElementType::U8, 1)?;
            }
            for _ in 0..options.warmup_iterations + options.bandwidth_iterations {
                self.receive_on_rail(
                    rail,
                    Some(sender_rank),
                    bandwidth_tag,
                    ElementType::U8,
                    options.payload_bytes as u64,
                )?;
            }
        }
        self.barrier()?;
        if self.rank != sender_rank {
            return Ok(None);
        }
        let latency_ns = median(&mut latency_samples).saturating_add(1) / 2;
        let elapsed_ns = median(&mut bandwidth_samples);
        let transfer_ns = elapsed_ns
            .saturating_sub(latency_ns.saturating_mul(2))
            .max(1);
        let bandwidth_mbps = u64::try_from(
            (options.payload_bytes as u128)
                .saturating_mul(1_000)
                .checked_div(u128::from(transfer_ns))
                .unwrap_or(0),
        )
        .unwrap_or(u64::MAX)
        .max(1);
        Ok(Some(LinkProbe {
            bandwidth_mbps,
            latency_ns: latency_ns.max(1),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn probe_concurrent_rails_direction(
        &self,
        first_rank: u32,
        second_rank: u32,
        sender_rank: u32,
        receiver_rank: u32,
        direction: u64,
        options: TopologyProbeOptions,
    ) -> Result<Option<u64>, NetworkError> {
        let rails = self.p2p_rails();
        if rails <= 1 {
            return Err(NetworkError::InvalidConfiguration(
                "concurrent topology probing requires at least two P2P rails".into(),
            ));
        }
        let samples = options
            .warmup_iterations
            .saturating_add(options.bandwidth_iterations);
        let mut bandwidth_samples = Vec::with_capacity(options.bandwidth_iterations);
        if self.rank == sender_rank {
            for sample in 0..samples {
                let elapsed_ns = self.send_concurrent_probe_sample(
                    first_rank,
                    second_rank,
                    receiver_rank,
                    direction,
                    options.payload_bytes,
                )?;
                if sample >= options.warmup_iterations {
                    bandwidth_samples.push(elapsed_ns);
                }
            }
        } else if self.rank == receiver_rank {
            for _ in 0..samples {
                for rail in 0..rails {
                    self.receive_on_rail(
                        rail,
                        Some(sender_rank),
                        topology_probe_tag(first_rank, second_rank, rail, direction, 2),
                        ElementType::U8,
                        options.payload_bytes as u64,
                    )?;
                }
            }
        }
        self.barrier()?;
        if self.rank != sender_rank {
            return Ok(None);
        }
        let elapsed_ns = median(&mut bandwidth_samples).max(1);
        let total_bytes = (options.payload_bytes as u128).saturating_mul(rails as u128);
        let bandwidth_mbps = u64::try_from(
            total_bytes
                .saturating_mul(1_000)
                .checked_div(u128::from(elapsed_ns))
                .unwrap_or(0),
        )
        .unwrap_or(u64::MAX)
        .max(1);
        Ok(Some(bandwidth_mbps))
    }

    fn send_concurrent_probe_sample(
        &self,
        first_rank: u32,
        second_rank: u32,
        receiver_rank: u32,
        direction: u64,
        payload_bytes: usize,
    ) -> Result<u64, NetworkError> {
        let rails = self.p2p_rails();
        thread::scope(|scope| {
            let start = Arc::new(Barrier::new(rails + 1));
            let mut workers = Vec::with_capacity(rails);
            for rail in 0..rails {
                let start = Arc::clone(&start);
                let payload = vec![0x5a; payload_bytes];
                workers.push(scope.spawn(move || {
                    start.wait();
                    self.send_on_rail(
                        rail,
                        receiver_rank,
                        topology_probe_tag(first_rank, second_rank, rail, direction, 2),
                        ElementType::U8,
                        payload_bytes as u64,
                        payload,
                    )
                }));
            }
            let started = Instant::now();
            start.wait();
            let mut first_error = None;
            for worker in workers {
                match worker.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) if first_error.is_none() => first_error = Some(error),
                    Ok(Err(_)) => {}
                    Err(_) if first_error.is_none() => {
                        first_error = Some(NetworkError::InvalidConfiguration(
                            "concurrent topology probe worker panicked".into(),
                        ));
                    }
                    Err(_) => {}
                }
            }
            match first_error {
                Some(error) => Err(error),
                None => Ok(duration_ns(started.elapsed())),
            }
        })
    }

    pub fn heartbeat(&self, timeout: Duration) -> Result<Duration, NetworkError> {
        if timeout.is_zero() {
            return Err(NetworkError::InvalidConfiguration(
                "heartbeat timeout must be greater than zero".into(),
            ));
        }
        let started = Instant::now();
        let response = self.control()?.request_with_timeout(
            self.unique_id,
            self.rank,
            self.world_size,
            Opcode::Heartbeat,
            ANY_RANK,
            0,
            ElementType::None,
            0,
            Vec::new(),
            timeout,
        )?;
        if response.header.opcode != Opcode::Heartbeat || !response.payload.is_empty() {
            return Err(NetworkError::InvalidConfiguration(
                "invalid heartbeat response".into(),
            ));
        }
        Ok(started.elapsed())
    }

    pub fn send(
        &self,
        destination: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<(), NetworkError> {
        let rail = (tag % self.p2p_rails() as u64) as usize;
        self.send_on_rail(rail, destination, tag, element_type, element_count, payload)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn send_on_rail(
        &self,
        rail: usize,
        destination: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<(), NetworkError> {
        if destination >= self.world_size {
            return Err(ProtocolError::RankOutOfRange {
                name: "destination",
                rank: destination,
                world_size: self.world_size,
            }
            .into());
        }
        match &self.p2p {
            P2pDataPlane::Coordinator(_) => {
                let response = self.p2p_rail(rail)?.request(
                    self.unique_id,
                    self.rank,
                    self.world_size,
                    Opcode::Send,
                    destination,
                    tag,
                    element_type,
                    element_count,
                    payload,
                )?;
                if response.header.opcode != Opcode::Send || !response.payload.is_empty() {
                    return Err(NetworkError::InvalidConfiguration(
                        "invalid point-to-point send acknowledgement".into(),
                    ));
                }
                Ok(())
            }
            P2pDataPlane::Peer { control, mesh } => {
                let result =
                    mesh.send_on_rail(rail, destination, tag, element_type, element_count, payload);
                if let Err(error) = &result {
                    let message = error.to_string();
                    let _ = mesh.abort(&message);
                    let _ = control.abort(&message);
                }
                result
            }
        }
    }

    pub fn receive(
        &self,
        source: Option<u32>,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
    ) -> Result<Frame, NetworkError> {
        let rail = (tag % self.p2p_rails() as u64) as usize;
        self.receive_on_rail(rail, source, tag, element_type, element_count)
    }

    pub fn receive_on_rail(
        &self,
        rail: usize,
        source: Option<u32>,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
    ) -> Result<Frame, NetworkError> {
        if let Some(source) = source
            && source >= self.world_size
        {
            return Err(ProtocolError::RankOutOfRange {
                name: "source",
                rank: source,
                world_size: self.world_size,
            }
            .into());
        }
        match &self.p2p {
            P2pDataPlane::Coordinator(_) => {
                let response = self.p2p_rail(rail)?.request(
                    self.unique_id,
                    self.rank,
                    self.world_size,
                    Opcode::Receive,
                    source.unwrap_or(ANY_RANK),
                    tag,
                    element_type,
                    element_count,
                    Vec::new(),
                )?;
                if response.header.opcode != Opcode::Receive {
                    return Err(NetworkError::InvalidConfiguration(
                        "invalid point-to-point receive response".into(),
                    ));
                }
                Ok(response)
            }
            P2pDataPlane::Peer { control, mesh } => {
                let result = mesh.receive_on_rail(rail, source, tag, element_type, element_count);
                if let Err(error) = &result {
                    let message = error.to_string();
                    let _ = mesh.abort(&message);
                    let _ = control.abort(&message);
                }
                result
            }
        }
    }

    pub fn exchange(
        &self,
        opcode: Opcode,
        element_type: ElementType,
        root_rank: u32,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<Frame, NetworkError> {
        self.exchange_with_options(
            opcode,
            element_type,
            ExchangeOptions::new(root_rank, element_count),
            payload,
        )
    }

    pub fn exchange_with_options(
        &self,
        opcode: Opcode,
        element_type: ElementType,
        options: ExchangeOptions,
        payload: Vec<u8>,
    ) -> Result<Frame, NetworkError> {
        let mut inner = self.inner.lock().map_err(|_| NetworkError::Poisoned)?;
        let sequence = inner.next_sequence;
        let mut header = FrameHeader::collective(
            self.unique_id,
            opcode,
            element_type,
            self.rank,
            options.root_rank,
            self.world_size,
            sequence,
            options.element_count,
        );
        header.flags |= options.flags;
        header.tag = options.tag;
        let request = Frame::new(header, payload)?;
        write_frame(&mut inner.stream, &request)?;
        let response = read_frame(&mut inner.stream)?.ok_or_else(|| {
            NetworkError::InvalidConfiguration(
                "coordinator closed before collective response".into(),
            )
        })?;
        validate_response(
            &response,
            self.unique_id,
            self.rank,
            self.world_size,
            sequence,
        )?;
        if response.header.opcode == Opcode::Abort {
            return Err(NetworkError::RemoteAbort(
                String::from_utf8_lossy(&response.payload).into_owned(),
            ));
        }
        inner.next_sequence = inner
            .next_sequence
            .checked_add(1)
            .ok_or(ProtocolError::SequenceOverflow)?;
        Ok(response)
    }

    pub fn barrier(&self) -> Result<(), NetworkError> {
        let response =
            self.exchange(Opcode::Barrier, ElementType::None, ANY_RANK, 0, Vec::new())?;
        if response.header.opcode != Opcode::Barrier || !response.payload.is_empty() {
            return Err(NetworkError::InvalidConfiguration(
                "invalid barrier response".into(),
            ));
        }
        Ok(())
    }

    pub fn set_timeout(&self, timeout: Duration) -> Result<(), NetworkError> {
        if timeout.is_zero() {
            return Err(NetworkError::InvalidConfiguration(
                "collective timeout must be greater than zero".into(),
            ));
        }
        let timeout_ms = u64::try_from(timeout.as_millis().max(1)).map_err(|_| {
            NetworkError::InvalidConfiguration("collective timeout exceeds u64 milliseconds".into())
        })?;
        let mut inner = self.inner.lock().map_err(|_| NetworkError::Poisoned)?;
        let sequence = inner.next_sequence;
        let mut header = FrameHeader::collective(
            self.unique_id,
            Opcode::SetTimeout,
            ElementType::None,
            self.rank,
            ANY_RANK,
            self.world_size,
            sequence,
            0,
        );
        header.tag = timeout_ms;
        write_frame(&mut inner.stream, &Frame::new(header, Vec::new())?)?;
        let response = read_frame(&mut inner.stream)?.ok_or_else(|| {
            NetworkError::InvalidConfiguration(
                "coordinator closed before SET_TIMEOUT response".into(),
            )
        })?;
        validate_response(
            &response,
            self.unique_id,
            self.rank,
            self.world_size,
            sequence,
        )?;
        if response.header.opcode == Opcode::Abort {
            return Err(NetworkError::RemoteAbort(
                String::from_utf8_lossy(&response.payload).into_owned(),
            ));
        }
        if response.header.opcode != Opcode::SetTimeout || !response.payload.is_empty() {
            return Err(NetworkError::InvalidConfiguration(
                "invalid SET_TIMEOUT response".into(),
            ));
        }
        inner.next_sequence = inner
            .next_sequence
            .checked_add(1)
            .ok_or(ProtocolError::SequenceOverflow)?;
        inner
            .stream
            .set_read_timeout(Some(timeout.saturating_add(COLLECTIVE_RESPONSE_GRACE)))?;
        inner.stream.set_write_timeout(Some(timeout))?;
        match &self.p2p {
            P2pDataPlane::Coordinator(rails) => {
                for rail in rails {
                    rail.set_timeout(timeout)?;
                }
            }
            P2pDataPlane::Peer { control, mesh } => {
                control.set_timeout(timeout)?;
                mesh.set_timeout(timeout)?;
            }
        }
        Ok(())
    }

    /// Notify every rank that this process is abandoning the current session.
    /// The abort uses the independent P2P channel so it can interrupt a rank
    /// waiting in the ordered collective channel.
    pub fn abort(&self, message: &str) -> Result<(), NetworkError> {
        let mut first_error = None;
        match &self.p2p {
            P2pDataPlane::Coordinator(rails) => {
                for rail in rails {
                    if let Err(error) = rail.abort(message)
                        && first_error.is_none()
                    {
                        first_error = Some(error);
                    }
                }
            }
            P2pDataPlane::Peer { control, mesh } => {
                if let Err(error) = control.abort(message) {
                    first_error = Some(error);
                }
                if let Err(error) = mesh.abort(message)
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl super::RankTransport for TcpRankSession {
    fn rank(&self) -> u32 {
        Self::rank(self)
    }

    fn world_size(&self) -> u32 {
        Self::world_size(self)
    }

    fn transport(&self) -> CollectiveTransport {
        Self::transport(self)
    }

    fn p2p_rails(&self) -> usize {
        Self::p2p_rails(self)
    }

    fn heartbeat(&self, timeout: Duration) -> Result<Duration, NetworkError> {
        Self::heartbeat(self, timeout)
    }

    fn send_on_rail(
        &self,
        rail: usize,
        destination: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<(), NetworkError> {
        Self::send_on_rail(
            self,
            rail,
            destination,
            tag,
            element_type,
            element_count,
            payload,
        )
    }

    fn receive_on_rail(
        &self,
        rail: usize,
        source: Option<u32>,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
    ) -> Result<Frame, NetworkError> {
        Self::receive_on_rail(self, rail, source, tag, element_type, element_count)
    }

    fn exchange_with_options(
        &self,
        opcode: Opcode,
        element_type: ElementType,
        options: ExchangeOptions,
        payload: Vec<u8>,
    ) -> Result<Frame, NetworkError> {
        Self::exchange_with_options(self, opcode, element_type, options, payload)
    }

    fn set_timeout(&self, timeout: Duration) -> Result<(), NetworkError> {
        Self::set_timeout(self, timeout)
    }

    fn abort(&self, message: &str) -> Result<(), NetworkError> {
        Self::abort(self, message)
    }
}

fn connect_rank_channel(
    addresses: &[SocketAddr],
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    timeout: Duration,
    p2p: bool,
    rail: usize,
) -> Result<TcpStream, NetworkError> {
    let mut last_error = None;
    let mut stream = None;
    for address in addresses {
        match TcpStream::connect_timeout(address, timeout) {
            Ok(connected) => {
                stream = Some(connected);
                break;
            }
            Err(error) => last_error = Some(error),
        }
    }
    let mut stream = stream.ok_or_else(|| {
        NetworkError::Io(last_error.unwrap_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "no rendezvous address resolved",
            )
        }))
    })?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Join,
        ElementType::None,
        rank,
        ANY_RANK,
        world_size,
        0,
        0,
    );
    if p2p {
        header.flags |= FLAG_P2P_CHANNEL;
        header.tag = rail as u64;
    }
    write_frame(&mut stream, &Frame::new(header, Vec::new())?)?;
    let ready = read_frame(&mut stream)?.ok_or_else(|| {
        NetworkError::InvalidConfiguration("coordinator closed before READY".into())
    })?;
    validate_response(&ready, unique_id, rank, world_size, 0)?;
    if ready.header.opcode != Opcode::Ready
        || (ready.header.flags & FLAG_P2P_CHANNEL != 0) != p2p
        || ready.header.tag != if p2p { rail as u64 } else { 0 }
    {
        return Err(NetworkError::InvalidConfiguration(format!(
            "coordinator answered JOIN on the wrong channel/rail with {:?} tag {}",
            ready.header.opcode, ready.header.tag
        )));
    }
    Ok(stream)
}

fn run_p2p_client_reader(
    stream: &mut TcpStream,
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    pending: &Arc<Mutex<HashMap<P2pResponseKey, PendingP2pResponse>>>,
    failure: &Arc<Mutex<Option<String>>>,
) {
    loop {
        let frame = match read_frame(stream) {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                fail_pending(pending, failure, "point-to-point coordinator closed");
                return;
            }
            Err(error) => {
                fail_pending(
                    pending,
                    failure,
                    &format!("point-to-point read failed: {error}"),
                );
                return;
            }
        };
        if frame.header.unique_id != unique_id
            || frame.header.world_size != world_size
            || frame.header.destination_rank != rank
        {
            fail_pending(pending, failure, "invalid point-to-point response envelope");
            return;
        }
        if frame.header.opcode == Opcode::Abort {
            fail_pending(pending, failure, &String::from_utf8_lossy(&frame.payload));
            return;
        }
        if !matches!(
            frame.header.opcode,
            Opcode::Send | Opcode::Receive | Opcode::Heartbeat
        ) {
            fail_pending(
                pending,
                failure,
                &format!(
                    "unexpected {:?} response on point-to-point channel",
                    frame.header.opcode
                ),
            );
            return;
        }
        let key = P2pResponseKey {
            opcode: frame.header.opcode,
            sequence: frame.header.sequence,
        };
        let sender = match pending.lock() {
            Ok(mut pending) => pending.remove(&key),
            Err(_) => return,
        };
        if let Some(sender) = sender {
            let _ = sender.send(Ok(frame));
        }
    }
}

fn fail_pending(
    pending: &Arc<Mutex<HashMap<P2pResponseKey, PendingP2pResponse>>>,
    failure: &Arc<Mutex<Option<String>>>,
    message: &str,
) {
    if let Ok(mut failure) = failure.lock() {
        failure.get_or_insert_with(|| message.to_owned());
    }
    if let Ok(mut pending) = pending.lock() {
        for (_, sender) in pending.drain() {
            let _ = sender.send(Err(message.to_owned()));
        }
    }
}

fn validate_join(
    frame: &Frame,
    unique_id: UniqueId,
    world_size: usize,
    p2p: bool,
    rail: usize,
) -> Result<(), NetworkError> {
    if frame.header.opcode != Opcode::Join
        || frame.header.element_type != ElementType::None
        || !frame.payload.is_empty()
    {
        return Err(NetworkError::InvalidConfiguration(
            "first rank frame must be an empty JOIN".into(),
        ));
    }
    if frame.header.unique_id != unique_id {
        return Err(NetworkError::WrongSession);
    }
    if (frame.header.flags & FLAG_P2P_CHANNEL != 0) != p2p {
        return Err(NetworkError::InvalidConfiguration(
            "rank joined the wrong TCP channel".into(),
        ));
    }
    let expected_rail = if p2p { rail as u64 } else { 0 };
    if frame.header.tag != expected_rail {
        return Err(NetworkError::InvalidConfiguration(format!(
            "rank joined point-to-point rail {}, coordinator expects {expected_rail}",
            frame.header.tag
        )));
    }
    if frame.header.world_size as usize != world_size {
        return Err(NetworkError::InvalidConfiguration(format!(
            "rank declares world size {}, coordinator expects {world_size}",
            frame.header.world_size
        )));
    }
    Ok(())
}

fn p2p_rails_from_environment() -> Result<usize, NetworkError> {
    match env::var("GX1_P2P_RAILS") {
        Ok(value) => {
            let rails = value.trim().parse::<usize>().map_err(|_| {
                NetworkError::InvalidConfiguration(format!(
                    "GX1_P2P_RAILS must be an integer in 1..={MAX_P2P_RAILS}, got {value:?}"
                ))
            })?;
            validate_p2p_rails(rails)?;
            Ok(rails)
        }
        Err(env::VarError::NotPresent) => Ok(1),
        Err(env::VarError::NotUnicode(value)) => Err(NetworkError::InvalidConfiguration(format!(
            "GX1_P2P_RAILS is not Unicode: {:?}",
            value.to_string_lossy()
        ))),
    }
}

fn parse_probe_usize(name: &'static str, default: usize) -> Result<usize, NetworkError> {
    match env::var(name) {
        Ok(value) => value.trim().parse::<usize>().map_err(|_| {
            NetworkError::InvalidConfiguration(format!(
                "{name} must be a non-negative integer, got {value:?}"
            ))
        }),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(value)) => Err(NetworkError::InvalidConfiguration(format!(
            "{name} is not Unicode: {:?}",
            value.to_string_lossy()
        ))),
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn median(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn conservative_probe_bandwidth_bucket(value: u64) -> u64 {
    let exponent = 63_u32.saturating_sub(value.max(1).leading_zeros());
    1_u64 << (exponent & !1)
}

fn conservative_probe_latency_bucket(value: u64) -> u64 {
    let exponent = 63_u32.saturating_sub(value.max(1).leading_zeros());
    let upper_exponent = (exponent & !1).saturating_add(2);
    if upper_exponent >= u64::BITS {
        u64::MAX
    } else {
        1_u64 << upper_exponent
    }
}

fn topology_probe_tag(
    first_rank: u32,
    second_rank: u32,
    rail: usize,
    direction: u64,
    kind: u64,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in first_rank
        .to_le_bytes()
        .into_iter()
        .chain(second_rank.to_le_bytes())
        .chain((rail as u64).to_le_bytes())
        .chain(direction.to_le_bytes())
        .chain(kind.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    TOPOLOGY_PROBE_TAG_PREFIX | (hash & 0x0000_ffff_ffff_ffff)
}

fn tcp_transport_from_environment() -> Result<CollectiveTransport, NetworkError> {
    match env::var("GX1_COLLECTIVE_TRANSPORT") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "coordinator" | "tcp_coordinator" | "tcp_host_staged" => {
                Ok(CollectiveTransport::TcpHostStaged)
            }
            "peer" | "tcp_peer" => Ok(CollectiveTransport::TcpPeer),
            _ => Err(NetworkError::InvalidConfiguration(format!(
                "GX1_COLLECTIVE_TRANSPORT must be tcp_host_staged or tcp_peer, got {value:?}"
            ))),
        },
        Err(env::VarError::NotPresent) => Ok(CollectiveTransport::TcpHostStaged),
        Err(env::VarError::NotUnicode(value)) => Err(NetworkError::InvalidConfiguration(format!(
            "GX1_COLLECTIVE_TRANSPORT is not Unicode: {:?}",
            value.to_string_lossy()
        ))),
    }
}

fn validate_tcp_transport(transport: CollectiveTransport) -> Result<(), NetworkError> {
    match transport {
        CollectiveTransport::TcpHostStaged | CollectiveTransport::TcpPeer => Ok(()),
        CollectiveTransport::HostStaged
        | CollectiveTransport::PciePeer
        | CollectiveTransport::Rdma
        | CollectiveTransport::GxLink => Err(NetworkError::InvalidConfiguration(format!(
            "{transport:?} is not a TCP transport"
        ))),
    }
}

fn validate_p2p_rails(rails: usize) -> Result<(), NetworkError> {
    if (1..=MAX_P2P_RAILS).contains(&rails) {
        Ok(())
    } else {
        Err(NetworkError::InvalidConfiguration(format!(
            "point-to-point rail count {rails} is outside 1..={MAX_P2P_RAILS}"
        )))
    }
}

fn validate_response(
    frame: &Frame,
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    sequence: u64,
) -> Result<(), NetworkError> {
    if frame.header.unique_id != unique_id || frame.header.world_size != world_size {
        return Err(NetworkError::WrongSession);
    }
    if frame.header.destination_rank != rank {
        return Err(NetworkError::WrongDestination {
            expected: rank,
            actual: frame.header.destination_rank,
        });
    }
    if frame.header.sequence != sequence {
        return Err(ProtocolError::SequenceMismatch {
            expected: sequence,
            actual: frame.header.sequence,
        }
        .into());
    }
    Ok(())
}

#[derive(Debug)]
enum P2pEvent {
    Frame { rank: usize, frame: Frame },
    Closed { rank: usize },
    Failed { rank: usize, message: String },
}

fn run_p2p_loop(
    mut streams: Vec<TcpStream>,
    unique_id: UniqueId,
    world_size: usize,
    heartbeat_timeout: Option<Duration>,
    server_failure: SharedServerFailure,
) -> Result<(), NetworkError> {
    let (sender, receiver): (Sender<P2pEvent>, Receiver<P2pEvent>) = mpsc::channel();
    let mut readers = Vec::with_capacity(world_size);
    for (rank, stream) in streams.iter().enumerate() {
        let mut stream = stream.try_clone()?;
        stream.set_read_timeout(None)?;
        let sender = sender.clone();
        readers.push(
            thread::Builder::new()
                .name(format!("gx1-p2p-server-rank-{rank}"))
                .spawn(move || {
                    loop {
                        match read_frame(&mut stream) {
                            Ok(Some(frame)) => {
                                if sender.send(P2pEvent::Frame { rank, frame }).is_err() {
                                    return;
                                }
                            }
                            Ok(None) => {
                                let _ = sender.send(P2pEvent::Closed { rank });
                                return;
                            }
                            Err(error) => {
                                let _ = sender.send(P2pEvent::Failed {
                                    rank,
                                    message: error.to_string(),
                                });
                                return;
                            }
                        }
                    }
                })
                .map_err(|error| {
                    NetworkError::InvalidConfiguration(format!(
                        "cannot start point-to-point rank reader: {error}"
                    ))
                })?,
        );
    }
    drop(sender);
    let mut sends = VecDeque::<Frame>::new();
    let mut receives = VecDeque::<Frame>::new();
    let mut last_seen = vec![Instant::now(); world_size];
    let mut graceful = vec![false; world_size];
    let mut closed = vec![false; world_size];
    let poll_interval = heartbeat_timeout.map(|timeout| {
        (timeout / 4)
            .max(Duration::from_millis(1))
            .min(Duration::from_millis(250))
    });
    let result = loop {
        let event = match poll_interval {
            Some(interval) => match receiver.recv_timeout(interval) {
                Ok(event) => Some(event),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break Ok(()),
            },
            None => match receiver.recv() {
                Ok(event) => Some(event),
                Err(_) => break Ok(()),
            },
        };
        let routed = match event {
            Some(P2pEvent::Frame { rank, frame }) => {
                last_seen[rank] = Instant::now();
                if graceful[rank] {
                    Err(NetworkError::InvalidConfiguration(format!(
                        "point-to-point rank {rank} sent data after LEAVE"
                    )))
                } else {
                    let leaving = frame.header.opcode == Opcode::Leave;
                    let result = route_p2p_request(
                        &mut streams,
                        unique_id,
                        world_size,
                        rank,
                        frame,
                        &mut sends,
                        &mut receives,
                    );
                    if result.is_ok() && leaving {
                        graceful[rank] = true;
                    }
                    result
                }
            }
            Some(P2pEvent::Closed { rank }) => {
                closed[rank] = true;
                if heartbeat_timeout.is_some() && !graceful[rank] {
                    Err(NetworkError::InvalidConfiguration(format!(
                        "point-to-point rank {rank} disconnected without LEAVE"
                    )))
                } else {
                    Ok(())
                }
            }
            Some(P2pEvent::Failed { rank, message }) => {
                closed[rank] = true;
                if heartbeat_timeout.is_some() && !graceful[rank] {
                    Err(NetworkError::InvalidConfiguration(format!(
                        "point-to-point rank {rank} failed without LEAVE: {message}"
                    )))
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        };
        if let Err(error) = routed {
            let message = server_failure_text(&error);
            publish_server_failure(&server_failure, &message);
            broadcast_p2p_abort(&mut streams, unique_id, world_size, &message);
            break Err(error);
        }
        if closed.iter().all(|closed| *closed) {
            break Ok(());
        }
        if let Some(timeout) = heartbeat_timeout
            && let Some((rank, elapsed)) = last_seen
                .iter()
                .enumerate()
                .filter(|(rank, _)| !closed[*rank] && !graceful[*rank])
                .map(|(rank, last_seen)| (rank, last_seen.elapsed()))
                .find(|(_, elapsed)| *elapsed >= timeout)
        {
            let error = NetworkError::Timeout(format!(
                "heartbeat lease for rank {rank}; last contact {} ms ago",
                elapsed.as_millis()
            ));
            let message = server_failure_text(&error);
            publish_server_failure(&server_failure, &message);
            broadcast_p2p_abort(&mut streams, unique_id, world_size, &message);
            break Err(error);
        }
    };
    for stream in &streams {
        let _ = stream.shutdown(Shutdown::Both);
    }
    for reader in readers {
        let _ = reader.join();
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn route_p2p_request(
    streams: &mut [TcpStream],
    unique_id: UniqueId,
    world_size: usize,
    connection_rank: usize,
    frame: Frame,
    sends: &mut VecDeque<Frame>,
    receives: &mut VecDeque<Frame>,
) -> Result<(), NetworkError> {
    validate_p2p_request(&frame, unique_id, world_size, connection_rank)?;
    match frame.header.opcode {
        Opcode::Send => {
            let sender = frame.header.source_rank as usize;
            let acknowledgement = p2p_send_acknowledgement(&frame, unique_id, world_size)?;
            write_frame(&mut streams[sender], &acknowledgement)?;
            if let Some(index) = receives
                .iter()
                .position(|receive| p2p_matches(&frame, receive))
            {
                let receive = receives.remove(index).expect("matched receive exists");
                deliver_p2p_receive(streams, unique_id, world_size, &frame, &receive)
            } else {
                sends.push_back(frame);
                Ok(())
            }
        }
        Opcode::Receive => {
            if let Some(index) = sends.iter().position(|send| p2p_matches(send, &frame)) {
                let send = sends.remove(index).expect("matched send exists");
                deliver_p2p_receive(streams, unique_id, world_size, &send, &frame)
            } else {
                receives.push_back(frame);
                Ok(())
            }
        }
        Opcode::Heartbeat => {
            let response = p2p_heartbeat_response(&frame, unique_id, world_size)?;
            write_frame(&mut streams[connection_rank], &response)
        }
        Opcode::Leave => Ok(()),
        Opcode::Abort => Err(NetworkError::RemoteAbort(format!(
            "rank {connection_rank} aborted: {}",
            String::from_utf8_lossy(&frame.payload)
        ))),
        _ => unreachable!("point-to-point request was validated"),
    }
}

fn validate_p2p_request(
    frame: &Frame,
    unique_id: UniqueId,
    world_size: usize,
    connection_rank: usize,
) -> Result<(), NetworkError> {
    if frame.header.unique_id != unique_id {
        return Err(NetworkError::WrongSession);
    }
    if frame.header.world_size as usize != world_size
        || frame.header.source_rank as usize != connection_rank
    {
        return Err(NetworkError::InvalidConfiguration(format!(
            "point-to-point connection rank {connection_rank} submitted source {} and world {}",
            frame.header.source_rank, frame.header.world_size
        )));
    }
    if frame.header.root_rank != ANY_RANK {
        return Err(NetworkError::InvalidConfiguration(
            "point-to-point request cannot declare a collective root".into(),
        ));
    }
    match frame.header.opcode {
        Opcode::Send => {
            if frame.header.destination_rank == ANY_RANK {
                return Err(NetworkError::InvalidConfiguration(
                    "point-to-point send requires a destination rank".into(),
                ));
            }
        }
        Opcode::Receive => {
            if !frame.payload.is_empty() {
                return Err(NetworkError::InvalidConfiguration(
                    "point-to-point receive request cannot carry data".into(),
                ));
            }
        }
        Opcode::Abort => {
            if frame.header.destination_rank != ANY_RANK
                || frame.header.element_type != ElementType::U8
                || frame.payload.is_empty()
            {
                return Err(NetworkError::InvalidConfiguration(
                    "point-to-point abort requires a non-empty U8 message".into(),
                ));
            }
        }
        Opcode::Heartbeat => {
            if frame.header.destination_rank != ANY_RANK
                || frame.header.element_type != ElementType::None
                || frame.header.element_count != 0
                || !frame.payload.is_empty()
            {
                return Err(NetworkError::InvalidConfiguration(
                    "point-to-point heartbeat must be an empty control frame".into(),
                ));
            }
        }
        Opcode::Leave => {
            if frame.header.destination_rank != ANY_RANK
                || frame.header.element_type != ElementType::None
                || frame.header.element_count != 0
                || !frame.payload.is_empty()
            {
                return Err(NetworkError::InvalidConfiguration(
                    "point-to-point LEAVE must be an empty control frame".into(),
                ));
            }
        }
        opcode => {
            return Err(NetworkError::InvalidConfiguration(format!(
                "opcode {opcode:?} is invalid on the point-to-point channel"
            )));
        }
    }
    Ok(())
}

fn p2p_matches(send: &Frame, receive: &Frame) -> bool {
    send.header.destination_rank == receive.header.source_rank
        && (receive.header.destination_rank == ANY_RANK
            || receive.header.destination_rank == send.header.source_rank)
        && send.header.tag == receive.header.tag
}

fn p2p_send_acknowledgement(
    send: &Frame,
    unique_id: UniqueId,
    world_size: usize,
) -> Result<Frame, NetworkError> {
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Send,
        ElementType::None,
        send.header.source_rank,
        ANY_RANK,
        world_size as u32,
        send.header.sequence,
        0,
    );
    header.destination_rank = send.header.source_rank;
    header.tag = send.header.tag;
    Ok(Frame::new(header, Vec::new())?)
}

fn p2p_heartbeat_response(
    heartbeat: &Frame,
    unique_id: UniqueId,
    world_size: usize,
) -> Result<Frame, NetworkError> {
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Heartbeat,
        ElementType::None,
        heartbeat.header.source_rank,
        ANY_RANK,
        world_size as u32,
        heartbeat.header.sequence,
        0,
    );
    header.destination_rank = heartbeat.header.source_rank;
    Ok(Frame::new(header, Vec::new())?)
}

fn deliver_p2p_receive(
    streams: &mut [TcpStream],
    unique_id: UniqueId,
    world_size: usize,
    send: &Frame,
    receive: &Frame,
) -> Result<(), NetworkError> {
    if send.header.element_type != receive.header.element_type
        || send.header.element_count != receive.header.element_count
    {
        return Err(NetworkError::InvalidConfiguration(format!(
            "point-to-point tag {} contract mismatch: send {:?}/{} receive {:?}/{}",
            send.header.tag,
            send.header.element_type,
            send.header.element_count,
            receive.header.element_type,
            receive.header.element_count
        )));
    }
    let receiver = receive.header.source_rank as usize;
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Receive,
        send.header.element_type,
        send.header.source_rank,
        ANY_RANK,
        world_size as u32,
        receive.header.sequence,
        send.header.element_count,
    );
    header.destination_rank = receive.header.source_rank;
    header.tag = send.header.tag;
    write_frame(
        &mut streams[receiver],
        &Frame::new(header, send.payload.clone())?,
    )
}

fn broadcast_p2p_abort(
    streams: &mut [TcpStream],
    unique_id: UniqueId,
    world_size: usize,
    message: &str,
) {
    for (rank, stream) in streams.iter_mut().enumerate() {
        if let Ok(frame) = abort_frame(unique_id, 0, rank as u32, world_size as u32, message) {
            let _ = write_frame(stream, &frame);
        }
    }
}

fn validate_and_route(
    agreement: &mut CollectiveAgreement,
    unique_id: UniqueId,
    sequence: u64,
    requests: &[Frame],
) -> Result<Vec<Frame>, NetworkError> {
    let world_size = requests.len();
    for (rank, request) in requests.iter().enumerate() {
        if request.header.unique_id != unique_id {
            return Err(NetworkError::WrongSession);
        }
        if request.header.source_rank as usize != rank
            || request.header.world_size as usize != world_size
        {
            return Err(NetworkError::InvalidConfiguration(format!(
                "connection rank {rank} submitted source rank {} and world size {}",
                request.header.source_rank, request.header.world_size
            )));
        }
        let variable = request.header.opcode == Opcode::AllToAllV;
        let descriptor = OperationDescriptor::new(
            request.header.opcode,
            request.header.element_type,
            request.header.root_rank,
            if variable {
                0
            } else {
                request.header.element_count
            },
        )
        .with_layout_hash(if variable { 0 } else { request.header.tag });
        agreement.submit(rank, sequence, descriptor)?;
    }
    let first = &requests[0].header;
    let element_bytes = first.element_type.byte_width();
    let count = usize::try_from(first.element_count).map_err(|_| {
        NetworkError::InvalidConfiguration("collective element count exceeds usize".into())
    })?;
    let rank_payload_bytes = count.checked_mul(element_bytes).ok_or_else(|| {
        NetworkError::InvalidConfiguration("collective payload size overflow".into())
    })?;
    let root = if first.root_rank == ANY_RANK {
        None
    } else {
        Some(first.root_rank as usize)
    };
    match first.opcode {
        Opcode::Barrier => route_empty(requests, unique_id, sequence, Opcode::Barrier),
        Opcode::SetTimeout => {
            if first.element_type != ElementType::None
                || first.root_rank != ANY_RANK
                || first.tag == 0
            {
                return Err(NetworkError::InvalidConfiguration(
                    "SET_TIMEOUT requires an empty control frame and non-zero millisecond tag"
                        .into(),
                ));
            }
            route_empty(requests, unique_id, sequence, Opcode::SetTimeout)
        }
        Opcode::Broadcast => {
            let root = required_root(root, world_size)?;
            require_only_root_payload(requests, root, rank_payload_bytes)?;
            route_same_payload(
                requests,
                unique_id,
                sequence,
                first.opcode,
                first.element_type,
                count,
                &requests[root].payload,
            )
        }
        Opcode::AllGather => {
            require_all_payloads(requests, rank_payload_bytes)?;
            let payload = concatenate_payloads(requests)?;
            route_same_payload(
                requests,
                unique_id,
                sequence,
                first.opcode,
                first.element_type,
                count.checked_mul(world_size).ok_or_else(|| {
                    NetworkError::InvalidConfiguration("all-gather count overflow".into())
                })?,
                &payload,
            )
        }
        Opcode::Gather | Opcode::Reduce => {
            let root = required_root(root, world_size)?;
            require_all_payloads(requests, rank_payload_bytes)?;
            let payload = concatenate_payloads(requests)?;
            route_root_payload(
                requests,
                unique_id,
                sequence,
                first.opcode,
                first.element_type,
                root,
                count.checked_mul(world_size).ok_or_else(|| {
                    NetworkError::InvalidConfiguration("rooted collective count overflow".into())
                })?,
                &payload,
            )
        }
        Opcode::Scatter => {
            let root = required_root(root, world_size)?;
            require_only_root_payload(requests, root, rank_payload_bytes)?;
            if !count.is_multiple_of(world_size) {
                return Err(NetworkError::InvalidConfiguration(
                    "scatter total element count is not divisible by world size".into(),
                ));
            }
            let shard_count = count / world_size;
            let shard_bytes = shard_count * element_bytes;
            let mut responses = Vec::with_capacity(world_size);
            for rank in 0..world_size {
                let start = rank * shard_bytes;
                responses.push(data_response(
                    requests,
                    unique_id,
                    sequence,
                    first.opcode,
                    first.element_type,
                    rank,
                    shard_count,
                    requests[root].payload[start..start + shard_bytes].to_vec(),
                )?);
            }
            Ok(responses)
        }
        Opcode::AllReduce => {
            require_all_payloads(requests, rank_payload_bytes)?;
            let payload = concatenate_payloads(requests)?;
            route_same_payload(
                requests,
                unique_id,
                sequence,
                first.opcode,
                first.element_type,
                count.checked_mul(world_size).ok_or_else(|| {
                    NetworkError::InvalidConfiguration("all-reduce count overflow".into())
                })?,
                &payload,
            )
        }
        Opcode::ReduceScatter => {
            require_all_payloads(requests, rank_payload_bytes)?;
            if !count.is_multiple_of(world_size) {
                return Err(NetworkError::InvalidConfiguration(
                    "reduce-scatter count is not divisible by world size".into(),
                ));
            }
            let shard_count = count / world_size;
            let shard_bytes = shard_count * element_bytes;
            let mut responses = Vec::with_capacity(world_size);
            for destination in 0..world_size {
                let mut payload = Vec::with_capacity(shard_bytes * world_size);
                for request in requests {
                    let start = destination * shard_bytes;
                    payload.extend_from_slice(&request.payload[start..start + shard_bytes]);
                }
                responses.push(data_response(
                    requests,
                    unique_id,
                    sequence,
                    first.opcode,
                    first.element_type,
                    destination,
                    shard_count * world_size,
                    payload,
                )?);
            }
            Ok(responses)
        }
        Opcode::AllToAll => {
            require_all_payloads(requests, rank_payload_bytes)?;
            if !count.is_multiple_of(world_size) {
                return Err(NetworkError::InvalidConfiguration(
                    "all-to-all count is not divisible by world size".into(),
                ));
            }
            let shard_count = count / world_size;
            let shard_bytes = shard_count * element_bytes;
            let mut responses = Vec::with_capacity(world_size);
            for destination in 0..world_size {
                let mut payload = Vec::with_capacity(rank_payload_bytes);
                for request in requests {
                    let start = destination * shard_bytes;
                    payload.extend_from_slice(&request.payload[start..start + shard_bytes]);
                }
                responses.push(data_response(
                    requests,
                    unique_id,
                    sequence,
                    first.opcode,
                    first.element_type,
                    destination,
                    count,
                    payload,
                )?);
            }
            Ok(responses)
        }
        Opcode::AllToAllV => route_all_to_all_v(
            requests,
            unique_id,
            sequence,
            first.element_type,
            element_bytes,
        ),
        Opcode::Join
        | Opcode::Ready
        | Opcode::PeerEndpoint
        | Opcode::Send
        | Opcode::Receive
        | Opcode::Abort
        | Opcode::Heartbeat
        | Opcode::Leave => Err(NetworkError::InvalidConfiguration(format!(
            "opcode {:?} is not a coordinator collective request",
            first.opcode
        ))),
    }
}

fn route_all_to_all_v(
    requests: &[Frame],
    unique_id: UniqueId,
    sequence: u64,
    element_type: ElementType,
    element_bytes: usize,
) -> Result<Vec<Frame>, NetworkError> {
    if element_type == ElementType::None || element_bytes == 0 {
        return Err(NetworkError::InvalidConfiguration(
            "all-to-all-v requires a concrete element type".into(),
        ));
    }
    let world_size = requests.len();
    let mut rows = Vec::with_capacity(world_size);
    for request in requests {
        rows.push(decode_counts_payload(request, world_size, element_bytes)?);
    }
    let mut responses = Vec::with_capacity(world_size);
    for destination in 0..world_size {
        let receive_counts = rows
            .iter()
            .map(|(counts, _)| counts[destination])
            .collect::<Vec<_>>();
        let receive_total = receive_counts
            .iter()
            .try_fold(0_usize, |total, count| total.checked_add(*count));
        let receive_total = receive_total.ok_or_else(|| {
            NetworkError::InvalidConfiguration("all-to-all-v receive count overflow".into())
        })?;
        let mut payload = encode_counts(&receive_counts)?;
        payload.reserve(receive_total.checked_mul(element_bytes).ok_or_else(|| {
            NetworkError::InvalidConfiguration("all-to-all-v response size overflow".into())
        })?);
        for (counts, data) in &rows {
            let start_elements = counts[..destination]
                .iter()
                .try_fold(0_usize, |total, count| total.checked_add(*count))
                .ok_or_else(|| {
                    NetworkError::InvalidConfiguration("all-to-all-v source offset overflow".into())
                })?;
            let start = start_elements.checked_mul(element_bytes).ok_or_else(|| {
                NetworkError::InvalidConfiguration("all-to-all-v byte offset overflow".into())
            })?;
            let bytes = counts[destination]
                .checked_mul(element_bytes)
                .ok_or_else(|| {
                    NetworkError::InvalidConfiguration("all-to-all-v shard size overflow".into())
                })?;
            payload.extend_from_slice(&data[start..start + bytes]);
        }
        let mut header = response_header(
            unique_id,
            Opcode::AllToAllV,
            element_type,
            sequence,
            destination,
            world_size,
            receive_total,
        );
        header.flags |= FLAG_COUNTS_PREFIX;
        responses.push(Frame::new(header, payload)?);
    }
    Ok(responses)
}

fn decode_counts_payload(
    frame: &Frame,
    world_size: usize,
    element_bytes: usize,
) -> Result<(Vec<usize>, &[u8]), NetworkError> {
    if frame.header.flags & FLAG_COUNTS_PREFIX == 0 {
        return Err(NetworkError::InvalidConfiguration(
            "all-to-all-v frame is missing its counts prefix".into(),
        ));
    }
    let prefix_bytes = world_size.checked_mul(8).ok_or_else(|| {
        NetworkError::InvalidConfiguration("all-to-all-v prefix size overflow".into())
    })?;
    if frame.payload.len() < prefix_bytes {
        return Err(NetworkError::InvalidConfiguration(
            "all-to-all-v counts prefix is truncated".into(),
        ));
    }
    let counts = frame.payload[..prefix_bytes]
        .chunks_exact(8)
        .map(|bytes| {
            usize::try_from(u64::from_le_bytes(bytes.try_into().unwrap())).map_err(|_| {
                NetworkError::InvalidConfiguration("all-to-all-v count exceeds host usize".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total = counts
        .iter()
        .try_fold(0_usize, |sum, count| sum.checked_add(*count));
    let total = total.ok_or_else(|| {
        NetworkError::InvalidConfiguration("all-to-all-v count sum overflow".into())
    })?;
    if total as u64 != frame.header.element_count {
        return Err(NetworkError::InvalidConfiguration(format!(
            "all-to-all-v counts sum to {total}, header declares {}",
            frame.header.element_count
        )));
    }
    let expected_data = total.checked_mul(element_bytes).ok_or_else(|| {
        NetworkError::InvalidConfiguration("all-to-all-v data size overflow".into())
    })?;
    let data = &frame.payload[prefix_bytes..];
    if data.len() != expected_data {
        return Err(NetworkError::InvalidConfiguration(format!(
            "all-to-all-v data has {} bytes, expected {expected_data}",
            data.len()
        )));
    }
    Ok((counts, data))
}

fn encode_counts(counts: &[usize]) -> Result<Vec<u8>, NetworkError> {
    let mut encoded = Vec::with_capacity(counts.len().saturating_mul(8));
    for count in counts {
        encoded.extend_from_slice(
            &u64::try_from(*count)
                .map_err(|_| {
                    NetworkError::InvalidConfiguration(
                        "all-to-all-v count exceeds protocol u64".into(),
                    )
                })?
                .to_le_bytes(),
        );
    }
    Ok(encoded)
}

fn route_empty(
    requests: &[Frame],
    unique_id: UniqueId,
    sequence: u64,
    opcode: Opcode,
) -> Result<Vec<Frame>, NetworkError> {
    if requests.iter().any(|request| !request.payload.is_empty()) {
        return Err(NetworkError::InvalidConfiguration(format!(
            "{opcode:?} cannot carry a payload"
        )));
    }
    (0..requests.len())
        .map(|rank| {
            Frame::new(
                response_header(
                    unique_id,
                    opcode,
                    ElementType::None,
                    sequence,
                    rank,
                    requests.len(),
                    0,
                ),
                Vec::new(),
            )
            .map_err(NetworkError::from)
        })
        .collect()
}

fn route_same_payload(
    requests: &[Frame],
    unique_id: UniqueId,
    sequence: u64,
    opcode: Opcode,
    element_type: ElementType,
    count: usize,
    payload: &[u8],
) -> Result<Vec<Frame>, NetworkError> {
    (0..requests.len())
        .map(|rank| {
            data_response(
                requests,
                unique_id,
                sequence,
                opcode,
                element_type,
                rank,
                count,
                payload.to_vec(),
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn route_root_payload(
    requests: &[Frame],
    unique_id: UniqueId,
    sequence: u64,
    opcode: Opcode,
    element_type: ElementType,
    root: usize,
    count: usize,
    payload: &[u8],
) -> Result<Vec<Frame>, NetworkError> {
    (0..requests.len())
        .map(|rank| {
            if rank == root {
                data_response(
                    requests,
                    unique_id,
                    sequence,
                    opcode,
                    element_type,
                    rank,
                    count,
                    payload.to_vec(),
                )
            } else {
                Frame::new(
                    response_header(
                        unique_id,
                        opcode,
                        ElementType::None,
                        sequence,
                        rank,
                        requests.len(),
                        0,
                    ),
                    Vec::new(),
                )
                .map_err(NetworkError::from)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn data_response(
    requests: &[Frame],
    unique_id: UniqueId,
    sequence: u64,
    opcode: Opcode,
    element_type: ElementType,
    destination: usize,
    count: usize,
    payload: Vec<u8>,
) -> Result<Frame, NetworkError> {
    Ok(Frame::new(
        response_header(
            unique_id,
            opcode,
            element_type,
            sequence,
            destination,
            requests.len(),
            count,
        ),
        payload,
    )?)
}

fn response_header(
    unique_id: UniqueId,
    opcode: Opcode,
    element_type: ElementType,
    sequence: u64,
    destination: usize,
    world_size: usize,
    count: usize,
) -> FrameHeader {
    let mut header = FrameHeader::collective(
        unique_id,
        opcode,
        element_type,
        0,
        ANY_RANK,
        world_size as u32,
        sequence,
        count as u64,
    );
    header.destination_rank = destination as u32;
    header
}

fn control_frame(
    unique_id: UniqueId,
    opcode: Opcode,
    sequence: u64,
    destination: u32,
    world_size: u32,
) -> Result<Frame, NetworkError> {
    let mut header = FrameHeader::collective(
        unique_id,
        opcode,
        ElementType::None,
        0,
        ANY_RANK,
        world_size,
        sequence,
        0,
    );
    header.destination_rank = destination;
    Ok(Frame::new(header, Vec::new())?)
}

fn abort_frame(
    unique_id: UniqueId,
    sequence: u64,
    destination: u32,
    world_size: u32,
    message: &str,
) -> Result<Frame, NetworkError> {
    let payload = message.as_bytes().to_vec();
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Abort,
        ElementType::U8,
        0,
        ANY_RANK,
        world_size,
        sequence,
        payload.len() as u64,
    );
    header.destination_rank = destination;
    Ok(Frame::new(header, payload)?)
}

fn required_root(root: Option<usize>, world_size: usize) -> Result<usize, NetworkError> {
    match root {
        Some(root) if root < world_size => Ok(root),
        _ => Err(NetworkError::InvalidConfiguration(
            "rooted collective requires a valid root rank".into(),
        )),
    }
}

fn require_all_payloads(requests: &[Frame], bytes: usize) -> Result<(), NetworkError> {
    if let Some((rank, actual)) = requests
        .iter()
        .enumerate()
        .map(|(rank, request)| (rank, request.payload.len()))
        .find(|(_, actual)| *actual != bytes)
    {
        return Err(NetworkError::InvalidConfiguration(format!(
            "rank {rank} payload has {actual} bytes, expected {bytes}"
        )));
    }
    Ok(())
}

fn require_only_root_payload(
    requests: &[Frame],
    root: usize,
    bytes: usize,
) -> Result<(), NetworkError> {
    for (rank, request) in requests.iter().enumerate() {
        let expected = if rank == root { bytes } else { 0 };
        if request.payload.len() != expected {
            return Err(NetworkError::InvalidConfiguration(format!(
                "rank {rank} payload has {} bytes, expected {expected}",
                request.payload.len()
            )));
        }
    }
    Ok(())
}

fn concatenate_payloads(requests: &[Frame]) -> Result<Vec<u8>, NetworkError> {
    let capacity = requests
        .iter()
        .try_fold(0_usize, |total, request| {
            total.checked_add(request.payload.len())
        })
        .ok_or_else(|| NetworkError::InvalidConfiguration("response payload overflow".into()))?;
    let mut payload = Vec::with_capacity(capacity);
    for request in requests {
        payload.extend_from_slice(&request.payload);
    }
    Ok(payload)
}

pub(crate) fn write_frame(stream: &mut TcpStream, frame: &Frame) -> Result<(), NetworkError> {
    let header = frame.encode_transport_header()?;
    let total = header.len() + frame.payload.len();
    let written = stream.write_vectored(&[
        IoSlice::new(&header),
        IoSlice::new(frame.payload.as_slice()),
    ])?;
    if written < header.len() {
        stream.write_all(&header[written..])?;
        stream.write_all(&frame.payload)?;
    } else if written < total {
        stream.write_all(&frame.payload[written - header.len()..])?;
    }
    stream.flush()?;
    Ok(())
}

pub(crate) fn read_frame(stream: &mut TcpStream) -> Result<Option<Frame>, NetworkError> {
    let mut header_bytes = [0_u8; super::protocol::FRAME_HEADER_BYTES];
    match stream.read(&mut header_bytes[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!("one-byte read cannot return more than one byte"),
        Err(error) => return Err(error.into()),
    }
    stream.read_exact(&mut header_bytes[1..])?;
    let header = FrameHeader::decode(&header_bytes)?;
    let payload_length = usize::try_from(header.payload_bytes).map_err(|_| {
        NetworkError::InvalidConfiguration("frame payload length exceeds usize".into())
    })?;
    let mut payload = vec![0_u8; payload_length];
    stream.read_exact(&mut payload)?;
    header.validate(Some(&payload))?;
    Ok(Some(Frame { header, payload }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn topology_probe_metrics_use_stable_conservative_buckets() {
        assert_eq!(conservative_probe_bandwidth_bucket(1), 1);
        assert_eq!(conservative_probe_bandwidth_bucket(63), 16);
        assert_eq!(conservative_probe_bandwidth_bucket(64), 64);
        assert_eq!(conservative_probe_bandwidth_bucket(255), 64);
        assert_eq!(conservative_probe_bandwidth_bucket(256), 256);
        assert_eq!(conservative_probe_latency_bucket(1), 4);
        assert_eq!(conservative_probe_latency_bucket(32_769), 65_536);
        assert_eq!(conservative_probe_latency_bucket(65_535), 65_536);
        assert_eq!(conservative_probe_latency_bucket(65_536), 262_144);
    }

    #[test]
    fn peer_rail_address_configuration_requires_one_address_per_rail() {
        let listen =
            parse_rail_socket_addresses("GX1_P2P_RAIL_LISTEN_ADDRS", "127.0.0.1:0, 127.0.0.2:0", 2)
                .unwrap();
        assert_eq!(listen.len(), 2);
        assert_eq!(listen[0].ip(), IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(listen[1].ip(), IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)));

        let advertise =
            parse_rail_ip_addresses("GX1_P2P_RAIL_ADVERTISE_ADDRS", "127.0.0.1,127.0.0.2", 2)
                .unwrap();
        assert_eq!(advertise.len(), 2);
        assert!(
            parse_rail_socket_addresses("GX1_P2P_RAIL_LISTEN_ADDRS", "127.0.0.1:0", 2,)
                .unwrap_err()
                .to_string()
                .contains("expected 2")
        );
        assert!(
            parse_rail_ip_addresses("GX1_P2P_RAIL_ADVERTISE_ADDRS", "127.0.0.1,not-an-ip", 2,)
                .unwrap_err()
                .to_string()
                .contains("rail 1")
        );
    }

    #[test]
    fn peer_rail_address_configuration_binds_dedicated_listeners() {
        let configuration = PeerEndpointConfiguration {
            listen_addresses: vec![
                "127.0.0.1:0".parse().unwrap(),
                "127.0.0.2:0".parse().unwrap(),
            ],
            advertise_addresses: vec![
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)),
            ],
        };
        let (listeners, endpoints) =
            bind_peer_listeners_with_configuration(configuration, 2).unwrap();
        assert_eq!(listeners.len(), 2);
        assert_eq!(endpoints.len(), 2);
        assert_eq!(endpoints[0].ip(), IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(endpoints[1].ip(), IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)));
        assert_ne!(endpoints[0].port(), 0);
        assert_ne!(endpoints[1].port(), 0);
    }

    #[test]
    fn tcp_rendezvous_routes_all_gather_all_to_all_and_barrier() {
        let unique_id = UniqueId::from_bytes([9; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 3).unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let ranks = (0..3_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let session = TcpRankSession::connect(
                        address,
                        unique_id,
                        rank,
                        3,
                        Duration::from_secs(5),
                    )
                    .unwrap();
                    let payload = [rank * 10, rank * 10 + 1]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>();
                    let gathered = session
                        .exchange(Opcode::AllGather, ElementType::U32, ANY_RANK, 2, payload)
                        .unwrap();
                    let values = gathered
                        .payload
                        .chunks_exact(4)
                        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    assert_eq!(values, vec![0, 1, 10, 11, 20, 21]);

                    let all_to_all_input = (0..3_u32)
                        .map(|destination| rank * 100 + destination)
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>();
                    let all_to_all = session
                        .exchange(
                            Opcode::AllToAll,
                            ElementType::U32,
                            ANY_RANK,
                            3,
                            all_to_all_input,
                        )
                        .unwrap();
                    let values = all_to_all
                        .payload
                        .chunks_exact(4)
                        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    assert_eq!(values, vec![rank, 100 + rank, 200 + rank]);
                    session.barrier().unwrap();
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn tcp_point_to_point_is_tagged_multiplexed_and_collective_independent() {
        let unique_id = UniqueId::from_bytes([7; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 3).unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let ranks = (0..3_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let session = Arc::new(
                        TcpRankSession::connect(
                            address,
                            unique_id,
                            rank,
                            3,
                            Duration::from_secs(5),
                        )
                        .unwrap(),
                    );
                    match rank {
                        0 => {
                            session
                                .send(
                                    1,
                                    101,
                                    ElementType::U32,
                                    2,
                                    [101_u32, 102]
                                        .into_iter()
                                        .flat_map(u32::to_le_bytes)
                                        .collect(),
                                )
                                .unwrap();
                            session
                                .send(
                                    1,
                                    100,
                                    ElementType::U32,
                                    2,
                                    [100_u32, 101]
                                        .into_iter()
                                        .flat_map(u32::to_le_bytes)
                                        .collect(),
                                )
                                .unwrap();
                        }
                        1 => {
                            let receives = [100_u64, 101]
                                .into_iter()
                                .map(|tag| {
                                    let session = Arc::clone(&session);
                                    thread::spawn(move || {
                                        let frame = session
                                            .receive(Some(0), tag, ElementType::U32, 2)
                                            .unwrap();
                                        let values = frame
                                            .payload
                                            .chunks_exact(4)
                                            .map(|bytes| {
                                                u32::from_le_bytes(bytes.try_into().unwrap())
                                            })
                                            .collect::<Vec<_>>();
                                        (tag, values)
                                    })
                                })
                                .collect::<Vec<_>>();
                            let mut received = receives
                                .into_iter()
                                .map(|receive| receive.join().unwrap())
                                .collect::<Vec<_>>();
                            received.sort_by_key(|(tag, _)| *tag);
                            assert_eq!(
                                received,
                                vec![(100, vec![100, 101]), (101, vec![101, 102])]
                            );
                        }
                        2 => {}
                        _ => unreachable!(),
                    }
                    let gathered = session
                        .exchange(
                            Opcode::AllGather,
                            ElementType::U32,
                            ANY_RANK,
                            1,
                            rank.to_le_bytes().to_vec(),
                        )
                        .unwrap();
                    assert_eq!(
                        gathered
                            .payload
                            .chunks_exact(4)
                            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                            .collect::<Vec<_>>(),
                        vec![0, 1, 2]
                    );
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn tcp_peer_mesh_routes_rank_to_rank_data_on_multiple_rails() {
        let unique_id = UniqueId::from_bytes([0x51; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 3)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let ranks = (0..3_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let session = TcpRankSession::connect_with_transport(
                        address,
                        unique_id,
                        rank,
                        3,
                        Duration::from_secs(5),
                        2,
                        CollectiveTransport::TcpPeer,
                    )
                    .unwrap();
                    assert_eq!(session.transport(), CollectiveTransport::TcpPeer);
                    assert_eq!(session.p2p_rails(), 2);
                    match rank {
                        0 => {
                            session
                                .send_on_rail(
                                    1,
                                    1,
                                    101,
                                    ElementType::U32,
                                    2,
                                    [10_u32, 11]
                                        .into_iter()
                                        .flat_map(u32::to_le_bytes)
                                        .collect(),
                                )
                                .unwrap();
                            let frame = session
                                .receive_on_rail(0, Some(2), 202, ElementType::U32, 1)
                                .unwrap();
                            assert_eq!(frame.payload, 22_u32.to_le_bytes());
                        }
                        1 => {
                            let frame = session
                                .receive_on_rail(1, None, 101, ElementType::U32, 2)
                                .unwrap();
                            assert_eq!(
                                frame
                                    .payload
                                    .chunks_exact(4)
                                    .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                                    .collect::<Vec<_>>(),
                                vec![10, 11]
                            );
                            session
                                .send_on_rail(
                                    0,
                                    2,
                                    303,
                                    ElementType::U32,
                                    1,
                                    13_u32.to_le_bytes().to_vec(),
                                )
                                .unwrap();
                        }
                        2 => {
                            session
                                .send_on_rail(
                                    0,
                                    0,
                                    202,
                                    ElementType::U32,
                                    1,
                                    22_u32.to_le_bytes().to_vec(),
                                )
                                .unwrap();
                            let frame = session
                                .receive_on_rail(0, Some(1), 303, ElementType::U32, 1)
                                .unwrap();
                            assert_eq!(frame.payload, 13_u32.to_le_bytes());
                        }
                        _ => unreachable!(),
                    }
                    let gathered = session
                        .exchange(
                            Opcode::AllGather,
                            ElementType::U32,
                            ANY_RANK,
                            1,
                            rank.to_le_bytes().to_vec(),
                        )
                        .unwrap();
                    assert_eq!(
                        gathered
                            .payload
                            .chunks_exact(4)
                            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                            .collect::<Vec<_>>(),
                        vec![0, 1, 2]
                    );
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn tcp_peer_mesh_routes_each_rail_through_its_own_listen_address() {
        let unique_id = UniqueId::from_bytes([0x5a; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let ranks = (0..2_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let configuration = PeerEndpointConfiguration {
                        listen_addresses: vec![
                            "127.0.0.1:0".parse().unwrap(),
                            "127.0.0.2:0".parse().unwrap(),
                        ],
                        advertise_addresses: vec![
                            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)),
                        ],
                    };
                    let session = TcpRankSession::connect_with_transport_configuration(
                        address,
                        unique_id,
                        rank,
                        2,
                        Duration::from_secs(5),
                        2,
                        CollectiveTransport::TcpPeer,
                        Some(configuration),
                    )
                    .unwrap();
                    if rank == 0 {
                        let first = session
                            .receive_on_rail(0, Some(1), 700, ElementType::U32, 1)
                            .unwrap();
                        let second = session
                            .receive_on_rail(1, Some(1), 701, ElementType::U32, 1)
                            .unwrap();
                        assert_eq!(first.payload, 10_u32.to_le_bytes());
                        assert_eq!(second.payload, 11_u32.to_le_bytes());
                    } else {
                        session
                            .send_on_rail(
                                0,
                                0,
                                700,
                                ElementType::U32,
                                1,
                                10_u32.to_le_bytes().to_vec(),
                            )
                            .unwrap();
                        session
                            .send_on_rail(
                                1,
                                0,
                                701,
                                ElementType::U32,
                                1,
                                11_u32.to_le_bytes().to_vec(),
                            )
                            .unwrap();
                    }
                    session
                        .exchange(Opcode::Barrier, ElementType::None, ANY_RANK, 0, Vec::new())
                        .unwrap();
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn tcp_peer_probe_builds_the_same_complete_topology_on_every_rank() {
        let unique_id = UniqueId::from_bytes([0x57; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 4)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let options = TopologyProbeOptions {
            payload_bytes: 16 * 1024,
            latency_iterations: 3,
            bandwidth_iterations: 2,
            warmup_iterations: 0,
        };
        let ranks = (0..4_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let session = TcpRankSession::connect_with_transport(
                        address,
                        unique_id,
                        rank,
                        4,
                        Duration::from_secs(5),
                        2,
                        CollectiveTransport::TcpPeer,
                    )
                    .unwrap();
                    let topology = session.probe_topology(options).unwrap();
                    let links = topology.links();
                    assert_eq!(links.len(), 6);
                    assert!(
                        links
                            .iter()
                            .all(|link| { link.bandwidth_mbps > 0 && link.latency_ns > 0 })
                    );
                    let rail_links = topology.rail_links();
                    assert_eq!(rail_links.len(), 12);
                    assert!(rail_links.iter().all(|link| {
                        link.rail < 2 && link.bandwidth_mbps > 0 && link.latency_ns > 0
                    }));
                    let aggregate_links = topology.aggregate_links();
                    assert_eq!(aggregate_links.len(), 6);
                    assert!(aggregate_links.iter().all(|link| link.bandwidth_mbps > 0));
                    (
                        links,
                        rail_links,
                        aggregate_links,
                        topology.best_ring_order().unwrap(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let results = ranks
            .into_iter()
            .map(|rank| rank.join().unwrap())
            .collect::<Vec<_>>();
        for result in &results[1..] {
            assert_eq!(result, &results[0]);
        }
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn topology_probe_rejects_coordinator_data_transport() {
        let unique_id = UniqueId::from_bytes([0x58; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 1).unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let session = TcpRankSession::connect_with_transport(
            address,
            unique_id,
            0,
            1,
            Duration::from_secs(2),
            1,
            CollectiveTransport::TcpHostStaged,
        )
        .unwrap();
        let error = session
            .probe_topology(TopologyProbeOptions::default())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires GX1_COLLECTIVE_TRANSPORT=tcp_peer")
        );
        drop(session);
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn peer_abort_interrupts_topology_probe_on_all_other_ranks() {
        let unique_id = UniqueId::from_bytes([0x59; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 3)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let options = TopologyProbeOptions {
            payload_bytes: 4096,
            latency_iterations: 3,
            bandwidth_iterations: 2,
            warmup_iterations: 0,
        };
        let connected = Arc::new(std::sync::Barrier::new(3));
        let ranks = (0..3_u32)
            .map(|rank| {
                let connected = Arc::clone(&connected);
                thread::spawn(move || {
                    let session = TcpRankSession::connect_with_transport(
                        address,
                        unique_id,
                        rank,
                        3,
                        Duration::from_secs(3),
                        1,
                        CollectiveTransport::TcpPeer,
                    )
                    .unwrap();
                    connected.wait();
                    if rank == 2 {
                        session.abort("topology probe failure injection").unwrap();
                        return;
                    }
                    let error = session.probe_topology(options).unwrap_err();
                    assert!(
                        matches!(error, NetworkError::RemoteAbort(_)),
                        "topology probe returned a non-abort failure: {error:?}"
                    );
                    assert!(
                        error
                            .to_string()
                            .contains("topology probe failure injection"),
                        "unexpected probe abort reason: {error}"
                    );
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn tcp_peer_abort_interrupts_pending_receive_on_another_rail() {
        let unique_id = UniqueId::from_bytes([0x52; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let (ready_sender, ready_receiver) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let session = TcpRankSession::connect_with_transport(
                address,
                unique_id,
                0,
                2,
                Duration::from_secs(3),
                2,
                CollectiveTransport::TcpPeer,
            )
            .unwrap();
            ready_sender.send(()).unwrap();
            let started = Instant::now();
            let error = session
                .receive_on_rail(1, Some(1), 404, ElementType::U32, 1)
                .unwrap_err();
            assert!(
                matches!(error, NetworkError::RemoteAbort(_)),
                "unexpected disconnect error: {error:?}"
            );
            assert!(error.to_string().contains("injected direct peer failure"));
            assert!(started.elapsed() < Duration::from_secs(1));
        });
        let aborter = thread::spawn(move || {
            let session = TcpRankSession::connect_with_transport(
                address,
                unique_id,
                1,
                2,
                Duration::from_secs(3),
                2,
                CollectiveTransport::TcpPeer,
            )
            .unwrap();
            ready_receiver.recv().unwrap();
            session.abort("injected direct peer failure").unwrap();
        });
        receiver.join().unwrap();
        aborter.join().unwrap();
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn eight_rank_peer_abort_fans_out_across_both_data_rails() {
        const WORLD_SIZE: u32 = 8;

        let unique_id = UniqueId::from_bytes([0x74; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, WORLD_SIZE as usize)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let connected = Arc::new(std::sync::Barrier::new(WORLD_SIZE as usize));
        let ranks = (0..WORLD_SIZE)
            .map(|rank| {
                let connected = Arc::clone(&connected);
                thread::spawn(move || {
                    let session = TcpRankSession::connect_with_transport(
                        address,
                        unique_id,
                        rank,
                        WORLD_SIZE,
                        Duration::from_secs(5),
                        2,
                        CollectiveTransport::TcpPeer,
                    )
                    .unwrap();
                    connected.wait();
                    if rank == WORLD_SIZE - 1 {
                        session.abort("injected eight-rank peer failure").unwrap();
                        return;
                    }

                    let rail = rank as usize % 2;
                    let source = (rank + 1) % (WORLD_SIZE - 1);
                    let started = Instant::now();
                    let error = session
                        .receive_on_rail(
                            rail,
                            Some(source),
                            0x8000 + u64::from(rank),
                            ElementType::U32,
                            1,
                        )
                        .unwrap_err();
                    assert!(
                        matches!(error, NetworkError::RemoteAbort(_)),
                        "rank {rank} returned a non-abort failure: {error:?}"
                    );
                    assert!(
                        error
                            .to_string()
                            .contains("injected eight-rank peer failure")
                    );
                    assert!(started.elapsed() < Duration::from_secs(1));
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn tcp_peer_endpoint_exchange_times_out_when_a_rank_never_publishes() {
        let unique_id = UniqueId::from_bytes([0x54; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_collective_timeout(Duration::from_millis(120))
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let publishing = thread::spawn(move || {
            let started = Instant::now();
            let result = TcpRankSession::connect_with_transport(
                address,
                unique_id,
                0,
                2,
                Duration::from_secs(2),
                1,
                CollectiveTransport::TcpPeer,
            );
            assert!(result.is_err());
            assert!(started.elapsed() < Duration::from_secs(1));
        });
        let stalled = thread::spawn(move || {
            let addresses = [address];
            let _stream = connect_rank_channel(
                &addresses,
                unique_id,
                1,
                2,
                Duration::from_secs(2),
                false,
                0,
            )
            .unwrap();
            thread::sleep(Duration::from_millis(250));
        });
        publishing.join().unwrap();
        stalled.join().unwrap();
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn tcp_peer_mesh_setup_failure_is_broadcast_on_control_plane() {
        let unique_id = UniqueId::from_bytes([0x56; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let fake_rank = thread::spawn(move || {
            let addresses = [address];
            let mut collective = connect_rank_channel(
                &addresses,
                unique_id,
                0,
                2,
                Duration::from_secs(2),
                false,
                0,
            )
            .unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let unreachable = listener.local_addr().unwrap();
            drop(listener);
            exchange_peer_endpoints_client(&mut collective, &[unreachable], unique_id, 0, 2, 1)
                .unwrap();
            let mut control =
                connect_rank_channel(&addresses, unique_id, 0, 2, Duration::from_secs(2), true, 0)
                    .unwrap();
            let abort = read_frame(&mut control)
                .unwrap()
                .expect("control plane returns setup abort");
            assert_eq!(abort.header.opcode, Opcode::Abort);
            assert!(
                String::from_utf8_lossy(&abort.payload)
                    .contains("rank 1 direct peer mesh setup failed")
            );
        });
        let failing_rank = thread::spawn(move || {
            let started = Instant::now();
            let result = TcpRankSession::connect_with_transport(
                address,
                unique_id,
                1,
                2,
                Duration::from_millis(300),
                1,
                CollectiveTransport::TcpPeer,
            );
            assert!(result.is_err());
            assert!(started.elapsed() < Duration::from_secs(1));
        });
        fake_rank.join().unwrap();
        failing_rank.join().unwrap();
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn tcp_peer_disconnect_wakes_pending_receive_without_waiting_for_timeout() {
        let unique_id = UniqueId::from_bytes([0x55; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let (ready_sender, ready_receiver) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let session = TcpRankSession::connect_with_transport(
                address,
                unique_id,
                0,
                2,
                Duration::from_secs(3),
                1,
                CollectiveTransport::TcpPeer,
            )
            .unwrap();
            ready_sender.send(()).unwrap();
            let started = Instant::now();
            let error = session
                .receive(Some(1), 505, ElementType::U32, 1)
                .unwrap_err();
            assert!(
                matches!(error, NetworkError::RemoteAbort(_)),
                "unexpected disconnect error: {error:?}"
            );
            assert!(error.to_string().contains("direct peer rank 1 left"));
            assert!(started.elapsed() < Duration::from_secs(1));
        });
        let disconnecting = thread::spawn(move || {
            let session = TcpRankSession::connect_with_transport(
                address,
                unique_id,
                1,
                2,
                Duration::from_secs(3),
                1,
                CollectiveTransport::TcpPeer,
            )
            .unwrap();
            ready_receiver.recv().unwrap();
            drop(session);
        });
        receiver.join().unwrap();
        disconnecting.join().unwrap();
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn tcp_rendezvous_aborts_mismatched_collective_order() {
        let unique_id = UniqueId::from_bytes([5; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2).unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let ranks = (0..2_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let session = TcpRankSession::connect(
                        address,
                        unique_id,
                        rank,
                        2,
                        Duration::from_secs(5),
                    )
                    .unwrap();
                    let opcode = if rank == 0 {
                        Opcode::AllGather
                    } else {
                        Opcode::AllReduce
                    };
                    assert!(matches!(
                        session.exchange(
                            opcode,
                            ElementType::U32,
                            ANY_RANK,
                            1,
                            rank.to_le_bytes().to_vec()
                        ),
                        Err(NetworkError::RemoteAbort(_))
                    ));
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn collective_timeout_names_missing_rank_and_aborts_waiter() {
        let unique_id = UniqueId::from_bytes([0x31; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_collective_timeout(Duration::from_millis(150))
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let waiting = thread::spawn(move || {
            let session =
                TcpRankSession::connect(address, unique_id, 0, 2, Duration::from_secs(2)).unwrap();
            session.barrier()
        });
        let missing = thread::spawn(move || {
            let _session =
                TcpRankSession::connect(address, unique_id, 1, 2, Duration::from_secs(2)).unwrap();
            thread::sleep(Duration::from_millis(300));
        });
        let error = waiting.join().unwrap().unwrap_err();
        assert!(matches!(error, NetworkError::RemoteAbort(_)));
        assert!(error.to_string().contains("missing ranks [1]"));
        missing.join().unwrap();
        assert!(matches!(
            coordinator.join().unwrap(),
            Err(NetworkError::Timeout(_))
        ));
    }

    #[test]
    fn explicit_rank_abort_interrupts_pending_point_to_point_work() {
        let unique_id = UniqueId::from_bytes([0x32; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2).unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let waiting = thread::spawn(move || {
            let session =
                TcpRankSession::connect(address, unique_id, 0, 2, Duration::from_secs(2)).unwrap();
            session.receive(Some(1), 99, ElementType::U32, 1)
        });
        let aborting = thread::spawn(move || {
            let session =
                TcpRankSession::connect(address, unique_id, 1, 2, Duration::from_secs(2)).unwrap();
            thread::sleep(Duration::from_millis(25));
            session.abort("injected rank failure").unwrap();
            thread::sleep(Duration::from_millis(25));
        });
        let error = waiting.join().unwrap().unwrap_err();
        assert!(matches!(error, NetworkError::RemoteAbort(_)));
        assert!(error.to_string().contains("injected rank failure"));
        aborting.join().unwrap();
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn explicit_rank_abort_interrupts_collective_without_closing_its_session() {
        for (index, transport) in [
            CollectiveTransport::TcpHostStaged,
            CollectiveTransport::TcpPeer,
        ]
        .into_iter()
        .enumerate()
        {
            let unique_id = UniqueId::from_bytes([0x5a + index as u8; 16]);
            let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
                .unwrap()
                .with_transport(transport)
                .unwrap();
            let address = server.local_addr().unwrap();
            let coordinator = thread::spawn(move || server.run());
            let (ready_sender, ready_receiver) = mpsc::channel();
            let (release_sender, release_receiver) = mpsc::channel();
            let waiting = thread::spawn(move || {
                let session = TcpRankSession::connect_with_transport(
                    address,
                    unique_id,
                    0,
                    2,
                    Duration::from_secs(2),
                    1,
                    transport,
                )
                .unwrap();
                ready_sender.send(()).unwrap();
                let error = session.barrier().unwrap_err();
                assert!(matches!(error, NetworkError::RemoteAbort(_)));
                assert!(
                    error.to_string().contains("collective failure reason"),
                    "unexpected collective abort reason: {error}"
                );
                release_sender.send(()).unwrap();
            });
            let aborting = thread::spawn(move || {
                let session = TcpRankSession::connect_with_transport(
                    address,
                    unique_id,
                    1,
                    2,
                    Duration::from_secs(2),
                    1,
                    transport,
                )
                .unwrap();
                ready_receiver.recv().unwrap();
                session.abort("collective failure reason").unwrap();
                release_receiver.recv().unwrap();
                drop(session);
            });
            waiting.join().unwrap();
            aborting.join().unwrap();
            let error = coordinator.join().unwrap().unwrap_err();
            assert!(error.to_string().contains("collective failure reason"));
        }
    }

    #[test]
    fn abort_is_broadcast_to_pending_work_on_every_point_to_point_rail() {
        let unique_id = UniqueId::from_bytes([0x46; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let (ready_sender, ready_receiver) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let session = TcpRankSession::connect_with_p2p_rails(
                address,
                unique_id,
                0,
                2,
                Duration::from_secs(2),
                2,
            )
            .unwrap();
            ready_sender.send(()).unwrap();
            let error = session
                .receive_on_rail(1, Some(1), 77, ElementType::U32, 1)
                .unwrap_err();
            assert!(matches!(error, NetworkError::RemoteAbort(_)));
        });
        let aborter = thread::spawn(move || {
            let session = TcpRankSession::connect_with_p2p_rails(
                address,
                unique_id,
                1,
                2,
                Duration::from_secs(2),
                2,
            )
            .unwrap();
            ready_receiver.recv().unwrap();
            session.abort("multi-rail failure").unwrap();
        });
        receiver.join().unwrap();
        aborter.join().unwrap();
        assert!(coordinator.join().unwrap().is_err());
    }

    #[test]
    fn heartbeat_round_trip_is_independent_of_collective_sequence() {
        let unique_id = UniqueId::from_bytes([0x33; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2).unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let ranks = (0..2_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let session = TcpRankSession::connect(
                        address,
                        unique_id,
                        rank,
                        2,
                        Duration::from_secs(2),
                    )
                    .unwrap();
                    for _ in 0..3 {
                        assert!(session.heartbeat(Duration::from_secs(1)).unwrap().as_secs() < 1);
                    }
                    session.barrier().unwrap();
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn heartbeat_enabled_session_closes_cleanly_with_leave() {
        let unique_id = UniqueId::from_bytes([0x35; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_heartbeat_timeout(Duration::from_secs(1))
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let ranks = (0..2_u32)
            .map(|rank| {
                thread::spawn(move || {
                    let session = TcpRankSession::connect(
                        address,
                        unique_id,
                        rank,
                        2,
                        Duration::from_secs(2),
                    )
                    .unwrap();
                    session.heartbeat(Duration::from_secs(1)).unwrap();
                    session.barrier().unwrap();
                })
            })
            .collect::<Vec<_>>();
        for rank in ranks {
            rank.join().unwrap();
        }
        coordinator.join().unwrap().unwrap();
    }

    #[test]
    fn collective_timeout_is_reconfigured_for_all_future_sequences() {
        let unique_id = UniqueId::from_bytes([0x36; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_collective_timeout(Duration::from_secs(2))
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let waiting = thread::spawn(move || {
            let session =
                TcpRankSession::connect(address, unique_id, 0, 2, Duration::from_secs(2)).unwrap();
            session.set_timeout(Duration::from_millis(120)).unwrap();
            session.barrier()
        });
        let missing = thread::spawn(move || {
            let session =
                TcpRankSession::connect(address, unique_id, 1, 2, Duration::from_secs(2)).unwrap();
            session.set_timeout(Duration::from_millis(120)).unwrap();
            thread::sleep(Duration::from_millis(300));
        });
        let error = waiting.join().unwrap().unwrap_err();
        assert!(matches!(error, NetworkError::RemoteAbort(_)));
        assert!(error.to_string().contains("missing ranks [1]"));
        missing.join().unwrap();
        assert!(matches!(
            coordinator.join().unwrap(),
            Err(NetworkError::Timeout(_))
        ));
    }

    #[test]
    fn heartbeat_lease_aborts_a_stale_idle_rank() {
        let unique_id = UniqueId::from_bytes([0x34; 16]);
        let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2)
            .unwrap()
            .with_heartbeat_timeout(Duration::from_millis(150))
            .unwrap();
        let address = server.local_addr().unwrap();
        let coordinator = thread::spawn(move || server.run());
        let active = thread::spawn(move || {
            let session =
                TcpRankSession::connect(address, unique_id, 0, 2, Duration::from_secs(2)).unwrap();
            for _ in 0..20 {
                match session.heartbeat(Duration::from_secs(1)) {
                    Ok(_) => thread::sleep(Duration::from_millis(30)),
                    Err(error) => return error,
                }
            }
            panic!("stale rank did not expire")
        });
        let stale = thread::spawn(move || {
            let _session =
                TcpRankSession::connect(address, unique_id, 1, 2, Duration::from_secs(2)).unwrap();
            thread::sleep(Duration::from_millis(300));
        });
        let error = active.join().unwrap();
        assert!(
            matches!(error, NetworkError::RemoteAbort(_)),
            "unexpected heartbeat failure: {error:?}"
        );
        assert!(error.to_string().contains("heartbeat lease for rank 1"));
        stale.join().unwrap();
        assert!(coordinator.join().unwrap().is_err());
    }
}
