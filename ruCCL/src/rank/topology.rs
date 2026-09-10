//! Topology-aware collective planning shared by the TCP prototype and future
//! peer-memory/GX-Link transports.

use super::{CollectiveAlgorithm, CollectiveTransport};
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

const DEFAULT_RING_THRESHOLD_BYTES: usize = 256 * 1024;
const DEFAULT_BANDWIDTH_MBPS: u64 = 32_000;
const DEFAULT_LATENCY_NS: u64 = 1_000;
const DEFAULT_HIERARCHY_INFERENCE_BYTES: usize = 16 * 1024 * 1024;
const EXACT_RING_SEARCH_MAX_RANKS: u32 = 9;
const MAX_RING_CHANNELS: usize = 64;
const MAX_P2P_RAILS: usize = 64;
pub const COLLECTIVE_AUTOTUNE_PROFILE_VERSION: u32 = 2;
const COLLECTIVE_EXECUTION_REVISION: u64 = 6;

const fn default_ring_channels() -> usize {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmPolicy {
    Auto,
    Direct,
    Ring,
    Hierarchical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectiveKind {
    #[serde(rename = "allreduce")]
    AllReduce,
    #[serde(rename = "allgather")]
    AllGather,
    #[serde(rename = "reduce_scatter")]
    ReduceScatter,
    #[serde(rename = "alltoall")]
    AllToAll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectivePlanSource {
    Policy,
    Autotune,
    Model,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectivePlan {
    pub algorithm: CollectiveAlgorithm,
    pub source: CollectivePlanSource,
    pub estimated_time_ns: u64,
    pub direct_estimated_time_ns: u64,
    pub peer_estimated_time_ns: Option<u64>,
    pub hierarchical_estimated_time_ns: Option<u64>,
    pub ring_channels: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectiveAutotuneProfile {
    pub version: u32,
    pub world_size: u32,
    pub topology_fingerprint: String,
    #[serde(default)]
    pub execution_fingerprint: String,
    pub entries: Vec<CollectiveAutotuneEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectiveAutotuneEntry {
    pub operation: CollectiveKind,
    pub min_payload_bytes: u64,
    pub max_payload_bytes: u64,
    pub algorithm: CollectiveAlgorithm,
    #[serde(default = "default_ring_channels")]
    pub ring_channels: usize,
    pub measured_time_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyLink {
    pub first_rank: u32,
    pub second_rank: u32,
    pub bandwidth_mbps: u64,
    pub latency_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyRailLink {
    pub rail: usize,
    pub first_rank: u32,
    pub second_rank: u32,
    pub bandwidth_mbps: u64,
    pub latency_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyAggregateLink {
    pub first_rank: u32,
    pub second_rank: u32,
    pub bandwidth_mbps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveTopology {
    world_size: u32,
    links: Vec<Vec<Option<LinkCost>>>,
    rail_links: Vec<Vec<Vec<Option<LinkCost>>>>,
    aggregate_links: Vec<Vec<Option<u64>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LinkCost {
    bandwidth_mbps: u64,
    latency_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveTuning {
    pub algorithm_policy: AlgorithmPolicy,
    pub ring_threshold_bytes: usize,
    pub ring_order: Vec<u32>,
    ring_orders: Vec<Vec<u32>>,
    ring_channels: usize,
    p2p_rails: usize,
    rail_order_override: Option<Vec<usize>>,
    hierarchy_groups: Option<Vec<Vec<u32>>>,
    topology: CollectiveTopology,
    direct_bandwidth_mbps: u64,
    direct_latency_ns: u64,
    transport: CollectiveTransport,
    autotune_profile: Option<CollectiveAutotuneProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    EmptyWorld,
    RankOutOfRange {
        rank: u32,
        world_size: u32,
    },
    SelfLink(u32),
    ZeroBandwidth {
        first_rank: u32,
        second_rank: u32,
    },
    ZeroDirectBandwidth,
    ProfileRead {
        path: String,
        message: String,
    },
    InvalidProfile(String),
    InvalidRing(String),
    InvalidRingChannels(usize),
    InvalidP2pRails(usize),
    InvalidRailOrder(String),
    P2pRailMismatch {
        tuning: usize,
        session: usize,
    },
    TransportMismatch {
        tuning: CollectiveTransport,
        communicator: CollectiveTransport,
    },
    InvalidHierarchy(String),
    InvalidEnvironment {
        name: &'static str,
        value: String,
    },
}

impl Display for TopologyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyWorld => formatter.write_str("collective topology cannot be empty"),
            Self::RankOutOfRange { rank, world_size } => {
                write!(
                    formatter,
                    "topology rank {rank} is outside world size {world_size}"
                )
            }
            Self::SelfLink(rank) => write!(formatter, "topology rank {rank} links to itself"),
            Self::ZeroBandwidth {
                first_rank,
                second_rank,
            } => write!(
                formatter,
                "topology link {first_rank}-{second_rank} has zero bandwidth"
            ),
            Self::ZeroDirectBandwidth => {
                formatter.write_str("direct collective transport has zero bandwidth")
            }
            Self::ProfileRead { path, message } => {
                write!(
                    formatter,
                    "cannot read collective autotune profile {path:?}: {message}"
                )
            }
            Self::InvalidProfile(message) => {
                write!(formatter, "invalid collective autotune profile: {message}")
            }
            Self::InvalidRing(message) => write!(formatter, "invalid collective ring: {message}"),
            Self::InvalidRingChannels(channels) => write!(
                formatter,
                "invalid collective ring channel count {channels}; expected 1..={MAX_RING_CHANNELS}"
            ),
            Self::InvalidP2pRails(rails) => write!(
                formatter,
                "invalid point-to-point rail count {rails}; expected 1..={MAX_P2P_RAILS}"
            ),
            Self::InvalidRailOrder(message) => {
                write!(formatter, "invalid collective rail order: {message}")
            }
            Self::P2pRailMismatch { tuning, session } => write!(
                formatter,
                "collective tuning declares {tuning} point-to-point rails, session has {session}"
            ),
            Self::TransportMismatch {
                tuning,
                communicator,
            } => write!(
                formatter,
                "collective tuning targets {tuning:?}, communicator uses {communicator:?}"
            ),
            Self::InvalidHierarchy(message) => {
                write!(formatter, "invalid collective hierarchy: {message}")
            }
            Self::InvalidEnvironment { name, value } => {
                write!(formatter, "invalid {name} value {value:?}")
            }
        }
    }
}

impl Error for TopologyError {}

impl CollectiveTopology {
    pub fn uniform(world_size: u32) -> Result<Self, TopologyError> {
        if world_size == 0 {
            return Err(TopologyError::EmptyWorld);
        }
        let mut topology = Self::empty(world_size)?;
        for first in 0..world_size {
            for second in first + 1..world_size {
                topology.add_link(TopologyLink {
                    first_rank: first,
                    second_rank: second,
                    bandwidth_mbps: DEFAULT_BANDWIDTH_MBPS,
                    latency_ns: DEFAULT_LATENCY_NS,
                })?;
            }
        }
        Ok(topology)
    }

    pub fn empty(world_size: u32) -> Result<Self, TopologyError> {
        if world_size == 0 {
            return Err(TopologyError::EmptyWorld);
        }
        Ok(Self {
            world_size,
            links: vec![vec![None; world_size as usize]; world_size as usize],
            rail_links: Vec::new(),
            aggregate_links: vec![vec![None; world_size as usize]; world_size as usize],
        })
    }

    pub const fn world_size(&self) -> u32 {
        self.world_size
    }

    pub fn add_link(&mut self, link: TopologyLink) -> Result<(), TopologyError> {
        self.validate_rank(link.first_rank)?;
        self.validate_rank(link.second_rank)?;
        if link.first_rank == link.second_rank {
            return Err(TopologyError::SelfLink(link.first_rank));
        }
        if link.bandwidth_mbps == 0 {
            return Err(TopologyError::ZeroBandwidth {
                first_rank: link.first_rank,
                second_rank: link.second_rank,
            });
        }
        let cost = Some(LinkCost {
            bandwidth_mbps: link.bandwidth_mbps,
            latency_ns: link.latency_ns,
        });
        self.links[link.first_rank as usize][link.second_rank as usize] = cost;
        self.links[link.second_rank as usize][link.first_rank as usize] = cost;
        Ok(())
    }

    pub fn links(&self) -> Vec<TopologyLink> {
        let mut links = Vec::new();
        for first_rank in 0..self.world_size {
            for second_rank in first_rank + 1..self.world_size {
                if let Some(link) = self.link(first_rank, second_rank) {
                    links.push(TopologyLink {
                        first_rank,
                        second_rank,
                        bandwidth_mbps: link.bandwidth_mbps,
                        latency_ns: link.latency_ns,
                    });
                }
            }
        }
        links
    }

    pub fn add_rail_link(&mut self, link: TopologyRailLink) -> Result<(), TopologyError> {
        validate_p2p_rails(link.rail.saturating_add(1))?;
        self.validate_rank(link.first_rank)?;
        self.validate_rank(link.second_rank)?;
        if link.first_rank == link.second_rank {
            return Err(TopologyError::SelfLink(link.first_rank));
        }
        if link.bandwidth_mbps == 0 {
            return Err(TopologyError::ZeroBandwidth {
                first_rank: link.first_rank,
                second_rank: link.second_rank,
            });
        }
        while self.rail_links.len() <= link.rail {
            self.rail_links.push(vec![
                vec![None; self.world_size as usize];
                self.world_size as usize
            ]);
        }
        let cost = Some(LinkCost {
            bandwidth_mbps: link.bandwidth_mbps,
            latency_ns: link.latency_ns,
        });
        self.rail_links[link.rail][link.first_rank as usize][link.second_rank as usize] = cost;
        self.rail_links[link.rail][link.second_rank as usize][link.first_rank as usize] = cost;
        Ok(())
    }

    pub fn rail_links(&self) -> Vec<TopologyRailLink> {
        let mut links = Vec::new();
        for (rail, rail_links) in self.rail_links.iter().enumerate() {
            for first_rank in 0..self.world_size {
                for second_rank in first_rank + 1..self.world_size {
                    if let Some(link) = rail_links[first_rank as usize][second_rank as usize] {
                        links.push(TopologyRailLink {
                            rail,
                            first_rank,
                            second_rank,
                            bandwidth_mbps: link.bandwidth_mbps,
                            latency_ns: link.latency_ns,
                        });
                    }
                }
            }
        }
        links
    }

    pub fn add_aggregate_link(&mut self, link: TopologyAggregateLink) -> Result<(), TopologyError> {
        self.validate_rank(link.first_rank)?;
        self.validate_rank(link.second_rank)?;
        if link.first_rank == link.second_rank {
            return Err(TopologyError::SelfLink(link.first_rank));
        }
        if link.bandwidth_mbps == 0 {
            return Err(TopologyError::ZeroBandwidth {
                first_rank: link.first_rank,
                second_rank: link.second_rank,
            });
        }
        self.aggregate_links[link.first_rank as usize][link.second_rank as usize] =
            Some(link.bandwidth_mbps);
        self.aggregate_links[link.second_rank as usize][link.first_rank as usize] =
            Some(link.bandwidth_mbps);
        Ok(())
    }

    pub fn aggregate_links(&self) -> Vec<TopologyAggregateLink> {
        let mut links = Vec::new();
        for first_rank in 0..self.world_size {
            for second_rank in first_rank + 1..self.world_size {
                if let Some(bandwidth_mbps) =
                    self.aggregate_links[first_rank as usize][second_rank as usize]
                {
                    links.push(TopologyAggregateLink {
                        first_rank,
                        second_rank,
                        bandwidth_mbps,
                    });
                }
            }
        }
        links
    }

    pub fn best_ring_order(&self) -> Result<Vec<u32>, TopologyError> {
        Ok(self.best_ring_orders(1)?.remove(0))
    }

    pub fn best_ring_orders(&self, maximum: usize) -> Result<Vec<Vec<u32>>, TopologyError> {
        if maximum == 0 {
            return Ok(Vec::new());
        }
        if self.world_size == 1 {
            return Ok(vec![vec![0]]);
        }
        if self.world_size <= EXACT_RING_SEARCH_MAX_RANKS {
            let mut order = Vec::with_capacity(self.world_size as usize);
            let mut used = vec![false; self.world_size as usize];
            let mut candidates = Vec::new();
            order.push(0);
            used[0] = true;
            collect_rings(self, &mut order, &mut used, &mut candidates)?;
            return select_diverse_rings(self, candidates, maximum);
        }
        let mut candidates = Vec::<Vec<u32>>::new();
        for start in 0..self.world_size {
            let mut order = Vec::with_capacity(self.world_size as usize);
            let mut used = vec![false; self.world_size as usize];
            order.push(start);
            used[start as usize] = true;
            while order.len() < self.world_size as usize {
                let current = *order.last().expect("ring always has a current rank");
                let next = (0..self.world_size)
                    .filter(|rank| !used[*rank as usize])
                    .filter_map(|rank| {
                        self.link(current, rank)
                            .map(|cost| (rank, cost.bandwidth_mbps, cost.latency_ns))
                    })
                    .max_by_key(|(rank, bandwidth, latency)| {
                        (*bandwidth, u64::MAX - *latency, u32::MAX - *rank)
                    })
                    .map(|(rank, _, _)| rank);
                let Some(next) = next else {
                    order.clear();
                    break;
                };
                used[next as usize] = true;
                order.push(next);
            }
            if order.len() != self.world_size as usize {
                continue;
            }
            if self.link(*order.last().unwrap(), order[0]).is_none() {
                continue;
            }
            rotate_ring_to_zero(&mut order);
            if !candidates.contains(&order) {
                candidates.push(order);
            }
        }
        select_diverse_rings(self, candidates, maximum)
    }

    fn ring_score(&self, order: &[u32]) -> Result<RingScore, TopologyError> {
        validate_ring_order(self.world_size, order)?;
        let mut bottleneck = u64::MAX;
        let mut total_bandwidth = 0_u128;
        let mut total_latency = 0_u128;
        for index in 0..order.len() {
            let first = order[index];
            let second = order[(index + 1) % order.len()];
            let link = self.link(first, second).ok_or_else(|| {
                TopologyError::InvalidRing(format!("ring edge {first}-{second} is not connected"))
            })?;
            bottleneck = bottleneck.min(link.bandwidth_mbps);
            total_bandwidth += u128::from(link.bandwidth_mbps);
            total_latency += u128::from(link.latency_ns);
        }
        Ok(RingScore {
            bottleneck,
            total_bandwidth,
            inverse_latency: u128::MAX - total_latency,
        })
    }

    fn link(&self, first: u32, second: u32) -> Option<LinkCost> {
        self.links[first as usize][second as usize]
    }

    fn link_on_rail(&self, first: u32, second: u32, rail: usize) -> Option<LinkCost> {
        self.rail_links
            .get(rail)
            .and_then(|links| links[first as usize][second as usize])
            .or_else(|| self.link(first, second))
    }

    fn aggregate_bandwidth_mbps(&self, first: u32, second: u32) -> Option<u64> {
        self.aggregate_links[first as usize][second as usize]
    }

    fn aggregate_transfer_cap_ns(&self, bytes_by_pair: &[Vec<usize>]) -> u64 {
        let mut slowest = 0_u64;
        for (first, forward) in bytes_by_pair
            .iter()
            .enumerate()
            .take(self.world_size as usize)
        {
            for (second, reverse) in bytes_by_pair
                .iter()
                .enumerate()
                .take(self.world_size as usize)
                .skip(first + 1)
            {
                let bytes = forward[second].max(reverse[first]);
                if bytes == 0 {
                    continue;
                }
                if let Some(bandwidth_mbps) =
                    self.aggregate_bandwidth_mbps(first as u32, second as u32)
                {
                    slowest = slowest.max(transfer_time_ns(bytes, bandwidth_mbps));
                }
            }
        }
        slowest
    }

    fn preferred_rail_order(&self, rail_count: usize) -> Vec<usize> {
        let mut scores = (0..rail_count)
            .map(|rail| (rail, self.rail_score(rail)))
            .collect::<Vec<_>>();
        scores.sort_by(|(first_rail, first_score), (second_rail, second_score)| {
            second_score
                .cmp(first_score)
                .then_with(|| first_rail.cmp(second_rail))
        });
        scores.into_iter().map(|(rail, _)| rail).collect()
    }

    fn rail_score(&self, rail: usize) -> RingScore {
        let mut bottleneck = u64::MAX;
        let mut total_bandwidth = 0_u128;
        let mut total_latency = 0_u128;
        let mut link_count = 0_usize;
        for first in 0..self.world_size {
            for second in first + 1..self.world_size {
                let Some(link) = self.link_on_rail(first, second, rail) else {
                    continue;
                };
                bottleneck = bottleneck.min(link.bandwidth_mbps);
                total_bandwidth = total_bandwidth.saturating_add(u128::from(link.bandwidth_mbps));
                total_latency = total_latency.saturating_add(u128::from(link.latency_ns));
                link_count += 1;
            }
        }
        if link_count == 0 {
            bottleneck = 0;
        }
        RingScore {
            bottleneck,
            total_bandwidth,
            inverse_latency: u128::MAX - total_latency,
        }
    }

    fn ring_step_time_ns_on_rail(&self, order: &[u32], bytes: usize, rail: usize) -> Option<u64> {
        (0..order.len())
            .map(|index| {
                let first = order[index];
                let second = order[(index + 1) % order.len()];
                self.link_on_rail(first, second, rail)
                    .map(|link| link_transfer_time_ns(link, bytes))
            })
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .max()
    }

    fn pairwise_time_ns(
        &self,
        order: &[u32],
        bytes_per_peer: usize,
        channel_count: usize,
        rail_order: &[usize],
    ) -> Option<u64> {
        if order.len() <= 1 {
            return Some(0);
        }
        if rail_order.is_empty() {
            return None;
        }
        let channels = channel_count.min(bytes_per_peer.max(1)).max(1);
        let rails = rail_order.len().min(channels).max(1);
        let channel_bytes = balanced_sizes(bytes_per_peer, channels);
        let mut total = 0_u64;
        for step in 1..order.len() {
            let mut rail_times = vec![0_u64; rails];
            let mut aggregate_bytes =
                vec![vec![0_usize; self.world_size as usize]; self.world_size as usize];
            for (channel, bytes) in channel_bytes.iter().copied().enumerate() {
                let rail_slot = channel % rails;
                let rail = rail_order[rail_slot];
                let mut slowest = 0_u64;
                for position in 0..order.len() {
                    let source = order[position];
                    let destination = order[(position + step) % order.len()];
                    aggregate_bytes[source as usize][destination as usize] = aggregate_bytes
                        [source as usize][destination as usize]
                        .saturating_add(bytes);
                    slowest = slowest.max(self.shortest_transfer_time_ns_on_rail(
                        source,
                        destination,
                        bytes,
                        rail,
                    )?);
                }
                rail_times[rail_slot] = rail_times[rail_slot].saturating_add(slowest);
            }
            let rail_time = rail_times.into_iter().max().unwrap_or(0);
            let aggregate_time = if rails > 1 {
                self.aggregate_transfer_cap_ns(&aggregate_bytes)
            } else {
                0
            };
            total = total.saturating_add(rail_time.max(aggregate_time));
        }
        Some(total)
    }

    fn shortest_transfer_time_ns(
        &self,
        source: u32,
        destination: u32,
        bytes: usize,
    ) -> Option<u64> {
        self.shortest_transfer_time_ns_for_rail(source, destination, bytes, None)
    }

    fn shortest_transfer_time_ns_on_rail(
        &self,
        source: u32,
        destination: u32,
        bytes: usize,
        rail: usize,
    ) -> Option<u64> {
        self.shortest_transfer_time_ns_for_rail(source, destination, bytes, Some(rail))
    }

    fn shortest_transfer_time_ns_for_rail(
        &self,
        source: u32,
        destination: u32,
        bytes: usize,
        rail: Option<usize>,
    ) -> Option<u64> {
        if source == destination {
            return Some(0);
        }
        let rank_count = self.world_size as usize;
        let mut distances = vec![u64::MAX; rank_count];
        let mut visited = vec![false; rank_count];
        distances[source as usize] = 0;
        for _ in 0..rank_count {
            let current = (0..rank_count)
                .filter(|rank| !visited[*rank])
                .min_by_key(|rank| distances[*rank])?;
            if distances[current] == u64::MAX {
                break;
            }
            if current == destination as usize {
                return Some(distances[current]);
            }
            visited[current] = true;
            for next in 0..rank_count {
                if visited[next] {
                    continue;
                }
                let link = match rail {
                    Some(rail) => self.link_on_rail(current as u32, next as u32, rail),
                    None => self.links[current][next],
                };
                let Some(link) = link else {
                    continue;
                };
                let candidate =
                    distances[current].saturating_add(link_transfer_time_ns(link, bytes));
                distances[next] = distances[next].min(candidate);
            }
        }
        (distances[destination as usize] != u64::MAX).then_some(distances[destination as usize])
    }

    fn validate_rank(&self, rank: u32) -> Result<(), TopologyError> {
        if rank >= self.world_size {
            return Err(TopologyError::RankOutOfRange {
                rank,
                world_size: self.world_size,
            });
        }
        Ok(())
    }

    fn hierarchy_candidates(&self, payload_bytes: usize) -> Vec<Vec<Vec<u32>>> {
        let mut thresholds = Vec::new();
        for first in 0..self.world_size {
            for second in first + 1..self.world_size {
                if let Some(link) = self.link(first, second) {
                    thresholds.push(link_transfer_time_ns(link, payload_bytes));
                }
            }
        }
        thresholds.sort_unstable();
        thresholds.dedup();

        let mut candidates = Vec::new();
        for threshold in thresholds {
            let mut visited = vec![false; self.world_size as usize];
            let mut groups = Vec::new();
            for first_rank in 0..self.world_size {
                if visited[first_rank as usize] {
                    continue;
                }
                visited[first_rank as usize] = true;
                let mut pending = vec![first_rank];
                let mut group = Vec::new();
                while let Some(rank) = pending.pop() {
                    group.push(rank);
                    for peer in 0..self.world_size {
                        if visited[peer as usize] {
                            continue;
                        }
                        let fast = self.link(rank, peer).is_some_and(|link| {
                            link_transfer_time_ns(link, payload_bytes) <= threshold
                        });
                        if fast {
                            visited[peer as usize] = true;
                            pending.push(peer);
                        }
                    }
                }
                group.sort_unstable();
                groups.push(group);
            }
            groups.sort_unstable_by_key(|group| group[0]);
            if groups.len() < 2
                || groups.len() == self.world_size as usize
                || candidates.contains(&groups)
            {
                continue;
            }
            if let Some(groups) = self.optimize_hierarchy(groups, payload_bytes)
                && !candidates.contains(&groups)
            {
                candidates.push(groups);
            }
        }
        candidates
    }

    fn optimize_hierarchy(
        &self,
        mut groups: Vec<Vec<u32>>,
        payload_bytes: usize,
    ) -> Option<Vec<Vec<u32>>> {
        for group in &mut groups {
            let leader = group
                .iter()
                .copied()
                .filter_map(|candidate| {
                    let total = group.iter().copied().try_fold(0_u64, |total, rank| {
                        self.shortest_transfer_time_ns_within(candidate, rank, payload_bytes, group)
                            .map(|cost| total.saturating_add(cost))
                    })?;
                    Some((total, candidate))
                })
                .min_by_key(|(total, candidate)| (*total, *candidate))?
                .1;
            group.sort_unstable();
            let leader_position = group.iter().position(|rank| *rank == leader)?;
            group.swap(0, leader_position);
        }
        let leaders = groups.iter().map(|group| group[0]).collect::<Vec<_>>();
        let order = self.best_rank_cycle_order(&leaders, payload_bytes)?;
        Some(
            order
                .into_iter()
                .map(|index| groups[index].clone())
                .collect(),
        )
    }

    fn shortest_transfer_time_ns_within(
        &self,
        source: u32,
        destination: u32,
        bytes: usize,
        allowed: &[u32],
    ) -> Option<u64> {
        if source == destination {
            return Some(0);
        }
        let mut permitted = vec![false; self.world_size as usize];
        for rank in allowed {
            permitted[*rank as usize] = true;
        }
        let rank_count = self.world_size as usize;
        let mut distances = vec![u64::MAX; rank_count];
        let mut visited = vec![false; rank_count];
        distances[source as usize] = 0;
        for _ in 0..allowed.len() {
            let current = (0..rank_count)
                .filter(|rank| permitted[*rank] && !visited[*rank])
                .min_by_key(|rank| distances[*rank])?;
            if distances[current] == u64::MAX {
                break;
            }
            if current == destination as usize {
                return Some(distances[current]);
            }
            visited[current] = true;
            for next in 0..rank_count {
                if !permitted[next] || visited[next] {
                    continue;
                }
                let Some(link) = self.links[current][next] else {
                    continue;
                };
                let candidate =
                    distances[current].saturating_add(link_transfer_time_ns(link, bytes));
                distances[next] = distances[next].min(candidate);
            }
        }
        (distances[destination as usize] != u64::MAX).then_some(distances[destination as usize])
    }

    fn best_rank_cycle_order(&self, ranks: &[u32], payload_bytes: usize) -> Option<Vec<usize>> {
        if ranks.is_empty() {
            return None;
        }
        if ranks.len() == 1 {
            return Some(vec![0]);
        }
        if ranks.len() <= EXACT_RING_SEARCH_MAX_RANKS as usize {
            let mut order = vec![0_usize];
            let mut used = vec![false; ranks.len()];
            used[0] = true;
            let mut best = None;
            collect_rank_cycles(self, ranks, payload_bytes, &mut order, &mut used, &mut best);
            return best.map(|(_, order)| order);
        }

        let mut order = vec![0_usize];
        let mut used = vec![false; ranks.len()];
        used[0] = true;
        while order.len() < ranks.len() {
            let current = *order.last()?;
            let next = (0..ranks.len())
                .filter(|candidate| !used[*candidate])
                .filter_map(|candidate| {
                    self.shortest_transfer_time_ns(ranks[current], ranks[candidate], payload_bytes)
                        .map(|cost| (cost, ranks[candidate], candidate))
                })
                .min_by_key(|(cost, rank, _)| (*cost, *rank))?
                .2;
            used[next] = true;
            order.push(next);
        }
        self.shortest_transfer_time_ns(ranks[*order.last()?], ranks[order[0]], payload_bytes)?;
        Some(order)
    }
}

fn collect_rank_cycles(
    topology: &CollectiveTopology,
    ranks: &[u32],
    payload_bytes: usize,
    order: &mut Vec<usize>,
    used: &mut [bool],
    best: &mut Option<((u64, u128), Vec<usize>)>,
) {
    if order.len() == ranks.len() {
        let mut maximum = 0_u64;
        let mut total = 0_u128;
        for position in 0..order.len() {
            let source = ranks[order[position]];
            let destination = ranks[order[(position + 1) % order.len()]];
            let Some(cost) = topology.shortest_transfer_time_ns(source, destination, payload_bytes)
            else {
                return;
            };
            maximum = maximum.max(cost);
            total = total.saturating_add(u128::from(cost));
        }
        let score = (maximum, total);
        if best.as_ref().is_none_or(|(best_score, best_order)| {
            score < *best_score || score == *best_score && order.as_slice() < best_order.as_slice()
        }) {
            *best = Some((score, order.clone()));
        }
        return;
    }
    for candidate in 1..ranks.len() {
        if used[candidate] {
            continue;
        }
        used[candidate] = true;
        order.push(candidate);
        collect_rank_cycles(topology, ranks, payload_bytes, order, used, best);
        order.pop();
        used[candidate] = false;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RingScore {
    bottleneck: u64,
    total_bandwidth: u128,
    inverse_latency: u128,
}

fn collect_rings(
    topology: &CollectiveTopology,
    order: &mut Vec<u32>,
    used: &mut [bool],
    candidates: &mut Vec<Vec<u32>>,
) -> Result<(), TopologyError> {
    if order.len() == topology.world_size as usize {
        if topology.link(*order.last().unwrap(), order[0]).is_none() {
            return Ok(());
        }
        let reversed = std::iter::once(0)
            .chain(order[1..].iter().rev().copied())
            .collect::<Vec<_>>();
        if order.as_slice() <= reversed.as_slice() {
            candidates.push(order.clone());
        }
        return Ok(());
    }
    let current = *order.last().expect("ring search has a current rank");
    for candidate in 1..topology.world_size {
        if used[candidate as usize] || topology.link(current, candidate).is_none() {
            continue;
        }
        used[candidate as usize] = true;
        order.push(candidate);
        collect_rings(topology, order, used, candidates)?;
        order.pop();
        used[candidate as usize] = false;
    }
    Ok(())
}

fn rotate_ring_to_zero(order: &mut [u32]) {
    if let Some(position) = order.iter().position(|rank| *rank == 0) {
        order.rotate_left(position);
    }
    let reversed = std::iter::once(0)
        .chain(order[1..].iter().rev().copied())
        .collect::<Vec<_>>();
    if reversed.as_slice() < order {
        order.copy_from_slice(&reversed);
    }
}

fn select_diverse_rings(
    topology: &CollectiveTopology,
    mut candidates: Vec<Vec<u32>>,
    maximum: usize,
) -> Result<Vec<Vec<u32>>, TopologyError> {
    if candidates.is_empty() {
        return Err(TopologyError::InvalidRing(
            "topology does not contain a closed rank ring".into(),
        ));
    }
    candidates.sort_unstable();
    candidates.dedup();
    let rank_count = topology.world_size as usize;
    let mut edge_usage = vec![vec![0_u16; rank_count]; rank_count];
    let mut selected = Vec::with_capacity(maximum.min(candidates.len()));
    while selected.len() < maximum && !candidates.is_empty() {
        let mut best_index = 0;
        let mut best_key = None::<(u16, u64, RingScore)>;
        for (index, order) in candidates.iter().enumerate() {
            let mut maximum_usage = 0_u16;
            let mut total_usage = 0_u64;
            for position in 0..order.len() {
                let first = order[position] as usize;
                let second = order[(position + 1) % order.len()] as usize;
                let usage = edge_usage[first][second];
                maximum_usage = maximum_usage.max(usage);
                total_usage = total_usage.saturating_add(u64::from(usage));
            }
            let key = (
                u16::MAX - maximum_usage,
                u64::MAX - total_usage,
                topology.ring_score(order)?,
            );
            if best_key.is_none_or(|current| key > current) {
                best_key = Some(key);
                best_index = index;
            }
        }
        let order = candidates.remove(best_index);
        for position in 0..order.len() {
            let first = order[position] as usize;
            let second = order[(position + 1) % order.len()] as usize;
            edge_usage[first][second] = edge_usage[first][second].saturating_add(1);
            edge_usage[second][first] = edge_usage[second][first].saturating_add(1);
        }
        selected.push(order);
    }
    Ok(selected)
}

fn balanced_sizes(total: usize, parts: usize) -> Vec<usize> {
    let base = total / parts;
    let remainder = total % parts;
    (0..parts)
        .map(|part| base + usize::from(part < remainder))
        .collect()
}

fn link_transfer_time_ns(link: LinkCost, bytes: usize) -> u64 {
    link.latency_ns
        .saturating_add(transfer_time_ns(bytes, link.bandwidth_mbps))
}

fn transfer_time_ns(bytes: usize, bandwidth_mbps: u64) -> u64 {
    saturating_u64(transfer_time_ns_u128(bytes as u128, bandwidth_mbps))
}

fn transfer_time_ns_u128(bytes: u128, bandwidth_mbps: u64) -> u128 {
    let numerator = bytes.saturating_mul(1_000);
    let denominator = u128::from(bandwidth_mbps);
    numerator.saturating_add(denominator - 1) / denominator
}

fn saturating_u64(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn fingerprint_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn ring_edges(order: &[u32]) -> Vec<(u32, u32)> {
    let mut edges = (0..order.len())
        .map(|index| {
            let first = order[index];
            let second = order[(index + 1) % order.len()];
            (first.min(second), first.max(second))
        })
        .collect::<Vec<_>>();
    edges.sort_unstable();
    edges
}

fn ring_orders_with_primary(topology: &CollectiveTopology, primary: &[u32]) -> Vec<Vec<u32>> {
    let mut orders = vec![primary.to_vec()];
    let primary_edges = ring_edges(primary);
    if let Ok(candidates) = topology.best_ring_orders(MAX_RING_CHANNELS) {
        for candidate in candidates {
            let edges = ring_edges(&candidate);
            if edges != primary_edges && !orders.iter().any(|order| ring_edges(order) == edges) {
                orders.push(candidate);
            }
        }
    }
    orders
}

impl CollectiveTuning {
    pub fn from_environment(world_size: u32) -> Result<Self, TopologyError> {
        Self::from_environment_impl(world_size, None)
    }

    pub fn from_environment_with_topology(
        topology: CollectiveTopology,
    ) -> Result<Self, TopologyError> {
        let world_size = topology.world_size();
        Self::from_environment_impl(world_size, Some(topology))
    }

    pub fn topology_probe_requested() -> Result<bool, TopologyError> {
        match env::var("GX1_TOPOLOGY_LINKS") {
            Ok(value) => Ok(value.trim().eq_ignore_ascii_case("probe")),
            Err(env::VarError::NotPresent) => Ok(false),
            Err(env::VarError::NotUnicode(value)) => Err(TopologyError::InvalidEnvironment {
                name: "GX1_TOPOLOGY_LINKS",
                value: value.to_string_lossy().into_owned(),
            }),
        }
    }

    fn from_environment_impl(
        world_size: u32,
        probed_topology: Option<CollectiveTopology>,
    ) -> Result<Self, TopologyError> {
        let algorithm_policy = match env::var("GX1_COLLECTIVE_ALGORITHM") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "auto" => AlgorithmPolicy::Auto,
                "direct" => AlgorithmPolicy::Direct,
                "ring" => AlgorithmPolicy::Ring,
                "hierarchical" => AlgorithmPolicy::Hierarchical,
                _ => {
                    return Err(TopologyError::InvalidEnvironment {
                        name: "GX1_COLLECTIVE_ALGORITHM",
                        value,
                    });
                }
            },
            Err(env::VarError::NotPresent) => AlgorithmPolicy::Auto,
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_COLLECTIVE_ALGORITHM",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        };
        let ring_threshold_bytes =
            parse_usize_environment("GX1_RING_THRESHOLD_BYTES", DEFAULT_RING_THRESHOLD_BYTES)?;
        let ring_channels = parse_usize_environment("GX1_RING_CHANNELS", 1)?;
        validate_ring_channels(ring_channels)?;
        let p2p_rails = parse_usize_environment("GX1_P2P_RAILS", 1)?;
        validate_p2p_rails(p2p_rails)?;
        let direct_bandwidth_mbps =
            parse_u64_environment("GX1_DIRECT_BANDWIDTH_MBPS", DEFAULT_BANDWIDTH_MBPS)?;
        if direct_bandwidth_mbps == 0 {
            return Err(TopologyError::InvalidEnvironment {
                name: "GX1_DIRECT_BANDWIDTH_MBPS",
                value: "0".into(),
            });
        }
        let direct_latency_ns = parse_u64_environment("GX1_DIRECT_LATENCY_NS", DEFAULT_LATENCY_NS)?;
        let transport = parse_collective_transport_environment()?;
        let mut topology = match env::var("GX1_TOPOLOGY_LINKS") {
            Ok(value) if value.trim().eq_ignore_ascii_case("probe") => {
                probed_topology.ok_or_else(|| TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.clone(),
                })?
            }
            Ok(value) => parse_topology_links(world_size, &value)?,
            Err(env::VarError::NotPresent) => match probed_topology {
                Some(topology) => topology,
                None => CollectiveTopology::uniform(world_size)?,
            },
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        };
        match env::var("GX1_TOPOLOGY_RAIL_LINKS") {
            Ok(value) => parse_topology_rail_links(&mut topology, &value, p2p_rails)?,
            Err(env::VarError::NotPresent) => {}
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_RAIL_LINKS",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        }
        match env::var("GX1_TOPOLOGY_AGGREGATE_LINKS") {
            Ok(value) => parse_topology_aggregate_links(&mut topology, &value)?,
            Err(env::VarError::NotPresent) => {}
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_AGGREGATE_LINKS",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        }
        let rail_order_override = match env::var("GX1_RAIL_ORDER") {
            Ok(value) if value.trim().eq_ignore_ascii_case("auto") => None,
            Ok(value) => Some(parse_rail_order(p2p_rails, &value)?),
            Err(env::VarError::NotPresent) => None,
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_RAIL_ORDER",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        };
        let ring_order = match env::var("GX1_RING_ORDER") {
            Ok(value) => parse_ring_order(world_size, &value)?,
            Err(env::VarError::NotPresent) => topology.best_ring_order()?,
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_RING_ORDER",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        };
        let (hierarchy_groups, inferred_hierarchy_bytes) = match env::var("GX1_TOPOLOGY_GROUPS") {
            Ok(value) if value.trim().eq_ignore_ascii_case("auto") => {
                let payload_bytes = parse_usize_environment(
                    "GX1_TOPOLOGY_GROUP_PAYLOAD_BYTES",
                    DEFAULT_HIERARCHY_INFERENCE_BYTES,
                )?;
                if payload_bytes == 0 {
                    return Err(TopologyError::InvalidEnvironment {
                        name: "GX1_TOPOLOGY_GROUP_PAYLOAD_BYTES",
                        value: "0".into(),
                    });
                }
                (None, Some(payload_bytes))
            }
            Ok(value) => (Some(parse_hierarchy_groups(world_size, &value)?), None),
            Err(env::VarError::NotPresent) => (None, None),
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_GROUPS",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        };
        let ring_orders = ring_orders_with_primary(&topology, &ring_order);
        let mut tuning = Self {
            algorithm_policy,
            ring_threshold_bytes,
            ring_order,
            ring_orders,
            ring_channels,
            p2p_rails,
            rail_order_override,
            hierarchy_groups,
            topology,
            direct_bandwidth_mbps,
            direct_latency_ns,
            transport,
            autotune_profile: None,
        };
        if let Some(payload_bytes) = inferred_hierarchy_bytes {
            tuning = tuning.with_inferred_hierarchy(payload_bytes)?;
        }
        if algorithm_policy == AlgorithmPolicy::Hierarchical && tuning.hierarchy_groups.is_none() {
            return Err(TopologyError::InvalidHierarchy(
                "GX1_COLLECTIVE_ALGORITHM=hierarchical requires explicit or inferred topology groups"
                    .into(),
            ));
        }
        match env::var("GX1_COLLECTIVE_AUTOTUNE_PROFILE") {
            Ok(path) => tuning = tuning.with_autotune_profile_path(path)?,
            Err(env::VarError::NotPresent) => {}
            Err(env::VarError::NotUnicode(value)) => {
                return Err(TopologyError::InvalidEnvironment {
                    name: "GX1_COLLECTIVE_AUTOTUNE_PROFILE",
                    value: value.to_string_lossy().into_owned(),
                });
            }
        }
        Ok(tuning)
    }

    pub fn new(
        algorithm_policy: AlgorithmPolicy,
        ring_threshold_bytes: usize,
        ring_order: Vec<u32>,
    ) -> Result<Self, TopologyError> {
        let world_size = u32::try_from(ring_order.len())
            .map_err(|_| TopologyError::InvalidRing("ring is too large".into()))?;
        let topology = CollectiveTopology::uniform(world_size)?;
        Self::from_topology(
            algorithm_policy,
            ring_threshold_bytes,
            topology,
            Some(ring_order),
        )
    }

    pub fn from_topology(
        algorithm_policy: AlgorithmPolicy,
        ring_threshold_bytes: usize,
        topology: CollectiveTopology,
        ring_order: Option<Vec<u32>>,
    ) -> Result<Self, TopologyError> {
        let world_size = topology.world_size();
        let ring_order = match ring_order {
            Some(order) => order,
            None => topology.best_ring_order()?,
        };
        validate_ring_order(world_size, &ring_order)?;
        let ring_orders = ring_orders_with_primary(&topology, &ring_order);
        Ok(Self {
            algorithm_policy,
            ring_threshold_bytes,
            ring_order,
            ring_orders,
            ring_channels: 1,
            p2p_rails: 1,
            rail_order_override: None,
            hierarchy_groups: None,
            topology,
            direct_bandwidth_mbps: DEFAULT_BANDWIDTH_MBPS,
            direct_latency_ns: DEFAULT_LATENCY_NS,
            transport: CollectiveTransport::TcpHostStaged,
            autotune_profile: None,
        })
    }

    pub fn with_direct_transport(
        mut self,
        bandwidth_mbps: u64,
        latency_ns: u64,
    ) -> Result<Self, TopologyError> {
        if bandwidth_mbps == 0 {
            return Err(TopologyError::ZeroDirectBandwidth);
        }
        self.direct_bandwidth_mbps = bandwidth_mbps;
        self.direct_latency_ns = latency_ns;
        if let Some(profile) = &self.autotune_profile {
            self.validate_autotune_profile(profile)?;
        }
        Ok(self)
    }

    pub fn with_transport(mut self, transport: CollectiveTransport) -> Result<Self, TopologyError> {
        self.transport = transport;
        if let Some(profile) = &self.autotune_profile {
            self.validate_autotune_profile(profile)?;
        }
        Ok(self)
    }

    pub const fn transport(&self) -> CollectiveTransport {
        self.transport
    }

    pub fn with_hierarchy(mut self, groups: Vec<Vec<u32>>) -> Result<Self, TopologyError> {
        validate_hierarchy_groups(self.topology.world_size(), &groups)?;
        self.hierarchy_groups = Some(groups);
        if let Some(profile) = &self.autotune_profile {
            self.validate_autotune_profile(profile)?;
        }
        Ok(self)
    }

    pub fn with_inferred_hierarchy(mut self, payload_bytes: usize) -> Result<Self, TopologyError> {
        if payload_bytes == 0 {
            return Err(TopologyError::InvalidHierarchy(
                "automatic grouping payload must be greater than zero".into(),
            ));
        }
        let mut best = None::<(u128, Vec<Vec<u32>>)>;
        for groups in self.topology.hierarchy_candidates(payload_bytes) {
            self.hierarchy_groups = Some(groups.clone());
            let score = [
                CollectiveKind::AllReduce,
                CollectiveKind::AllGather,
                CollectiveKind::ReduceScatter,
            ]
            .into_iter()
            .try_fold(0_u128, |total, kind| {
                self.hierarchical_estimated_time_ns(kind, payload_bytes)
                    .map(|estimate| total.saturating_add(u128::from(estimate)))
            });
            let Some(score) = score else {
                continue;
            };
            if best.as_ref().is_none_or(|(best_score, best_groups)| {
                score < *best_score || score == *best_score && groups < *best_groups
            }) {
                best = Some((score, groups));
            }
        }
        let Some((_, groups)) = best else {
            self.hierarchy_groups = None;
            return Err(TopologyError::InvalidHierarchy(
                "automatic grouping found no distinct fast-link domains".into(),
            ));
        };
        self.hierarchy_groups = Some(groups);
        if let Some(profile) = &self.autotune_profile {
            self.validate_autotune_profile(profile)?;
        }
        Ok(self)
    }

    pub fn hierarchy_groups(&self) -> Option<&[Vec<u32>]> {
        self.hierarchy_groups.as_deref()
    }

    pub fn with_ring_channels(mut self, channels: usize) -> Result<Self, TopologyError> {
        validate_ring_channels(channels)?;
        self.ring_channels = channels;
        if let Some(profile) = &self.autotune_profile {
            self.validate_autotune_profile(profile)?;
        }
        Ok(self)
    }

    pub const fn ring_channels(&self) -> usize {
        self.ring_channels
    }

    pub fn ring_orders(&self) -> &[Vec<u32>] {
        &self.ring_orders
    }

    pub fn topology_links(&self) -> Vec<TopologyLink> {
        self.topology.links()
    }

    pub fn topology_rail_links(&self) -> Vec<TopologyRailLink> {
        self.topology.rail_links()
    }

    pub fn topology_aggregate_links(&self) -> Vec<TopologyAggregateLink> {
        self.topology.aggregate_links()
    }

    pub fn ring_order_for_channel(&self, channel: usize) -> &[u32] {
        &self.ring_orders[channel % self.ring_orders.len()]
    }

    pub fn rail_order(&self) -> Vec<usize> {
        self.rail_order_override
            .clone()
            .unwrap_or_else(|| self.topology.preferred_rail_order(self.p2p_rails))
    }

    pub fn rail_for_channel(&self, channel: usize) -> usize {
        let rail_order = self.rail_order();
        rail_order[channel % rail_order.len()]
    }

    pub fn with_p2p_rails(mut self, rails: usize) -> Result<Self, TopologyError> {
        validate_p2p_rails(rails)?;
        if let Some(order) = &self.rail_order_override {
            validate_rail_order(rails, order)?;
        }
        self.p2p_rails = rails;
        if let Some(profile) = &self.autotune_profile {
            self.validate_autotune_profile(profile)?;
        }
        Ok(self)
    }

    pub fn with_rail_order(mut self, order: Vec<usize>) -> Result<Self, TopologyError> {
        validate_rail_order(self.p2p_rails, &order)?;
        self.rail_order_override = Some(order);
        if let Some(profile) = &self.autotune_profile {
            self.validate_autotune_profile(profile)?;
        }
        Ok(self)
    }

    pub const fn p2p_rails(&self) -> usize {
        self.p2p_rails
    }

    pub fn with_autotune_profile_path(
        mut self,
        path: impl AsRef<Path>,
    ) -> Result<Self, TopologyError> {
        let path = path.as_ref();
        let encoded = fs::read_to_string(path).map_err(|error| TopologyError::ProfileRead {
            path: path.display().to_string(),
            message: error.to_string(),
        })?;
        let document = serde_json::from_str::<serde_json::Value>(&encoded).map_err(|error| {
            TopologyError::InvalidProfile(format!("{} is not valid JSON: {error}", path.display()))
        })?;
        let detected_topology_links = match document.get("detected_topology_links") {
            Some(serde_json::Value::String(value)) => Some(value.as_str()),
            Some(serde_json::Value::Null) | None => None,
            Some(_) => {
                return Err(TopologyError::InvalidProfile(
                    "detected_topology_links must be a string or null".into(),
                ));
            }
        };
        let detected_topology_rail_links = match document.get("detected_topology_rail_links") {
            Some(serde_json::Value::String(value)) => Some(value.as_str()),
            Some(serde_json::Value::Null) | None => None,
            Some(_) => {
                return Err(TopologyError::InvalidProfile(
                    "detected_topology_rail_links must be a string or null".into(),
                ));
            }
        };
        let detected_topology_aggregate_links =
            match document.get("detected_topology_aggregate_links") {
                Some(serde_json::Value::String(value)) => Some(value.as_str()),
                Some(serde_json::Value::Null) | None => None,
                Some(_) => {
                    return Err(TopologyError::InvalidProfile(
                        "detected_topology_aggregate_links must be a string or null".into(),
                    ));
                }
            };
        let profile = serde_json::from_str(&encoded).map_err(|error| {
            TopologyError::InvalidProfile(format!("{} is not valid JSON: {error}", path.display()))
        })?;
        self.validate_autotune_profile_with_detected(
            &profile,
            detected_topology_links,
            detected_topology_rail_links,
            detected_topology_aggregate_links,
            Self::topology_probe_requested()?,
        )?;
        self.autotune_profile = Some(profile);
        Ok(self)
    }

    pub fn with_autotune_profile(
        mut self,
        profile: CollectiveAutotuneProfile,
    ) -> Result<Self, TopologyError> {
        self.validate_autotune_profile(&profile)?;
        self.autotune_profile = Some(profile);
        Ok(self)
    }

    pub fn topology_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        fingerprint_u64(&mut hash, u64::from(self.topology.world_size));
        for first in 0..self.topology.world_size {
            for second in first + 1..self.topology.world_size {
                fingerprint_u64(&mut hash, u64::from(first));
                fingerprint_u64(&mut hash, u64::from(second));
                match self.topology.link(first, second) {
                    Some(link) => {
                        fingerprint_u64(&mut hash, 1);
                        fingerprint_u64(&mut hash, link.bandwidth_mbps);
                        fingerprint_u64(&mut hash, link.latency_ns);
                    }
                    None => fingerprint_u64(&mut hash, 0),
                }
            }
        }
        if !self.topology.rail_links.is_empty() {
            fingerprint_u64(&mut hash, 0x4758_5241_494c_4c4b);
            fingerprint_u64(&mut hash, self.topology.rail_links.len() as u64);
            for (rail, links) in self.topology.rail_links.iter().enumerate() {
                fingerprint_u64(&mut hash, rail as u64);
                for first in 0..self.topology.world_size {
                    for second in first + 1..self.topology.world_size {
                        fingerprint_u64(&mut hash, u64::from(first));
                        fingerprint_u64(&mut hash, u64::from(second));
                        match links[first as usize][second as usize] {
                            Some(link) => {
                                fingerprint_u64(&mut hash, 1);
                                fingerprint_u64(&mut hash, link.bandwidth_mbps);
                                fingerprint_u64(&mut hash, link.latency_ns);
                            }
                            None => fingerprint_u64(&mut hash, 0),
                        }
                    }
                }
            }
        }
        if self
            .topology
            .aggregate_links
            .iter()
            .flatten()
            .any(Option::is_some)
        {
            fingerprint_u64(&mut hash, 0x4758_4147_4752_4547);
            for first in 0..self.topology.world_size {
                for second in first + 1..self.topology.world_size {
                    fingerprint_u64(&mut hash, u64::from(first));
                    fingerprint_u64(&mut hash, u64::from(second));
                    match self.topology.aggregate_bandwidth_mbps(first, second) {
                        Some(bandwidth_mbps) => {
                            fingerprint_u64(&mut hash, 1);
                            fingerprint_u64(&mut hash, bandwidth_mbps);
                        }
                        None => fingerprint_u64(&mut hash, 0),
                    }
                }
            }
        }
        fingerprint_u64(&mut hash, self.ring_orders.len() as u64);
        for order in &self.ring_orders {
            fingerprint_u64(&mut hash, order.len() as u64);
            for rank in order {
                fingerprint_u64(&mut hash, u64::from(*rank));
            }
        }
        match &self.hierarchy_groups {
            Some(groups) => {
                fingerprint_u64(&mut hash, groups.len() as u64);
                for group in groups {
                    fingerprint_u64(&mut hash, group.len() as u64);
                    for rank in group {
                        fingerprint_u64(&mut hash, u64::from(*rank));
                    }
                }
            }
            None => fingerprint_u64(&mut hash, 0),
        }
        fingerprint_u64(&mut hash, self.direct_bandwidth_mbps);
        fingerprint_u64(&mut hash, self.direct_latency_ns);
        if self.p2p_rails != 1 {
            fingerprint_u64(&mut hash, 0x4758_5241_494c_0001);
            fingerprint_u64(&mut hash, self.p2p_rails as u64);
        }
        let rail_order = self.rail_order();
        if !rail_order.iter().copied().eq(0..self.p2p_rails) {
            fingerprint_u64(&mut hash, 0x4758_5241_494c_4f52);
            for rail in rail_order {
                fingerprint_u64(&mut hash, rail as u64);
            }
        }
        hash
    }

    pub fn topology_fingerprint_hex(&self) -> String {
        format!("{:016x}", self.topology_fingerprint())
    }

    pub fn execution_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        fingerprint_u64(&mut hash, COLLECTIVE_EXECUTION_REVISION);
        fingerprint_u64(
            &mut hash,
            match self.transport {
                CollectiveTransport::HostStaged => 1,
                CollectiveTransport::TcpHostStaged => 2,
                CollectiveTransport::TcpPeer => 3,
                CollectiveTransport::PciePeer => 4,
                CollectiveTransport::Rdma => 5,
                CollectiveTransport::GxLink => 6,
            },
        );
        hash
    }

    pub fn execution_fingerprint_hex(&self) -> String {
        format!("{:016x}", self.execution_fingerprint())
    }

    fn validate_autotune_profile(
        &self,
        profile: &CollectiveAutotuneProfile,
    ) -> Result<(), TopologyError> {
        self.validate_autotune_profile_with_detected(profile, None, None, None, false)
    }

    fn validate_autotune_profile_with_detected(
        &self,
        profile: &CollectiveAutotuneProfile,
        detected_topology_links: Option<&str>,
        detected_topology_rail_links: Option<&str>,
        detected_topology_aggregate_links: Option<&str>,
        allow_detected_topology: bool,
    ) -> Result<(), TopologyError> {
        if profile.version != COLLECTIVE_AUTOTUNE_PROFILE_VERSION {
            return Err(TopologyError::InvalidProfile(format!(
                "unsupported version {}, expected {COLLECTIVE_AUTOTUNE_PROFILE_VERSION}",
                profile.version,
            )));
        }
        let world_size = self.ring_order.len() as u32;
        if profile.world_size != world_size {
            return Err(TopologyError::InvalidProfile(format!(
                "world size {} does not match configured world size {world_size}",
                profile.world_size
            )));
        }
        let fingerprint = profile
            .topology_fingerprint
            .strip_prefix("0x")
            .unwrap_or(&profile.topology_fingerprint)
            .to_ascii_lowercase();
        let expected = self.topology_fingerprint_hex();
        if fingerprint != expected {
            let detected_matches = match detected_topology_links {
                Some(encoded) if allow_detected_topology => {
                    let mut detected = parse_topology_links(world_size, encoded)?;
                    if let Some(encoded) = detected_topology_rail_links {
                        parse_topology_rail_links(&mut detected, encoded, self.p2p_rails)?;
                    }
                    if let Some(encoded) = detected_topology_aggregate_links {
                        parse_topology_aggregate_links(&mut detected, encoded)?;
                    }
                    let mut pinned = self.clone();
                    pinned.topology = detected.clone();
                    pinned.autotune_profile = None;
                    probed_topology_matches(&self.topology, &detected)
                        && self.rail_order() == pinned.rail_order()
                        && pinned.topology_fingerprint_hex() == fingerprint
                }
                _ => false,
            };
            if !detected_matches {
                return Err(TopologyError::InvalidProfile(format!(
                    "topology fingerprint {fingerprint:?} does not match {expected:?}"
                )));
            }
        }
        let execution_fingerprint = profile
            .execution_fingerprint
            .strip_prefix("0x")
            .unwrap_or(&profile.execution_fingerprint)
            .to_ascii_lowercase();
        let expected_execution = self.execution_fingerprint_hex();
        if execution_fingerprint != expected_execution {
            return Err(TopologyError::InvalidProfile(format!(
                "execution fingerprint {execution_fingerprint:?} does not match {expected_execution:?} for transport {:?}",
                self.transport
            )));
        }
        if profile.entries.is_empty() {
            return Err(TopologyError::InvalidProfile(
                "profile contains no measurements".into(),
            ));
        }
        for entry in &profile.entries {
            if entry.min_payload_bytes > entry.max_payload_bytes {
                return Err(TopologyError::InvalidProfile(format!(
                    "{:?} range {}..={} is reversed",
                    entry.operation, entry.min_payload_bytes, entry.max_payload_bytes
                )));
            }
            if entry.measured_time_ns == 0 {
                return Err(TopologyError::InvalidProfile(format!(
                    "{:?} range {}..={} has zero measured time",
                    entry.operation, entry.min_payload_bytes, entry.max_payload_bytes
                )));
            }
            validate_ring_channels(entry.ring_channels).map_err(|_| {
                TopologyError::InvalidProfile(format!(
                    "{:?} range {}..={} has invalid ring channel count {}",
                    entry.operation,
                    entry.min_payload_bytes,
                    entry.max_payload_bytes,
                    entry.ring_channels
                ))
            })?;
            let valid_algorithm = match entry.operation {
                CollectiveKind::AllToAll => matches!(
                    entry.algorithm,
                    CollectiveAlgorithm::Direct | CollectiveAlgorithm::Pairwise
                ),
                CollectiveKind::AllReduce
                | CollectiveKind::AllGather
                | CollectiveKind::ReduceScatter => matches!(
                    entry.algorithm,
                    CollectiveAlgorithm::Direct
                        | CollectiveAlgorithm::Ring
                        | CollectiveAlgorithm::Hierarchical
                ),
            };
            if !valid_algorithm {
                return Err(TopologyError::InvalidProfile(format!(
                    "algorithm {:?} cannot execute {:?}",
                    entry.algorithm, entry.operation
                )));
            }
            if entry.algorithm == CollectiveAlgorithm::Hierarchical
                && self.hierarchy_groups.is_none()
            {
                return Err(TopologyError::InvalidProfile(
                    "hierarchical measurement requires configured topology groups".into(),
                ));
            }
        }
        for kind in [
            CollectiveKind::AllReduce,
            CollectiveKind::AllGather,
            CollectiveKind::ReduceScatter,
            CollectiveKind::AllToAll,
        ] {
            let mut ranges = profile
                .entries
                .iter()
                .filter(|entry| entry.operation == kind)
                .map(|entry| (entry.min_payload_bytes, entry.max_payload_bytes))
                .collect::<Vec<_>>();
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                if pair[1].0 <= pair[0].1 {
                    return Err(TopologyError::InvalidProfile(format!(
                        "{kind:?} ranges {}..={} and {}..={} overlap",
                        pair[0].0, pair[0].1, pair[1].0, pair[1].1
                    )));
                }
            }
        }
        Ok(())
    }

    fn autotune_entry(
        &self,
        kind: CollectiveKind,
        payload_bytes: usize,
    ) -> Option<&CollectiveAutotuneEntry> {
        let payload_bytes = u64::try_from(payload_bytes).unwrap_or(u64::MAX);
        self.autotune_profile
            .as_ref()?
            .entries
            .iter()
            .find(|entry| {
                entry.operation == kind
                    && payload_bytes >= entry.min_payload_bytes
                    && payload_bytes <= entry.max_payload_bytes
            })
    }

    pub fn all_reduce_algorithm(&self, payload_bytes: usize) -> CollectiveAlgorithm {
        self.plan(CollectiveKind::AllReduce, payload_bytes)
            .algorithm
    }

    pub fn all_gather_algorithm(&self, payload_bytes: usize) -> CollectiveAlgorithm {
        self.plan(CollectiveKind::AllGather, payload_bytes)
            .algorithm
    }

    pub fn reduce_scatter_algorithm(&self, payload_bytes: usize) -> CollectiveAlgorithm {
        self.plan(CollectiveKind::ReduceScatter, payload_bytes)
            .algorithm
    }

    pub fn all_to_all_algorithm(&self, payload_bytes: usize) -> CollectiveAlgorithm {
        self.plan(CollectiveKind::AllToAll, payload_bytes).algorithm
    }

    pub fn plan(&self, kind: CollectiveKind, payload_bytes: usize) -> CollectivePlan {
        let direct_estimated_time_ns = self.direct_estimated_time_ns(kind, payload_bytes);
        let peer_algorithm = match kind {
            CollectiveKind::AllToAll => CollectiveAlgorithm::Pairwise,
            CollectiveKind::AllReduce
            | CollectiveKind::AllGather
            | CollectiveKind::ReduceScatter => CollectiveAlgorithm::Ring,
        };
        let peer_estimated_time_ns = self.peer_estimated_time_ns(kind, payload_bytes);
        let hierarchical_estimated_time_ns =
            self.hierarchical_estimated_time_ns(kind, payload_bytes);
        let mut ring_channels = self.ring_channels;
        let (algorithm, source, measured_time_ns) = if self.ring_order.len() <= 1 {
            (
                CollectiveAlgorithm::Direct,
                match self.algorithm_policy {
                    AlgorithmPolicy::Auto => CollectivePlanSource::Model,
                    AlgorithmPolicy::Direct
                    | AlgorithmPolicy::Ring
                    | AlgorithmPolicy::Hierarchical => CollectivePlanSource::Policy,
                },
                None,
            )
        } else {
            match self.algorithm_policy {
                AlgorithmPolicy::Direct => (
                    CollectiveAlgorithm::Direct,
                    CollectivePlanSource::Policy,
                    None,
                ),
                AlgorithmPolicy::Ring => (peer_algorithm, CollectivePlanSource::Policy, None),
                AlgorithmPolicy::Hierarchical => (
                    if kind == CollectiveKind::AllToAll {
                        peer_algorithm
                    } else if hierarchical_estimated_time_ns.is_some() {
                        CollectiveAlgorithm::Hierarchical
                    } else {
                        peer_algorithm
                    },
                    CollectivePlanSource::Policy,
                    None,
                ),
                AlgorithmPolicy::Auto => {
                    if let Some(entry) = self.autotune_entry(kind, payload_bytes) {
                        ring_channels = entry.ring_channels;
                        (
                            entry.algorithm,
                            CollectivePlanSource::Autotune,
                            Some(entry.measured_time_ns),
                        )
                    } else if payload_bytes < self.ring_threshold_bytes {
                        (
                            CollectiveAlgorithm::Direct,
                            CollectivePlanSource::Model,
                            None,
                        )
                    } else {
                        let mut best = (CollectiveAlgorithm::Direct, direct_estimated_time_ns);
                        if let Some(estimate) = peer_estimated_time_ns
                            && estimate < best.1
                        {
                            best = (peer_algorithm, estimate);
                        }
                        if kind != CollectiveKind::AllToAll
                            && let Some(estimate) = hierarchical_estimated_time_ns
                            && estimate < best.1
                        {
                            best = (CollectiveAlgorithm::Hierarchical, estimate);
                        }
                        (best.0, CollectivePlanSource::Model, None)
                    }
                }
            }
        };
        let estimated_time_ns = measured_time_ns.unwrap_or_else(|| match algorithm {
            CollectiveAlgorithm::Direct => direct_estimated_time_ns,
            CollectiveAlgorithm::Ring | CollectiveAlgorithm::Pairwise => {
                peer_estimated_time_ns.unwrap_or(u64::MAX)
            }
            CollectiveAlgorithm::Hierarchical => hierarchical_estimated_time_ns.unwrap_or(u64::MAX),
        });
        CollectivePlan {
            algorithm,
            source,
            estimated_time_ns,
            direct_estimated_time_ns,
            peer_estimated_time_ns,
            hierarchical_estimated_time_ns,
            ring_channels,
        }
    }

    fn direct_estimated_time_ns(&self, kind: CollectiveKind, payload_bytes: usize) -> u64 {
        let ranks = self.ring_order.len() as u128;
        if ranks <= 1 {
            return 0;
        }
        let payload = payload_bytes as u128;
        let response = match kind {
            CollectiveKind::AllReduce | CollectiveKind::AllToAll => payload,
            CollectiveKind::AllGather => payload.saturating_mul(ranks),
            CollectiveKind::ReduceScatter => payload.div_ceil(ranks),
        };
        let aggregate_bytes = ranks.saturating_mul(payload.saturating_add(response));
        let transfer_ns = transfer_time_ns_u128(aggregate_bytes, self.direct_bandwidth_mbps);
        let latency_ns = (u128::from(self.direct_latency_ns))
            .saturating_mul(2)
            .saturating_mul(ranks);
        saturating_u64(transfer_ns.saturating_add(latency_ns))
    }

    fn peer_estimated_time_ns(&self, kind: CollectiveKind, payload_bytes: usize) -> Option<u64> {
        let ranks = self.ring_order.len();
        if ranks <= 1 {
            return Some(0);
        }
        let rail_order = self.rail_order();
        let chunk_bytes = payload_bytes.div_ceil(ranks);
        match kind {
            CollectiveKind::AllReduce => self
                .multi_ring_step_time_ns(chunk_bytes)
                .map(|step| step.saturating_mul((2 * (ranks - 1)) as u64)),
            CollectiveKind::AllGather => self
                .multi_ring_step_time_ns(payload_bytes)
                .map(|step| step.saturating_mul((ranks - 1) as u64)),
            CollectiveKind::ReduceScatter => self
                .multi_ring_step_time_ns(chunk_bytes)
                .map(|step| step.saturating_mul((ranks - 1) as u64)),
            CollectiveKind::AllToAll => self.topology.pairwise_time_ns(
                &self.ring_order,
                chunk_bytes,
                self.ring_channels,
                &rail_order,
            ),
        }
    }

    fn hierarchical_estimated_time_ns(
        &self,
        kind: CollectiveKind,
        payload_bytes: usize,
    ) -> Option<u64> {
        let groups = self.hierarchy_groups.as_ref()?;
        if kind == CollectiveKind::AllToAll {
            return None;
        }
        let leaders = groups.iter().map(|group| group[0]).collect::<Vec<_>>();
        let local_gather = groups
            .iter()
            .map(|group| {
                let leader = group[0];
                group[1..].iter().try_fold(0_u64, |total, rank| {
                    self.topology
                        .shortest_transfer_time_ns(*rank, leader, payload_bytes)
                        .map(|cost| total.saturating_add(cost))
                })
            })
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let leader_count = leaders.len();
        let inter_collective = match kind {
            CollectiveKind::AllReduce => {
                let chunk_bytes = payload_bytes.div_ceil(leader_count);
                let step =
                    self.striped_ring_step_time_ns(&leaders, &vec![chunk_bytes; leader_count])?;
                step.saturating_mul((2 * (leader_count - 1)) as u64)
            }
            CollectiveKind::ReduceScatter => {
                let shard_bytes = payload_bytes.div_ceil(self.ring_order.len());
                let mut total = 0_u64;
                for step in 0..leader_count - 1 {
                    let bytes = (0..leader_count)
                        .map(|position| {
                            let group_index = (position + leader_count - 1 - step) % leader_count;
                            groups[group_index].len().saturating_mul(shard_bytes)
                        })
                        .collect::<Vec<_>>();
                    total = total.saturating_add(self.striped_ring_step_time_ns(&leaders, &bytes)?);
                }
                total
            }
            CollectiveKind::AllGather => {
                let mut total = 0_u64;
                for step in 0..leader_count - 1 {
                    let bytes = (0..leader_count)
                        .map(|position| {
                            let group_index = (position + leader_count - step) % leader_count;
                            groups[group_index].len().saturating_mul(payload_bytes)
                        })
                        .collect::<Vec<_>>();
                    total = total.saturating_add(self.striped_ring_step_time_ns(&leaders, &bytes)?);
                }
                total
            }
            CollectiveKind::AllToAll => unreachable!(),
        };
        let world_size = self.ring_order.len();
        let shard_bytes = payload_bytes.div_ceil(world_size);
        let local_distribute = groups
            .iter()
            .map(|group| {
                let leader = group[0];
                let bytes = match kind {
                    CollectiveKind::AllReduce => payload_bytes,
                    CollectiveKind::AllGather => payload_bytes.saturating_mul(world_size),
                    CollectiveKind::ReduceScatter => shard_bytes,
                    CollectiveKind::AllToAll => unreachable!(),
                };
                group[1..].iter().try_fold(0_u64, |total, rank| {
                    self.topology
                        .shortest_transfer_time_ns(leader, *rank, bytes)
                        .map(|cost| total.saturating_add(cost))
                })
            })
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        Some(
            local_gather
                .saturating_add(inter_collective)
                .saturating_add(local_distribute),
        )
    }

    fn striped_ring_step_time_ns(&self, order: &[u32], bytes_by_position: &[usize]) -> Option<u64> {
        if order.len() != bytes_by_position.len() {
            return None;
        }
        let maximum_bytes = bytes_by_position.iter().copied().max().unwrap_or(0);
        let channels = self.ring_channels.min(maximum_bytes.max(1)).max(1);
        let rail_order = self.rail_order();
        let rails = rail_order.len().min(channels).max(1);
        let split = bytes_by_position
            .iter()
            .map(|bytes| balanced_sizes(*bytes, channels))
            .collect::<Vec<_>>();
        let split_by_channel = (0..channels)
            .map(|channel| {
                split
                    .iter()
                    .map(|position| position[channel])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut rail_times = vec![0_u64; rails];
        let mut aggregate_bytes = vec![vec![0_usize; self.ring_order.len()]; self.ring_order.len()];
        for (channel, channel_bytes) in split_by_channel.into_iter().enumerate() {
            let rail_slot = channel % rails;
            let rail = rail_order[rail_slot];
            let mut slowest = 0_u64;
            for ((source, destination), bytes) in order
                .iter()
                .copied()
                .zip(order.iter().copied().cycle().skip(1))
                .take(order.len())
                .zip(channel_bytes)
            {
                aggregate_bytes[source as usize][destination as usize] =
                    aggregate_bytes[source as usize][destination as usize].saturating_add(bytes);
                slowest = slowest.max(self.topology.shortest_transfer_time_ns_on_rail(
                    source,
                    destination,
                    bytes,
                    rail,
                )?);
            }
            rail_times[rail_slot] = rail_times[rail_slot].saturating_add(slowest);
        }
        let rail_time = rail_times.into_iter().max().unwrap_or(0);
        let aggregate_time = if rails > 1 {
            self.topology.aggregate_transfer_cap_ns(&aggregate_bytes)
        } else {
            0
        };
        Some(rail_time.max(aggregate_time))
    }

    fn multi_ring_step_time_ns(&self, bytes: usize) -> Option<u64> {
        let channels = self.ring_channels.min(bytes.max(1)).max(1);
        let rail_order = self.rail_order();
        let rails = rail_order.len().min(channels).max(1);
        let mut rail_times = vec![0_u64; rails];
        let mut aggregate_bytes = vec![vec![0_usize; self.ring_order.len()]; self.ring_order.len()];
        let base = bytes / channels;
        let remainder = bytes % channels;
        for channel in 0..channels {
            let channel_bytes = base + usize::from(channel < remainder);
            let rail_slot = channel % rails;
            let rail = rail_order[rail_slot];
            let order = self.ring_order_for_channel(channel);
            for (source, destination) in order
                .iter()
                .copied()
                .zip(order.iter().copied().cycle().skip(1))
                .take(order.len())
            {
                aggregate_bytes[source as usize][destination as usize] = aggregate_bytes
                    [source as usize][destination as usize]
                    .saturating_add(channel_bytes);
            }
            let step = self
                .topology
                .ring_step_time_ns_on_rail(order, channel_bytes, rail)?;
            rail_times[rail_slot] = rail_times[rail_slot].saturating_add(step);
        }
        let rail_time = rail_times.into_iter().max().unwrap_or(0);
        let aggregate_time = if rails > 1 {
            self.topology.aggregate_transfer_cap_ns(&aggregate_bytes)
        } else {
            0
        };
        Some(rail_time.max(aggregate_time))
    }
}

fn parse_usize_environment(name: &'static str, default: usize) -> Result<usize, TopologyError> {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|_| TopologyError::InvalidEnvironment { name, value }),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(value)) => Err(TopologyError::InvalidEnvironment {
            name,
            value: value.to_string_lossy().into_owned(),
        }),
    }
}

fn parse_u64_environment(name: &'static str, default: u64) -> Result<u64, TopologyError> {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|_| TopologyError::InvalidEnvironment { name, value }),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(value)) => Err(TopologyError::InvalidEnvironment {
            name,
            value: value.to_string_lossy().into_owned(),
        }),
    }
}

fn parse_collective_transport_environment() -> Result<CollectiveTransport, TopologyError> {
    match env::var("GX1_COLLECTIVE_TRANSPORT") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "coordinator" | "tcp_coordinator" | "tcp_host_staged" => {
                Ok(CollectiveTransport::TcpHostStaged)
            }
            "peer" | "tcp_peer" => Ok(CollectiveTransport::TcpPeer),
            _ => Err(TopologyError::InvalidEnvironment {
                name: "GX1_COLLECTIVE_TRANSPORT",
                value,
            }),
        },
        Err(env::VarError::NotPresent) => Ok(CollectiveTransport::TcpHostStaged),
        Err(env::VarError::NotUnicode(value)) => Err(TopologyError::InvalidEnvironment {
            name: "GX1_COLLECTIVE_TRANSPORT",
            value: value.to_string_lossy().into_owned(),
        }),
    }
}

fn parse_ring_order(world_size: u32, value: &str) -> Result<Vec<u32>, TopologyError> {
    let order = value
        .split(',')
        .map(|rank| {
            rank.trim()
                .parse::<u32>()
                .map_err(|_| TopologyError::InvalidEnvironment {
                    name: "GX1_RING_ORDER",
                    value: value.to_owned(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_ring_order(world_size, &order)?;
    Ok(order)
}

fn parse_rail_order(rail_count: usize, value: &str) -> Result<Vec<usize>, TopologyError> {
    let order = value
        .split(',')
        .map(|rail| {
            rail.trim()
                .parse::<usize>()
                .map_err(|_| TopologyError::InvalidEnvironment {
                    name: "GX1_RAIL_ORDER",
                    value: value.to_owned(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_rail_order(rail_count, &order)?;
    Ok(order)
}

fn parse_hierarchy_groups(world_size: u32, value: &str) -> Result<Vec<Vec<u32>>, TopologyError> {
    let groups = value
        .split(';')
        .map(|group| {
            group
                .split(',')
                .filter(|rank| !rank.trim().is_empty())
                .map(|rank| {
                    rank.trim()
                        .parse::<u32>()
                        .map_err(|_| TopologyError::InvalidEnvironment {
                            name: "GX1_TOPOLOGY_GROUPS",
                            value: value.to_owned(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_hierarchy_groups(world_size, &groups)?;
    Ok(groups)
}

fn validate_hierarchy_groups(world_size: u32, groups: &[Vec<u32>]) -> Result<(), TopologyError> {
    if groups.len() < 2 {
        return Err(TopologyError::InvalidHierarchy(
            "requires at least two non-empty groups".into(),
        ));
    }
    let mut seen = vec![false; world_size as usize];
    for (group_index, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(TopologyError::InvalidHierarchy(format!(
                "group {group_index} is empty"
            )));
        }
        for rank in group {
            if *rank >= world_size {
                return Err(TopologyError::RankOutOfRange {
                    rank: *rank,
                    world_size,
                });
            }
            if std::mem::replace(&mut seen[*rank as usize], true) {
                return Err(TopologyError::InvalidHierarchy(format!(
                    "rank {rank} appears more than once"
                )));
            }
        }
    }
    let missing = seen
        .iter()
        .enumerate()
        .filter_map(|(rank, included)| (!included).then_some(rank.to_string()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(TopologyError::InvalidHierarchy(format!(
            "does not cover ranks [{}]",
            missing.join(",")
        )));
    }
    Ok(())
}

fn parse_topology_links(world_size: u32, value: &str) -> Result<CollectiveTopology, TopologyError> {
    let mut topology = CollectiveTopology::empty(world_size)?;
    for encoded in value.split(',').filter(|link| !link.trim().is_empty()) {
        let (ranks, metrics) =
            encoded
                .trim()
                .split_once(':')
                .ok_or_else(|| TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.to_owned(),
                })?;
        let (first, second) =
            ranks
                .split_once('-')
                .ok_or_else(|| TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.to_owned(),
                })?;
        let (bandwidth, latency) =
            metrics
                .split_once(':')
                .ok_or_else(|| TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.to_owned(),
                })?;
        let parse = |part: &str| {
            part.trim()
                .parse::<u64>()
                .map_err(|_| TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.to_owned(),
                })
        };
        topology.add_link(TopologyLink {
            first_rank: first
                .trim()
                .parse()
                .map_err(|_| TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.to_owned(),
                })?,
            second_rank: second
                .trim()
                .parse()
                .map_err(|_| TopologyError::InvalidEnvironment {
                    name: "GX1_TOPOLOGY_LINKS",
                    value: value.to_owned(),
                })?,
            bandwidth_mbps: parse(bandwidth)?,
            latency_ns: parse(latency)?,
        })?;
    }
    Ok(topology)
}

fn parse_topology_rail_links(
    topology: &mut CollectiveTopology,
    value: &str,
    p2p_rails: usize,
) -> Result<(), TopologyError> {
    for encoded in value.split(',').filter(|link| !link.trim().is_empty()) {
        let invalid = || TopologyError::InvalidEnvironment {
            name: "GX1_TOPOLOGY_RAIL_LINKS",
            value: value.to_owned(),
        };
        let (rail, link) = encoded.trim().split_once('@').ok_or_else(invalid)?;
        let rail = rail.trim().parse::<usize>().map_err(|_| invalid())?;
        if rail >= p2p_rails {
            return Err(invalid());
        }
        let (ranks, metrics) = link.split_once(':').ok_or_else(invalid)?;
        let (first, second) = ranks.split_once('-').ok_or_else(invalid)?;
        let (bandwidth, latency) = metrics.split_once(':').ok_or_else(invalid)?;
        topology.add_rail_link(TopologyRailLink {
            rail,
            first_rank: first.trim().parse().map_err(|_| invalid())?,
            second_rank: second.trim().parse().map_err(|_| invalid())?,
            bandwidth_mbps: bandwidth.trim().parse().map_err(|_| invalid())?,
            latency_ns: latency.trim().parse().map_err(|_| invalid())?,
        })?;
    }
    Ok(())
}

fn parse_topology_aggregate_links(
    topology: &mut CollectiveTopology,
    value: &str,
) -> Result<(), TopologyError> {
    for encoded in value.split(',').filter(|link| !link.trim().is_empty()) {
        let invalid = || TopologyError::InvalidEnvironment {
            name: "GX1_TOPOLOGY_AGGREGATE_LINKS",
            value: value.to_owned(),
        };
        let (ranks, bandwidth) = encoded.trim().split_once(':').ok_or_else(invalid)?;
        let (first, second) = ranks.split_once('-').ok_or_else(invalid)?;
        topology.add_aggregate_link(TopologyAggregateLink {
            first_rank: first.trim().parse().map_err(|_| invalid())?,
            second_rank: second.trim().parse().map_err(|_| invalid())?,
            bandwidth_mbps: bandwidth.trim().parse().map_err(|_| invalid())?,
        })?;
    }
    Ok(())
}

fn probed_topology_matches(current: &CollectiveTopology, detected: &CollectiveTopology) -> bool {
    if current.world_size != detected.world_size {
        return false;
    }
    for first in 0..current.world_size {
        for second in first + 1..current.world_size {
            match (current.link(first, second), detected.link(first, second)) {
                (None, None) => {}
                (Some(current), Some(detected))
                    if metric_within_factor(current.bandwidth_mbps, detected.bandwidth_mbps, 4)
                        && metric_within_factor(current.latency_ns, detected.latency_ns, 4) => {}
                _ => return false,
            }
        }
    }
    if current.rail_links.len() != detected.rail_links.len() {
        return false;
    }
    for (current_links, detected_links) in current.rail_links.iter().zip(&detected.rail_links) {
        for first in 0..current.world_size {
            for second in first + 1..current.world_size {
                match (
                    current_links[first as usize][second as usize],
                    detected_links[first as usize][second as usize],
                ) {
                    (None, None) => {}
                    (Some(current), Some(detected))
                        if metric_within_factor(
                            current.bandwidth_mbps,
                            detected.bandwidth_mbps,
                            4,
                        ) && metric_within_factor(
                            current.latency_ns,
                            detected.latency_ns,
                            4,
                        ) => {}
                    _ => return false,
                }
            }
        }
    }
    for first in 0..current.world_size {
        for second in first + 1..current.world_size {
            match (
                current.aggregate_bandwidth_mbps(first, second),
                detected.aggregate_bandwidth_mbps(first, second),
            ) {
                (None, None) => {}
                (Some(current), Some(detected)) if metric_within_factor(current, detected, 4) => {}
                _ => return false,
            }
        }
    }
    true
}

fn metric_within_factor(first: u64, second: u64, factor: u64) -> bool {
    if first == 0 || second == 0 {
        return first == second;
    }
    let (smaller, larger) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    larger <= smaller.saturating_mul(factor)
}

fn validate_ring_order(world_size: u32, order: &[u32]) -> Result<(), TopologyError> {
    if world_size == 0 {
        return Err(TopologyError::EmptyWorld);
    }
    if order.len() != world_size as usize {
        return Err(TopologyError::InvalidRing(format!(
            "contains {} ranks, expected {world_size}",
            order.len()
        )));
    }
    let mut seen = vec![false; world_size as usize];
    for rank in order {
        if *rank >= world_size {
            return Err(TopologyError::RankOutOfRange {
                rank: *rank,
                world_size,
            });
        }
        if std::mem::replace(&mut seen[*rank as usize], true) {
            return Err(TopologyError::InvalidRing(format!(
                "rank {rank} appears more than once"
            )));
        }
    }
    Ok(())
}

fn validate_ring_channels(channels: usize) -> Result<(), TopologyError> {
    if channels == 0 || channels > MAX_RING_CHANNELS {
        return Err(TopologyError::InvalidRingChannels(channels));
    }
    Ok(())
}

fn validate_p2p_rails(rails: usize) -> Result<(), TopologyError> {
    if rails == 0 || rails > MAX_P2P_RAILS {
        return Err(TopologyError::InvalidP2pRails(rails));
    }
    Ok(())
}

fn validate_rail_order(rail_count: usize, order: &[usize]) -> Result<(), TopologyError> {
    if order.len() != rail_count {
        return Err(TopologyError::InvalidRailOrder(format!(
            "contains {} rails, expected {rail_count}",
            order.len()
        )));
    }
    let mut seen = vec![false; rail_count];
    for rail in order {
        if *rail >= rail_count {
            return Err(TopologyError::InvalidRailOrder(format!(
                "rail {rail} is outside configured rail count {rail_count}"
            )));
        }
        if std::mem::replace(&mut seen[*rail], true) {
            return Err(TopologyError::InvalidRailOrder(format!(
                "rail {rail} appears more than once"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_four_rank_topology_builds_three_edge_balanced_rings() {
        let topology = CollectiveTopology::uniform(4).unwrap();
        let orders = topology.best_ring_orders(MAX_RING_CHANNELS).unwrap();
        assert_eq!(orders.len(), 3);
        let mut edge_counts = [[0_u32; 4]; 4];
        for order in &orders {
            validate_ring_order(4, order).unwrap();
            for index in 0..order.len() {
                let first = order[index] as usize;
                let second = order[(index + 1) % order.len()] as usize;
                edge_counts[first][second] += 1;
                edge_counts[second][first] += 1;
            }
        }
        for (first, counts) in edge_counts.iter().enumerate() {
            for (second, count) in counts.iter().enumerate().skip(first + 1) {
                assert_eq!(*count, 2, "edge {first}-{second}");
            }
        }

        let tuning = CollectiveTuning::from_topology(
            AlgorithmPolicy::Ring,
            0,
            topology,
            Some(vec![0, 2, 1, 3]),
        )
        .unwrap();
        assert_eq!(tuning.ring_orders().len(), 3);
        assert_eq!(tuning.ring_order_for_channel(0), [0, 2, 1, 3]);
        assert_ne!(
            ring_edges(tuning.ring_order_for_channel(0)),
            ring_edges(tuning.ring_order_for_channel(1))
        );
    }

    #[test]
    fn multi_ring_cost_serializes_channels_that_share_a_rail() {
        let base = CollectiveTuning::new(AlgorithmPolicy::Ring, 0, vec![0, 1]).unwrap();
        let single = base
            .clone()
            .with_ring_channels(1)
            .unwrap()
            .plan(CollectiveKind::AllReduce, 16 * 1024 * 1024)
            .peer_estimated_time_ns
            .unwrap();
        let shared_rail = base
            .clone()
            .with_ring_channels(4)
            .unwrap()
            .plan(CollectiveKind::AllReduce, 16 * 1024 * 1024)
            .peer_estimated_time_ns
            .unwrap();
        let independent_rails = base
            .with_ring_channels(4)
            .unwrap()
            .with_p2p_rails(4)
            .unwrap()
            .plan(CollectiveKind::AllReduce, 16 * 1024 * 1024)
            .peer_estimated_time_ns
            .unwrap();
        assert!(shared_rail > single);
        assert!(independent_rails < single);
    }

    #[test]
    fn multi_ring_cost_uses_the_link_metrics_for_each_assigned_rail() {
        let build = |second_rail_bandwidth: u64| {
            let mut topology = CollectiveTopology::empty(2).unwrap();
            topology
                .add_link(TopologyLink {
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: second_rail_bandwidth.min(100_000),
                    latency_ns: 100,
                })
                .unwrap();
            for (rail, bandwidth_mbps) in [(0, 100_000), (1, second_rail_bandwidth)] {
                topology
                    .add_rail_link(TopologyRailLink {
                        rail,
                        first_rank: 0,
                        second_rank: 1,
                        bandwidth_mbps,
                        latency_ns: 100,
                    })
                    .unwrap();
            }
            CollectiveTuning::from_topology(AlgorithmPolicy::Ring, 0, topology, None)
                .unwrap()
                .with_ring_channels(2)
                .unwrap()
                .with_p2p_rails(2)
                .unwrap()
        };
        let balanced = build(100_000);
        let slow_second_rail = build(1_000);
        let balanced_time = balanced.multi_ring_step_time_ns(16 * 1024 * 1024).unwrap();
        let heterogeneous_time = slow_second_rail
            .multi_ring_step_time_ns(16 * 1024 * 1024)
            .unwrap();
        assert!(heterogeneous_time > balanced_time.saturating_mul(50));
        assert_ne!(
            balanced.topology_fingerprint(),
            slow_second_rail.topology_fingerprint()
        );
    }

    #[test]
    fn concurrent_rail_aggregate_capacity_caps_ring_pairwise_and_hierarchy_costs() {
        let build = |aggregate_bandwidth_mbps: Option<u64>, channels: usize| {
            let mut topology = CollectiveTopology::empty(4).unwrap();
            for first_rank in 0..4_u32 {
                for second_rank in first_rank + 1..4_u32 {
                    topology
                        .add_link(TopologyLink {
                            first_rank,
                            second_rank,
                            bandwidth_mbps: 100_000,
                            latency_ns: 100,
                        })
                        .unwrap();
                    for rail in 0..2 {
                        topology
                            .add_rail_link(TopologyRailLink {
                                rail,
                                first_rank,
                                second_rank,
                                bandwidth_mbps: 100_000,
                                latency_ns: 100,
                            })
                            .unwrap();
                    }
                    if let Some(bandwidth_mbps) = aggregate_bandwidth_mbps {
                        topology
                            .add_aggregate_link(TopologyAggregateLink {
                                first_rank,
                                second_rank,
                                bandwidth_mbps,
                            })
                            .unwrap();
                    }
                }
            }
            CollectiveTuning::from_topology(
                AlgorithmPolicy::Ring,
                0,
                topology,
                Some(vec![0, 1, 2, 3]),
            )
            .unwrap()
            .with_ring_channels(channels)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
        };
        let payload = 64 * 1024 * 1024;
        let uncapped = build(None, 2);
        let capped = build(Some(1_000), 2);
        assert!(
            capped.multi_ring_step_time_ns(payload).unwrap()
                > uncapped
                    .multi_ring_step_time_ns(payload)
                    .unwrap()
                    .saturating_mul(20)
        );
        assert!(
            capped
                .topology
                .pairwise_time_ns(&capped.ring_order, payload / 4, 2, &capped.rail_order())
                .unwrap()
                > uncapped
                    .topology
                    .pairwise_time_ns(&uncapped.ring_order, payload / 4, 2, &uncapped.rail_order(),)
                    .unwrap()
                    .saturating_mul(20)
        );

        let uncapped_hierarchy = uncapped
            .with_hierarchy(vec![vec![0], vec![1], vec![2], vec![3]])
            .unwrap();
        let capped_hierarchy = capped
            .with_hierarchy(vec![vec![0], vec![1], vec![2], vec![3]])
            .unwrap();
        assert!(
            capped_hierarchy
                .hierarchical_estimated_time_ns(CollectiveKind::AllReduce, payload)
                .unwrap()
                > uncapped_hierarchy
                    .hierarchical_estimated_time_ns(CollectiveKind::AllReduce, payload)
                    .unwrap()
                    .saturating_mul(20)
        );

        let uncapped_single = build(None, 1);
        let capped_single = build(Some(1_000), 1);
        assert_eq!(
            capped_single.multi_ring_step_time_ns(payload),
            uncapped_single.multi_ring_step_time_ns(payload)
        );
        assert_ne!(
            capped_single.topology_fingerprint(),
            uncapped_single.topology_fingerprint()
        );
    }

    #[test]
    fn topology_aware_rail_order_skips_slow_noncontiguous_rails_first() {
        let mut topology = CollectiveTopology::empty(2).unwrap();
        topology
            .add_link(TopologyLink {
                first_rank: 0,
                second_rank: 1,
                bandwidth_mbps: 1_000,
                latency_ns: 1_000,
            })
            .unwrap();
        for (rail, bandwidth_mbps, latency_ns) in [
            (0, 100_000, 100),
            (1, 1_000, 1_000),
            (2, 80_000, 150),
            (3, 2_000, 900),
        ] {
            topology
                .add_rail_link(TopologyRailLink {
                    rail,
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps,
                    latency_ns,
                })
                .unwrap();
        }
        let automatic = CollectiveTuning::from_topology(AlgorithmPolicy::Ring, 0, topology, None)
            .unwrap()
            .with_p2p_rails(4)
            .unwrap()
            .with_ring_channels(2)
            .unwrap();
        assert_eq!(automatic.rail_order(), vec![0, 2, 3, 1]);
        assert_eq!(
            (0..6)
                .map(|channel| automatic.rail_for_channel(channel))
                .collect::<Vec<_>>(),
            vec![0, 2, 3, 1, 0, 2]
        );

        let same_explicit = automatic.clone().with_rail_order(vec![0, 2, 3, 1]).unwrap();
        assert_eq!(
            automatic.topology_fingerprint(),
            same_explicit.topology_fingerprint()
        );
        let natural = automatic.clone().with_rail_order(vec![0, 1, 2, 3]).unwrap();
        let automatic_time = automatic
            .plan(CollectiveKind::AllReduce, 16 * 1024 * 1024)
            .peer_estimated_time_ns
            .unwrap();
        let natural_time = natural
            .plan(CollectiveKind::AllReduce, 16 * 1024 * 1024)
            .peer_estimated_time_ns
            .unwrap();
        assert!(natural_time > automatic_time.saturating_mul(50));
        assert_ne!(
            automatic.topology_fingerprint(),
            natural.topology_fingerprint()
        );

        assert!(matches!(
            automatic.clone().with_rail_order(vec![0, 1, 1, 3]),
            Err(TopologyError::InvalidRailOrder(_))
        ));
        assert!(matches!(
            automatic.clone().with_rail_order(vec![0, 1, 2]),
            Err(TopologyError::InvalidRailOrder(_))
        ));
        assert!(matches!(
            automatic.with_rail_order(vec![0, 1, 2, 4]),
            Err(TopologyError::InvalidRailOrder(_))
        ));
    }

    #[test]
    fn pairwise_cost_uses_the_link_metrics_for_each_assigned_rail() {
        let build = |second_rail_bandwidth: u64, channels: usize| {
            let mut topology = CollectiveTopology::empty(2).unwrap();
            topology
                .add_link(TopologyLink {
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: 100_000,
                    latency_ns: 100,
                })
                .unwrap();
            for (rail, bandwidth_mbps) in [(0, 100_000), (1, second_rail_bandwidth)] {
                topology
                    .add_rail_link(TopologyRailLink {
                        rail,
                        first_rank: 0,
                        second_rank: 1,
                        bandwidth_mbps,
                        latency_ns: 100,
                    })
                    .unwrap();
            }
            CollectiveTuning::from_topology(AlgorithmPolicy::Ring, 0, topology, None)
                .unwrap()
                .with_ring_channels(channels)
                .unwrap()
                .with_p2p_rails(2)
                .unwrap()
                .plan(CollectiveKind::AllToAll, 16 * 1024 * 1024)
                .peer_estimated_time_ns
                .unwrap()
        };

        let balanced = build(100_000, 2);
        let slow_second_rail = build(1_000, 2);
        let rail_zero_only = build(1_000, 1);
        assert!(slow_second_rail > balanced.saturating_mul(50));
        assert!(slow_second_rail > rail_zero_only.saturating_mul(40));
    }

    #[test]
    fn hierarchical_inter_leader_cost_uses_assigned_rail_metrics() {
        let build = |second_rail_bandwidth: u64, channels: usize| {
            let mut topology = CollectiveTopology::empty(4).unwrap();
            for first_rank in 0..4_u32 {
                for second_rank in first_rank + 1..4_u32 {
                    topology
                        .add_link(TopologyLink {
                            first_rank,
                            second_rank,
                            bandwidth_mbps: 100_000,
                            latency_ns: 100,
                        })
                        .unwrap();
                    for (rail, bandwidth_mbps) in [(0, 100_000), (1, second_rail_bandwidth)] {
                        topology
                            .add_rail_link(TopologyRailLink {
                                rail,
                                first_rank,
                                second_rank,
                                bandwidth_mbps,
                                latency_ns: 100,
                            })
                            .unwrap();
                    }
                }
            }
            CollectiveTuning::from_topology(
                AlgorithmPolicy::Hierarchical,
                0,
                topology,
                Some(vec![0, 1, 2, 3]),
            )
            .unwrap()
            .with_hierarchy(vec![vec![0, 1], vec![2, 3]])
            .unwrap()
            .with_ring_channels(channels)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
        };

        let balanced = build(100_000, 2);
        let slow_second_rail = build(1_000, 2);
        let rail_zero_only = build(1_000, 1);
        for kind in [
            CollectiveKind::AllReduce,
            CollectiveKind::AllGather,
            CollectiveKind::ReduceScatter,
        ] {
            let balanced_time = balanced
                .plan(kind, 64 * 1024 * 1024)
                .hierarchical_estimated_time_ns
                .unwrap();
            let heterogeneous_time = slow_second_rail
                .plan(kind, 64 * 1024 * 1024)
                .hierarchical_estimated_time_ns
                .unwrap();
            let rail_zero_time = rail_zero_only
                .plan(kind, 64 * 1024 * 1024)
                .hierarchical_estimated_time_ns
                .unwrap();
            assert!(heterogeneous_time > balanced_time.saturating_mul(4));
            assert!(heterogeneous_time > rail_zero_time.saturating_mul(4));
        }
    }

    #[test]
    fn explicit_rail_topology_parser_validates_the_configured_rail_count() {
        let mut topology = CollectiveTopology::uniform(2).unwrap();
        parse_topology_rail_links(&mut topology, "0@0-1:100000:100,1@0-1:25000:900", 2).unwrap();
        assert_eq!(
            topology.rail_links(),
            vec![
                TopologyRailLink {
                    rail: 0,
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: 100_000,
                    latency_ns: 100,
                },
                TopologyRailLink {
                    rail: 1,
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: 25_000,
                    latency_ns: 900,
                },
            ]
        );
        assert!(parse_topology_rail_links(&mut topology, "2@0-1:1:1", 2).is_err());
    }

    #[test]
    fn explicit_aggregate_topology_parser_validates_and_preserves_bandwidth() {
        let mut topology = CollectiveTopology::uniform(3).unwrap();
        parse_topology_aggregate_links(&mut topology, "0-1:150000,1-2:75000").unwrap();
        assert_eq!(
            topology.aggregate_links(),
            vec![
                TopologyAggregateLink {
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: 150_000,
                },
                TopologyAggregateLink {
                    first_rank: 1,
                    second_rank: 2,
                    bandwidth_mbps: 75_000,
                },
            ]
        );
        assert!(parse_topology_aggregate_links(&mut topology, "0-1:0").is_err());
        assert!(parse_topology_aggregate_links(&mut topology, "0@1:100").is_err());
    }

    #[test]
    fn topology_prefers_the_high_bandwidth_closed_ring() {
        let mut topology = CollectiveTopology::empty(4).unwrap();
        for (first, second, bandwidth) in [
            (0, 1, 100_000),
            (1, 2, 100_000),
            (2, 3, 100_000),
            (3, 0, 100_000),
            (0, 2, 10_000),
            (1, 3, 10_000),
        ] {
            topology
                .add_link(TopologyLink {
                    first_rank: first,
                    second_rank: second,
                    bandwidth_mbps: bandwidth,
                    latency_ns: 500,
                })
                .unwrap();
        }
        let ring = topology.best_ring_order().unwrap();
        let edges = (0..ring.len())
            .map(|index| {
                let first = ring[index];
                let second = ring[(index + 1) % ring.len()];
                (first.min(second), first.max(second))
            })
            .collect::<Vec<_>>();
        assert!(edges.iter().all(|edge| !matches!(edge, (0, 2) | (1, 3))));
    }

    #[test]
    fn tuning_selects_direct_for_small_and_ring_for_large_payloads() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Auto, 1024, vec![0, 1]).unwrap();
        assert_eq!(
            tuning.all_reduce_algorithm(1023),
            CollectiveAlgorithm::Direct
        );
        assert_eq!(tuning.all_reduce_algorithm(1024), CollectiveAlgorithm::Ring);
    }

    #[test]
    fn auto_planner_uses_collective_specific_peer_algorithms_and_costs() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Auto, 0, vec![0, 1, 2, 3]).unwrap();
        for kind in [
            CollectiveKind::AllReduce,
            CollectiveKind::AllGather,
            CollectiveKind::ReduceScatter,
        ] {
            let plan = tuning.plan(kind, 4 * 1024 * 1024);
            assert_eq!(plan.algorithm, CollectiveAlgorithm::Ring);
            assert!(plan.peer_estimated_time_ns.unwrap() < plan.direct_estimated_time_ns);
            assert_eq!(plan.estimated_time_ns, plan.peer_estimated_time_ns.unwrap());
        }
        let all_to_all = tuning.plan(CollectiveKind::AllToAll, 4 * 1024 * 1024);
        assert_eq!(all_to_all.algorithm, CollectiveAlgorithm::Pairwise);
        assert!(all_to_all.peer_estimated_time_ns.unwrap() < all_to_all.direct_estimated_time_ns);
    }

    #[test]
    fn auto_planner_keeps_direct_when_peer_links_are_slower() {
        let mut topology = CollectiveTopology::empty(2).unwrap();
        topology
            .add_link(TopologyLink {
                first_rank: 0,
                second_rank: 1,
                bandwidth_mbps: 1,
                latency_ns: 1_000_000,
            })
            .unwrap();
        let tuning = CollectiveTuning::from_topology(AlgorithmPolicy::Auto, 0, topology, None)
            .unwrap()
            .with_direct_transport(100_000, 100)
            .unwrap();
        let plan = tuning.plan(CollectiveKind::AllReduce, 1024 * 1024);
        assert_eq!(plan.algorithm, CollectiveAlgorithm::Direct);
        assert!(plan.direct_estimated_time_ns < plan.peer_estimated_time_ns.unwrap());
    }

    #[test]
    fn hierarchy_validation_requires_unique_full_rank_coverage() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Auto, 0, vec![0, 1, 2, 3]).unwrap();
        assert!(matches!(
            tuning
                .clone()
                .with_hierarchy(vec![vec![0, 1], vec![1, 2, 3]]),
            Err(TopologyError::InvalidHierarchy(_))
        ));
        assert!(matches!(
            tuning.clone().with_hierarchy(vec![vec![0, 1], vec![2]]),
            Err(TopologyError::InvalidHierarchy(_))
        ));
        assert!(matches!(
            tuning.with_hierarchy(vec![vec![0, 1, 2, 3]]),
            Err(TopologyError::InvalidHierarchy(_))
        ));
    }

    #[test]
    fn auto_planner_selects_hierarchy_for_fast_local_and_slow_cross_group_links() {
        let mut topology = CollectiveTopology::empty(4).unwrap();
        for first_rank in 0..4_u32 {
            for second_rank in first_rank + 1..4_u32 {
                let local = (first_rank < 2) == (second_rank < 2);
                topology
                    .add_link(TopologyLink {
                        first_rank,
                        second_rank,
                        bandwidth_mbps: if local { 200_000 } else { 10_000 },
                        latency_ns: if local { 200 } else { 50_000 },
                    })
                    .unwrap();
            }
        }
        let tuning = CollectiveTuning::from_topology(
            AlgorithmPolicy::Auto,
            0,
            topology,
            Some(vec![0, 1, 2, 3]),
        )
        .unwrap()
        .with_direct_transport(1_000, 100_000)
        .unwrap()
        .with_hierarchy(vec![vec![0, 1], vec![2, 3]])
        .unwrap();

        for kind in [
            CollectiveKind::AllReduce,
            CollectiveKind::AllGather,
            CollectiveKind::ReduceScatter,
        ] {
            let plan = tuning.plan(kind, 16 * 1024 * 1024);
            assert_eq!(plan.algorithm, CollectiveAlgorithm::Hierarchical);
            assert!(
                plan.hierarchical_estimated_time_ns.unwrap() < plan.peer_estimated_time_ns.unwrap()
            );
            assert_eq!(
                plan.estimated_time_ns,
                plan.hierarchical_estimated_time_ns.unwrap()
            );
        }
        let all_to_all = tuning.plan(CollectiveKind::AllToAll, 16 * 1024 * 1024);
        assert_eq!(all_to_all.algorithm, CollectiveAlgorithm::Pairwise);
        assert_eq!(all_to_all.hierarchical_estimated_time_ns, None);
    }

    #[test]
    fn inferred_hierarchy_finds_fast_domains_and_selects_low_cost_leaders() {
        let mut topology = CollectiveTopology::empty(6).unwrap();
        for first_rank in 0..6_u32 {
            for second_rank in first_rank + 1..6_u32 {
                let first_domain = first_rank / 3;
                let second_domain = second_rank / 3;
                let (bandwidth_mbps, latency_ns) = if first_domain != second_domain {
                    (10_000, 50_000)
                } else if first_rank.abs_diff(second_rank) == 1 {
                    (200_000, 200)
                } else {
                    (120_000, 300)
                };
                topology
                    .add_link(TopologyLink {
                        first_rank,
                        second_rank,
                        bandwidth_mbps,
                        latency_ns,
                    })
                    .unwrap();
            }
        }
        let base = CollectiveTuning::from_topology(AlgorithmPolicy::Auto, 0, topology, None)
            .unwrap()
            .with_ring_channels(2)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap();
        let inferred = base
            .clone()
            .with_inferred_hierarchy(16 * 1024 * 1024)
            .unwrap();
        assert_eq!(
            inferred.hierarchy_groups(),
            Some([vec![1, 0, 2], vec![4, 3, 5]].as_slice())
        );
        for kind in [
            CollectiveKind::AllReduce,
            CollectiveKind::AllGather,
            CollectiveKind::ReduceScatter,
        ] {
            let plan = inferred.plan(kind, 16 * 1024 * 1024);
            assert_eq!(plan.algorithm, CollectiveAlgorithm::Hierarchical);
            assert!(
                plan.hierarchical_estimated_time_ns.unwrap() < plan.peer_estimated_time_ns.unwrap()
            );
        }
        let explicit = base
            .with_hierarchy(vec![vec![1, 0, 2], vec![4, 3, 5]])
            .unwrap();
        assert_eq!(
            inferred.topology_fingerprint(),
            explicit.topology_fingerprint()
        );
    }

    #[test]
    fn inferred_hierarchy_rejects_uniform_topology_without_distinct_domains() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Auto, 0, vec![0, 1, 2, 3]).unwrap();
        assert!(matches!(
            tuning.with_inferred_hierarchy(16 * 1024 * 1024),
            Err(TopologyError::InvalidHierarchy(message))
                if message.contains("no distinct fast-link domains")
        ));
    }

    #[test]
    fn hierarchy_changes_the_autotune_fingerprint() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Auto, 0, vec![0, 1, 2, 3]).unwrap();
        let flat = tuning.topology_fingerprint();
        let hierarchical = tuning
            .with_hierarchy(vec![vec![0, 1], vec![2, 3]])
            .unwrap()
            .topology_fingerprint();
        assert_ne!(flat, hierarchical);
    }

    #[test]
    fn ring_channel_count_is_validated_without_changing_hardware_fingerprint() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Ring, 0, vec![0, 1]).unwrap();
        assert!(matches!(
            tuning.clone().with_ring_channels(0),
            Err(TopologyError::InvalidRingChannels(0))
        ));
        assert!(matches!(
            tuning.clone().with_ring_channels(MAX_RING_CHANNELS + 1),
            Err(TopologyError::InvalidRingChannels(_))
        ));
        let original_fingerprint = tuning.topology_fingerprint();
        let tuning = tuning.with_ring_channels(4).unwrap();
        assert_eq!(tuning.ring_channels(), 4);
        assert_eq!(
            tuning.plan(CollectiveKind::AllReduce, 1024).ring_channels,
            4
        );
        assert_eq!(original_fingerprint, tuning.topology_fingerprint());
    }

    #[test]
    fn point_to_point_rails_are_validated_and_change_hardware_fingerprint() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Ring, 0, vec![0, 1]).unwrap();
        assert!(matches!(
            tuning.clone().with_p2p_rails(0),
            Err(TopologyError::InvalidP2pRails(0))
        ));
        assert!(matches!(
            tuning.clone().with_p2p_rails(MAX_P2P_RAILS + 1),
            Err(TopologyError::InvalidP2pRails(_))
        ));
        let original_fingerprint = tuning.topology_fingerprint();
        let single_rail_time = tuning
            .clone()
            .with_ring_channels(4)
            .unwrap()
            .plan(CollectiveKind::AllReduce, 16 * 1024 * 1024)
            .peer_estimated_time_ns
            .unwrap();
        let tuning = tuning.with_p2p_rails(4).unwrap();
        assert_eq!(tuning.p2p_rails(), 4);
        assert_ne!(original_fingerprint, tuning.topology_fingerprint());
        let four_rail_time = tuning
            .with_ring_channels(4)
            .unwrap()
            .plan(CollectiveKind::AllReduce, 16 * 1024 * 1024)
            .peer_estimated_time_ns
            .unwrap();
        assert!(four_rail_time < single_rail_time);
    }

    #[test]
    fn tcp_peer_transport_changes_execution_but_not_hardware_fingerprint() {
        let coordinator = CollectiveTuning::new(AlgorithmPolicy::Ring, 0, vec![0, 1, 2]).unwrap();
        let peer = coordinator
            .clone()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap();
        assert_eq!(peer.transport(), CollectiveTransport::TcpPeer);
        assert_eq!(
            coordinator.topology_fingerprint(),
            peer.topology_fingerprint()
        );
        assert_ne!(
            coordinator.execution_fingerprint(),
            peer.execution_fingerprint()
        );
    }

    #[test]
    fn pairwise_estimate_routes_across_sparse_connected_topology() {
        let mut topology = CollectiveTopology::empty(4).unwrap();
        for (first_rank, second_rank) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            topology
                .add_link(TopologyLink {
                    first_rank,
                    second_rank,
                    bandwidth_mbps: 50_000,
                    latency_ns: 400,
                })
                .unwrap();
        }
        let tuning = CollectiveTuning::from_topology(
            AlgorithmPolicy::Ring,
            0,
            topology,
            Some(vec![0, 1, 2, 3]),
        )
        .unwrap();
        let plan = tuning.plan(CollectiveKind::AllToAll, 4096);
        assert_eq!(plan.algorithm, CollectiveAlgorithm::Pairwise);
        assert!(plan.peer_estimated_time_ns.is_some());
        assert_ne!(plan.estimated_time_ns, u64::MAX);
    }

    #[test]
    fn autotune_profile_overrides_model_only_for_measured_ranges() {
        let tuning =
            CollectiveTuning::new(AlgorithmPolicy::Auto, 16 * 1024 * 1024, vec![0, 1]).unwrap();
        assert_eq!(
            tuning
                .plan(CollectiveKind::AllReduce, 1024 * 1024)
                .algorithm,
            CollectiveAlgorithm::Direct
        );
        let profile = CollectiveAutotuneProfile {
            version: COLLECTIVE_AUTOTUNE_PROFILE_VERSION,
            world_size: 2,
            topology_fingerprint: tuning.topology_fingerprint_hex(),
            execution_fingerprint: tuning.execution_fingerprint_hex(),
            entries: vec![CollectiveAutotuneEntry {
                operation: CollectiveKind::AllReduce,
                min_payload_bytes: 512 * 1024,
                max_payload_bytes: 2 * 1024 * 1024,
                algorithm: CollectiveAlgorithm::Ring,
                ring_channels: 3,
                measured_time_ns: 123_456,
            }],
        };
        let tuning = tuning.with_autotune_profile(profile).unwrap();
        let measured = tuning.plan(CollectiveKind::AllReduce, 1024 * 1024);
        assert_eq!(measured.algorithm, CollectiveAlgorithm::Ring);
        assert_eq!(measured.source, CollectivePlanSource::Autotune);
        assert_eq!(measured.ring_channels, 3);
        assert_eq!(measured.estimated_time_ns, 123_456);
        let unmeasured = tuning.plan(CollectiveKind::AllReduce, 4 * 1024 * 1024);
        assert_eq!(unmeasured.algorithm, CollectiveAlgorithm::Direct);
        assert_eq!(unmeasured.source, CollectivePlanSource::Model);
    }

    #[test]
    fn probed_profile_accepts_one_adjacent_metric_bucket_but_not_a_topology_change() {
        let topology = |bandwidth_mbps, latency_ns| {
            let mut topology = CollectiveTopology::empty(2).unwrap();
            topology
                .add_link(TopologyLink {
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps,
                    latency_ns,
                })
                .unwrap();
            for rail in 0..2 {
                topology
                    .add_rail_link(TopologyRailLink {
                        rail,
                        first_rank: 0,
                        second_rank: 1,
                        bandwidth_mbps,
                        latency_ns,
                    })
                    .unwrap();
            }
            topology
        };
        let pinned =
            CollectiveTuning::from_topology(AlgorithmPolicy::Auto, 0, topology(64, 65_536), None)
                .unwrap()
                .with_transport(CollectiveTransport::TcpPeer)
                .unwrap()
                .with_p2p_rails(2)
                .unwrap();
        let profile = CollectiveAutotuneProfile {
            version: COLLECTIVE_AUTOTUNE_PROFILE_VERSION,
            world_size: 2,
            topology_fingerprint: pinned.topology_fingerprint_hex(),
            execution_fingerprint: pinned.execution_fingerprint_hex(),
            entries: vec![CollectiveAutotuneEntry {
                operation: CollectiveKind::AllReduce,
                min_payload_bytes: 0,
                max_payload_bytes: u64::MAX,
                algorithm: CollectiveAlgorithm::Ring,
                ring_channels: 1,
                measured_time_ns: 1,
            }],
        };
        let adjacent =
            CollectiveTuning::from_topology(AlgorithmPolicy::Auto, 0, topology(256, 262_144), None)
                .unwrap()
                .with_transport(CollectiveTransport::TcpPeer)
                .unwrap()
                .with_p2p_rails(2)
                .unwrap();
        assert!(
            adjacent
                .clone()
                .with_autotune_profile(profile.clone())
                .is_err()
        );
        adjacent
            .validate_autotune_profile_with_detected(
                &profile,
                Some("0-1:64:65536"),
                Some("0@0-1:64:65536,1@0-1:64:65536"),
                None,
                true,
            )
            .unwrap();

        let changed = CollectiveTuning::from_topology(
            AlgorithmPolicy::Auto,
            0,
            topology(1024, 1_048_576),
            None,
        )
        .unwrap()
        .with_transport(CollectiveTransport::TcpPeer)
        .unwrap()
        .with_p2p_rails(2)
        .unwrap();
        assert!(
            changed
                .validate_autotune_profile_with_detected(
                    &profile,
                    Some("0-1:64:65536"),
                    Some("0@0-1:64:65536,1@0-1:64:65536"),
                    None,
                    true,
                )
                .is_err()
        );
    }

    #[test]
    fn probed_profile_tolerates_adjacent_aggregate_capacity_but_requires_the_cap() {
        let topology = |aggregate_bandwidth_mbps| {
            let mut topology = CollectiveTopology::empty(2).unwrap();
            topology
                .add_link(TopologyLink {
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: 64,
                    latency_ns: 65_536,
                })
                .unwrap();
            for rail in 0..2 {
                topology
                    .add_rail_link(TopologyRailLink {
                        rail,
                        first_rank: 0,
                        second_rank: 1,
                        bandwidth_mbps: 64,
                        latency_ns: 65_536,
                    })
                    .unwrap();
            }
            topology
                .add_aggregate_link(TopologyAggregateLink {
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: aggregate_bandwidth_mbps,
                })
                .unwrap();
            topology
        };
        let tuning = |aggregate_bandwidth_mbps| {
            CollectiveTuning::from_topology(
                AlgorithmPolicy::Auto,
                0,
                topology(aggregate_bandwidth_mbps),
                None,
            )
            .unwrap()
            .with_transport(CollectiveTransport::TcpPeer)
            .unwrap()
            .with_p2p_rails(2)
            .unwrap()
        };
        let pinned = tuning(64);
        let profile = CollectiveAutotuneProfile {
            version: COLLECTIVE_AUTOTUNE_PROFILE_VERSION,
            world_size: 2,
            topology_fingerprint: pinned.topology_fingerprint_hex(),
            execution_fingerprint: pinned.execution_fingerprint_hex(),
            entries: vec![CollectiveAutotuneEntry {
                operation: CollectiveKind::AllReduce,
                min_payload_bytes: 0,
                max_payload_bytes: u64::MAX,
                algorithm: CollectiveAlgorithm::Ring,
                ring_channels: 2,
                measured_time_ns: 1,
            }],
        };

        let adjacent = tuning(256);
        adjacent
            .validate_autotune_profile_with_detected(
                &profile,
                Some("0-1:64:65536"),
                Some("0@0-1:64:65536,1@0-1:64:65536"),
                Some("0-1:64"),
                true,
            )
            .unwrap();
        assert!(
            adjacent
                .validate_autotune_profile_with_detected(
                    &profile,
                    Some("0-1:64:65536"),
                    Some("0@0-1:64:65536,1@0-1:64:65536"),
                    None,
                    true,
                )
                .is_err()
        );
        assert!(
            tuning(1024)
                .validate_autotune_profile_with_detected(
                    &profile,
                    Some("0-1:64:65536"),
                    Some("0@0-1:64:65536,1@0-1:64:65536"),
                    Some("0-1:64"),
                    true,
                )
                .is_err()
        );
    }

    #[test]
    fn probed_profile_rejects_adjacent_metrics_when_the_preferred_rail_order_flips() {
        let topology = |first_rail_bandwidth, second_rail_bandwidth| {
            let mut topology = CollectiveTopology::empty(2).unwrap();
            topology
                .add_link(TopologyLink {
                    first_rank: 0,
                    second_rank: 1,
                    bandwidth_mbps: 64,
                    latency_ns: 65_536,
                })
                .unwrap();
            for (rail, bandwidth_mbps) in [(0, first_rail_bandwidth), (1, second_rail_bandwidth)] {
                topology
                    .add_rail_link(TopologyRailLink {
                        rail,
                        first_rank: 0,
                        second_rank: 1,
                        bandwidth_mbps,
                        latency_ns: 65_536,
                    })
                    .unwrap();
            }
            topology
        };
        let pinned =
            CollectiveTuning::from_topology(AlgorithmPolicy::Auto, 0, topology(64, 256), None)
                .unwrap()
                .with_transport(CollectiveTransport::TcpPeer)
                .unwrap()
                .with_p2p_rails(2)
                .unwrap();
        assert_eq!(pinned.rail_order(), vec![1, 0]);
        let profile = CollectiveAutotuneProfile {
            version: COLLECTIVE_AUTOTUNE_PROFILE_VERSION,
            world_size: 2,
            topology_fingerprint: pinned.topology_fingerprint_hex(),
            execution_fingerprint: pinned.execution_fingerprint_hex(),
            entries: vec![CollectiveAutotuneEntry {
                operation: CollectiveKind::AllReduce,
                min_payload_bytes: 0,
                max_payload_bytes: u64::MAX,
                algorithm: CollectiveAlgorithm::Ring,
                ring_channels: 2,
                measured_time_ns: 1,
            }],
        };

        let current =
            CollectiveTuning::from_topology(AlgorithmPolicy::Auto, 0, topology(256, 64), None)
                .unwrap()
                .with_transport(CollectiveTransport::TcpPeer)
                .unwrap()
                .with_p2p_rails(2)
                .unwrap();
        assert_eq!(current.rail_order(), vec![0, 1]);
        assert!(
            current
                .validate_autotune_profile_with_detected(
                    &profile,
                    Some("0-1:64:65536"),
                    Some("0@0-1:64:65536,1@0-1:256:65536"),
                    None,
                    true,
                )
                .is_err()
        );
    }

    #[test]
    fn autotune_profile_rejects_stale_overlapping_and_wrong_algorithm_entries() {
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Auto, 0, vec![0, 1]).unwrap();
        let entry = CollectiveAutotuneEntry {
            operation: CollectiveKind::AllReduce,
            min_payload_bytes: 0,
            max_payload_bytes: 1024,
            algorithm: CollectiveAlgorithm::Ring,
            ring_channels: 1,
            measured_time_ns: 100,
        };
        let profile = |fingerprint: String, entries: Vec<CollectiveAutotuneEntry>| {
            CollectiveAutotuneProfile {
                version: COLLECTIVE_AUTOTUNE_PROFILE_VERSION,
                world_size: 2,
                topology_fingerprint: fingerprint,
                execution_fingerprint: tuning.execution_fingerprint_hex(),
                entries,
            }
        };
        assert!(matches!(
            tuning
                .clone()
                .with_autotune_profile(profile("0000000000000000".into(), vec![entry])),
            Err(TopologyError::InvalidProfile(_))
        ));
        let mut wrong_execution = profile(tuning.topology_fingerprint_hex(), vec![entry]);
        wrong_execution.execution_fingerprint = "0000000000000000".into();
        assert!(matches!(
            tuning.clone().with_autotune_profile(wrong_execution),
            Err(TopologyError::InvalidProfile(_))
        ));
        assert!(matches!(
            tuning.clone().with_autotune_profile(profile(
                tuning.topology_fingerprint_hex(),
                vec![
                    entry,
                    CollectiveAutotuneEntry {
                        min_payload_bytes: 1024,
                        max_payload_bytes: 2048,
                        ..entry
                    },
                ]
            )),
            Err(TopologyError::InvalidProfile(_))
        ));
        let profiled = tuning
            .clone()
            .with_autotune_profile(profile(tuning.topology_fingerprint_hex(), vec![entry]))
            .unwrap();
        assert!(matches!(
            profiled.with_transport(CollectiveTransport::HostStaged),
            Err(TopologyError::InvalidProfile(_))
        ));
        assert!(matches!(
            tuning.clone().with_autotune_profile(profile(
                tuning.topology_fingerprint_hex(),
                vec![CollectiveAutotuneEntry {
                    operation: CollectiveKind::AllToAll,
                    algorithm: CollectiveAlgorithm::Ring,
                    ..entry
                }]
            )),
            Err(TopologyError::InvalidProfile(_))
        ));
        assert!(matches!(
            tuning.clone().with_autotune_profile(profile(
                tuning.topology_fingerprint_hex(),
                vec![CollectiveAutotuneEntry {
                    algorithm: CollectiveAlgorithm::Hierarchical,
                    ..entry
                }]
            )),
            Err(TopologyError::InvalidProfile(_))
        ));
    }

    #[test]
    fn autotune_profile_json_uses_stable_wire_names() {
        let profile = CollectiveAutotuneProfile {
            version: COLLECTIVE_AUTOTUNE_PROFILE_VERSION,
            world_size: 2,
            topology_fingerprint: "0123456789abcdef".into(),
            execution_fingerprint: "fedcba9876543210".into(),
            entries: vec![CollectiveAutotuneEntry {
                operation: CollectiveKind::AllToAll,
                min_payload_bytes: 0,
                max_payload_bytes: 4095,
                algorithm: CollectiveAlgorithm::Pairwise,
                ring_channels: 1,
                measured_time_ns: 42,
            }],
        };
        let encoded = serde_json::to_string(&profile).unwrap();
        assert!(encoded.contains("\"operation\":\"alltoall\""));
        assert!(encoded.contains("\"algorithm\":\"pairwise\""));
        assert!(encoded.contains("\"ring_channels\":1"));
        assert!(encoded.contains("\"execution_fingerprint\":\"fedcba9876543210\""));
        assert_eq!(
            serde_json::from_str::<CollectiveAutotuneProfile>(&encoded).unwrap(),
            profile
        );
    }

    #[test]
    fn legacy_autotune_profile_defaults_to_one_ring_channel() {
        let encoded = r#"{
            "version": 1,
            "world_size": 2,
            "topology_fingerprint": "0123456789abcdef",
            "entries": [{
                "operation": "allreduce",
                "min_payload_bytes": 0,
                "max_payload_bytes": 4095,
                "algorithm": "ring",
                "measured_time_ns": 42
            }]
        }"#;
        let profile = serde_json::from_str::<CollectiveAutotuneProfile>(encoded).unwrap();
        assert_eq!(profile.entries[0].ring_channels, 1);
        assert!(profile.execution_fingerprint.is_empty());
        let tuning = CollectiveTuning::new(AlgorithmPolicy::Auto, 0, vec![0, 1]).unwrap();
        assert!(matches!(
            tuning.with_autotune_profile(profile),
            Err(TopologyError::InvalidProfile(_))
        ));
    }

    #[test]
    fn duplicate_rank_is_rejected() {
        assert!(matches!(
            CollectiveTuning::new(AlgorithmPolicy::Ring, 0, vec![0, 0]),
            Err(TopologyError::InvalidRing(_))
        ));
    }
}
