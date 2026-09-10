use super::device::RankDevice;
use super::error::RankError;
use super::host::HostStagedExchange;
use super::layout::{balanced_ranges, prefix_offsets};
use super::payload::{decode_counts_response, encode_counts};
use super::tags::{reduction_exchange_options, ring_agreement_tag, ring_data_tag};
use super::work::WorkError;
use super::{
    ANY_RANK, FLAG_COUNTS_PREFIX, CollectiveKind, CollectiveAlgorithm, CollectiveStats, CollectiveTuning, ElementType, ExchangeOptions,
    NetworkError, Opcode, RankTransport, ReductionOperation, TopologyError,
};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

mod all_gather;
mod all_to_all;
mod direct;
mod output;
mod point_to_point;
mod reduce_dispatch;
mod hierarchical;
mod reduce_scatter;
mod reduction;
mod ring;

pub use output::{ReceivedMessage, VariableCollectiveOutput};

pub struct DeviceCollective<'a, T, D: RankDevice<T>> {
    execution: &'a D,
    session: &'a dyn RankTransport,
    tuning: &'a CollectiveTuning,
    collective_rail_order: &'a [usize],
    internal_sequence: &'a AtomicU64,
    element_type: ElementType,
    element: PhantomData<fn() -> T>,
}

impl<T, D: RankDevice<T>> Clone for DeviceCollective<'_, T, D> {
    fn clone(&self) -> Self {
        Self {
            execution: self.execution,
            session: self.session,
            tuning: self.tuning,
            collective_rail_order: self.collective_rail_order,
            internal_sequence: self.internal_sequence,
            element_type: self.element_type,
            element: PhantomData,
        }
    }
}

impl<'a, T, D> DeviceCollective<'a, T, D>
where
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn new(
        execution: &'a D,
        session: &'a dyn RankTransport,
        tuning: &'a CollectiveTuning,
        collective_rail_order: &'a [usize],
        internal_sequence: &'a AtomicU64,
        element_type: ElementType,
    ) -> Self {
        Self {
            execution,
            session,
            tuning,
            collective_rail_order,
            internal_sequence,
            element_type,
            element: PhantomData,
        }
    }
    pub fn rank(&self) -> u32 {
        self.session.rank()
    }

    pub fn world_size(&self) -> u32 {
        self.session.world_size()
    }

    fn effective_ring_channels(
        &self,
        requested_channels: usize,
        elements_per_rank: usize,
    ) -> usize {
        requested_channels.min(elements_per_rank.max(1))
    }

    fn collective_rail_for_channel(&self, channel: usize) -> usize {
        self.collective_rail_order[channel % self.collective_rail_order.len()]
    }

    fn propagate_channel_failure<R>(
        &self,
        operation: &str,
        result: Result<R, D::Error>,
    ) -> Result<R, D::Error> {
        if let Err(error) = &result {
            let _ = self
                .session
                .abort(&format!("{operation} channel failed: {error}"));
        }
        result
    }

    fn ring_position(&self, ring: &[u32]) -> Result<usize, D::Error> {
        ring.iter()
            .position(|rank| *rank == self.rank())
            .ok_or_else(|| {
                TopologyError::InvalidRing(format!(
                    "rank {} is absent from the configured ring",
                    self.rank()
                ))
                .into()
            })
    }

    fn ring_agreement(
        &self,
        operation: u64,
        length: usize,
        discriminator: u32,
        channels: usize,
    ) -> Result<(), D::Error> {
        let rings = (0..channels)
            .map(|channel| self.tuning.ring_order_for_channel(channel))
            .collect::<Vec<_>>();
        let channel_rails = (0..channels)
            .map(|channel| self.collective_rail_for_channel(channel))
            .collect::<Vec<_>>();
        let tag = ring_agreement_tag(
            operation,
            length,
            &rings,
            discriminator,
            channels,
            &channel_rails,
        );
        let response = self.session.exchange_with_options(
            Opcode::Barrier,
            ElementType::None,
            ExchangeOptions {
                root_rank: ANY_RANK,
                element_count: 0,
                flags: 0,
                tag,
            },
            Vec::new(),
        )?;
        if response.header.opcode != Opcode::Barrier || !response.payload.is_empty() {
            return Err(NetworkError::InvalidConfiguration(
                "invalid ring collective agreement response".into(),
            )
            .into());
        }
        Ok(())
    }

    fn stats(
        &self,
        algorithm: CollectiveAlgorithm,
        steps: u32,
        transferred_bytes: usize,
        reduction_kernel_launches: u32,
    ) -> Result<CollectiveStats, D::Error> {
        Ok(HostStagedExchange::new(self.session).stats(
            algorithm,
            steps,
            transferred_bytes,
            reduction_kernel_launches,
        )?)
    }

    fn validate_root(&self, root: u32) -> Result<(), D::Error> {
        Ok(HostStagedExchange::new(self.session).validate_root(root)?)
    }

    fn decode_exact(&self, payload: &[u8], expected_elements: usize) -> Result<Vec<T>, D::Error> {
        super::payload::decode_exact::<T, D::Error>(
            payload,
            expected_elements,
            D::ELEMENT_SIZE,
            D::decode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn exchange_host_payload(
        &self,
        element_type: ElementType,
        payload: Vec<u8>,
        receive_bytes: usize,
        destination: u32,
        source: u32,
        rail_hint: usize,
        tag: u64,
    ) -> Result<Vec<u8>, D::Error> {
        Ok(HostStagedExchange::new(self.session).exchange_host_payload(
            self.collective_rail_order,
            element_type,
            payload,
            receive_bytes,
            destination,
            source,
            rail_hint,
            tag,
        )?)
    }
}
