use super::*;
use crate::rank::layout::hierarchy_steps;
use crate::rank::payload::decode_exact;
use crate::rank::tags::{hierarchy_agreement_tag, hierarchy_data_tag, hierarchy_ring_data_tag};

mod all_gather;
mod all_reduce;
mod reduce_scatter;

type LeaderGatherPieces<T> = Vec<Option<Vec<T>>>;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    fn hierarchy_membership(&self) -> Result<(&[Vec<u32>], usize, usize), D::Error> {
        let groups = self.tuning.hierarchy_groups().ok_or_else(|| {
            TopologyError::InvalidHierarchy(
                "hierarchical collective selected without topology groups".into(),
            )
        })?;
        for (group_index, group) in groups.iter().enumerate() {
            if let Some(member_index) = group.iter().position(|rank| *rank == self.rank()) {
                return Ok((groups, group_index, member_index));
            }
        }
        Err(TopologyError::InvalidHierarchy(format!(
            "rank {} is absent from the configured hierarchy",
            self.rank()
        ))
        .into())
    }

    fn hierarchy_agreement(
        &self,
        operation: u64,
        length: usize,
        discriminator: u32,
        ring_channels: usize,
    ) -> Result<(), D::Error> {
        let groups = self.tuning.hierarchy_groups().ok_or_else(|| {
            TopologyError::InvalidHierarchy(
                "hierarchical collective selected without topology groups".into(),
            )
        })?;
        let channel_rails = (0..ring_channels)
            .map(|channel| self.collective_rail_for_channel(channel))
            .collect::<Vec<_>>();
        let tag = hierarchy_agreement_tag(
            operation,
            length,
            groups,
            discriminator,
            ring_channels,
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
                "invalid hierarchical collective agreement response".into(),
            )
            .into());
        }
        Ok(())
    }

    fn receive_values(&self, source: u32, tag: u64, length: usize) -> Result<Vec<T>, D::Error> {
        let response = self
            .session
            .receive(Some(source), tag, self.element_type, length as u64)?;
        decode_exact::<T, D::Error>(&response.payload, length, D::ELEMENT_SIZE, D::decode)
    }

}
