use super::buffer::{DistributedBuffer, RootedBuffer, VariableDistributedBuffer};
use super::device::InProcessDevice;
use super::error::InProcessError;
use crate::rank::{CollectiveAlgorithm, CollectiveStats, CollectiveTransport};
use std::marker::PhantomData;
use std::ops::Range;

mod all_to_all;
mod exchange;
mod memory;
mod reduction;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointToPointTransfer {
    pub source_rank: usize,
    pub source_range: Range<usize>,
    pub destination_rank: usize,
    pub destination_offset: usize,
    pub tag: u64,
}

pub struct InProcessCollective<'a, T, D: InProcessDevice<T>, E> {
    pub(super) id: u64,
    pub(super) contexts: &'a [D::Context],
    pub(super) marker: PhantomData<fn() -> (T, E)>,
}

impl<T, D, E> InProcessCollective<'_, T, D, E>
where
    T: Copy + Send + Sync + 'static,
    D: InProcessDevice<T>,
    E: From<D::Error> + From<InProcessError>,
{
    pub fn world_size(&self) -> usize {
        self.contexts.len()
    }

    pub const fn transport(&self) -> CollectiveTransport {
        CollectiveTransport::HostStaged
    }

    fn context(&self, rank: usize) -> Result<&D::Context, E> {
        self.contexts
            .get(rank)
            .ok_or(InProcessError::RankOutOfRange {
                rank,
                world_size: self.world_size(),
            })
            .map_err(Into::into)
    }

    fn validate_buffer(&self, buffer: &DistributedBuffer<T, D>) -> Result<(), E> {
        if buffer.communicator_id != self.id || buffer.world_size() != self.world_size() {
            return Err(InProcessError::WrongCommunicator.into());
        }
        Ok(())
    }

    fn validate_rooted_buffer(&self, buffer: &RootedBuffer<T, D>) -> Result<(), E> {
        if buffer.communicator_id != self.id {
            return Err(InProcessError::WrongCommunicator.into());
        }
        self.context(buffer.root)?;
        Ok(())
    }

    fn validate_variable_buffer(&self, buffer: &VariableDistributedBuffer<T, D>) -> Result<(), E> {
        if buffer.communicator_id != self.id || buffer.world_size() != self.world_size() {
            return Err(InProcessError::WrongCommunicator.into());
        }
        Ok(())
    }
}

fn ring_chunks(length: u32, world_size: usize) -> Result<Vec<Range<usize>>, InProcessError> {
    if world_size == 0 {
        return Err(InProcessError::EmptyWorld);
    }
    let length = length as usize;
    let base = length / world_size;
    let remainder = length % world_size;
    let mut start = 0;
    Ok((0..world_size)
        .map(|rank| {
            let chunk_length = base + usize::from(rank < remainder);
            let range = start..start + chunk_length;
            start += chunk_length;
            range
        })
        .collect())
}

fn transferred_bytes(
    transfers: usize,
    elements: usize,
    element_bytes: usize,
) -> Result<u64, InProcessError> {
    let bytes = transfers
        .checked_mul(elements)
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or(InProcessError::Overflow("transferred bytes"))?;
    u64::try_from(bytes).map_err(|_| InProcessError::Overflow("transferred bytes"))
}
