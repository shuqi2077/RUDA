use super::collective::InProcessCollective;
use super::device::InProcessDevice;
use super::error::InProcessError;
use crate::rank::{CollectiveAlgorithm, CollectiveStats, CollectiveTransport};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_COMMUNICATOR_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct Communicator<C> {
    id: u64,
    contexts: Vec<C>,
}

impl<C> Communicator<C> {
    pub fn new<K, E>(
        contexts: Vec<C>,
        initialize: impl FnOnce(&C) -> Result<K, E>,
    ) -> Result<(Self, K), E>
    where
        E: From<InProcessError>,
    {
        let Some(first) = contexts.first() else {
            return Err(InProcessError::EmptyWorld.into());
        };
        let kernels = initialize(first)?;
        Ok((
            Self {
                id: NEXT_COMMUNICATOR_ID.fetch_add(1, Ordering::Relaxed),
                contexts,
            },
            kernels,
        ))
    }

    pub fn world_size(&self) -> usize {
        self.contexts.len()
    }

    pub const fn transport(&self) -> CollectiveTransport {
        CollectiveTransport::HostStaged
    }

    pub fn collective<T, D, E>(&self) -> InProcessCollective<'_, T, D, E>
    where
        D: InProcessDevice<T, Context = C>,
        E: From<D::Error> + From<InProcessError>,
    {
        InProcessCollective {
            id: self.id,
            contexts: &self.contexts,
            marker: PhantomData,
        }
    }

    pub fn barrier(&self) -> Result<CollectiveStats, InProcessError> {
        if self.world_size() == 0 {
            return Err(InProcessError::EmptyWorld);
        }
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        stats.steps = u32::from(self.world_size() > 1);
        Ok(stats)
    }
}
