use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvPageId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvReservationId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvCacheConfig {
    pub block_size: usize,
    pub num_pages: usize,
    pub max_sequence_length: usize,
}

impl PagedKvCacheConfig {
    pub fn validate(self) -> Result<(), PagedKvError> {
        if self.block_size == 0 || self.num_pages == 0 || self.max_sequence_length == 0 {
            return Err(PagedKvError(
                "paged KV block_size, num_pages, and max_sequence_length must be non-zero".into(),
            ));
        }
        if self.num_pages > u32::MAX as usize {
            return Err(PagedKvError(
                "paged KV physical page count exceeds u32 page-table capacity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvAppendReservation {
    pub id: KvReservationId,
    pub request_id: RequestId,
    pub start_position: usize,
    pub token_count: usize,
    pub context_length: usize,
    pub newly_allocated_pages: Vec<KvPageId>,
    pub block_table: Vec<KvPageId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvSequenceSnapshot {
    pub request_id: RequestId,
    pub committed_tokens: usize,
    pub block_table: Vec<KvPageId>,
    pub pending_reservation: Option<KvReservationId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvCacheSnapshot {
    pub block_size: usize,
    pub total_pages: usize,
    pub free_pages: usize,
    pub sequences: Vec<PagedKvSequenceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvError(pub String);

impl Display for PagedKvError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for PagedKvError {}

#[derive(Debug)]
struct SequenceState {
    committed_tokens: usize,
    pages: Vec<KvPageId>,
    pending: Option<KvReservationId>,
}

#[derive(Debug)]
struct ReservationState {
    request_id: RequestId,
    previous_page_count: usize,
    start_position: usize,
    token_count: usize,
}

/// Owns physical-page metadata for every live KV sequence. Appends use an
/// explicit reserve/commit protocol so a failed kernel launch can return all
/// newly acquired pages without exposing a partially advanced sequence.
#[derive(Debug)]
pub struct PagedKvCacheManager {
    config: PagedKvCacheConfig,
    free_pages: BTreeSet<KvPageId>,
    sequences: BTreeMap<RequestId, SequenceState>,
    reservations: BTreeMap<KvReservationId, ReservationState>,
    next_reservation: u64,
}

impl PagedKvCacheManager {
    pub fn new(config: PagedKvCacheConfig) -> Result<Self, PagedKvError> {
        config.validate()?;
        let free_pages = (0..config.num_pages)
            .map(|page| KvPageId(page as u32))
            .collect();
        Ok(Self {
            config,
            free_pages,
            sequences: BTreeMap::new(),
            reservations: BTreeMap::new(),
            next_reservation: 1,
        })
    }

    pub const fn config(&self) -> PagedKvCacheConfig {
        self.config
    }

    /// Constant-time counters; unlike `snapshot`, these never clone page tables.
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    pub fn allocated_page_count(&self) -> usize {
        self.config.num_pages - self.free_pages.len()
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn reservation_count(&self) -> usize {
        self.reservations.len()
    }

    pub fn create_sequence(&mut self, request_id: RequestId) -> Result<(), PagedKvError> {
        if self.sequences.contains_key(&request_id) {
            return Err(PagedKvError(format!(
                "paged KV sequence {} already exists",
                request_id.0
            )));
        }
        self.sequences.insert(
            request_id,
            SequenceState {
                committed_tokens: 0,
                pages: Vec::new(),
                pending: None,
            },
        );
        Ok(())
    }

    pub fn contains_sequence(&self, request_id: RequestId) -> bool {
        self.sequences.contains_key(&request_id)
    }

    pub fn committed_tokens(&self, request_id: RequestId) -> Result<usize, PagedKvError> {
        Ok(self.sequence(request_id)?.committed_tokens)
    }

    pub fn block_table(&self, request_id: RequestId) -> Result<Vec<KvPageId>, PagedKvError> {
        Ok(self.sequence(request_id)?.pages.clone())
    }

    pub fn can_append(
        &self,
        request_id: RequestId,
        token_count: usize,
    ) -> Result<bool, PagedKvError> {
        if token_count == 0 {
            return Ok(false);
        }
        let sequence = self.sequence(request_id)?;
        if sequence.pending.is_some() {
            return Ok(false);
        }
        let Some(end) = sequence.committed_tokens.checked_add(token_count) else {
            return Ok(false);
        };
        if end > self.config.max_sequence_length {
            return Ok(false);
        }
        let required_pages = end.div_ceil(self.config.block_size);
        Ok(required_pages.saturating_sub(sequence.pages.len()) <= self.free_pages.len())
    }

    pub fn begin_append(
        &mut self,
        request_id: RequestId,
        token_count: usize,
    ) -> Result<PagedKvAppendReservation, PagedKvError> {
        if token_count == 0 {
            return Err(PagedKvError(
                "paged KV append reservation must contain at least one token".into(),
            ));
        }
        let sequence = self.sequence(request_id)?;
        if let Some(reservation) = sequence.pending {
            return Err(PagedKvError(format!(
                "paged KV sequence {} already has pending reservation {}",
                request_id.0, reservation.0
            )));
        }
        let start_position = sequence.committed_tokens;
        let context_length = start_position
            .checked_add(token_count)
            .ok_or_else(|| PagedKvError("paged KV sequence length overflow".into()))?;
        if context_length > self.config.max_sequence_length {
            return Err(PagedKvError(format!(
                "paged KV sequence {} would grow to {context_length}, above capacity {}",
                request_id.0, self.config.max_sequence_length
            )));
        }
        let previous_page_count = sequence.pages.len();
        let required_pages = context_length.div_ceil(self.config.block_size);
        let additional_pages = required_pages.saturating_sub(previous_page_count);
        if additional_pages > self.free_pages.len() {
            return Err(PagedKvError(format!(
                "paged KV append needs {additional_pages} new pages but only {} are free",
                self.free_pages.len()
            )));
        }
        let id = KvReservationId(self.next_reservation);
        self.next_reservation = self
            .next_reservation
            .checked_add(1)
            .ok_or_else(|| PagedKvError("paged KV reservation id overflow".into()))?;

        let mut newly_allocated_pages = Vec::with_capacity(additional_pages);
        for _ in 0..additional_pages {
            let page = self
                .free_pages
                .pop_first()
                .expect("free-page count was checked before allocation");
            newly_allocated_pages.push(page);
        }
        let sequence = self
            .sequences
            .get_mut(&request_id)
            .expect("sequence was checked before allocation");
        sequence.pages.extend(newly_allocated_pages.iter().copied());
        sequence.pending = Some(id);
        let block_table = sequence.pages.clone();
        self.reservations.insert(
            id,
            ReservationState {
                request_id,
                previous_page_count,
                start_position,
                token_count,
            },
        );
        Ok(PagedKvAppendReservation {
            id,
            request_id,
            start_position,
            token_count,
            context_length,
            newly_allocated_pages,
            block_table,
        })
    }

    pub fn commit_append(
        &mut self,
        reservation_id: KvReservationId,
    ) -> Result<usize, PagedKvError> {
        self.validate_reservation(reservation_id)?;
        let reservation = self.take_reservation(reservation_id)?;
        let sequence = self
            .sequences
            .get_mut(&reservation.request_id)
            .ok_or_else(|| PagedKvError("reservation references a removed sequence".into()))?;
        if sequence.pending != Some(reservation_id) {
            return Err(PagedKvError(format!(
                "reservation {} does not own sequence {} pending state",
                reservation_id.0, reservation.request_id.0
            )));
        }
        let context_length = reservation
            .start_position
            .checked_add(reservation.token_count)
            .ok_or_else(|| PagedKvError("paged KV committed length overflow".into()))?;
        sequence.committed_tokens = context_length;
        sequence.pending = None;
        Ok(context_length)
    }

    pub fn cancel_append(&mut self, reservation_id: KvReservationId) -> Result<(), PagedKvError> {
        self.validate_reservation(reservation_id)?;
        let reservation = self.take_reservation(reservation_id)?;
        let sequence = self
            .sequences
            .get_mut(&reservation.request_id)
            .ok_or_else(|| PagedKvError("reservation references a removed sequence".into()))?;
        if sequence.pending != Some(reservation_id) {
            return Err(PagedKvError(format!(
                "reservation {} does not own sequence {} pending state",
                reservation_id.0, reservation.request_id.0
            )));
        }
        let released = sequence.pages.split_off(reservation.previous_page_count);
        sequence.pending = None;
        for page in released {
            let inserted = self.free_pages.insert(page);
            debug_assert!(inserted, "released page must not already be free");
        }
        Ok(())
    }

    /// Commit a group atomically with respect to validation errors. Unknown or
    /// duplicate IDs leave every reservation, sequence and page unchanged.
    pub fn commit_appends(
        &mut self,
        reservation_ids: &[KvReservationId],
    ) -> Result<Vec<usize>, PagedKvError> {
        self.validate_reservations(reservation_ids)?;
        Ok(reservation_ids.iter().map(|&id| {
            self.commit_append(id).expect("all reservations were validated before commit")
        }).collect())
    }

    /// Cancel a group atomically with respect to validation errors.
    pub fn cancel_appends(
        &mut self,
        reservation_ids: &[KvReservationId],
    ) -> Result<(), PagedKvError> {
        self.validate_reservations(reservation_ids)?;
        for &id in reservation_ids {
            self.cancel_append(id).expect("all reservations were validated before cancellation");
        }
        Ok(())
    }

    fn validate_reservations(&self, ids: &[KvReservationId]) -> Result<(), PagedKvError> {
        let mut seen = BTreeSet::new();
        for &id in ids {
            if !seen.insert(id) {
                return Err(PagedKvError(format!("duplicate reservation {}", id.0)));
            }
            self.validate_reservation(id)?;
        }
        Ok(())
    }

    fn validate_reservation(&self, id: KvReservationId) -> Result<(), PagedKvError> {
        let reservation = self.reservations.get(&id).ok_or_else(|| {
            PagedKvError(format!("paged KV reservation {} does not exist", id.0))
        })?;
        let sequence = self.sequence(reservation.request_id)?;
        if sequence.pending != Some(id)
            || sequence.committed_tokens != reservation.start_position
            || sequence.pages.len() < reservation.previous_page_count
        {
            return Err(PagedKvError(format!(
                "reservation {} does not own a consistent sequence state", id.0
            )));
        }
        reservation.start_position.checked_add(reservation.token_count)
            .ok_or_else(|| PagedKvError("paged KV committed length overflow".into()))?;
        Ok(())
    }

    pub fn remove_sequence(&mut self, request_id: RequestId) -> Result<(), PagedKvError> {
        let sequence = self.sequence(request_id)?;
        if let Some(reservation) = sequence.pending {
            return Err(PagedKvError(format!(
                "cannot remove sequence {} while reservation {} is pending",
                request_id.0, reservation.0
            )));
        }
        let sequence = self
            .sequences
            .remove(&request_id)
            .expect("sequence was checked before removal");
        for page in sequence.pages {
            let inserted = self.free_pages.insert(page);
            debug_assert!(inserted, "removed sequence page must not already be free");
        }
        Ok(())
    }

    pub fn snapshot(&self) -> PagedKvCacheSnapshot {
        PagedKvCacheSnapshot {
            block_size: self.config.block_size,
            total_pages: self.config.num_pages,
            free_pages: self.free_pages.len(),
            sequences: self
                .sequences
                .iter()
                .map(|(&request_id, sequence)| PagedKvSequenceSnapshot {
                    request_id,
                    committed_tokens: sequence.committed_tokens,
                    block_table: sequence.pages.clone(),
                    pending_reservation: sequence.pending,
                })
                .collect(),
        }
    }

    fn sequence(&self, request_id: RequestId) -> Result<&SequenceState, PagedKvError> {
        self.sequences.get(&request_id).ok_or_else(|| {
            PagedKvError(format!("paged KV sequence {} does not exist", request_id.0))
        })
    }

    fn take_reservation(
        &mut self,
        reservation_id: KvReservationId,
    ) -> Result<ReservationState, PagedKvError> {
        self.reservations.remove(&reservation_id).ok_or_else(|| {
            PagedKvError(format!(
                "paged KV reservation {} does not exist",
                reservation_id.0
            ))
        })
    }
}

#[cfg(test)]
mod tests;
