use crate::{
    GreedyGenerationConfig, KvPageId, KvReservationId, PagedKvAppendReservation,
    PagedKvCacheConfig, PagedKvCacheManager, PagedKvError, RequestId, TokenSampler,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

mod sampling;
mod lifecycle;
pub use lifecycle::{CancelledGeneration, ContinuousBatchOptions, KvAdmissionPolicy};
pub(crate) use sampling::BatchTokenSelection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuousBatchConfig {
    pub max_active_sequences: usize,
    pub max_batch_tokens: usize,
}

impl ContinuousBatchConfig {
    pub fn validate(self) -> Result<(), ContinuousBatchError> {
        if self.max_active_sequences == 0 || self.max_batch_tokens == 0 {
            return Err(ContinuousBatchError(
                "continuous batch limits must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledBatchKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledSequence {
    pub request_id: RequestId,
    pub token_ids: Vec<i32>,
    pub start_position: usize,
    pub context_length: usize,
    pub block_table: Vec<KvPageId>,
    pub reservation_id: KvReservationId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledBatch {
    pub id: u64,
    pub kind: ScheduledBatchKind,
    pub sequences: Vec<ScheduledSequence>,
}

impl ScheduledBatch {
    pub fn batch_size(&self) -> usize {
        self.sequences.len()
    }

    pub fn next_n(&self) -> usize {
        self.sequences.first().map_or(0, |row| row.token_ids.len())
    }

    /// Row-major `[batch, next_n]` input expected by the Llama embedding path.
    pub fn token_matrix(&self) -> Vec<i32> {
        self.sequences
            .iter()
            .flat_map(|sequence| sequence.token_ids.iter().copied())
            .collect()
    }

    /// One causal length per query token, in the exact order required by
    /// `Fp8Fp4PagedAttention::context_lens`.
    pub fn context_lengths(&self) -> Result<Vec<u32>, ContinuousBatchError> {
        let mut lengths = Vec::with_capacity(self.batch_size().saturating_mul(self.next_n()));
        for sequence in &self.sequences {
            for offset in 0..sequence.token_ids.len() {
                let length = sequence
                    .start_position
                    .checked_add(offset)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| ContinuousBatchError("context length overflow".into()))?;
                lengths.push(u32::try_from(length).map_err(|_| {
                    ContinuousBatchError("context length exceeds DeepGEMM u32 ABI".into())
                })?);
            }
        }
        Ok(lengths)
    }

    /// Return the padded row-major physical page table and its row stride.
    pub fn flattened_block_table(&self) -> Result<(Vec<u32>, usize), ContinuousBatchError> {
        let stride = self
            .sequences
            .iter()
            .map(|sequence| sequence.block_table.len())
            .max()
            .unwrap_or(0);
        if stride == 0 && !self.sequences.is_empty() {
            return Err(ContinuousBatchError(
                "scheduled sequence has no physical KV page".into(),
            ));
        }
        let mut table = vec![0_u32; self.batch_size().saturating_mul(stride)];
        for (row, sequence) in self.sequences.iter().enumerate() {
            for (column, page) in sequence.block_table.iter().enumerate() {
                table[row * stride + column] = page.0;
            }
        }
        Ok((table, stride))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedGeneration {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<i32>,
    pub generated_token_ids: Vec<i32>,
    pub stopped_on_eos: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousBatchSnapshot {
    pub pending_requests: usize,
    pub active_requests: usize,
    pub has_in_flight_batch: bool,
    pub finished_requests: usize,
    pub free_kv_pages: usize,
    pub total_kv_pages: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousBatchError(pub String);

impl Display for ContinuousBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ContinuousBatchError {}

impl From<PagedKvError> for ContinuousBatchError {
    fn from(value: PagedKvError) -> Self {
        Self(value.0)
    }
}

#[derive(Debug)]
struct PendingRequest {
    id: RequestId,
    prompt_token_ids: Vec<i32>,
    generation: GreedyGenerationConfig,
    sampler: Option<TokenSampler>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestPhase {
    NeedsPrefill,
    AwaitingDecode(i32),
    InFlight(u64),
}

#[derive(Debug)]
struct ActiveRequest {
    prompt_token_ids: Vec<i32>,
    generation: GreedyGenerationConfig,
    sampler: Option<TokenSampler>,
    eos: BTreeSet<i32>,
    generated_token_ids: Vec<i32>,
    phase: RequestPhase,
}

/// Metadata scheduler for paged Llama inference. It allows prefill requests to
/// enter while older requests are decoding, batches all ready one-token decode
/// rows, and commits KV positions only after the execution result succeeds.
#[derive(Debug)]
pub struct ContinuousBatchScheduler {
    config: ContinuousBatchConfig,
    options: ContinuousBatchOptions,
    kv_cache: PagedKvCacheManager,
    pending: VecDeque<PendingRequest>,
    active: BTreeMap<RequestId, ActiveRequest>,
    finished: VecDeque<FinishedGeneration>,
    in_flight: Option<ScheduledBatch>,
    prefer_prefill: bool,
    next_request: u64,
    next_batch: u64,
}

impl ContinuousBatchScheduler {
    pub fn new(
        config: ContinuousBatchConfig,
        kv_config: PagedKvCacheConfig,
    ) -> Result<Self, ContinuousBatchError> {
        Self::with_options(config, kv_config, ContinuousBatchOptions::default())
    }

    /// Configure bounded queuing and optional full-sequence KV admission.
    pub fn with_options(
        config: ContinuousBatchConfig,
        kv_config: PagedKvCacheConfig,
        options: ContinuousBatchOptions,
    ) -> Result<Self, ContinuousBatchError> {
        config.validate()?;
        options.validate()?;
        Ok(Self {
            config,
            options,
            kv_cache: PagedKvCacheManager::new(kv_config)?,
            pending: VecDeque::new(),
            active: BTreeMap::new(),
            finished: VecDeque::new(),
            in_flight: None,
            prefer_prefill: true,
            next_request: 1,
            next_batch: 1,
        })
    }

    pub fn submit(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: GreedyGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        self.submit_with_sampler(prompt_token_ids, generation, None)
    }

    fn submit_with_sampler(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: GreedyGenerationConfig,
        sampler: Option<TokenSampler>,
    ) -> Result<RequestId, ContinuousBatchError> {
        if prompt_token_ids.is_empty() {
            return Err(ContinuousBatchError(
                "continuous generation prompt must not be empty".into(),
            ));
        }
        if prompt_token_ids.len() > self.config.max_batch_tokens {
            return Err(ContinuousBatchError(format!(
                "prompt has {} tokens, above max_batch_tokens {}",
                prompt_token_ids.len(),
                self.config.max_batch_tokens
            )));
        }
        let total_length = prompt_token_ids
            .len()
            .checked_add(generation.max_new_tokens)
            .ok_or_else(|| ContinuousBatchError("request sequence length overflow".into()))?;
        if total_length > self.kv_cache.config().max_sequence_length {
            return Err(ContinuousBatchError(format!(
                "request length {total_length} exceeds paged KV capacity {}",
                self.kv_cache.config().max_sequence_length
            )));
        }
        if prompt_token_ids.iter().chain(&generation.eos_token_ids).any(|&id| id < 0) {
            return Err(ContinuousBatchError("prompt and EOS token IDs must be non-negative".into()));
        }
        if generation.max_new_tokens > 0 {
            let kv = self.kv_cache.config();
            let required_pages = match self.options.kv_admission {
                KvAdmissionPolicy::OnDemand => prompt_token_ids.len().div_ceil(kv.block_size),
                KvAdmissionPolicy::ReserveSequenceCapacity =>
                    Self::request_page_budget(prompt_token_ids.len(), &generation, kv.block_size),
            };
            if required_pages > kv.num_pages {
                return Err(ContinuousBatchError(format!(
                    "request needs {required_pages} KV pages under the admission policy, but the pool has {}",
                    kv.num_pages
                )));
            }
            if self.options.max_pending_requests.is_some_and(|limit| self.pending.len() >= limit) {
                return Err(ContinuousBatchError("pending request queue is full".into()));
            }
        }
        let id = RequestId(self.next_request);
        self.next_request = self
            .next_request
            .checked_add(1)
            .ok_or_else(|| ContinuousBatchError("request id overflow".into()))?;
        if generation.max_new_tokens == 0 {
            self.finished.push_back(FinishedGeneration {
                request_id: id,
                prompt_token_ids,
                generated_token_ids: Vec::new(),
                stopped_on_eos: false,
            });
        } else {
            self.pending.push_back(PendingRequest {
                id,
                prompt_token_ids,
                generation,
                sampler,
            });
        }
        Ok(id)
    }

    /// Build the next device batch. At most one batch is in flight so page
    /// reservations have an unambiguous commit/rollback owner.
    pub fn schedule(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        if let Some(batch) = &self.in_flight {
            return Err(ContinuousBatchError(format!(
                "batch {} is still in flight",
                batch.id
            )));
        }
        // Check before acquiring reservations: an exhausted batch ID must not
        // leak pages or strand requests in a partially scheduled batch.
        if self.next_batch == u64::MAX {
            return Err(ContinuousBatchError("batch id overflow".into()));
        }
        self.promote_pending()?;
        if self.prefer_prefill
            && let Some(batch) = self.schedule_prefill()?
        {
            self.prefer_prefill = false;
            self.in_flight = Some(batch.clone());
            return Ok(Some(batch));
        }
        if let Some(batch) = self.schedule_decode()? {
            self.prefer_prefill = true;
            self.in_flight = Some(batch.clone());
            return Ok(Some(batch));
        }
        if let Some(batch) = self.schedule_prefill()? {
            self.prefer_prefill = false;
            self.in_flight = Some(batch.clone());
            return Ok(Some(batch));
        }
        Ok(None)
    }

    /// Commit every page reservation and apply one sampled/selected token per
    /// request. The caller must preserve the scheduled row order.
    pub fn complete_batch(
        &mut self,
        batch_id: u64,
        generated_token_ids: &[i32],
    ) -> Result<(), ContinuousBatchError> {
        let batch = self.take_batch(batch_id)?;
        if generated_token_ids.len() != batch.sequences.len() {
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError(format!(
                "batch {batch_id} has {} rows but {} generated tokens were supplied",
                self.in_flight.as_ref().unwrap().sequences.len(),
                generated_token_ids.len()
            )));
        }
        if generated_token_ids.iter().any(|&token| token < 0) {
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError("generated token IDs must be non-negative".into()));
        }
        let ownership_error = batch.sequences.iter().find_map(|sequence| {
            let Some(request) = self.active.get(&sequence.request_id) else {
                return Some(ContinuousBatchError(format!(
                    "batch references missing request {}",
                    sequence.request_id.0
                )));
            };
            (request.phase != RequestPhase::InFlight(batch_id)).then(|| {
                ContinuousBatchError(format!(
                    "request {} is not owned by batch {batch_id}",
                    sequence.request_id.0
                ))
            })
        });
        if let Some(error) = ownership_error {
            self.in_flight = Some(batch);
            return Err(error);
        }
        let reservations = batch.sequences.iter().map(|row| row.reservation_id).collect::<Vec<_>>();
        if let Err(error) = self.kv_cache.commit_appends(&reservations) {
            self.in_flight = Some(batch);
            return Err(error.into());
        }
        for (sequence, &token) in batch.sequences.iter().zip(generated_token_ids) {
            let request = self
                .active
                .get_mut(&sequence.request_id)
                .expect("batch ownership was validated before KV commit");
            request.generated_token_ids.push(token);
            let stopped_on_eos = request.eos.contains(&token);
            let finished = stopped_on_eos
                || request.generated_token_ids.len() >= request.generation.max_new_tokens;
            if finished {
                let request = self
                    .active
                    .remove(&sequence.request_id)
                    .expect("active request was checked before removal");
                self.kv_cache.remove_sequence(sequence.request_id)?;
                self.finished.push_back(FinishedGeneration {
                    request_id: sequence.request_id,
                    prompt_token_ids: request.prompt_token_ids,
                    generated_token_ids: request.generated_token_ids,
                    stopped_on_eos,
                });
            } else {
                request.phase = RequestPhase::AwaitingDecode(token);
            }
        }
        Ok(())
    }

    /// Roll back all page allocations from a failed device batch and restore
    /// requests to the exact state from which that batch was scheduled.
    pub fn fail_batch(&mut self, batch_id: u64) -> Result<(), ContinuousBatchError> {
        let batch = self.take_batch(batch_id)?;
        if batch.sequences.iter().any(|row| {
            self.active.get(&row.request_id)
                .is_none_or(|request| request.phase != RequestPhase::InFlight(batch_id))
        }) {
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError("failed batch does not own every request".into()));
        }
        let reservations = batch.sequences.iter().map(|row| row.reservation_id).collect::<Vec<_>>();
        if let Err(error) = self.kv_cache.cancel_appends(&reservations) {
            self.in_flight = Some(batch);
            return Err(error.into());
        }
        for sequence in &batch.sequences {
            let request = self.active.get_mut(&sequence.request_id)
                .expect("batch ownership was validated before cancellation");
            request.phase = match batch.kind {
                ScheduledBatchKind::Prefill => RequestPhase::NeedsPrefill,
                ScheduledBatchKind::Decode => RequestPhase::AwaitingDecode(sequence.token_ids[0]),
            };
        }
        Ok(())
    }

    pub fn pop_finished(&mut self) -> Option<FinishedGeneration> {
        self.finished.pop_front()
    }

    pub fn snapshot(&self) -> ContinuousBatchSnapshot {
        ContinuousBatchSnapshot {
            pending_requests: self.pending.len(),
            active_requests: self.active.len(),
            has_in_flight_batch: self.in_flight.is_some(),
            finished_requests: self.finished.len(),
            free_kv_pages: self.kv_cache.free_page_count(),
            total_kv_pages: self.kv_cache.config().num_pages,
        }
    }

    pub fn kv_cache(&self) -> &PagedKvCacheManager {
        &self.kv_cache
    }

    fn promote_pending(&mut self) -> Result<(), ContinuousBatchError> {
        let mut available_pages = self.kv_cache.config().num_pages;
        if self.options.kv_admission == KvAdmissionPolicy::ReserveSequenceCapacity {
            for request in self.active.values() {
                available_pages = available_pages.checked_sub(Self::request_page_budget(
                    request.prompt_token_ids.len(), &request.generation, self.kv_cache.config().block_size,
                )).ok_or_else(|| ContinuousBatchError("active KV page budgets exceed the pool".into()))?;
            }
        }
        while self.active.len() < self.config.max_active_sequences {
            if self.options.kv_admission == KvAdmissionPolicy::ReserveSequenceCapacity {
                let Some(pending) = self.pending.front() else { break; };
                let pages = Self::request_page_budget(
                    pending.prompt_token_ids.len(), &pending.generation, self.kv_cache.config().block_size,
                );
                // FIFO admission prevents a stream of short arrivals from
                // indefinitely overtaking an older, larger request.
                if pages > available_pages { break; }
                available_pages -= pages;
            }
            let Some(pending) = self.pending.pop_front() else {
                break;
            };
            self.kv_cache.create_sequence(pending.id)?;
            self.active.insert(
                pending.id,
                ActiveRequest {
                    eos: pending.generation.eos_token_ids.iter().copied().collect(),
                    prompt_token_ids: pending.prompt_token_ids,
                    generation: pending.generation,
                    sampler: pending.sampler,
                    generated_token_ids: Vec::new(),
                    phase: RequestPhase::NeedsPrefill,
                },
            );
        }
        Ok(())
    }

    fn schedule_decode(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        let candidates = self
            .active
            .iter()
            .filter_map(|(&id, request)| match request.phase {
                RequestPhase::AwaitingDecode(token) => Some((id, token)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut rows = Vec::new();
        for (request_id, token) in candidates {
            if rows.len() >= self.config.max_batch_tokens {
                break;
            }
            if !self.kv_cache.can_append(request_id, 1)? {
                continue;
            }
            let reservation = match self.kv_cache.begin_append(request_id, 1) {
                Ok(reservation) => reservation,
                Err(error) => {
                    self.rollback_scheduled_rows(&rows)?;
                    return Err(error.into());
                }
            };
            rows.push(Self::scheduled_sequence(reservation, vec![token]));
        }
        self.finish_schedule(ScheduledBatchKind::Decode, rows)
    }

    fn schedule_prefill(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        let mut buckets = BTreeMap::<usize, Vec<RequestId>>::new();
        for (&request_id, request) in &self.active {
            if request.phase == RequestPhase::NeedsPrefill
                && self
                    .kv_cache
                    .can_append(request_id, request.prompt_token_ids.len())?
            {
                buckets
                    .entry(request.prompt_token_ids.len())
                    .or_default()
                    .push(request_id);
            }
        }
        let Some((&next_n, candidates)) =
            buckets
                .iter()
                .max_by(|(left_len, left), (right_len, right)| {
                    let left_tokens =
                        left.len().min(self.config.max_batch_tokens / **left_len) * **left_len;
                    let right_tokens =
                        right.len().min(self.config.max_batch_tokens / **right_len) * **right_len;
                    left_tokens
                        .cmp(&right_tokens)
                        // For equally full batches, keep the oldest request first.
                        .then_with(|| right[0].cmp(&left[0]))
                })
        else {
            return Ok(None);
        };
        let max_rows = self.config.max_batch_tokens / next_n;
        let mut rows = Vec::new();
        for &request_id in candidates {
            if rows.len() >= max_rows {
                break;
            }
            if !self.kv_cache.can_append(request_id, next_n)? {
                continue;
            }
            let prompt = self
                .active
                .get(&request_id)
                .expect("prefill candidate came from the active request map")
                .prompt_token_ids
                .clone();
            let reservation = match self.kv_cache.begin_append(request_id, next_n) {
                Ok(reservation) => reservation,
                Err(error) => {
                    self.rollback_scheduled_rows(&rows)?;
                    return Err(error.into());
                }
            };
            rows.push(Self::scheduled_sequence(reservation, prompt));
        }
        self.finish_schedule(ScheduledBatchKind::Prefill, rows)
    }

    fn rollback_scheduled_rows(&mut self, rows: &[ScheduledSequence]) -> Result<(), ContinuousBatchError> {
        let ids = rows.iter().map(|row| row.reservation_id).collect::<Vec<_>>();
        self.kv_cache.cancel_appends(&ids)?;
        Ok(())
    }

    fn finish_schedule(
        &mut self,
        kind: ScheduledBatchKind,
        rows: Vec<ScheduledSequence>,
    ) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        if rows.is_empty() {
            return Ok(None);
        }
        let id = self.next_batch;
        let Some(next_batch) = self.next_batch.checked_add(1) else {
            self.rollback_scheduled_rows(&rows)?;
            return Err(ContinuousBatchError("batch id overflow".into()));
        };
        if rows.iter().any(|row| !self.active.contains_key(&row.request_id)) {
            self.rollback_scheduled_rows(&rows)?;
            return Err(ContinuousBatchError("scheduled request disappeared".into()));
        }
        self.next_batch = next_batch;
        for row in &rows {
            let request = self.active.get_mut(&row.request_id).ok_or_else(|| {
                ContinuousBatchError(format!(
                    "scheduled request {} disappeared",
                    row.request_id.0
                ))
            })?;
            request.phase = RequestPhase::InFlight(id);
        }
        Ok(Some(ScheduledBatch {
            id,
            kind,
            sequences: rows,
        }))
    }

    fn scheduled_sequence(
        reservation: PagedKvAppendReservation,
        token_ids: Vec<i32>,
    ) -> ScheduledSequence {
        ScheduledSequence {
            request_id: reservation.request_id,
            token_ids,
            start_position: reservation.start_position,
            context_length: reservation.context_length,
            block_table: reservation.block_table,
            reservation_id: reservation.id,
        }
    }

    fn take_batch(&mut self, batch_id: u64) -> Result<ScheduledBatch, ContinuousBatchError> {
        let batch = self
            .in_flight
            .take()
            .ok_or_else(|| ContinuousBatchError(format!("batch {batch_id} is not in flight")))?;
        if batch.id != batch_id {
            let actual = batch.id;
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError(format!(
                "batch {batch_id} cannot complete while batch {actual} is in flight"
            )));
        }
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generation() -> GreedyGenerationConfig {
        GreedyGenerationConfig {
            max_new_tokens: 2,
            eos_token_ids: Vec::new(),
        }
    }

    fn scheduler(max_active_sequences: usize, max_batch_tokens: usize) -> ContinuousBatchScheduler {
        ContinuousBatchScheduler::new(
            ContinuousBatchConfig {
                max_active_sequences,
                max_batch_tokens,
            },
            PagedKvCacheConfig {
                block_size: 8,
                num_pages: 32,
                max_sequence_length: 32,
            },
        )
        .unwrap()
    }

    #[test]
    fn prefill_selects_the_shape_bucket_with_the_most_useful_tokens() {
        let mut scheduler = scheduler(3, 8);
        let oldest = scheduler.submit(vec![1, 2], generation()).unwrap();
        let first_full = scheduler.submit(vec![3, 4, 5], generation()).unwrap();
        let second_full = scheduler.submit(vec![6, 7, 8], generation()).unwrap();

        let batch = scheduler.schedule().unwrap().unwrap();

        assert_eq!(batch.kind, ScheduledBatchKind::Prefill);
        assert_eq!(batch.next_n(), 3);
        assert_eq!(
            batch
                .sequences
                .iter()
                .map(|row| row.request_id)
                .collect::<Vec<_>>(),
            vec![first_full, second_full]
        );
        assert!(!batch.sequences.iter().any(|row| row.request_id == oldest));
    }

    #[test]
    fn decode_skips_unreservable_rows_without_underfilling_the_batch() {
        let mut scheduler = ContinuousBatchScheduler::new(
            ContinuousBatchConfig {
                max_active_sequences: 2,
                max_batch_tokens: 1,
            },
            PagedKvCacheConfig {
                block_size: 2,
                num_pages: 2,
                max_sequence_length: 4,
            },
        )
        .unwrap();
        let blocked = RequestId(1);
        let runnable = RequestId(2);
        for request_id in [blocked, runnable] {
            scheduler.kv_cache.create_sequence(request_id).unwrap();
            scheduler.active.insert(
                request_id,
                ActiveRequest {
                    prompt_token_ids: vec![1],
                    generation: generation(),
                    sampler: None,
                    eos: BTreeSet::new(),
                    generated_token_ids: Vec::new(),
                    phase: RequestPhase::AwaitingDecode(9),
                },
            );
        }
        let blocked_prefill = scheduler.kv_cache.begin_append(blocked, 2).unwrap();
        scheduler
            .kv_cache
            .commit_append(blocked_prefill.id)
            .unwrap();
        let runnable_prefill = scheduler.kv_cache.begin_append(runnable, 1).unwrap();
        scheduler
            .kv_cache
            .commit_append(runnable_prefill.id)
            .unwrap();

        let batch = scheduler.schedule_decode().unwrap().unwrap();

        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.sequences[0].request_id, runnable);
    }
}
