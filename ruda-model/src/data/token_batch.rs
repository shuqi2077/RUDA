//! Variable-sized batches with exact token cost and recoverable sample cursors.
use super::sampler::{ConsolidatedSamplerState, SamplerError, SamplerState, StatefulShardSampler};
use crate::{record::{PrecisionSettings, Record}, tensor::backend::Backend};
use serde::{Deserialize, Serialize};

fn invalid(message: impl Into<String>) -> SamplerError { SamplerError(message.into()) }

/// The actual token-axis layout used by the caller's batcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenBatchLayout {
    /// Flat document-isolated tokens; cost is the sum of sequence lengths.
    Packed,
    /// Rectangular rows; cost is example count times maximum sequence length.
    Padded,
}

/// Explicit batch limits; these do not truncate, repeat, reorder or pad samples.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBatchOptions {
    /// Maximum token slots in one batch under the selected layout.
    pub maximum_tokens: usize,
    /// Optional independent maximum number of actual examples in one batch.
    pub maximum_examples: Option<usize>,
    /// Packed or rectangular cost accounting; the batcher must honor this layout.
    pub layout: TokenBatchLayout,
}

impl TokenBatchOptions {
    /// Reject zero limits and empty/oversized sequence metadata before issuance.
    pub fn validate_lengths(&self, lengths: &[usize]) -> Result<(), SamplerError> {
        if self.maximum_tokens == 0 || self.maximum_examples == Some(0) {
            return Err(invalid("token batch limits must be positive"));
        }
        if let Some(index) = lengths.iter().position(|&length| length == 0 || length > self.maximum_tokens) {
            return Err(invalid(format!("sequence {index} is empty or exceeds the token budget; preprocess explicitly")));
        }
        Ok(())
    }
}

/// One actual batch's original source indices and exact work accounting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenBatchPlan {
    /// Source indices, in the distributed sampler's recorded relative order.
    pub indices: Vec<usize>,
    /// Real tokens in these examples, before any caller-side rectangular padding.
    pub real_tokens: usize,
    /// Maximum actual example length.
    pub maximum_sequence_length: usize,
    /// Layout-dependent token slots: sum for packed, rows times maximum for padded.
    pub token_slots: usize,
}

/// Actual examples paired with the plan used to load them.
pub struct TokenBatch<I> {
    /// The exact source indices and token cost, not inferred from padded tensors.
    pub plan: TokenBatchPlan,
    /// Loaded source examples; no filler rows.
    pub items: Vec<I>,
}

/// Complete batch continuation, including the actual immutable length table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBatchState {
    /// Record format, currently one.
    pub version: u32,
    /// Committed sample order and explicit immutable source identity.
    pub sampler: SamplerState,
    /// One caller-supplied encoded length per original source example.
    pub lengths: Vec<usize>,
    /// Exact token accounting/limits used to partition that order.
    pub options: TokenBatchOptions,
}

impl TokenBatchState {
    /// Check source geometry and all length/limit metadata before restoration.
    pub fn validate(&self) -> Result<(), SamplerError> {
        self.sampler.validate()?;
        if self.version != 1 || self.lengths.len() != self.sampler.source_length {
            return Err(invalid("token batch checkpoint length table/source geometry differs"));
        }
        self.options.validate_lengths(&self.lengths)
    }
}

impl<B: Backend> Record<B> for TokenBatchState {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self { self }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self { item }
}

/// Topology-independent remaining epoch, retaining the exact length table/policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsolidatedTokenBatchState {
    /// Record format, currently one.
    pub version: u32,
    /// Only actual unconsumed original source examples.
    pub sampler: ConsolidatedSamplerState,
    /// Immutable encoded lengths indexed by original source position.
    pub lengths: Vec<usize>,
    /// Batch partitioning policy; no additional epoch tail is dropped.
    pub options: TokenBatchOptions,
}

impl ConsolidatedTokenBatchState {
    /// Validate a continuation without changing any data-rank cursor.
    pub fn validate(&self) -> Result<(), SamplerError> {
        self.sampler.validate()?;
        if self.version != 1 || self.lengths.len() != self.sampler.source_length {
            return Err(invalid("consolidated token batch metadata differs from its source"));
        }
        self.options.validate_lengths(&self.lengths)
    }

    /// Consolidate every data coordinate, accepting identical TP/PP copies.
    pub fn consolidate(states: &[TokenBatchState]) -> Result<Self, SamplerError> {
        let first = states.first().ok_or_else(|| invalid("supply every data-rank token batch state"))?;
        for state in states {
            state.validate()?;
            if state.lengths != first.lengths || state.options != first.options {
                return Err(invalid("data ranks use different encoded lengths or token batch policies"));
            }
        }
        let sampler_states: Vec<_> = states.iter().map(|state| state.sampler.clone()).collect();
        let result = Self {
            version: 1, sampler: ConsolidatedSamplerState::consolidate(&sampler_states)?,
            lengths: first.lengths.clone(), options: first.options.clone(),
        };
        result.validate()?;
        Ok(result)
    }
}

impl<B: Backend> Record<B> for ConsolidatedTokenBatchState {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self { self }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self { item }
}

/// Greedy contiguous token-budget batches over the actual distributed order.
///
/// This changes batch boundaries only. Encoded lengths must describe the actual
/// examples returned by the immutable source. No tokenization, truncation,
/// length sorting, duplicate tail rows or cross-document supervision is inferred.
/// Different DP coordinates can have different batch counts; the caller must
/// use its training coordinator's actual participation/accumulation contract.
pub struct StatefulTokenBatchSampler {
    sampler: StatefulShardSampler,
    lengths: Vec<usize>,
    options: TokenBatchOptions,
}

impl StatefulTokenBatchSampler {
    /// Wrap a committed distributed order with exact original-source lengths.
    pub fn new(sampler: StatefulShardSampler, lengths: Vec<usize>, options: TokenBatchOptions) -> Result<Self, SamplerError> {
        if lengths.len() != sampler.source_length() {
            return Err(invalid("supply one encoded length per original source example"));
        }
        options.validate_lengths(&lengths)?;
        Ok(Self { sampler, lengths, options })
    }

    /// Restore committed samples; all prefetched-but-unconsumed rows are replayed.
    pub fn from_state(state: TokenBatchState) -> Result<Self, SamplerError> {
        state.validate()?;
        Self::new(StatefulShardSampler::from_state(state.sampler)?, state.lengths, state.options)
    }

    /// Capture the committed continuation without saving an issued prefetch cursor.
    pub fn state(&self) -> TokenBatchState {
        TokenBatchState { version: 1, sampler: self.sampler.state(), lengths: self.lengths.clone(), options: self.options.clone() }
    }

    /// Replace a matching continuation; reject silently changed lengths/limits.
    pub fn load_state(&mut self, state: TokenBatchState) -> Result<(), SamplerError> {
        state.validate()?;
        if state.lengths != self.lengths || state.options != self.options {
            return Err(invalid("token batch restoration requires unchanged encoded lengths and limits"));
        }
        self.sampler.load_state(state.sampler)
    }

    /// Assign unconsumed examples to a new data topology and repartition batches.
    pub fn reshard(state: &ConsolidatedTokenBatchState, rank: usize, world_size: usize) -> Result<Self, SamplerError> {
        state.validate()?;
        Self::new(StatefulShardSampler::reshard(&state.sampler, rank, world_size)?, state.lengths.clone(), state.options.clone())
    }

    /// Underlying actual committed/issued order, without mutable aliasing.
    pub fn sampler(&self) -> &StatefulShardSampler { &self.sampler }

    /// Commit actual consumed examples, never token slots or prefetched batches.
    pub fn commit(&mut self, examples: usize) -> Result<(), SamplerError> { self.sampler.commit(examples) }

    /// Replay all outstanding prefetch when constructing a fresh loader iterator.
    pub fn rewind_uncommitted(&mut self) { self.sampler.rewind_uncommitted(); }

    /// Start the next explicitly selected epoch on the current data topology.
    pub fn start_epoch(&mut self, epoch: u64) -> Result<(), SamplerError> { self.sampler.start_epoch(epoch) }

    /// Inspect the next exact batch without advancing either sample cursor.
    pub fn peek(&self) -> Result<Option<TokenBatchPlan>, SamplerError> {
        let indices = self.sampler.unissued_indices();
        if indices.is_empty() { return Ok(None); }
        let mut real_tokens = 0usize;
        let mut maximum_sequence_length = 0usize;
        let mut count = 0usize;
        let mut token_slots = 0usize;
        for &index in indices {
            if self.options.maximum_examples.is_some_and(|limit| count == limit) { break; }
            let length = self.lengths[index];
            let maximum = maximum_sequence_length.max(length);
            let total = real_tokens.checked_add(length);
            let slots = match self.options.layout {
                TokenBatchLayout::Packed => total,
                TokenBatchLayout::Padded => (count + 1).checked_mul(maximum),
            };
            let Some(slots) = slots.filter(|&slots| slots <= self.options.maximum_tokens) else { break; };
            real_tokens = total.ok_or_else(|| invalid("real token count overflow"))?;
            maximum_sequence_length = maximum;
            token_slots = slots;
            count += 1;
        }
        if count == 0 { return Err(invalid("next actual sample cannot fit the explicit token budget")); }
        Ok(Some(TokenBatchPlan { indices: indices[..count].to_vec(), real_tokens, maximum_sequence_length, token_slots }))
    }

    /// Issue the next batch of real indices; no lookahead sample is prematurely issued.
    pub fn next_plan(&mut self) -> Result<Option<TokenBatchPlan>, SamplerError> {
        let Some(plan) = self.peek()? else { return Ok(None); };
        self.sampler.next_indices(plan.indices.len())?;
        Ok(Some(plan))
    }

    /// Load every actual example before advancing the issued cursor.
    #[cfg(feature = "dataset")]
    pub fn load_next<I, D: super::dataset::Dataset<I>>(&mut self, dataset: &D) -> Result<Option<TokenBatch<I>>, SamplerError> {
        let Some(plan) = self.peek()? else { return Ok(None); };
        let items = self.sampler.load_next(dataset, plan.indices.len())?
            .ok_or_else(|| invalid("source order changed while loading a token batch"))?;
        Ok(Some(TokenBatch { plan, items }))
    }

    /// Batch the loaded actual records on the explicit backend device.
    #[cfg(feature = "dataset")]
    pub fn load_batch<B: Backend, I, O, D, F>(&mut self, dataset: &D, batcher: &F, device: &B::Device)
        -> Result<Option<(TokenBatchPlan, O)>, SamplerError>
    where D: super::dataset::Dataset<I>, F: super::dataloader::batcher::Batcher<B, I, O> {
        Ok(self.load_next(dataset)?.map(|batch| (batch.plan, batcher.batch(batch.items, device))))
    }
}
