//! Dataset-position recovery independent of GPU completion and loader prefetch.
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, error::Error, fmt};
use crate::record::{PrecisionSettings, Record};
use crate::tensor::backend::Backend;

/// Explicit handling of an epoch that does not divide the data rank count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleTail {
    /// Unequal final shard lengths; never repeat or pad examples.
    Uneven,
    /// Exclude the final incomplete group of data-rank samples for this epoch.
    Drop,
}

/// Invalid sampler configuration, incompatible continuation, or missing data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerError(pub String);

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl Error for SamplerError {}

fn invalid(message: impl Into<String>) -> SamplerError { SamplerError(message.into()) }
fn quota(length: usize, rank: usize, world: usize) -> usize {
    length / world + usize::from(rank < length % world)
}
fn check_indices(indices: &[usize], length: usize) -> Result<(), SamplerError> {
    let mut seen = HashSet::with_capacity(indices.len());
    if indices.iter().any(|&index| index >= length || !seen.insert(index)) {
        return Err(invalid("epoch indices repeat or lie outside the immutable dataset"));
    }
    Ok(())
}

/// Actual rank-local sample order and its committed cursor.
///
/// The order itself is recorded, so restoration does not depend on reconstructing
/// a previous rand version's permutation. Outstanding prefetch is NOT committed:
/// restore begins at `committed`, replaying issued-but-unconsumed samples.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplerState {
    /// Record format, currently one.
    pub version: u32,
    /// Caller-supplied identity of the immutable dataset/version.
    pub source_id: String,
    /// Original source length, not a rank-local length.
    pub source_length: usize,
    /// Data rank; tensor/pipeline ranks may use the same data coordinate.
    pub rank: usize,
    /// Data replicas assigning distinct examples.
    pub world_size: usize,
    /// Current caller-selected epoch number.
    pub epoch: u64,
    /// Private shuffle seed; None selects original source order.
    pub shuffle_seed: Option<u64>,
    /// Tail policy used when starting a fresh epoch.
    pub tail: SampleTail,
    /// Size of the originally selected epoch, after its initial tail policy.
    pub epoch_size: usize,
    /// Global consumption preceding the most recent topology change.
    pub carried_consumed: usize,
    /// Remaining global scope assigned at that topology change.
    pub scope_size: usize,
    /// Actual local order; positions are original dataset indices.
    pub indices: Vec<usize>,
    /// Samples actually consumed by this rank from this scope.
    pub committed: usize,
}

impl SamplerState {
    /// Validate rank geometry, ordering, bounds and committed accounting.
    pub fn validate(&self) -> Result<(), SamplerError> {
        if self.version != 1 || self.source_id.is_empty() || self.world_size == 0 || self.rank >= self.world_size {
            return Err(invalid("invalid sampler record identity or data topology"));
        }
        if self.epoch_size > self.source_length || self.carried_consumed.checked_add(self.scope_size) != Some(self.epoch_size)
            || self.indices.len() != quota(self.scope_size, self.rank, self.world_size) || self.committed > self.indices.len() {
            return Err(invalid("sampler record scope/order/consumption counts differ"));
        }
        check_indices(&self.indices, self.source_length)
    }
}

impl<B: Backend> Record<B> for SamplerState {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self { self }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self { item }
}

/// Topology-independent continuation containing ONLY unconsumed epoch examples.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsolidatedSamplerState {
    /// Record format, currently one.
    pub version: u32,
    /// Immutable dataset identity.
    pub source_id: String,
    /// Original dataset length.
    pub source_length: usize,
    /// Epoch whose remaining order is retained.
    pub epoch: u64,
    /// Shuffle configuration for subsequent epochs.
    pub shuffle_seed: Option<u64>,
    /// Tail policy for subsequent epochs, not reapplied to this continuation.
    pub tail: SampleTail,
    /// Original effective epoch size.
    pub epoch_size: usize,
    /// Global samples already committed across all previous data layouts.
    pub consumed: usize,
    /// Remaining source indices in their original relative global order.
    pub remaining: Vec<usize>,
}

impl ConsolidatedSamplerState {
    /// Validate an explicit continuation before loading it on any topology.
    pub fn validate(&self) -> Result<(), SamplerError> {
        if self.version != 1 || self.source_id.is_empty() || self.epoch_size > self.source_length
            || self.consumed.checked_add(self.remaining.len()) != Some(self.epoch_size) {
            return Err(invalid("invalid consolidated sampler identity or accounting"));
        }
        check_indices(&self.remaining, self.source_length)
    }

    /// Join every data rank, accepting identical duplicate TP/PP copies.
    /// Conflicting copies, missing ranks or overlapping source indices fail.
    pub fn consolidate(records: &[SamplerState]) -> Result<Self, SamplerError> {
        let first = records.first().ok_or_else(|| invalid("consolidation needs every data rank"))?;
        first.validate()?;
        let mut ranks = vec![None; first.world_size];
        for state in records {
            state.validate()?;
            if (state.source_id.as_str(), state.source_length, state.world_size, state.epoch, state.shuffle_seed,
                state.tail, state.epoch_size, state.carried_consumed, state.scope_size)
                != (first.source_id.as_str(), first.source_length, first.world_size, first.epoch, first.shuffle_seed,
                    first.tail, first.epoch_size, first.carried_consumed, first.scope_size) {
                return Err(invalid("sampler source, epoch, policy or scope differs between data ranks"));
            }
            match ranks[state.rank] {
                Some(old) if old != state => return Err(invalid("duplicate sampler coordinate has different committed state")),
                _ => ranks[state.rank] = Some(state),
            }
        }
        if ranks.iter().any(Option::is_none) { return Err(invalid("sampler consolidation is missing a data coordinate")); }
        let ranks: Vec<&SamplerState> = ranks.into_iter().map(Option::unwrap).collect();
        let mut seen = HashSet::with_capacity(first.scope_size);
        let mut remaining = Vec::new();
        for position in 0..first.scope_size {
            let state = ranks[position % first.world_size];
            let local_position = position / first.world_size;
            let index = state.indices[local_position];
            if !seen.insert(index) { return Err(invalid("data shards contain the same source sample")); }
            if local_position >= state.committed { remaining.push(index); }
        }
        let consumed = first.carried_consumed + ranks.iter().map(|state| state.committed).sum::<usize>();
        let result = Self { version: 1, source_id: first.source_id.clone(), source_length: first.source_length,
            epoch: first.epoch, shuffle_seed: first.shuffle_seed, tail: first.tail, epoch_size: first.epoch_size,
            consumed, remaining };
        result.validate()?;
        Ok(result)
    }
}

impl<B: Backend> Record<B> for ConsolidatedSamplerState {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self { self }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self { item }
}

/// A data-shard iterator with separate issued and actually-consumed positions.
///
/// Use `commit` after the caller's batch consumption boundary, not when workers
/// prefetch. State captures only that boundary. This sampler never pads examples,
/// automatically advances epochs, guesses source identity, or alters GPU state.
pub struct StatefulShardSampler {
    state: SamplerState,
    issued: usize,
}

impl StatefulShardSampler {
    /// Start an explicit epoch on a caller-selected data coordinate.
    pub fn new(source_id: String, source_length: usize, rank: usize, world_size: usize,
               epoch: u64, shuffle_seed: Option<u64>, tail: SampleTail) -> Result<Self, SamplerError> {
        if source_id.is_empty() || world_size == 0 || rank >= world_size {
            return Err(invalid("supply dataset identity and a valid data rank/world size"));
        }
        let mut order: Vec<usize> = (0..source_length).collect();
        if let Some(seed) = shuffle_seed { order.shuffle(&mut StdRng::seed_from_u64(seed.wrapping_add(epoch))); }
        let epoch_size = match tail { SampleTail::Uneven => source_length, SampleTail::Drop => source_length / world_size * world_size };
        order.truncate(epoch_size);
        let indices = order.into_iter().skip(rank).step_by(world_size).collect();
        let state = SamplerState { version: 1, source_id, source_length, rank, world_size, epoch, shuffle_seed,
            tail, epoch_size, carried_consumed: 0, scope_size: epoch_size, indices, committed: 0 };
        state.validate()?;
        Ok(Self { state, issued: 0 })
    }

    /// Restore the recorded actual order; outstanding prefetch is replayed.
    pub fn from_state(state: SamplerState) -> Result<Self, SamplerError> {
        state.validate()?;
        let issued = state.committed;
        Ok(Self { state, issued })
    }

    /// Replace a matching source/configuration/data-coordinate continuation.
    pub fn load_state(&mut self, state: SamplerState) -> Result<(), SamplerError> {
        state.validate()?;
        if (state.source_id.as_str(), state.source_length, state.rank, state.world_size, state.shuffle_seed, state.tail)
            != (self.state.source_id.as_str(), self.state.source_length, self.state.rank, self.state.world_size,
                self.state.shuffle_seed, self.state.tail) {
            return Err(invalid("sampler source, configuration or topology changed; explicitly consolidate and reshard"));
        }
        self.issued = state.committed;
        self.state = state;
        Ok(())
    }

    /// Assign unconsumed samples on a new data topology, preserving their order.
    /// The original epoch tail is retained; no additional samples are dropped.
    pub fn reshard(state: &ConsolidatedSamplerState, rank: usize, world_size: usize) -> Result<Self, SamplerError> {
        state.validate()?;
        if world_size == 0 || rank >= world_size { return Err(invalid("invalid target data coordinate")); }
        Self::from_state(SamplerState { version: 1, source_id: state.source_id.clone(), source_length: state.source_length,
            rank, world_size, epoch: state.epoch, shuffle_seed: state.shuffle_seed, tail: state.tail,
            epoch_size: state.epoch_size, carried_consumed: state.consumed, scope_size: state.remaining.len(),
            indices: state.remaining.iter().copied().skip(rank).step_by(world_size).collect(), committed: 0 })
    }

    /// Capture a lossless sampler record for TrainingRecord's application state.
    pub fn state(&self) -> SamplerState { self.state.clone() }

    /// Number of samples actually consumed in this local scope.
    pub fn committed(&self) -> usize { self.state.committed }

    /// Number of samples issued, including uncommitted prefetch.
    pub fn issued(&self) -> usize { self.issued }

    /// Actual rank-local work remaining to be consumed, including prefetch.
    pub fn remaining(&self) -> usize { self.state.indices.len() - self.state.committed }

    /// Commit only a prefix of samples already issued by this sampler.
    pub fn commit(&mut self, count: usize) -> Result<(), SamplerError> {
        let committed = self.state.committed.checked_add(count).ok_or_else(|| invalid("committed sample count overflow"))?;
        if committed > self.issued { return Err(invalid("cannot commit samples that have not been issued")); }
        self.state.committed = committed;
        Ok(())
    }

    /// Explicitly discard outstanding prefetch and replay it from the commit.
    pub fn rewind_uncommitted(&mut self) { self.issued = self.state.committed; }

    /// Explicitly start a fresh epoch using the current data coordinate/policy.
    pub fn start_epoch(&mut self, epoch: u64) -> Result<(), SamplerError> {
        *self = Self::new(self.state.source_id.clone(), self.state.source_length, self.state.rank,
            self.state.world_size, epoch, self.state.shuffle_seed, self.state.tail)?;
        Ok(())
    }

    /// Issue at most the requested number of actual indices; no padded batch.
    pub fn next_indices(&mut self, maximum: usize) -> Result<Option<Vec<usize>>, SamplerError> {
        if maximum == 0 { return Err(invalid("sample batch size must be positive")); }
        let end = self.issued.saturating_add(maximum).min(self.state.indices.len());
        if end == self.issued { return Ok(None); }
        let indices = self.state.indices[self.issued..end].to_vec();
        self.issued = end;
        Ok(Some(indices))
    }

    /// Fetch actual dataset items before advancing the issued position.
    /// Missing records or a changed source length leave the cursor unchanged.
    #[cfg(feature = "dataset")]
    pub fn load_next<I, D: super::dataset::Dataset<I>>(&mut self, dataset: &D, maximum: usize)
        -> Result<Option<Vec<I>>, SamplerError> {
        if maximum == 0 { return Err(invalid("sample batch size must be positive")); }
        if dataset.len() != self.state.source_length { return Err(invalid("dataset length changed since sampler construction")); }
        let end = self.issued.saturating_add(maximum).min(self.state.indices.len());
        if end == self.issued { return Ok(None); }
        let items = self.state.indices[self.issued..end].iter().map(|&index|
            dataset.get(index).ok_or_else(|| invalid(format!("immutable dataset is missing index {index}"))))
            .collect::<Result<Vec<_>, _>>()?;
        self.issued = end;
        Ok(Some(items))
    }

    /// Fetch and batch on the explicit backend device; caller still owns commit.
    #[cfg(feature = "dataset")]
    pub fn load_batch<B: Backend, I, O, D, F>(&mut self, dataset: &D, batcher: &F,
        device: &B::Device, maximum: usize) -> Result<Option<O>, SamplerError>
    where D: super::dataset::Dataset<I>, F: super::dataloader::batcher::Batcher<B, I, O> {
        Ok(self.load_next(dataset, maximum)?.map(|items| batcher.batch(items, device)))
    }
}

impl Iterator for StatefulShardSampler {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let item = self.state.indices.get(self.issued).copied()?;
        self.issued += 1;
        Some(item)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.state.indices.len() - self.issued;
        (count, Some(count))
    }
}
impl ExactSizeIterator for StatefulShardSampler {}
