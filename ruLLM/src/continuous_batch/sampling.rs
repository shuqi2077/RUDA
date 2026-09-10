use super::{ContinuousBatchError, ContinuousBatchScheduler, RequestPhase};
use crate::{GreedyGenerationConfig, RequestId, SamplingGenerationConfig, TokenSampler};

/// Selection state is committed only together with the corresponding batch.
pub(crate) struct BatchTokenSelection {
    batch_id: u64,
    tokens: Vec<i32>,
    samplers: Vec<(RequestId, TokenSampler)>,
}

impl ContinuousBatchScheduler {
    /// Submit a sampled request with its own RNG, independent of batch membership.
    pub fn submit_sampled(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: SamplingGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        let sampler = TokenSampler::new(generation.sampling)
            .map_err(|error| ContinuousBatchError(error.to_string()))?;
        self.submit_with_sampler(
            prompt_token_ids,
            GreedyGenerationConfig {
                max_new_tokens: generation.max_new_tokens,
                eos_token_ids: generation.eos_token_ids,
            },
            Some(sampler),
        )
    }

    pub(crate) fn select_batch_tokens(
        &self,
        batch_id: u64,
        logits: &[f32],
        vocabulary: usize,
    ) -> Result<BatchTokenSelection, ContinuousBatchError> {
        let batch = self
            .in_flight
            .as_ref()
            .filter(|batch| batch.id == batch_id)
            .ok_or_else(|| ContinuousBatchError(format!("batch {batch_id} is not in flight")))?;
        if vocabulary == 0 || batch.sequences.len().checked_mul(vocabulary) != Some(logits.len()) {
            return Err(ContinuousBatchError(
                "logit rows do not match the scheduled batch".into(),
            ));
        }
        let mut tokens = Vec::with_capacity(batch.sequences.len());
        let mut samplers = Vec::new();
        for (row, sequence) in batch.sequences.iter().enumerate() {
            let request = self.active.get(&sequence.request_id).ok_or_else(|| {
                ContinuousBatchError(format!(
                    "batch references missing request {}",
                    sequence.request_id.0
                ))
            })?;
            if request.phase != RequestPhase::InFlight(batch_id) {
                return Err(ContinuousBatchError(format!(
                    "request {} is not owned by batch {batch_id}",
                    sequence.request_id.0
                )));
            }
            let values = &logits[row * vocabulary..(row + 1) * vocabulary];
            if let Some(sampler) = &request.sampler {
                let mut next = sampler.clone();
                tokens.push(next.sample(values).map_err(|error| {
                    ContinuousBatchError(format!("sampling batch row {row}: {error}"))
                })?);
                // Batch previews retain RNG checkpoints, not one O(vocabulary)
                // buffer per request. Streaming single requests still reuse
                // their TokenSampler workspace across every decode step.
                next.clear_workspace();
                samplers.push((sequence.request_id, next));
            } else {
                tokens.push(greedy_token(values, row)?);
            }
        }
        Ok(BatchTokenSelection {
            batch_id,
            tokens,
            samplers,
        })
    }

    pub(crate) fn complete_selected_batch(
        &mut self,
        selection: BatchTokenSelection,
    ) -> Result<Vec<i32>, ContinuousBatchError> {
        self.complete_batch(selection.batch_id, &selection.tokens)?;
        for (request_id, sampler) in selection.samplers {
            if let Some(request) = self.active.get_mut(&request_id) {
                request.sampler = Some(sampler);
            }
        }
        Ok(selection.tokens)
    }
}

fn greedy_token(row_values: &[f32], row: usize) -> Result<i32, ContinuousBatchError> {
    let mut best = None;
    for (index, &value) in row_values.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((index, value));
        }
    }
    let (index, _) = best.ok_or_else(|| {
        ContinuousBatchError(format!("all vocabulary logits are NaN for batch row {row}"))
    })?;
    i32::try_from(index).map_err(|_| ContinuousBatchError("vocabulary index exceeds i32".into()))
}

#[cfg(test)]
mod tests;
