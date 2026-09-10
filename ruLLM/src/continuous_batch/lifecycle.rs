use super::*;

/// Trade throughput-oriented overcommit for a conservative progress guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvAdmissionPolicy {
    /// Allocate only when needed, preserving the original scheduling policy.
    /// Overcommitted requests can exhaust the pool and require cancellation.
    #[default]
    OnDemand,
    /// Admit only when every active request can fit its maximum cached length.
    /// This reserves a metadata budget, not physical pages in advance.
    ReserveSequenceCapacity,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContinuousBatchOptions {
    /// Maximum queued, not active or completed, requests. None is unbounded.
    pub max_pending_requests: Option<usize>,
    pub kv_admission: KvAdmissionPolicy,
}

impl ContinuousBatchOptions {
    pub(super) fn validate(self) -> Result<(), ContinuousBatchError> {
        if self.max_pending_requests == Some(0) {
            return Err(ContinuousBatchError("max_pending_requests must be positive when set".into()));
        }
        Ok(())
    }
}

/// Returned directly by `cancel`; never mixed into the normal completion queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelledGeneration {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<i32>,
    pub generated_token_ids: Vec<i32>,
}

impl ContinuousBatchScheduler {
    /// Cancel a queued or ready request and reclaim its pages. Unknown and
    /// already-finished IDs return None. In-flight requests return an error:
    /// first wait for device work to stop, then complete/fail the owning batch.
    /// Reusing pages while a kernel is still writing them would be unsafe.
    pub fn cancel(
        &mut self,
        request_id: RequestId,
    ) -> Result<Option<CancelledGeneration>, ContinuousBatchError> {
        if let Some(position) = self.pending.iter().position(|request| request.id == request_id) {
            let request = self.pending.remove(position).expect("pending position was found");
            return Ok(Some(CancelledGeneration {
                request_id,
                prompt_token_ids: request.prompt_token_ids,
                generated_token_ids: Vec::new(),
            }));
        }
        let Some(request) = self.active.get(&request_id) else { return Ok(None); };
        if let RequestPhase::InFlight(batch_id) = request.phase {
            return Err(ContinuousBatchError(format!(
                "request {} belongs to in-flight batch {batch_id}; finish device work before cancelling",
                request_id.0
            )));
        }
        // Fallible cleanup happens before removing the request metadata.
        self.kv_cache.remove_sequence(request_id)?;
        let request = self.active.remove(&request_id).expect("active request was found");
        Ok(Some(CancelledGeneration {
            request_id,
            prompt_token_ids: request.prompt_token_ids,
            generated_token_ids: request.generated_token_ids,
        }))
    }

    /// True when no queued, active or in-flight work remains. Completed results
    /// can still be waiting in `pop_finished`.
    pub fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.active.is_empty() && self.in_flight.is_none()
    }

    pub(super) fn request_page_budget(
        prompt_length: usize,
        generation: &GreedyGenerationConfig,
        block_size: usize,
    ) -> usize {
        // The final selected token is returned, never fed through the model.
        // submit validates prompt_length + max_new_tokens before this helper.
        (prompt_length + generation.max_new_tokens.saturating_sub(1)).div_ceil(block_size)
    }
}

#[cfg(test)]
mod tests;

impl ContinuousBatchScheduler {
    /// Lowest useful token limit without implementing chunked prefill. This
    /// avoids an OOM retry policy reducing the limit below a waiting prompt and
    /// silently stranding it forever. Decode-only work has a minimum of one.
    pub fn minimum_batch_token_limit(&self) -> usize {
        self.pending.iter().map(|r| r.prompt_token_ids.len())
            .chain(self.active.values().filter(|r| r.phase == RequestPhase::NeedsPrefill)
                .map(|r| r.prompt_token_ids.len()))
            .max().unwrap_or(1).max(1)
    }

    pub fn batch_token_limit(&self) -> usize { self.config.max_batch_tokens }

    /// Reconfigure between batches after an OOM or workload change. No global
    /// mutable setting and no change to the original public config fields.
    pub fn set_batch_token_limit(&mut self, limit: usize) -> Result<(), ContinuousBatchError> {
        if self.in_flight.is_some() {
            return Err(ContinuousBatchError("cannot change token limit while a batch is in flight".into()));
        }
        let minimum = self.minimum_batch_token_limit();
        if limit < minimum {
            return Err(ContinuousBatchError(format!("token limit {limit} would strand a prompt requiring {minimum} tokens")));
        }
        self.config.max_batch_tokens = limit;
        Ok(())
    }
}
