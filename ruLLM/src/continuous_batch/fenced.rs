//! Opt-in synchronization gate around the existing metadata scheduler.
use super::*;
use crate::runtime::{RuntimeError, RuntimeErrorKind, WorkState, checked_argmax};
use std::task::Poll;

/// Adapter-owned completion fence. Ready(Ok) MUST mean every device stream and
/// host readback touching this batch is complete. A timeout returns Pending,
/// not success. Errors do not establish that device writes have stopped.
pub trait BatchFence {
    fn poll_complete(&mut self) -> Poll<Result<(), RuntimeError>>;
}

#[derive(Debug, Clone)]
pub struct BatchLaunchFailure {
    pub error: RuntimeError,
    pub work: WorkState,
    pub committed_cache_intact: bool,
}

/// A fence-gated scheduler, not a CUDA/HIP event implementation. Physical KV
/// storage remains owned by the adapter. Once quarantined, do not recycle that
/// storage: tear down its isolated worker process or prove quiescence externally.
/// No mutable access to the wrapped scheduler is exposed to bypass the fence.
pub struct FencedBatchScheduler<F> {
    scheduler: ContinuousBatchScheduler,
    fence: Option<(u64, F, bool)>,
    quarantined: bool,
}
impl<F: BatchFence> FencedBatchScheduler<F> {
    pub fn new(scheduler: ContinuousBatchScheduler) -> Result<Self, RuntimeError> {
        if scheduler.snapshot().has_in_flight_batch {
            return Err(RuntimeError::invalid("cannot wrap a scheduler with untracked in-flight work"));
        }
        Ok(Self { scheduler, fence: None, quarantined: false })
    }
    fn ensure_healthy(&self) -> Result<(), RuntimeError> {
        if self.quarantined {
            Err(RuntimeError::new(RuntimeErrorKind::DeviceLost, "scheduler is quarantined; device work may still be in flight"))
        } else { Ok(()) }
    }
    fn map_error(error: ContinuousBatchError) -> RuntimeError {
        RuntimeError::new(RuntimeErrorKind::InvalidInput, error.to_string())
    }
    pub fn submit(&mut self, prompt: Vec<i32>, generation: GreedyGenerationConfig) -> Result<RequestId, RuntimeError> {
        self.ensure_healthy()?;
        self.scheduler.submit(prompt, generation).map_err(Self::map_error)
    }
    pub fn submit_sampled(&mut self, prompt: Vec<i32>, generation: crate::SamplingGenerationConfig) -> Result<RequestId, RuntimeError> {
        self.ensure_healthy()?;
        self.scheduler.submit_sampled(prompt, generation).map_err(Self::map_error)
    }
    /// A launch panic leaves the gate quarantined, even if caught by its caller.
    /// A recoverable error is rollback-safe only before submission, or after
    /// verified completion AND with all previously committed KV entries intact.
    pub fn launch(
        &mut self,
        submit: impl FnOnce(&ScheduledBatch) -> Result<F, BatchLaunchFailure>,
    ) -> Result<Option<ScheduledBatch>, RuntimeError> {
        self.ensure_healthy()?;
        let Some(batch) = self.scheduler.schedule().map_err(Self::map_error)? else { return Ok(None); };
        self.quarantined = true;
        match submit(&batch) {
            Ok(fence) => {
                self.fence = Some((batch.id, fence, false));
                self.quarantined = false;
                Ok(Some(batch))
            }
            Err(failure) => {
                let fatal = matches!(failure.error.kind(), RuntimeErrorKind::DeviceLost
                    | RuntimeErrorKind::Synchronization | RuntimeErrorKind::WorkerPanicked
                    | RuntimeErrorKind::Internal);
                if failure.work != WorkState::Unknown && failure.committed_cache_intact && !fatal {
                    self.scheduler.fail_batch(batch.id).map_err(Self::map_error)?;
                    self.quarantined = false;
                }
                Err(failure.error)
            }
        }
    }
    fn poll_fence(&mut self, batch_id: u64) -> Result<Poll<()>, RuntimeError> {
        self.ensure_healthy()?;
        let (id, fence, completed) = self.fence.as_mut().ok_or_else(|| RuntimeError::invalid("no tracked device batch"))?;
        if *id != batch_id { return Err(RuntimeError::invalid("stale or wrong batch fence ID")); }
        if *completed { return Ok(Poll::Ready(())); }
        // A panic while polling is not evidence of completion either.
        self.quarantined = true;
        match fence.poll_complete() {
            Poll::Pending => { self.quarantined = false; Ok(Poll::Pending) }
            Poll::Ready(Ok(())) => { *completed = true; self.quarantined = false; Ok(Poll::Ready(())) }
            Poll::Ready(Err(error)) => Err(error),
        }
    }
    /// Validate host logits, preview request-local RNGs, then atomically commit.
    /// Neither a pending event nor invalid logits advances RNG/cache positions.
    pub fn try_complete_logits(&mut self, batch_id: u64, logits: &[f32], vocabulary: usize) -> Result<Poll<Vec<i32>>, RuntimeError> {
        if self.poll_fence(batch_id)?.is_pending() { return Ok(Poll::Pending); }
        let rows = self.scheduler.in_flight.as_ref().expect("fence tracks an in-flight batch").batch_size();
        if vocabulary == 0 || rows.checked_mul(vocabulary) != Some(logits.len()) {
            return Err(RuntimeError::invalid("logit shape does not match the fenced batch"));
        }
        for row in logits.chunks_exact(vocabulary) { checked_argmax(row)?; }
        let selected = self.scheduler.select_batch_tokens(batch_id, logits, vocabulary).map_err(Self::map_error)?;
        let tokens = self.scheduler.complete_selected_batch(selected).map_err(Self::map_error)?;
        self.fence.take();
        Ok(Poll::Ready(tokens))
    }
    /// Roll back only an append-only failed tail after its completion fence.
    /// The adapter must not have modified any earlier committed KV entries.
    pub fn try_rollback(&mut self, batch_id: u64) -> Result<Poll<()>, RuntimeError> {
        if self.poll_fence(batch_id)?.is_pending() { return Ok(Poll::Pending); }
        self.scheduler.fail_batch(batch_id).map_err(Self::map_error)?;
        self.fence.take();
        Ok(Poll::Ready(()))
    }
    pub fn cancel(&mut self, id: RequestId) -> Result<Option<CancelledGeneration>, RuntimeError> {
        self.ensure_healthy()?;
        self.scheduler.cancel(id).map_err(Self::map_error)
    }
    pub fn set_batch_token_limit(&mut self, limit: usize) -> Result<(), RuntimeError> {
        self.ensure_healthy()?;
        self.scheduler.set_batch_token_limit(limit).map_err(Self::map_error)
    }
    pub fn is_quarantined(&self) -> bool { self.quarantined }
    pub fn snapshot(&self) -> ContinuousBatchSnapshot { self.scheduler.snapshot() }
    pub fn pop_finished(&mut self) -> Option<FinishedGeneration> { self.scheduler.pop_finished() }
}

impl<F> Drop for FencedBatchScheduler<F> {
    fn drop(&mut self) {
        // Physical allocations used by a batch must be held by its fence (or
        // another owner with at least this lifetime). No Drop-time completion
        // assumption: unresolved fences intentionally survive until teardown.
        if let Some((_, fence, completed)) = self.fence.take() {
            if !completed { std::mem::forget(fence); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
    struct Fence { ready: Arc<AtomicBool>, failure: bool, polls: Arc<AtomicUsize> }
    impl BatchFence for Fence {
        fn poll_complete(&mut self) -> Poll<Result<(), RuntimeError>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            if self.failure { Poll::Ready(Err(RuntimeError::new(RuntimeErrorKind::Synchronization,"injected event failure"))) }
            else if self.ready.load(Ordering::Acquire) { Poll::Ready(Ok(())) }
            else { Poll::Pending }
        }
    }
    fn scheduler() -> FencedBatchScheduler<Fence> {
        FencedBatchScheduler::new(ContinuousBatchScheduler::new(
            ContinuousBatchConfig { max_active_sequences: 2, max_batch_tokens: 8 },
            PagedKvCacheConfig { block_size: 2, num_pages: 8, max_sequence_length: 16 },
        ).unwrap()).unwrap()
    }
    fn generation() -> GreedyGenerationConfig { GreedyGenerationConfig { max_new_tokens: 1, eos_token_ids: vec![] } }
    fn fence(ready: &Arc<AtomicBool>) -> Fence { Fence { ready: ready.clone(), failure: false, polls: Arc::new(AtomicUsize::new(0)) } }
    #[test]
    fn pending_fence_never_commits_or_reclaims_pages() {
        let mut gate = scheduler(); let request = gate.submit(vec![1],generation()).unwrap();
        let ready = Arc::new(AtomicBool::new(false));
        let batch = gate.launch(|_| Ok(fence(&ready))).unwrap().unwrap();
        let before = gate.snapshot();
        assert!(gate.try_complete_logits(batch.id,&[1.0,2.0],2).unwrap().is_pending());
        assert!(gate.try_rollback(batch.id).unwrap().is_pending());
        assert!(gate.cancel(request).is_err());
        assert_eq!(gate.snapshot(),before);
        ready.store(true,Ordering::Release);
        assert_eq!(gate.try_complete_logits(batch.id,&[1.0,2.0],2).unwrap(),Poll::Ready(vec![1]));
        assert_eq!(gate.snapshot().free_kv_pages,8);
        assert_eq!(gate.pop_finished().unwrap().generated_token_ids,vec![1]);
    }
    #[test]
    fn stale_completion_does_not_poll_the_current_event() {
        let mut gate=scheduler();gate.submit(vec![1],generation()).unwrap();
        let ready=Arc::new(AtomicBool::new(true));let f=fence(&ready);let polls=f.polls.clone();
        let batch=gate.launch(|_|Ok(f)).unwrap().unwrap();
        assert!(gate.try_complete_logits(batch.id+1,&[1.0],1).is_err());
        assert_eq!(polls.load(Ordering::SeqCst),0);
        assert!(gate.try_complete_logits(batch.id,&[1.0],1).unwrap().is_ready());
        assert!(gate.try_complete_logits(batch.id,&[1.0],1).is_err());
    }
    #[test]
    fn event_error_quarantines_and_retains_reservations() {
        let mut gate=scheduler();let id=gate.submit(vec![1],generation()).unwrap();
        let ready=Arc::new(AtomicBool::new(true));let mut f=fence(&ready);f.failure=true;
        let batch=gate.launch(|_|Ok(f)).unwrap().unwrap();let before=gate.snapshot();
        assert!(gate.try_complete_logits(batch.id,&[1.0],1).is_err());
        assert!(gate.is_quarantined());assert_eq!(gate.snapshot(),before);
        assert!(gate.cancel(id).is_err());assert!(gate.launch(|_|Ok(fence(&ready))).is_err());
    }
    #[test]
    fn invalid_logits_do_not_advance_a_ready_batch() {
        let mut gate=scheduler();gate.submit(vec![1],generation()).unwrap();
        let ready=Arc::new(AtomicBool::new(true));
        let f=fence(&ready);let polls=f.polls.clone();
        let batch=gate.launch(|_|Ok(f)).unwrap().unwrap();let before=gate.snapshot();
        for row in [vec![f32::NAN,1.0],vec![f32::INFINITY,1.0],vec![f32::NEG_INFINITY;2]] {
            assert_eq!(gate.try_complete_logits(batch.id,&row,2).unwrap_err().kind(),RuntimeErrorKind::Numerical);
            assert_eq!(gate.snapshot(),before);
        }
        assert!(gate.try_complete_logits(batch.id,&[1.0,2.0],2).unwrap().is_ready());
        assert_eq!(polls.load(Ordering::SeqCst),1,"completed events are not polled twice");
    }
    #[test]
    fn proven_prelaunch_oom_rolls_back_without_quarantine() {
        let mut gate=scheduler();gate.submit(vec![1],generation()).unwrap();
        assert!(gate.launch(|_|Err(BatchLaunchFailure {
            error:RuntimeError::new(RuntimeErrorKind::OutOfMemory,"prelaunch allocation failed"),
            work:WorkState::NotSubmitted,committed_cache_intact:true,
        })).is_err());
        assert!(!gate.is_quarantined());assert_eq!(gate.snapshot().free_kv_pages,8);
        let ready=Arc::new(AtomicBool::new(true));
        let batch=gate.launch(|_|Ok(fence(&ready))).unwrap().unwrap();
        assert!(gate.try_complete_logits(batch.id,&[1.0],1).unwrap().is_ready());
    }
    #[test]
    fn unknown_launch_failure_never_reclaims_pages() {
        let mut gate=scheduler();gate.submit(vec![1],generation()).unwrap();
        assert!(gate.launch(|_|Err(BatchLaunchFailure {
            error:RuntimeError::new(RuntimeErrorKind::OutOfMemory,"partial submission"),
            work:WorkState::Unknown,committed_cache_intact:true,
        })).is_err());
        assert!(gate.is_quarantined());assert_eq!(gate.snapshot().free_kv_pages,7);
    }
    #[test]
    fn token_limit_cannot_strand_pending_prefill() {
        let mut gate=scheduler();gate.submit(vec![1,2,3,4,5],generation()).unwrap();
        assert!(gate.set_batch_token_limit(4).is_err());assert!(gate.set_batch_token_limit(5).is_ok());
        let ready=Arc::new(AtomicBool::new(true));
        let batch=gate.launch(|_|Ok(fence(&ready))).unwrap().unwrap();
        assert!(gate.set_batch_token_limit(8).is_err());
        assert!(gate.try_rollback(batch.id).unwrap().is_ready());
        assert!(gate.set_batch_token_limit(8).is_ok());
    }
    #[test]
    fn rollback_and_retry_do_not_consume_sampled_rng_state() {
        let mut gate=scheduler();
        let options=crate::SamplingGenerationConfig { max_new_tokens:1,eos_token_ids:vec![],sampling:crate::SamplingConfig {
            seed:Some(17),..Default::default()
        }};
        let expected=crate::TokenSampler::new(options.sampling).unwrap().sample(&[0.0;8]).unwrap();
        gate.submit_sampled(vec![1],options).unwrap();
        let ready=Arc::new(AtomicBool::new(true));
        let first=gate.launch(|_|Ok(fence(&ready))).unwrap().unwrap();
        assert!(gate.try_rollback(first.id).unwrap().is_ready());
        let retry=gate.launch(|_|Ok(fence(&ready))).unwrap().unwrap();
        assert!(gate.try_complete_logits(first.id,&[0.0;8],8).is_err());
        assert_eq!(gate.try_complete_logits(retry.id,&[0.0;8],8).unwrap(),Poll::Ready(vec![expected]));
    }
    #[test]
    fn unresolved_fence_ownership_is_retained_on_drop() {
        struct OwnedFence(Arc<AtomicUsize>);
        impl Drop for OwnedFence { fn drop(&mut self) { self.0.fetch_add(1,Ordering::SeqCst); } }
        impl BatchFence for OwnedFence {
            fn poll_complete(&mut self)->Poll<Result<(),RuntimeError>> {Poll::Pending}
        }
        let dropped=Arc::new(AtomicUsize::new(0));
        let mut gate=FencedBatchScheduler::new(ContinuousBatchScheduler::new(
            ContinuousBatchConfig {max_active_sequences:1,max_batch_tokens:8},
            PagedKvCacheConfig {block_size:2,num_pages:8,max_sequence_length:16},
        ).unwrap()).unwrap();
        gate.submit(vec![1],generation()).unwrap();
        gate.launch(|_|Ok(OwnedFence(dropped.clone()))).unwrap();
        drop(gate);
        assert_eq!(dropped.load(Ordering::SeqCst),0);
    }
    #[test]
    fn launch_panic_leaves_scheduler_quarantined() {
        let mut gate=scheduler();gate.submit(vec![1],generation()).unwrap();
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _=gate.launch(|_|->Result<Fence,BatchLaunchFailure> {panic!("injected launch panic")});
        }));
        assert!(result.is_err());assert!(gate.is_quarantined());
        assert!(gate.snapshot().has_in_flight_batch);
    }

}
