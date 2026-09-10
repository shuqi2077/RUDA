use super::*;

fn generation(max_new_tokens: usize) -> GreedyGenerationConfig {
    GreedyGenerationConfig { max_new_tokens, eos_token_ids: vec![] }
}

fn scheduler(options: ContinuousBatchOptions, pages: usize) -> ContinuousBatchScheduler {
    ContinuousBatchScheduler::with_options(
        ContinuousBatchConfig { max_active_sequences: 4, max_batch_tokens: 8 },
        PagedKvCacheConfig { block_size: 2, num_pages: pages, max_sequence_length: 32 },
        options,
    ).unwrap()
}

#[test]
fn queue_limit_rejects_without_consuming_id_or_mutating_state() {
    let mut scheduler = scheduler(ContinuousBatchOptions {
        max_pending_requests: Some(1), ..Default::default()
    }, 16);
    let first = scheduler.submit(vec![1], generation(3)).unwrap();
    let before = scheduler.snapshot();
    let next_id = scheduler.next_request;
    assert!(scheduler.submit(vec![2], generation(3)).is_err());
    assert_eq!(scheduler.snapshot(), before);
    assert_eq!(scheduler.next_request, next_id);
    scheduler.cancel(first).unwrap();
    assert_eq!(scheduler.submit(vec![2], generation(3)).unwrap().0, next_id);
}

#[test]
fn zero_token_request_bypasses_full_pending_queue() {
    let mut scheduler = scheduler(ContinuousBatchOptions {
        max_pending_requests: Some(1), ..Default::default()
    }, 8);
    scheduler.submit(vec![1], generation(2)).unwrap();
    let immediate = scheduler.submit(vec![2], generation(0)).unwrap();
    assert_eq!(scheduler.pop_finished().unwrap().request_id, immediate);
    assert_eq!(scheduler.snapshot().pending_requests, 1);
}

#[test]
fn cancellation_is_idempotent_and_returns_partial_generation() {
    let mut scheduler = scheduler(Default::default(), 8);
    let queued = scheduler.submit(vec![1], generation(3)).unwrap();
    assert!(scheduler.cancel(queued).unwrap().unwrap().generated_token_ids.is_empty());
    assert!(scheduler.cancel(queued).unwrap().is_none());
    let active = scheduler.submit(vec![1], generation(3)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    scheduler.complete_batch(batch.id, &[7]).unwrap();
    let cancelled = scheduler.cancel(active).unwrap().unwrap();
    assert_eq!(cancelled.generated_token_ids, vec![7]);
    assert_eq!(scheduler.snapshot().free_kv_pages, 8);
    assert_eq!(scheduler.snapshot().finished_requests, 0);
    assert!(scheduler.is_idle());
}

#[test]
fn in_flight_cancellation_does_not_reclaim_pages() {
    let mut scheduler = scheduler(Default::default(), 8);
    let id = scheduler.submit(vec![1, 2], generation(3)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    let before = scheduler.kv_cache.snapshot();
    assert!(scheduler.cancel(id).is_err());
    assert_eq!(scheduler.kv_cache.snapshot(), before);
    assert!(scheduler.snapshot().has_in_flight_batch);
    scheduler.fail_batch(batch.id).unwrap();
    scheduler.cancel(id).unwrap();
    assert!(scheduler.is_idle());
    assert_eq!(scheduler.snapshot().free_kv_pages, 8);
}

#[test]
fn ready_prefill_can_be_cancelled_while_another_batch_runs() {
    let mut scheduler = scheduler(Default::default(), 8);
    let first = scheduler.submit(vec![1, 2, 3], generation(2)).unwrap();
    let second = scheduler.submit(vec![1], generation(2)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    assert_eq!(batch.sequences[0].request_id, first);
    scheduler.cancel(second).unwrap().unwrap();
    scheduler.complete_batch(batch.id, &[3]).unwrap();
    scheduler.cancel(first).unwrap();
    assert!(scheduler.is_idle());
}

#[test]
fn invalid_tokens_do_not_enqueue_or_commit() {
    let mut scheduler = scheduler(Default::default(), 8);
    let before = scheduler.snapshot();
    assert!(scheduler.submit(vec![-1], generation(1)).is_err());
    let invalid = GreedyGenerationConfig { max_new_tokens: 1, eos_token_ids: vec![-1] };
    assert!(scheduler.submit(vec![1], invalid).is_err());
    assert_eq!(scheduler.snapshot(), before);
    scheduler.submit(vec![1], generation(1)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    let before = scheduler.kv_cache.snapshot();
    assert!(scheduler.complete_batch(batch.id, &[-1]).is_err());
    assert_eq!(scheduler.kv_cache.snapshot(), before);
    assert!(scheduler.complete_batch(batch.id, &[]).is_err());
    scheduler.complete_batch(batch.id, &[1]).unwrap();
    assert!(scheduler.is_idle());
}

#[test]
fn conservative_admission_prevents_mutual_page_exhaustion() {
    let mut scheduler = scheduler(ContinuousBatchOptions {
        kv_admission: KvAdmissionPolicy::ReserveSequenceCapacity, ..Default::default()
    }, 2);
    // Each request needs ceil((2 prompt + 3 new - 1) / 2) == 2 pages.
    let first = scheduler.submit(vec![1, 2], generation(3)).unwrap();
    let second = scheduler.submit(vec![1, 2], generation(3)).unwrap();
    let mut finished = Vec::new();
    for _ in 0..6 {
        let batch = scheduler.schedule().unwrap().expect("admitted requests must make progress");
        assert_eq!(batch.batch_size(), 1);
        scheduler.complete_batch(batch.id, &[4]).unwrap();
        while let Some(result) = scheduler.pop_finished() { finished.push(result.request_id); }
    }
    assert_eq!(finished, vec![first, second]);
    assert!(scheduler.is_idle());
    assert_eq!(scheduler.snapshot().free_kv_pages, 2);
}

#[test]
fn conservative_admission_reclaims_cancelled_and_eos_budgets() {
    let mut scheduler = scheduler(ContinuousBatchOptions {
        kv_admission: KvAdmissionPolicy::ReserveSequenceCapacity, ..Default::default()
    }, 2);
    let first = scheduler.submit(vec![1, 2], generation(3)).unwrap();
    let second = scheduler.submit(vec![1, 2], GreedyGenerationConfig {
        max_new_tokens: 3, eos_token_ids: vec![7],
    }).unwrap();
    let third = scheduler.submit(vec![1, 2], generation(3)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    scheduler.complete_batch(batch.id, &[4]).unwrap();
    scheduler.cancel(first).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    assert_eq!(batch.sequences[0].request_id, second);
    scheduler.complete_batch(batch.id, &[7]).unwrap();
    assert!(scheduler.pop_finished().unwrap().stopped_on_eos);
    let batch = scheduler.schedule().unwrap().unwrap();
    assert_eq!(batch.sequences[0].request_id, third);
}

#[test]
fn impossible_requests_are_rejected_before_admission() {
    for policy in [KvAdmissionPolicy::OnDemand, KvAdmissionPolicy::ReserveSequenceCapacity] {
        let mut scheduler = scheduler(ContinuousBatchOptions { kv_admission: policy, ..Default::default() }, 1);
        assert!(scheduler.submit(vec![1, 2, 3], generation(1)).is_err());
        assert!(scheduler.is_idle());
        assert_eq!(scheduler.next_request, 1);
    }
    let mut scheduler = scheduler(ContinuousBatchOptions {
        kv_admission: KvAdmissionPolicy::ReserveSequenceCapacity, ..Default::default()
    }, 1);
    assert!(scheduler.submit(vec![1], generation(4)).is_err());
}

#[test]
fn last_selected_token_needs_no_extra_cache_page() {
    let mut scheduler = scheduler(ContinuousBatchOptions {
        kv_admission: KvAdmissionPolicy::ReserveSequenceCapacity, ..Default::default()
    }, 1);
    scheduler.submit(vec![1, 2], generation(1)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    scheduler.complete_batch(batch.id, &[3]).unwrap();
    assert!(scheduler.is_idle());
}

#[test]
fn batch_id_overflow_does_not_allocate_or_promote() {
    let mut scheduler = scheduler(Default::default(), 8);
    scheduler.submit(vec![1], generation(3)).unwrap();
    scheduler.next_batch = u64::MAX;
    let before = scheduler.snapshot();
    let kv_before = scheduler.kv_cache.snapshot();
    assert!(scheduler.schedule().is_err());
    assert_eq!(scheduler.snapshot(), before);
    assert_eq!(scheduler.kv_cache.snapshot(), kv_before);
    assert_eq!(scheduler.kv_cache.reservation_count(), 0);
}

#[test]
fn completed_results_do_not_prevent_idle_state() {
    let mut scheduler = scheduler(Default::default(), 8);
    let id = scheduler.submit(vec![1], generation(0)).unwrap();
    assert!(scheduler.is_idle());
    assert!(scheduler.cancel(id).unwrap().is_none());
    assert_eq!(scheduler.pop_finished().unwrap().request_id, id);
}
