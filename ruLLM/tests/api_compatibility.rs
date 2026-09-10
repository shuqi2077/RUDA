//! Compile-time smoke test of original public struct literals and entry points.
use rullm::*;

#[test]
fn existing_config_literals_and_scheduler_calls_still_compile() {
    let config = ContinuousBatchConfig { max_active_sequences: 2, max_batch_tokens: 8 };
    let kv = PagedKvCacheConfig { block_size: 2, num_pages: 8, max_sequence_length: 16 };
    let options = ContinuousBatchOptions { max_pending_requests: Some(8), kv_admission: KvAdmissionPolicy::OnDemand };
    let mut legacy = ContinuousBatchScheduler::new(config,kv).unwrap();
    let _configured = ContinuousBatchScheduler::with_options(config,kv,options).unwrap();
    let _ = legacy.submit(vec![1], GreedyGenerationConfig { max_new_tokens: 1, eos_token_ids: vec![] }).unwrap();
    let batch = legacy.schedule().unwrap().unwrap();
    legacy.complete_batch(batch.id, &[1]).unwrap();
    assert!(legacy.pop_finished().is_some());
    let error = GenerationError("unchanged public tuple error".into());
    let GenerationError(message) = error;
    assert!(!message.is_empty());
    let _old_sampling = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 1.0, seed: Some(7) };
    let _old_control = GenerationControl::default();
}

#[test]
fn new_host_controls_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<runtime::MemoryBudget>();
    check::<runtime::CancellationFlag>();
    check::<runtime::ReplicaPool<usize>>();
}
