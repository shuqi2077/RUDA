use ruda_tensor::api::{Tensor, TensorData};
use ruda_tensor_host::{Host, HostDevice};
use rullm::*;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

#[derive(Default)]
struct Model {
    fail: Cell<bool>,
    calls: RefCell<Vec<ScheduledBatch>>,
}
impl DeviceBatchModel<Host> for Model {
    type Cache = BTreeMap<RequestId, Vec<i32>>;
    fn batch_limits(&self) -> CausalModelLimits {
        CausalModelLimits {
            vocab_size: 7,
            max_sequence_length: 64,
        }
    }
    fn new_batch_cache(&self, _: PagedKvCacheConfig) -> Self::Cache {
        BTreeMap::new()
    }
    fn fork_batch_cache(&self, c: &Self::Cache) -> Self::Cache {
        c.clone()
    }
    fn retain_batch_cache(&self, c: &mut Self::Cache, s: &ContinuousBatchScheduler) {
        c.retain(|id, _| s.kv_cache().contains_sequence(*id));
    }
    fn forward_batch(
        &self,
        b: &ScheduledBatch,
        c: &mut Self::Cache,
        _: &HostDevice,
    ) -> Result<Tensor<Host, 3>, GenerationError> {
        self.calls.borrow_mut().push(b.clone());
        let mut values = vec![-10.; b.batch_size() * 7];
        for (i, row) in b.sequences.iter().enumerate() {
            let history = c.entry(row.request_id).or_default();
            assert_eq!(history.len(), row.start_position);
            history.extend(&row.token_ids);
            values[i * 7 + (history.iter().sum::<i32>() as usize + 1) % 7] = 10.;
        }
        if self.fail.replace(false) {
            return Err(GenerationError("injected after cache mutation".into()));
        }
        Ok(Tensor::from_data(
            TensorData::new(values, [b.batch_size(), 1, 7]),
            &HostDevice,
        ))
    }
}
fn config(n: usize) -> GreedyGenerationConfig {
    GreedyGenerationConfig {
        max_new_tokens: n,
        eos_token_ids: vec![],
    }
}
fn executor(m: &Model) -> DeviceBatchExecutor<'_, Host, Model> {
    DeviceBatchExecutor::new(
        m,
        HostDevice,
        ContinuousBatchConfig {
            max_active_sequences: 4,
            max_batch_tokens: 32,
        },
        PagedKvCacheConfig {
            block_size: 2,
            num_pages: 16,
            max_sequence_length: 32,
        },
        ContinuousBatchOptions::default(),
    )
    .unwrap()
}
fn expected(prompt: &[i32], n: usize) -> Vec<i32> {
    let mut sum = prompt.iter().sum::<i32>();
    (0..n)
        .map(|_| {
            let t = (sum + 1) % 7;
            sum += t;
            t
        })
        .collect()
}

#[test]
fn dynamic_batches_keep_different_contexts_and_recycle_cancelled_pages() {
    let m = Model::default();
    let mut e = executor(&m);
    let a = e.submit(vec![1, 2], config(8)).unwrap();
    let b = e.submit(vec![2, 3], config(8)).unwrap();
    assert_eq!(e.step().unwrap().unwrap().batch.batch_size(), 2);
    let c = e.submit(vec![3], config(4)).unwrap();
    e.step().unwrap();
    let cancelled = e.cancel(b).unwrap().unwrap();
    assert_eq!(cancelled.generated_token_ids, expected(&[2, 3], 2));
    assert!(!e.cache().contains_key(&b));
    let d = e.submit(vec![4, 1, 2], config(3)).unwrap();
    for _ in 0..30 {
        if e.is_idle() {
            break;
        }
        assert!(e.step().unwrap().is_some());
    }
    assert!(e.is_idle());
    let mut results = BTreeMap::new();
    while let Some(r) = e.pop_finished() {
        results.insert(r.request_id, r.generated_token_ids);
    }
    for (id, p, n) in [(a, vec![1, 2], 8), (c, vec![3], 4), (d, vec![4, 1, 2], 3)] {
        assert_eq!(results[&id], expected(&p, n));
    }
    assert!(m.calls.borrow().iter().any(|b| {
        b.kind == ScheduledBatchKind::Decode
            && b.sequences
                .iter()
                .any(|r| r.start_position != b.sequences[0].start_position)
    }));
    assert!(e.cache().is_empty());
    assert_eq!(e.snapshot().free_kv_pages, 16);
}

#[test]
fn failed_device_result_rolls_back_real_cache_and_scheduler_before_retry() {
    let m = Model::default();
    let mut e = executor(&m);
    let id = e.submit(vec![1, 2], config(5)).unwrap();
    e.step().unwrap();
    let before = e.cache().clone();
    let free = e.snapshot().free_kv_pages;
    m.fail.set(true);
    assert!(e.step().is_err());
    assert_eq!(e.cache(), &before);
    assert_eq!(e.snapshot().free_kv_pages, free);
    assert!(!e.snapshot().has_in_flight_batch);
    while !e.is_idle() {
        e.step().unwrap();
    }
    let r = e.pop_finished().unwrap();
    assert_eq!(r.request_id, id);
    assert_eq!(r.generated_token_ids, expected(&[1, 2], 5));
}

#[test]
fn eos_zero_length_and_queued_cancel_release_all_state() {
    let m = Model::default();
    let mut e = executor(&m);
    e.submit(vec![1], config(0)).unwrap();
    let queued = e.submit(vec![3], config(8)).unwrap();
    assert!(e.cancel(queued).unwrap().is_some());
    let id = e
        .submit(
            vec![1],
            GreedyGenerationConfig {
                max_new_tokens: 8,
                eos_token_ids: vec![2],
            },
        )
        .unwrap();
    e.step().unwrap();
    assert!(e.is_idle());
    assert!(e.pop_finished().unwrap().generated_token_ids.is_empty());
    let r = e.pop_finished().unwrap();
    assert_eq!(r.request_id, id);
    assert_eq!(r.generated_token_ids, [2]);
    assert!(r.stopped_on_eos);
    assert_eq!(e.snapshot().free_kv_pages, 16);
    assert!(e.cache().is_empty());
    assert!(e.submit(vec![7], config(1)).is_err());
    assert!(e.submit(vec![1], config(usize::MAX)).is_err());
}

#[test]
fn sampled_rng_survives_batch_changes_and_failed_attempts() {
    fn run(with_other: bool) -> Vec<i32> {
        let m = Model::default();
        let mut e = executor(&m);
        let c = SamplingGenerationConfig {
            max_new_tokens: 8,
            eos_token_ids: vec![],
            sampling: SamplingConfig {
                temperature: 100.,
                seed: Some(42),
                ..Default::default()
            },
        };
        let id = e.submit_sampled(vec![1, 2], c).unwrap();
        if with_other {
            e.submit(vec![3, 4], config(3)).unwrap();
            m.fail.set(true);
            assert!(e.step().is_err());
        }
        while !e.is_idle() {
            e.step().unwrap();
        }
        loop {
            let r = e.pop_finished().unwrap();
            if r.request_id == id {
                return r.generated_token_ids;
            }
        }
    }
    assert_eq!(run(false), run(true));
}
