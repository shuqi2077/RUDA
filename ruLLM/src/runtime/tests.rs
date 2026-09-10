use super::*;
use std::sync::{Arc, Barrier, atomic::{AtomicUsize, Ordering}, mpsc};
use std::time::Duration;
use std::task::{Context, Poll, Wake, Waker};
use std::future::Future;

fn key(backend: BackendKind, ordinal: u32) -> DeviceKey { DeviceKey { backend, ordinal } }
fn fingerprint() -> DeviceFingerprint {
    DeviceFingerprint { device: key(BackendKind::Host, 0), architecture: "test-arch".into(),
        driver: "test-driver-1".into(), runtime: "test-runtime-1".into(), compiler_build: "test-build-1".into() }
}
fn perf_key() -> PerformanceKey {
    PerformanceKey { device: fingerprint(), operation: "decode-test-v1".into(), dtype: TensorDType::F32,
        layouts: vec![TensorLayout::contiguous(vec![1, 4, 1, 64]).unwrap()], numerical_policy: "fp32-accumulate".into() }
}
fn error(kind: RuntimeErrorKind) -> RuntimeError { RuntimeError::new(kind, "injected failure") }

#[test]
fn cuda_encoding_is_not_a_driver_release() {
    let version = CudaApiVersion::from_encoded(12080).unwrap();
    assert_eq!((version.major(), version.minor()), (12, 8));
    for invalid in [0, 580, 12081] { assert!(CudaApiVersion::from_encoded(invalid).is_err()); }
}
#[test]
fn cuda_required_api_is_checked() {
    let requirement = CudaRequirements { minimum_api: CudaApiVersion::from_encoded(12000).unwrap(), minimum_ptx_api: None, uses_ptx: false };
    assert!(requirement.check(CudaApiVersion::from_encoded(12000).unwrap()).is_ok());
    assert_eq!(requirement.check(CudaApiVersion::from_encoded(11080).unwrap()).unwrap_err().kind(), RuntimeErrorKind::DriverIncompatible);
}
#[test]
fn unknown_ptx_requirement_fails_closed() {
    let mut requirement = CudaRequirements { minimum_api: CudaApiVersion::from_encoded(12000).unwrap(), minimum_ptx_api: None, uses_ptx: true };
    assert!(requirement.check(CudaApiVersion::from_encoded(13000).unwrap()).is_err());
    requirement.minimum_ptx_api = Some(CudaApiVersion::from_encoded(12080).unwrap());
    assert!(requirement.check(CudaApiVersion::from_encoded(12060).unwrap()).is_err());
    assert!(requirement.check(CudaApiVersion::from_encoded(13000).unwrap()).is_ok());
}
#[test]
fn nvidia_and_amd_zero_are_distinct() {
    assert_ne!(key(BackendKind::Nvidia, 0), key(BackendKind::Amd, 0));
}
#[test]
fn fixed_subgroups_follow_capabilities_not_vendor() {
    for width in [16, 32, 64] {
        assert_eq!(fixed_reduction_width(width, width, true, true, 256, 256), Some(width));
        assert!(supports_two_lane_decode(Some(width), width as usize * 2, 1));
    }
}
#[test]
fn partial_or_unknown_subgroups_are_not_specialized() {
    for (lo, hi, ops, packed, threads) in [(32,64,true,true,256), (0,0,true,true,256),
        (32,32,false,true,256), (64,64,true,false,256), (64,64,true,true,32), (48,48,true,true,256)] {
        assert_eq!(fixed_reduction_width(lo,hi,ops,packed,threads,threads), None);
    }
    assert!(!supports_two_lane_decode(Some(32), 128, 1));
    assert!(!supports_two_lane_decode(Some(64), 64, 1));
    assert!(!supports_two_lane_decode(Some(32), 64, 2));
    assert!(!supports_two_lane_decode(None, 64, 1));
}
#[test]
fn dtype_storage_counts_include_odd_int4_tail() {
    assert_eq!(TensorDType::I4.storage_bytes(3).unwrap(), 2);
    assert_eq!(TensorDType::BF16.storage_bytes(7).unwrap(), 14);
    assert_eq!(TensorDType::F64.storage_bytes(7).unwrap(), 56);
    assert!(TensorDType::F64.storage_bytes(u64::MAX).is_err());
}
#[test]
fn scalar_layout_contains_one_element() {
    let scalar = TensorLayout::contiguous(vec![]).unwrap();
    assert_eq!(scalar.elements().unwrap(), 1);
    assert_eq!(scalar.storage_bytes(TensorDType::F32).unwrap(), 4);
}
#[test]
fn empty_layout_does_not_overflow_unused_dimensions() {
    let empty = TensorLayout::contiguous(vec![usize::MAX, 0, usize::MAX]).unwrap();
    assert_eq!(empty.elements().unwrap(), 0);
    assert_eq!(empty.storage_span_elements().unwrap(), 0);
}
#[test]
fn tensor_shape_and_offset_overflow_are_rejected() {
    assert!(TensorLayout::contiguous(vec![usize::MAX, 2]).is_err());
    assert!(TensorLayout::new(vec![1], vec![1], usize::MAX).is_err());
    assert!(TensorLayout::new(vec![2, 3], vec![3], 0).is_err());
    assert!(TensorLayout::new(vec![2], vec![usize::MAX], 1).is_err());
}
#[test]
fn strided_views_account_for_holes_and_offset() {
    let view = TensorLayout::new(vec![3, 2], vec![8, 2], 4).unwrap();
    assert_eq!(view.elements().unwrap(), 6);
    assert_eq!(view.storage_span_elements().unwrap(), 23);
    assert!(!view.is_contiguous());
    assert!(view.validate_storage(TensorDType::F32, 92).is_ok());
    assert!(view.validate_storage(TensorDType::F32, 91).is_err());
}
#[test]
fn broadcast_and_singleton_strides_are_valid_views() {
    let broadcast = TensorLayout::new(vec![5, 2], vec![0, 1], 0).unwrap();
    assert_eq!(broadcast.storage_span_elements().unwrap(), 2);
    assert!(!broadcast.is_contiguous());
    assert!(TensorLayout::new(vec![2,1,3],vec![3,99,1],0).unwrap().is_contiguous());
}
#[test]
fn attention_dimensions_validate_group_ratio() {
    let shape = AttentionShape { batch: 2, query_heads: 8, kv_heads: 2, query_tokens: 7, cached_tokens: 31, head_dimension: 80 };
    assert!(shape.validate().is_ok());
    assert!(AttentionShape { kv_heads: 3, ..shape }.validate().is_err());
    assert!(AttentionShape { kv_heads: 0, ..shape }.validate().is_err());
    assert!(AttentionShape { query_tokens: 32, ..shape }.validate().is_err());
}
#[test]
fn kv_geometry_counts_both_key_and_value() {
    let geometry = KvMemoryGeometry { layers: 32, kv_heads: 8, head_dimension: 128, tokens_per_page: 16, dtype: TensorDType::BF16 };
    assert_eq!(geometry.bytes_per_page().unwrap(), 2 * 1024 * 1024);
    assert_eq!(geometry.page_capacity(20*1024*1024, 4*1024*1024, 2*1024*1024).unwrap(), 7);
}
#[test]
fn kv_geometry_does_not_pretend_quantized_caches_are_dense() {
    let geometry = KvMemoryGeometry { layers: 1, kv_heads: 1, head_dimension: 1, tokens_per_page: 1, dtype: TensorDType::I4 };
    assert!(geometry.bytes_per_page().is_err());
}
#[test]
fn kv_geometry_overflow_and_zero_capacity_fail() {
    let geometry = KvMemoryGeometry { layers: 1, kv_heads: 1, head_dimension: 64, tokens_per_page: 16, dtype: TensorDType::F32 };
    assert!(KvMemoryGeometry { layers: usize::MAX, ..geometry }.bytes_per_page().is_err());
    assert!(geometry.page_capacity(100, 200, 0).is_err());
    assert!(geometry.page_capacity(100, 0, 0).is_err());
}
#[test]
fn logical_budget_releases_without_double_accounting() {
    let budget = MemoryBudget::new(100, 10).unwrap();
    let first = budget.try_reserve(60).unwrap();
    assert!(budget.try_reserve(40).is_err());
    let second = budget.try_reserve(30).unwrap();
    assert_eq!(budget.snapshot().unwrap().reserved, 90);
    drop(first); drop(second);
    assert_eq!(budget.snapshot().unwrap(), MemoryBudgetSnapshot { limit: 90, reserved: 0, peak: 90 });
}
#[test]
fn concurrent_memory_reservations_cannot_oversubscribe() {
    let budget = MemoryBudget::new(100, 0).unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let threads: Vec<_> = (0..2).map(|_| {
        let budget = budget.clone(); let barrier = barrier.clone();
        std::thread::spawn(move || {
            let reservation = budget.try_reserve(60);
            barrier.wait(); barrier.wait();
            reservation.is_ok()
        })
    }).collect();
    barrier.wait();
    assert_eq!(budget.snapshot().unwrap().reserved, 60);
    barrier.wait();
    let accepted = threads.into_iter()
        .map(|t| usize::from(t.join().unwrap())).sum::<usize>();
    assert_eq!(accepted, 1);
    assert_eq!(budget.snapshot().unwrap().reserved, 0);
}
#[test]
fn invalid_memory_budgets_are_rejected() {
    assert!(MemoryBudget::new(0, 0).is_err());
    assert!(MemoryBudget::new(10, 10).is_err());
    assert!(MemoryBudget::new(10, 11).is_err());
    assert!(MemoryBudget::new(10, 0).unwrap().try_reserve(0).is_err());
}
#[test]
fn strict_argmax_keeps_mask_and_tie_semantics() {
    assert_eq!(checked_argmax(&[f32::NEG_INFINITY, 2.0, 2.0]).unwrap(), 1);
    assert_eq!(checked_argmax(&[-0.0, 0.0]).unwrap(), 0);
    for row in [vec![], vec![f32::NAN, 1.0], vec![f32::INFINITY], vec![f32::NEG_INFINITY]] {
        assert_eq!(checked_argmax(&row).unwrap_err().kind(), RuntimeErrorKind::Numerical);
    }
}
#[test]
fn softmax_extremes_remain_finite_and_normalized() {
    for row in [vec![f32::MAX, -f32::MAX, 0.0], vec![-10000.0, -10001.0], vec![1.0, f32::NEG_INFINITY, 1.0]] {
        let probabilities = stable_softmax(&row).unwrap();
        assert!(probabilities.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
}
#[test]
fn rms_norm_handles_large_and_small_finite_inputs() {
    for magnitude in [1e-300f64, 1.0, 1e300] {
        let output = stable_rms_norm(&[magnitude, -magnitude], &[1.0, 1.0], 1e-6).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
        assert_eq!(output[0], -output[1]);
        assert!(output[0] <= 1.0);
    }
    assert_eq!(stable_rms_norm(&[0.0; 3], &[1.0; 3], 1e-6).unwrap(), vec![0.0; 3]);
}
#[test]
fn rms_norm_rejects_invalid_values_and_shapes() {
    assert!(stable_rms_norm(&[f64::INFINITY], &[1.0], 1e-6).is_err());
    assert!(stable_rms_norm(&[1.0], &[1.0], 0.0).is_err());
    assert!(stable_rms_norm(&[], &[], 1e-6).is_err());
    assert!(stable_rms_norm(&[1.0], &[], 1e-6).is_err());
}
#[test]
fn tolerance_never_accepts_matching_nan_or_infinity() {
    let tolerance = NumericalTolerance { absolute: 1e-5, relative: 1e-3 };
    assert!(tolerance.compare(&[1.0001], &[1.0]).is_ok());
    assert!(tolerance.compare(&[1.1], &[1.0]).is_err());
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(tolerance.compare(&[value], &[value]).is_err());
    }
}
#[test]
fn opposite_extremes_do_not_pass_due_to_infinity_comparison() {
    let tolerance = NumericalTolerance { absolute: 0.0, relative: 1.0 };
    assert!(tolerance.compare(&[f64::MAX], &[-f64::MAX]).is_err());
}
#[test]
fn unknown_completion_and_corrupt_cache_never_retry() {
    let policy = RecoveryPolicy::default();
    for kind in [RuntimeErrorKind::OutOfMemory, RuntimeErrorKind::Unsupported, RuntimeErrorKind::Cancelled] {
        assert_eq!(policy.decide(kind,WorkState::Unknown,true,0,16,1,false), RecoveryAction::QuarantineDevice);
        assert_eq!(policy.decide(kind,WorkState::Quiescent,false,0,16,1,false), RecoveryAction::QuarantineDevice);
    }
}
#[test]
fn oom_retry_is_bounded_and_cannot_strand_prefill() {
    let policy = RecoveryPolicy::default();
    assert_eq!(policy.decide(RuntimeErrorKind::OutOfMemory,WorkState::Quiescent,true,0,16,7,false), RecoveryAction::RetryWithTokenLimit(8));
    assert_eq!(policy.decide(RuntimeErrorKind::OutOfMemory,WorkState::Quiescent,true,1,8,7,false), RecoveryAction::RetryWithTokenLimit(7));
    assert_eq!(policy.decide(RuntimeErrorKind::OutOfMemory,WorkState::Quiescent,true,2,8,7,false), RecoveryAction::FailRequest);
    assert_eq!(policy.decide(RuntimeErrorKind::OutOfMemory,WorkState::Quiescent,true,0,7,7,false), RecoveryAction::FailRequest);
}
#[test]
fn numerical_errors_are_not_hidden_by_retries() {
    assert_eq!(RecoveryPolicy::default().decide(RuntimeErrorKind::Numerical,WorkState::Quiescent,true,0,16,1,false), RecoveryAction::FailRequest);
}
#[test]
fn portable_retry_happens_at_most_once() {
    let p = RecoveryPolicy::default();
    assert_eq!(p.decide(RuntimeErrorKind::Unsupported,WorkState::NotSubmitted,true,0,16,1,false), RecoveryAction::RetryPortable);
    assert_eq!(p.decide(RuntimeErrorKind::Unsupported,WorkState::NotSubmitted,true,0,16,1,true), RecoveryAction::FailRequest);
}
#[test]
fn performance_guard_ignores_warmup_and_single_outliers() {
    let mut guard = PerformanceGuard::new(PerformanceGuardConfig { warmup_pairs: 1, measurement_pairs: 3, ..Default::default() }).unwrap();
    let key = perf_key();
    for elapsed in [1000, 1000, 90, 90] {
        assert_eq!(guard.record_pair(key.clone(),Duration::from_nanos(100),Duration::from_nanos(elapsed)).unwrap(), KernelMode::Automatic);
    }
}
#[test]
fn sustained_regression_switches_to_portable_and_latches() {
    let mut guard = PerformanceGuard::new(PerformanceGuardConfig { warmup_pairs: 0, measurement_pairs: 3, ..Default::default() }).unwrap();
    let key = perf_key();
    for _ in 0..3 { guard.record_pair(key.clone(),Duration::from_nanos(100),Duration::from_nanos(150)).unwrap(); }
    assert_eq!(guard.mode(&key), KernelMode::Portable);
    for _ in 0..3 { guard.record_pair(key.clone(),Duration::from_nanos(100),Duration::from_nanos(50)).unwrap(); }
    assert_eq!(guard.mode(&key), KernelMode::Portable);
}
#[test]
fn version_and_shape_changes_do_not_reuse_a_regressed_decision() {
    let mut guard = PerformanceGuard::new(PerformanceGuardConfig { warmup_pairs: 0, measurement_pairs: 3, ..Default::default() }).unwrap();
    let original = perf_key();
    for _ in 0..3 { guard.record_pair(original.clone(),Duration::from_nanos(100),Duration::from_nanos(150)).unwrap(); }
    let mut changed = original.clone(); changed.device.driver = "different-driver".into();
    assert_eq!(guard.mode(&changed), KernelMode::Automatic);
    changed = original.clone(); changed.layouts = vec![TensorLayout::contiguous(vec![2,4,1,64]).unwrap()];
    assert_eq!(guard.mode(&changed), KernelMode::Automatic);
    changed = original.clone(); changed.dtype = TensorDType::BF16;
    assert_eq!(guard.mode(&changed), KernelMode::Automatic);
}
#[test]
fn performance_cache_is_bounded_and_rejects_zero_timings() {
    let mut guard = PerformanceGuard::new(PerformanceGuardConfig { capacity: 2, ..Default::default() }).unwrap();
    for n in 0..10 {
        let mut key = perf_key(); key.operation = format!("test-{n}");
        guard.record_pair(key, Duration::from_nanos(100), Duration::from_nanos(90)).unwrap();
        assert!(guard.len() <= 2);
    }
    assert!(guard.record_pair(perf_key(), Duration::ZERO, Duration::from_nanos(1)).is_err());
    guard.reset(); assert!(guard.is_empty());
}
#[test]
fn worker_initializes_and_uses_state_on_one_thread() {
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),2,|| Ok(std::thread::current().id())).unwrap();
    let result = worker.try_submit(|owner,_| Ok(*owner == std::thread::current().id())).unwrap().wait().unwrap();
    assert!(result); worker.shutdown().unwrap();
}
#[test]
fn worker_bound_includes_running_and_waiting_jobs() {
    let (release_tx, release_rx) = mpsc::channel();
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),2,move || { release_rx.recv().unwrap(); Ok(0usize) }).unwrap();
    let first = worker.try_submit(|state,_| { *state += 1; Ok(*state) }).unwrap();
    let second = worker.try_submit(|state,_| { *state += 1; Ok(*state) }).unwrap();
    assert_eq!(worker.try_submit(|_,_| Ok(3)).err().unwrap().kind(), RuntimeErrorKind::QueueFull);
    release_tx.send(()).unwrap();
    assert_eq!(first.wait().unwrap(),1); assert_eq!(second.wait().unwrap(),2);
    worker.shutdown().unwrap(); assert_eq!(worker.outstanding(),0);
}
#[test]
fn cancelling_queued_work_does_not_mutate_replica_state() {
    let (release_tx, release_rx) = mpsc::channel();
    let calls = Arc::new(AtomicUsize::new(0)); let observed = calls.clone();
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),2,move || { release_rx.recv().unwrap(); Ok(()) }).unwrap();
    let job = worker.try_submit(move |_,_| { observed.fetch_add(1,Ordering::SeqCst); Ok(()) }).unwrap();
    job.cancel(); release_tx.send(()).unwrap();
    assert_eq!(job.wait().unwrap_err().kind(), RuntimeErrorKind::Cancelled);
    worker.shutdown().unwrap(); assert_eq!(calls.load(Ordering::SeqCst),0);
}
#[test]
fn running_cancellation_is_cooperative() {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),1,|| Ok(())).unwrap();
    let job = worker.try_submit(move |_,flag| { started_tx.send(()).unwrap(); release_rx.recv().unwrap(); Ok(flag.is_cancelled()) }).unwrap();
    started_rx.recv().unwrap(); job.cancel(); release_tx.send(()).unwrap();
    assert!(job.wait().unwrap()); worker.shutdown().unwrap();
}
#[test]
fn panic_quarantines_worker_and_rejects_queued_jobs() {
    let (release_tx, release_rx) = mpsc::channel();
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),2,move || { release_rx.recv().unwrap(); Ok(()) }).unwrap();
    let first = worker.try_submit::<()>(|_,_| panic!("injected host failure")).unwrap();
    let second = worker.try_submit(|_,_| Ok(7)).unwrap();
    release_tx.send(()).unwrap();
    assert_eq!(first.wait().unwrap_err().kind(),RuntimeErrorKind::WorkerPanicked);
    assert_eq!(second.wait().unwrap_err().kind(),RuntimeErrorKind::WorkerPanicked);
    assert!(worker.is_quarantined()); worker.shutdown().unwrap();
}
#[test]
fn recoverable_error_does_not_poison_later_jobs() {
    let (release_tx, release_rx) = mpsc::channel();
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),2,move || { release_rx.recv().unwrap(); Ok(()) }).unwrap();
    let first = worker.try_submit::<()>(|_,_| Err(error(RuntimeErrorKind::OutOfMemory))).unwrap();
    let second = worker.try_submit(|_,_| Ok(7)).unwrap();
    release_tx.send(()).unwrap();
    assert_eq!(first.wait().unwrap_err().kind(),RuntimeErrorKind::OutOfMemory);
    assert_eq!(second.wait().unwrap(),7); worker.shutdown().unwrap();
}
#[test]
fn synchronization_failure_quarantines_worker() {
    let (release_tx, release_rx) = mpsc::channel();
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),2,move || { release_rx.recv().unwrap(); Ok(()) }).unwrap();
    let first = worker.try_submit::<()>(|_,_| Err(error(RuntimeErrorKind::Synchronization))).unwrap();
    let second = worker.try_submit(|_,_| Ok(7)).unwrap();
    release_tx.send(()).unwrap();
    assert_eq!(first.wait().unwrap_err().kind(),RuntimeErrorKind::Synchronization);
    assert!(second.wait().is_err()); worker.shutdown().unwrap();
}
#[test]
fn replicas_have_independent_mutable_state() {
    let (release_tx, release_rx) = mpsc::channel();
    let first = BoundedWorker::spawn(key(BackendKind::Host,0),1,move || { release_rx.recv().unwrap(); Ok(10usize) }).unwrap();
    let second = BoundedWorker::spawn(key(BackendKind::Host,1),1,|| Ok(20usize)).unwrap();
    let mut pool = ReplicaPool::new(vec![first,second]).unwrap();
    let (device_a, a) = pool.try_submit(|state,_| { *state += 1; Ok(*state) }).unwrap();
    let (device_b, b) = pool.try_submit(|state,_| { *state += 1; Ok(*state) }).unwrap();
    assert_ne!(device_a,device_b); release_tx.send(()).unwrap();
    assert_eq!(a.wait().unwrap(),11); assert_eq!(b.wait().unwrap(),21);
    pool.shutdown().unwrap();
}
#[test]
fn pool_rejects_duplicate_device_identity() {
    let first = BoundedWorker::spawn(key(BackendKind::Host,0),1,|| Ok(())).unwrap();
    let second = BoundedWorker::spawn(key(BackendKind::Host,0),1,|| Ok(())).unwrap();
    assert!(ReplicaPool::new(vec![first,second]).is_err());
}
struct CountingWake(AtomicUsize);
impl Wake for CountingWake { fn wake(self: Arc<Self>) { self.0.fetch_add(1,Ordering::SeqCst); } }
#[test]
fn async_future_registers_waker_without_a_lost_wakeup() {
    let (release_tx,release_rx) = mpsc::channel();
    let mut worker = BoundedWorker::spawn(key(BackendKind::Host,0),1,move || { release_rx.recv().unwrap(); Ok(()) }).unwrap();
    let mut job = std::pin::pin!(worker.try_submit(|_,_| Ok(42)).unwrap());
    let counter = Arc::new(CountingWake(AtomicUsize::new(0)));
    let waker = Waker::from(counter.clone()); let mut context = Context::from_waker(&waker);
    assert!(job.as_mut().poll(&mut context).is_pending());
    release_tx.send(()).unwrap();
    // A separate completion barrier uses shutdown only after the job resolves.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let value = loop {
        assert!(std::time::Instant::now() < deadline, "job future did not complete");
        match job.as_mut().poll(&mut context) {
            Poll::Pending => std::thread::yield_now(),
            Poll::Ready(value) => break value.unwrap(),
        }
    };
    assert_eq!(value,42); worker.shutdown().unwrap();
    assert!(counter.0.load(Ordering::SeqCst) >= 1);
}
