//! Deterministic HOST tests: mock timings exercise selection logic, not GPU performance.
use super::*;
use alloc::{format, vec, vec::Vec};
use std::{collections::BTreeMap, sync::{Arc, atomic::{AtomicU64, Ordering}, mpsc}, time::Duration};

fn policy() -> StackPolicy { StackPolicy { warmups: 1, samples: 3, regression_pairs: 3,
    budget: Duration::from_secs(30), ..StackPolicy::default() } }
fn problem() -> Problem { Problem { scope: Scope::Operator, operation: "matmul-v1".into(),
    environment: "device=unit-test;driver=1;build=1".into(), workload: "dtype=f32;shape=17x31;stride=31,1".into(),
    execution_context: "isolated".into(), persistent: false } }
fn candidates() -> Vec<Candidate> { vec![Candidate::new("reference"), Candidate::new("fast"), Candidate::new("faster")] }
fn ns(v: u64) -> Duration { Duration::from_nanos(v) }
#[derive(Default)]
struct Mock { times: Vec<u64>, measured: Vec<usize>, checked: Vec<(usize,usize)>,
    validation: BTreeMap<usize, Result<Validation, TuneFailure>>, failures: BTreeMap<usize,TuneFailure> }
impl Mock { fn new(times: &[u64]) -> Self { Self { times: times.to_vec(), ..Self::default() } } }
impl TrialRunner for Mock {
    fn validate(&mut self, reference: usize, candidate: usize, _: Tolerance) -> Result<Validation,TuneFailure> {
        self.checked.push((reference,candidate)); self.validation.get(&candidate).cloned().unwrap_or(Ok(Validation::Passed))
    }
    fn measure(&mut self, index: usize) -> Result<Duration,TuneFailure> {
        self.measured.push(index);
        if let Some(e) = self.failures.get(&index) { return Err(e.clone()); }
        Ok(ns(self.times[index]))
    }
}
fn select(tuner: &StackTuner, runner: &mut Mock) -> Decision { tuner.select(&problem(), &candidates(), 0, runner).unwrap() }

#[test] fn selects_best_across_all_candidates() {
    let t=StackTuner::new(policy(),None).unwrap(); let d=select(&t,&mut Mock::new(&[100,80,60]));
    assert_eq!(d.index,2); assert!(d.verified); assert_eq!(d.ratio,Some(0.6)); assert_eq!(t.stats().tunes,1);
}
#[test] fn keeps_reference_without_material_gain() {
    let t=StackTuner::new(policy(),None).unwrap(); assert_eq!(select(&t,&mut Mock::new(&[100,98,99])).index,0);
}
#[test] fn wrong_output_cannot_win() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,1]);
    m.validation.insert(2,Err(TuneFailure::rejected("wrong output")));
    assert_eq!(select(&t,&mut m).index,1); assert!(t.reports()[0].candidates[2].note.contains("wrong"));
}
#[test] fn unsupported_candidate_is_not_promoted() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,1]);
    m.validation.insert(2,Ok(Validation::Unsupported)); assert_eq!(select(&t,&mut m).index,1);
}
#[test] fn unsupported_reference_is_bypassed_once() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,1]);
    m.validation.insert(0,Ok(Validation::Unsupported)); let a=select(&t,&mut m);
    assert_eq!(a.source,DecisionSource::ValidationUnavailable); assert!(!a.verified);
    let count=m.measured.len(); select(&t,&mut m); assert_eq!(count,m.measured.len());
}
#[test] fn explicit_unchecked_policy_never_claims_validation() {
    let mut p=policy(); p.require_validation=false; let t=StackTuner::new(p,None).unwrap(); let mut m=Mock::new(&[100,80,1]);
    for i in 0..3 { m.validation.insert(i,Ok(Validation::Unsupported)); }
    let d=select(&t,&mut m); assert_eq!(d.index,2); assert!(!d.verified);
}
#[test] fn rejected_candidate_does_not_abort_other_searches() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    m.failures.insert(1,TuneFailure::rejected("not supported on this device")); assert_eq!(select(&t,&mut m).index,2);
}
#[test] fn device_failure_aborts_and_quarantines_before_cache_reuse() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    m.failures.insert(1,TuneFailure::device("completion unknown"));
    assert_eq!(t.select(&problem(),&candidates(),0,&mut m).unwrap_err().kind,FailureKind::Device);
    assert!(!m.measured.contains(&2)); m.failures.clear();
    assert_eq!(t.select(&problem(),&candidates(),0,&mut m).unwrap_err().kind,FailureKind::Quarantined);
}
#[test] fn memory_hit_does_not_benchmark_or_validate_again() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,60]); select(&t,&mut m);
    let before=(m.measured.len(),m.checked.len()); let d=select(&t,&mut m);
    assert_eq!(d.source,DecisionSource::MemoryCache); assert_eq!(before,(m.measured.len(),m.checked.len()));
}
#[test] fn cache_only_miss_has_no_trials() {
    let mut p=policy(); p.mode=Mode::CacheOnly; let t=StackTuner::new(p,None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    assert_eq!(select(&t,&mut m).source,DecisionSource::CacheMiss); assert!(m.measured.is_empty()); assert!(m.checked.is_empty());
}
#[test] fn disabled_has_no_trials() {
    let mut p=policy(); p.mode=Mode::Disabled; let t=StackTuner::new(p,None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    assert_eq!(select(&t,&mut m).source,DecisionSource::Disabled); assert!(m.measured.is_empty());
}
#[test] fn candidate_budget_preserves_reference() {
    let mut p=policy(); p.max_candidates=1; let t=StackTuner::new(p,None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    assert_eq!(select(&t,&mut m).index,0); assert!(m.measured.iter().all(|&i|i==0)); assert!(t.reports()[0].budget_exhausted);
}
#[test] fn elapsed_budget_is_soft_but_reference_is_checked() {
    let mut p=policy(); p.budget=ns(1); let t=StackTuner::new(p,None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    assert_eq!(select(&t,&mut m).index,0); assert_eq!(m.checked,vec![(0,0)]);
}
#[test] fn zero_reference_time_is_not_a_performance_result() {
    let t=StackTuner::new(policy(),None).unwrap(); let d=select(&t,&mut Mock::new(&[0,80,60]));
    assert_eq!(d.source,DecisionSource::ValidationUnavailable); assert!(d.ratio.is_none());
}
#[test] fn workspace_unknown_is_rejected_under_explicit_limit() {
    let mut p=policy(); p.workspace_limit=Some(128); let t=StackTuner::new(p,None).unwrap(); let mut c=candidates();
    c[0].workspace_bytes=Some(0); c[2].workspace_bytes=Some(256);
    assert_eq!(t.select(&problem(),&c,0,&mut Mock::new(&[100,1,1])).unwrap().index,0);
}
#[test] fn declared_workspace_within_budget_can_win() {
    let mut p=policy(); p.workspace_limit=Some(128); let t=StackTuner::new(p,None).unwrap(); let mut c=candidates();
    c[0].workspace_bytes=Some(0); c[1].workspace_bytes=Some(128); c[2].workspace_bytes=Some(129);
    assert_eq!(t.select(&problem(),&c,0,&mut Mock::new(&[100,50,1])).unwrap().index,1);
}
#[test] fn invalid_reference_is_an_input_error() {
    let t=StackTuner::new(policy(),None).unwrap(); assert_eq!(t.select(&problem(),&candidates(),4,&mut Mock::default()).unwrap_err().kind,FailureKind::InvalidInput);
}
#[test] fn duplicate_candidates_are_rejected() {
    let t=StackTuner::new(policy(),None).unwrap(); let c=vec![Candidate::new("same"),Candidate::new("same")];
    assert!(t.select(&problem(),&c,0,&mut Mock::default()).is_err());
}
#[test] fn oversized_workload_key_is_rejected_before_trial() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut p=problem(); p.workload="x".repeat(384*1024);
    let mut m=Mock::default(); assert!(t.select(&p,&candidates(),0,&mut m).is_err()); assert!(m.measured.is_empty());
}
#[test] fn invalid_policies_are_rejected() {
    let mut p=policy(); p.samples=2; assert!(p.validate().is_err()); p=policy(); p.min_speedup=f64::NAN; assert!(p.validate().is_err());
    p=policy(); p.capacity=0; assert!(p.validate().is_err()); p=policy(); p.tolerance.max_bytes=0; assert!(p.validate().is_err());
}
#[test] fn key_separates_environment_workload_context_and_scope() {
    let p=problem(); let c=candidates(); let k=cache_key(&p,&c,0,&policy());
    for field in 0..5 { let mut q=p.clone(); match field { 0=>q.environment.push_str("newdriver"),1=>q.workload.push_str("newstride"),
        2=>q.execution_context.push_str("loaded"),3=>q.operation.push_str("newbuild"),_=>q.scope=Scope::Graph };
        assert_ne!(k,cache_key(&q,&c,0,&policy())); }
}
#[test] fn key_separates_candidate_revision_order_and_precision_policy() {
    let p=problem(); let c=candidates(); let k=cache_key(&p,&c,0,&policy()); let mut d=c.clone(); d[1].revision="2".into();
    assert_ne!(k,cache_key(&p,&d,0,&policy())); d=c.clone(); d.swap(1,2); assert_ne!(k,cache_key(&p,&d,0,&policy()));
    let mut s=policy(); s.tolerance.relative*=2.; assert_ne!(k,cache_key(&p,&c,0,&s));
}
#[test] fn explore_cache_is_readable_in_cache_only_mode() {
    let mut p=policy(); let before=cache_key(&problem(),&candidates(),0,&p); p.mode=Mode::CacheOnly;
    assert_eq!(before,cache_key(&problem(),&candidates(),0,&p));
}
#[test] fn fields_are_unambiguous_with_delimiters_and_unicode() {
    assert_ne!(fields(&["ab","c"]),fields(&["a","bc"])); assert_ne!(fields(&["a:1","b"]),fields(&["a","1:b"]));
    assert_eq!(fields(&["中"]),"3:中");
}
#[test] fn median_rejects_bad_samples_and_resists_an_outlier() {
    assert_eq!(median(&mut [100.,101.,10000.]),Some(101.)); assert!(median(&mut [1.,0.,3.]).is_none());
    assert!(median(&mut [1.,f64::NAN,3.]).is_none()); assert!(median(&mut []).is_none());
}
#[test] fn paired_score_rejects_noise_and_incomplete_trials() {
    let p=policy(); assert!(paired_score(&[(ns(100),ns(10)),(ns(100),ns(100)),(ns(100),ns(1000))],&p).is_none());
    assert!(paired_score(&[(ns(100),ns(10))],&p).is_none()); assert_eq!(paired_score(&[(ns(100),ns(50));3],&p),Some((0.5,0.0)));
}
#[test] fn regression_requires_verified_paired_observations() {
    let t=StackTuner::new(policy(),None).unwrap(); let d=select(&t,&mut Mock::new(&[100,80,60]));
    assert!(t.record_comparison(&d,ns(100),ns(200),false).is_err()); assert!(t.record_comparison(&d,ns(0),ns(200),true).is_err());
}
#[test] fn sustained_regression_bans_the_fast_plan_for_future_calls() {
    let t=StackTuner::new(policy(),None).unwrap(); let d=select(&t,&mut Mock::new(&[100,80,60]));
    assert!(!t.record_comparison(&d,ns(100),ns(130),true).unwrap()); assert!(!t.record_comparison(&d,ns(100),ns(140),true).unwrap());
    assert!(t.record_comparison(&d,ns(100),ns(150),true).unwrap());
    assert_eq!(select(&t,&mut Mock::new(&[100,80,1])).index,1);
}
#[test] fn isolated_regression_outlier_does_not_invalidate() {
    let t=StackTuner::new(policy(),None).unwrap(); let d=select(&t,&mut Mock::new(&[100,80,60]));
    for n in [60,1000,60] { assert!(!t.record_comparison(&d,ns(100),ns(n),true).unwrap()); }
}
#[test] fn nonzero_reference_index_does_not_protect_candidate_zero() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut c=candidates(); c[2].eligible=false;
    let d=t.select(&problem(),&c,1,&mut Mock::new(&[50,100,1])).unwrap(); assert_eq!(d.index,0);
    t.invalidate(&d,true); assert_eq!(t.select(&problem(),&c,1,&mut Mock::new(&[1,100,1])).unwrap().index,1);
}
#[test] fn explicit_invalidation_never_replays_the_current_request() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,60]); let d=select(&t,&mut m);
    let before=m.measured.len(); t.invalidate(&d,true); assert_eq!(m.measured.len(),before);
}
#[test] fn bounded_cache_evicts_old_workloads() {
    let mut p=policy(); p.capacity=1; let t=StackTuner::new(p,None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    select(&t,&mut m); let mut other=problem(); other.workload.push_str("new"); t.select(&other,&candidates(),0,&mut m).unwrap();
    assert_eq!(select(&t,&mut m).source,DecisionSource::Tuned); assert_eq!(t.reports().len(),1);
}
#[test] fn lower_level_dependency_changes_on_new_choice_and_invalidation() {
    let t=StackTuner::new(policy(),None).unwrap(); let before=t.lower_level_fingerprint(); let d=select(&t,&mut Mock::new(&[100,80,60]));
    let after=t.lower_level_fingerprint(); assert_ne!(before,after); t.invalidate(&d,false); assert_eq!(before,t.lower_level_fingerprint());
}
#[test] fn pipeline_records_do_not_change_their_own_dependency_fingerprint() {
    let t=StackTuner::new(policy(),None).unwrap(); let before=t.lower_level_fingerprint(); let mut p=problem(); p.scope=Scope::Pipeline;
    t.select(&p,&candidates(),0,&mut Mock::new(&[100,80,60])).unwrap(); assert_eq!(before,t.lower_level_fingerprint());
}
#[test] fn pipeline_bypass_does_not_change_lower_dependencies() {
    let t=StackTuner::new(policy(),None).unwrap(); let before=t.lower_level_fingerprint(); let mut p=problem(); p.scope=Scope::Pipeline;
    let mut m=Mock::new(&[100,80,60]); m.validation.insert(0,Ok(Validation::Unsupported));
    t.select(&p,&candidates(),0,&mut m).unwrap(); assert_eq!(before,t.lower_level_fingerprint());
}
#[test] fn reference_bypass_is_a_lower_level_dependency() {
    let t=StackTuner::new(policy(),None).unwrap(); let before=t.lower_level_fingerprint(); let mut m=Mock::new(&[100,80,60]);
    m.validation.insert(0,Ok(Validation::Unsupported)); select(&t,&mut m); assert_ne!(before,t.lower_level_fingerprint());
}

static TEMP: AtomicU64=AtomicU64::new(0);
struct Dir(std::path::PathBuf);
impl Dir { fn new()->Self { Self(std::env::temp_dir().join(format!("ruda-stack-engine-{}-{}",std::process::id(),TEMP.fetch_add(1,Ordering::Relaxed)))) } }
impl Drop for Dir { fn drop(&mut self) { let _=std::fs::remove_dir_all(&self.0); } }
#[test] fn disk_hit_revalidates_without_timing_search() {
    let dir=Dir::new(); let mut q=problem(); q.persistent=true;
    let t=StackTuner::new(policy(),Some(dir.0.clone())).unwrap(); t.select(&q,&candidates(),0,&mut Mock::new(&[100,80,60])).unwrap();
    let mut p=policy(); p.mode=Mode::CacheOnly; let t=StackTuner::new(p,Some(dir.0.clone())).unwrap(); let mut m=Mock::new(&[100,80,60]);
    let d=t.select(&q,&candidates(),0,&mut m).unwrap(); assert_eq!(d.source,DecisionSource::DiskCache); assert_eq!(m.checked,vec![(0,2)]); assert!(m.measured.is_empty());
}
#[test] fn disk_winner_with_wrong_outputs_is_not_reused() {
    let dir=Dir::new(); let mut q=problem(); q.persistent=true;
    let t=StackTuner::new(policy(),Some(dir.0.clone())).unwrap(); t.select(&q,&candidates(),0,&mut Mock::new(&[100,80,60])).unwrap();
    let mut p=policy(); p.mode=Mode::CacheOnly; let t=StackTuner::new(p,Some(dir.0.clone())).unwrap(); let mut m=Mock::new(&[100,80,60]);
    m.validation.insert(2,Err(TuneFailure::rejected("wrong output"))); let d=t.select(&q,&candidates(),0,&mut m).unwrap();
    assert_eq!(d.source,DecisionSource::CacheMiss); assert!(m.measured.is_empty());
}
#[test] fn unknown_driver_uses_memory_only_even_with_disk_directory() {
    let dir=Dir::new(); let t=StackTuner::new(policy(),Some(dir.0.clone())).unwrap(); select(&t,&mut Mock::new(&[100,80,60]));
    assert!(!dir.0.exists());
}
#[test] fn corrupt_disk_cache_is_a_miss_not_a_crash() {
    let dir=Dir::new(); let mut q=problem(); q.persistent=true;
    let t=StackTuner::new(policy(),Some(dir.0.clone())).unwrap(); t.select(&q,&candidates(),0,&mut Mock::new(&[100,80,60])).unwrap();
    let file=std::fs::read_dir(dir.0.join("stack-autotune-v1")).unwrap().next().unwrap().unwrap().path(); std::fs::write(file,b"bad cache").unwrap();
    let mut p=policy(); p.mode=Mode::CacheOnly; let t=StackTuner::new(p,Some(dir.0.clone())).unwrap();
    assert_eq!(t.select(&q,&candidates(),0,&mut Mock::default()).unwrap().source,DecisionSource::CacheMiss); assert_eq!(t.stats().cache_warnings,1);
}
#[test] fn new_driver_does_not_load_old_disk_choice() {
    let dir=Dir::new(); let mut q=problem(); q.persistent=true;
    let t=StackTuner::new(policy(),Some(dir.0.clone())).unwrap(); t.select(&q,&candidates(),0,&mut Mock::new(&[100,80,60])).unwrap();
    let mut p=policy(); p.mode=Mode::CacheOnly; let t=StackTuner::new(p,Some(dir.0.clone())).unwrap(); q.environment.push_str("driver2");
    assert_eq!(t.select(&q,&candidates(),0,&mut Mock::default()).unwrap().source,DecisionSource::CacheMiss);
}

struct Blocking { inner:Mock, started:Option<mpsc::Sender<()>>, release:mpsc::Receiver<()> }
impl TrialRunner for Blocking {
    fn validate(&mut self,r:usize,c:usize,t:Tolerance)->Result<Validation,TuneFailure>{ self.inner.validate(r,c,t) }
    fn measure(&mut self,c:usize)->Result<Duration,TuneFailure>{
        if let Some(s)=self.started.take() { s.send(()).unwrap(); self.release.recv().unwrap(); } self.inner.measure(c)
    }
}
fn concurrency_case(same_device:bool, max_parallel:usize)->DecisionSource {
    let mut p=policy(); p.max_parallel_tunes=max_parallel; let t=Arc::new(StackTuner::new(p,None).unwrap());
    let (stx,srx)=mpsc::channel(); let (rtx,rrx)=mpsc::channel(); let worker=t.clone();
    let handle=std::thread::spawn(move || { let mut m=Blocking {inner:Mock::new(&[100,80,60]),started:Some(stx),release:rrx};
        worker.select(&problem(),&candidates(),0,&mut m).unwrap() });
    srx.recv().unwrap(); let mut q=problem(); q.workload.push_str("other-request"); if !same_device { q.environment.push_str("device2"); }
    let result=t.select(&q,&candidates(),0,&mut Mock::new(&[100,80,60])); rtx.send(()).unwrap(); handle.join().unwrap(); result.unwrap().source
}
#[test] fn same_device_tunes_are_single_flight_across_threads() { assert_eq!(concurrency_case(true,2),DecisionSource::Busy); }
#[test] fn global_parallel_budget_is_respected() { assert_eq!(concurrency_case(false,1),DecisionSource::Busy); }
#[test] fn independent_devices_can_tune_in_parallel() { assert_eq!(concurrency_case(false,2),DecisionSource::Tuned); }
#[test] fn nested_miss_uses_reference_without_deadlock() {
    struct Nested<'a> { tuner:&'a StackTuner, inner:Mock, entered:bool }
    impl TrialRunner for Nested<'_> {
        fn validate(&mut self,r:usize,c:usize,t:Tolerance)->Result<Validation,TuneFailure>{ self.inner.validate(r,c,t) }
        fn measure(&mut self,c:usize)->Result<Duration,TuneFailure>{
            if !self.entered { self.entered=true; let mut q=problem(); q.operation.push_str("child");
                let d=self.tuner.select(&q,&candidates(),0,&mut Mock::default()).unwrap(); assert_eq!(d.source,DecisionSource::Nested); }
            self.inner.measure(c)
        }
    }
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Nested {tuner:&t,inner:Mock::new(&[100,80,60]),entered:false};
    t.select(&problem(),&candidates(),0,&mut m).unwrap(); assert!(m.entered); assert!(!is_tuning());
}
#[test] fn host_panic_releases_tuning_permit_and_depth() {
    struct Panics;
    impl TrialRunner for Panics {
        fn validate(&mut self,_:usize,_:usize,_:Tolerance)->Result<Validation,TuneFailure>{ Ok(Validation::Passed) }
        fn measure(&mut self,_:usize)->Result<Duration,TuneFailure>{ panic!("host mock panic, no device work") }
    }
    let t=StackTuner::new(policy(),None).unwrap();
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.select(&problem(),&candidates(),0,&mut Panics)));
    assert!(result.is_err()); assert!(!is_tuning()); assert_eq!(select(&t,&mut Mock::new(&[100,80,60])).source,DecisionSource::Tuned);
}

#[test] fn invalid_reference_output_is_an_error_not_unchecked_fallback() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    m.validation.insert(0,Err(TuneFailure::rejected("non-finite reference")));
    let error=t.select(&problem(),&candidates(),0,&mut m).unwrap_err();
    assert_eq!(error.kind,FailureKind::Rejected); assert!(!m.measured.contains(&1));
}
#[test] fn reference_execution_error_is_not_replayed() {
    let t=StackTuner::new(policy(),None).unwrap(); let mut m=Mock::new(&[100,80,60]);
    m.failures.insert(0,TuneFailure::rejected("reference launch rejected"));
    assert!(t.select(&problem(),&candidates(),0,&mut m).is_err()); assert_eq!(m.measured,vec![0]);
}
