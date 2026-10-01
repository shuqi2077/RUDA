//! Common selection engine for tensor operations, fused graphs and complete inference plans.
//! GPU code is behind TrialRunner: the built-in runtime adapter uses completed device profiles
//! or explicit synchronization, while application adapters must honor that same contract.
use super::{cache::{DiskCache, Record, now_seconds}, policy::*};
use std::{cell::Cell, collections::{BTreeMap, BTreeSet, VecDeque}, path::PathBuf,
    string::{String, ToString}, sync::{Mutex, MutexGuard}, time::{Duration, Instant}, vec::Vec};

std::thread_local! { static DEPTH: Cell<usize> = const { Cell::new(0) }; }
pub fn is_tuning() -> bool { DEPTH.with(|d| d.get() != 0) }
pub(super) struct DepthGuard;
impl DepthGuard { pub(super) fn enter() -> Self { DEPTH.with(|d| d.set(d.get()+1)); Self } }
impl Drop for DepthGuard { fn drop(&mut self) { DEPTH.with(|d| d.set(d.get()-1)); } }

/// Benchmarks MUST use isolated state. A completed measurement includes all device work whose
/// cost belongs to the plan. A submit-only host timestamp is not a valid measurement.
pub trait TrialRunner {
    fn validate(&mut self, reference: usize, candidate: usize, tolerance: Tolerance) -> Result<Validation, TuneFailure>;
    fn measure(&mut self, candidate: usize) -> Result<Duration, TuneFailure>;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionSource { Tuned, MemoryCache, DiskCache, CacheMiss, Disabled, Busy, Nested, ValidationUnavailable }
#[derive(Debug, Clone)]
pub struct Decision {
    pub index: usize,
    pub reference_index: usize,
    pub name: String,
    pub source: DecisionSource,
    /// True only after a validator reported Passed, never inferred from successful launch.
    pub verified: bool,
    pub ratio: Option<f64>,
    pub cache_key: String,
}
#[derive(Debug, Clone)]
pub struct CandidateReport {
    pub name: String,
    pub samples: usize,
    pub ratio: Option<f64>,
    pub relative_mad: Option<f64>,
    pub verified: bool,
    pub note: String,
}
#[derive(Debug, Clone)]
pub struct TuneReport {
    pub operation: String,
    pub environment: String,
    pub workload: String,
    pub execution_context: String,
    pub winner: String,
    pub elapsed: Duration,
    pub budget_exhausted: bool,
    pub candidates: Vec<CandidateReport>,
}
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub memory_hits: u64, pub disk_hits: u64, pub misses: u64,
    pub busy_fallbacks: u64, pub tunes: u64, pub cache_warnings: u64,
    pub invalidations: u64, pub device_failures: u64,
}
#[derive(Default)]
struct State {
    records: BTreeMap<String, Record>, order: VecDeque<String>,
    bypass: BTreeMap<String, Scope>,
    pending: BTreeSet<String>, devices: BTreeSet<String>, poisoned_devices: BTreeSet<String>,
    banned: BTreeMap<String, BTreeSet<String>>, regressions: BTreeMap<String, VecDeque<f64>>,
    reports: VecDeque<TuneReport>, stats: Stats,
}
/// No state mutex is held while running a candidate, validator, profile, or disk operation.
/// Contenders use the declared reference instead of waiting (also avoids nested-tuning deadlocks).
pub struct StackTuner { policy: StackPolicy, disk: Option<DiskCache>, state: Mutex<State> }
struct Permit<'a> { tuner: &'a StackTuner, key: String, device: String }
impl Drop for Permit<'_> {
    fn drop(&mut self) { let mut s = self.tuner.lock(); s.pending.remove(&self.key); s.devices.remove(&self.device); }
}
impl StackTuner {
    pub fn new(policy: StackPolicy, cache_directory: Option<PathBuf>) -> Result<Self, TuneFailure> {
        policy.validate()?;
        let disk = cache_directory.map(|dir| DiskCache::new(dir, policy.capacity));
        Ok(Self { policy, disk, state: Mutex::new(State::default()) })
    }
    fn lock(&self) -> MutexGuard<'_, State> { self.state.lock().unwrap_or_else(|e| e.into_inner()) }
    pub fn policy(&self) -> &StackPolicy { &self.policy }
    pub fn stats(&self) -> Stats { self.lock().stats.clone() }
    /// Conservative dependency snapshot for complete pipelines. It includes all currently
    /// cached lower-level decisions in this controller, including other workloads/devices.
    /// That may over-invalidate a pipeline but never hides a changed lower-level choice.
    pub fn lower_level_fingerprint(&self) -> String {
        let s = self.lock();
        let mut parts = Vec::new();
        for (key, record) in &s.records {
            if record.scope != Scope::Pipeline as u8 && record.is_fresh(now_seconds(), self.policy.ttl.as_secs().max(1)) { parts.push(fields(&[key, &record.winner])); }
        }
        // Bypassed operators retain their declared references, never an unchecked winner.
        for (key, scope) in &s.bypass {
            if *scope != Scope::Pipeline { parts.push(fields(&[key, "unverified-reference"])); }
        }
        parts.sort();
        super::cache::digest(fields(&parts.iter().map(String::as_str).collect::<Vec<_>>()).as_bytes())
    }
    pub fn reports(&self) -> Vec<TuneReport> { self.lock().reports.iter().cloned().collect() }
    fn insert(&self, record: Record) {
        let mut s = self.lock();
        if !s.order.contains(&record.key) {
            while s.order.len() >= self.policy.capacity {
                if let Some(key) = s.order.pop_front() {
                    s.records.remove(&key); s.banned.remove(&key); s.regressions.remove(&key); s.bypass.remove(&key);
                }
            }
            s.order.push_back(record.key.clone());
        }
        s.records.insert(record.key.clone(), record);
    }
    fn allowed(&self, key: &str, c: &Candidate) -> bool {
        c.fits(&self.policy) && !self.lock().banned.get(key).is_some_and(|b| b.contains(&c.name))
    }
    fn from_record(&self, r: &Record, candidates: &[Candidate], reference: usize, source: DecisionSource) -> Option<Decision> {
        if !r.is_fresh(now_seconds(), self.policy.ttl.as_secs().max(1)) || (self.policy.require_validation && !r.verified) { return None; }
        let index = candidates.iter().position(|c| c.name == r.winner && self.allowed(&r.key, c))?;
        Some(Decision { index, reference_index: reference, name: r.winner.clone(), source, verified: r.verified, ratio: Some(r.ratio), cache_key: r.key.clone() })
    }
    fn validation(&self, runner: &mut impl TrialRunner, reference: usize, candidate: usize) -> Result<bool, TuneFailure> {
        match runner.validate(reference, candidate, self.policy.tolerance)? {
            Validation::Passed => Ok(true),
            Validation::Unsupported if !self.policy.require_validation => Ok(false),
            Validation::Unsupported => Err(TuneFailure { kind: FailureKind::Unavailable, message: "no numerical validator for this output/size".into() }),
        }
    }
    fn fail_device(&self, environment: &str) {
        let mut s = self.lock(); s.poisoned_devices.insert(environment.to_string()); s.stats.device_failures += 1;
    }
    pub fn select(&self, problem: &Problem, candidates: &[Candidate], reference: usize, runner: &mut impl TrialRunner) -> Result<Decision, TuneFailure> {
        if candidates.is_empty() || candidates.len() > 4096 || reference >= candidates.len()
            || problem.operation.is_empty() || problem.environment.is_empty() || problem.workload.is_empty() {
            return Err(TuneFailure::invalid("a workload, environment and valid reference candidate are required"));
        }
        let mut names = BTreeSet::new();
        if candidates.iter().any(|c| c.name.is_empty() || c.name.len() > 4096 || !names.insert(&c.name)) {
            return Err(TuneFailure::invalid("candidate names must be nonempty, bounded and unique"));
        }
        if !candidates[reference].fits(&self.policy) { return Err(TuneFailure::invalid("reference does not satisfy eligibility/workspace policy")); }
        let key = cache_key(problem, candidates, reference, &self.policy);
        if key.len() > 384 * 1024 { return Err(TuneFailure::invalid("autotune key exceeds 384 KiB")); }
        let fallback = |source| Decision { index: reference, reference_index: reference, name: candidates[reference].name.clone(), source, verified: false, ratio: None, cache_key: key.clone() };
        if self.lock().poisoned_devices.contains(&problem.environment) {
            return Err(TuneFailure { kind: FailureKind::Quarantined, message: "device tuning lane is quarantined after an unconfirmed device completion".into() });
        }
        if self.policy.mode == Mode::Disabled { return Ok(fallback(DecisionSource::Disabled)); }
        let memory = { self.lock().records.get(&key).cloned() };
        if let Some(record) = memory {
            if let Some(d) = self.from_record(&record, candidates, reference, DecisionSource::MemoryCache) {
                self.lock().stats.memory_hits += 1; return Ok(d);
            }
        }
        if self.lock().bypass.contains_key(&key) { return Ok(fallback(DecisionSource::ValidationUnavailable)); }
        if is_tuning() { return Ok(fallback(DecisionSource::Nested)); }
        let _permit = {
            let mut s = self.lock();
            if s.pending.contains(&key) || s.devices.contains(&problem.environment) || s.pending.len() >= self.policy.max_parallel_tunes {
                s.stats.busy_fallbacks += 1; return Ok(fallback(DecisionSource::Busy));
            }
            s.pending.insert(key.clone()); s.devices.insert(problem.environment.clone()); s.stats.misses += 1;
            Permit { tuner: self, key: key.clone(), device: problem.environment.clone() }
        };
        let _depth = DepthGuard::enter();
        if problem.persistent {
            if let Some(disk) = &self.disk {
                match disk.load(&key) {
                    Ok(Some(record)) => {
                        if let Some(mut decision) = self.from_record(&record, candidates, reference, DecisionSource::DiskCache).filter(|_| record.scope == problem.scope as u8) {
                            // A disk record is not trusted as evidence for the current inputs until
                            // the candidate is checked once in this process. CacheOnly still validates.
                            match self.validation(runner, reference, decision.index) {
                                Ok(checked) => {
                                    decision.verified = checked;
                                    let mut record = record; record.verified = checked;
                                    self.insert(record); self.lock().stats.disk_hits += 1; return Ok(decision);
                                }
                                Err(e) if e.kind == FailureKind::Device => { self.fail_device(&problem.environment); return Err(e); }
                                Err(_) => { if disk.remove(&key).is_err() { self.lock().stats.cache_warnings += 1; } }
                            }
                        }
                    }
                    Err(_) => self.lock().stats.cache_warnings += 1,
                    Ok(None) => {}
                }
            }
        }
        if self.policy.mode == Mode::CacheOnly { return Ok(fallback(DecisionSource::CacheMiss)); }
        match self.explore(problem, candidates, reference, &key, runner) {
            Ok((decision, report, record)) => {
                if problem.persistent {
                    if let Some(disk) = &self.disk { if disk.save(&record).is_err() { self.lock().stats.cache_warnings += 1; } }
                }
                self.insert(record);
                let mut s = self.lock(); s.stats.tunes += 1;
                if s.reports.len() >= self.policy.capacity.min(128) { s.reports.pop_front(); }
                s.reports.push_back(report);
                Ok(decision)
            }
            Err(e) if e.kind == FailureKind::Unavailable => {
                // Reference benchmarking may exceed the validation/workspace budget. Preserve
                // normal execution on the declared reference; never promote an unchecked winner.
                let mut s = self.lock();
                if !s.order.contains(&key) { s.order.push_back(key.clone()); }
                s.bypass.insert(key.clone(), problem.scope);
                while s.order.len() > self.policy.capacity {
                    if let Some(old) = s.order.pop_front() { s.records.remove(&old); s.banned.remove(&old); s.regressions.remove(&old); s.bypass.remove(&old); }
                }
                if s.reports.len() >= self.policy.capacity.min(128) { s.reports.pop_front(); }
                s.reports.push_back(TuneReport {
                    operation: problem.operation.clone(), environment: problem.environment.clone(),
                    workload: problem.workload.clone(), execution_context: problem.execution_context.clone(),
                    winner: candidates[reference].name.clone(), elapsed: Duration::ZERO, budget_exhausted: false,
                    candidates: std::vec![CandidateReport { name: candidates[reference].name.clone(), samples: 0,
                        ratio: None, relative_mad: None, verified: false, note: e.message }],
                });
                Ok(fallback(DecisionSource::ValidationUnavailable))
            }
            Err(e) => { if e.kind == FailureKind::Device { self.fail_device(&problem.environment); } Err(e) }
        }
    }
    fn explore(&self, problem: &Problem, candidates: &[Candidate], reference: usize, key: &str, runner: &mut impl TrialRunner) -> Result<(Decision, TuneReport, Record), TuneFailure> {
        let started = Instant::now();
        // The mandatory reference is validated even if its compilation takes the whole budget.
        // Never treat a failed reference as permission to choose an unverified fast candidate.
        for _ in 0..self.policy.warmups { runner.measure(reference)?; }
        let reference_checked = self.validation(runner, reference, reference)?;
        let reference_time = runner.measure(reference)?;
        if reference_time.is_zero() { return Err(TuneFailure { kind: FailureKind::Unavailable, message: "reference has zero elapsed time".into() }); }
        let mut winner = reference; let mut winner_ratio = 1.0; let mut winner_time = reference_time;
        let mut winner_reference = reference_time; let mut winner_checked = reference_checked;
        let mut reports = Vec::new();
        reports.push(CandidateReport { name: candidates[reference].name.clone(), samples: 1, ratio: Some(1.0), relative_mad: Some(0.0), verified: reference_checked, note: "reference; retained unless a stable measured improvement qualifies".into() });
        let mut attempted = 1; let mut exhausted = false;
        for (index, candidate) in candidates.iter().enumerate() {
            if index == reference { continue; }
            let mut report = CandidateReport { name: candidate.name.clone(), samples: 0, ratio: None, relative_mad: None, verified: false, note: String::new() };
            if !self.allowed(key, candidate) { report.note = "ineligible, unknown/over-budget workspace, or regressed candidate".into(); reports.push(report); continue; }
            if attempted >= self.policy.max_candidates || started.elapsed() >= self.policy.budget {
                exhausted = true; report.note = "search budget exhausted".into(); reports.push(report); continue;
            }
            attempted += 1;
            let result = (|| -> Result<(bool, Vec<(Duration, Duration)>), TuneFailure> {
                for _ in 0..self.policy.warmups {
                    if started.elapsed() >= self.policy.budget { return Err(TuneFailure::rejected("budget exhausted during warmup")); }
                    runner.measure(index)?;
                }
                let checked = self.validation(runner, reference, index)?;
                let mut pairs = Vec::with_capacity(self.policy.samples);
                for sample in 0..self.policy.samples {
                    if started.elapsed() >= self.policy.budget { return Err(TuneFailure::rejected("budget exhausted before enough paired samples")); }
                    // Alternate order to reduce monotonic warmup/clock/thermal bias.
                    let pair = if (sample + index) % 2 == 0 {
                        let a = runner.measure(reference)?; let b = runner.measure(index)?; (a,b)
                    } else {
                        let b = runner.measure(index)?; let a = runner.measure(reference)?; (a,b)
                    };
                    if pair.0.is_zero() || pair.1.is_zero() { return Err(TuneFailure::rejected("zero duration cannot be scored")); }
                    pairs.push(pair);
                }
                Ok((checked, pairs))
            })();
            match result {
                Err(e) if e.kind == FailureKind::Device => return Err(e),
                Err(e) => { report.note = e.message; },
                Ok((checked, pairs)) => {
                    report.verified = checked; report.samples = pairs.len();
                    match paired_score(&pairs, &self.policy) {
                        None => { report.note = "timings too noisy or incomplete".into(); },
                        Some((ratio, mad)) => {
                            report.ratio = Some(ratio); report.relative_mad = Some(mad);
                            report.note = "paired synchronized measurements".into();
                            if ratio < winner_ratio && ratio * self.policy.min_speedup <= 1.0 {
                                winner = index; winner_ratio = ratio; winner_checked = checked;
                                let mut a: Vec<_> = pairs.iter().map(|p| p.0.as_secs_f64()).collect();
                                let mut b: Vec<_> = pairs.iter().map(|p| p.1.as_secs_f64()).collect();
                                winner_reference = Duration::from_secs_f64(median(&mut a).unwrap());
                                winner_time = Duration::from_secs_f64(median(&mut b).unwrap());
                            }
                        }
                    }
                }
            }
            reports.push(report);
        }
        exhausted |= started.elapsed() >= self.policy.budget;
        let record = Record { scope: problem.scope as u8, key: key.to_string(), winner: candidates[winner].name.clone(), created: now_seconds(),
            reference_ns: winner_reference.as_nanos().min(u64::MAX as u128) as u64,
            winner_ns: winner_time.as_nanos().min(u64::MAX as u128) as u64, ratio: winner_ratio, verified: winner_checked };
        let decision = Decision { index: winner, reference_index: reference, name: record.winner.clone(), source: DecisionSource::Tuned, verified: winner_checked, ratio: Some(winner_ratio), cache_key: key.to_string() };
        let report = TuneReport { operation: problem.operation.clone(), environment: problem.environment.clone(), workload: problem.workload.clone(), execution_context: problem.execution_context.clone(), winner: record.winner.clone(), elapsed: started.elapsed(), budget_exhausted: exhausted, candidates: reports };
        Ok((decision, report, record))
    }
    /// Invalidate future selections; NEVER re-run the current stateful request after a failure.
    pub fn invalidate(&self, decision: &Decision, ban_candidate: bool) {
        {
            let mut s = self.lock(); s.records.remove(&decision.cache_key); s.regressions.remove(&decision.cache_key); s.bypass.remove(&decision.cache_key);
            if ban_candidate && decision.index != decision.reference_index {
                s.banned.entry(decision.cache_key.clone()).or_default().insert(decision.name.clone());
            }
            // Retain one bounded FIFO slot while the key is banned.
            if !s.order.contains(&decision.cache_key) { s.order.push_back(decision.cache_key.clone()); }
            while s.order.len() > self.policy.capacity {
                if let Some(old) = s.order.pop_front() { s.records.remove(&old); s.banned.remove(&old); s.regressions.remove(&old); s.bypass.remove(&old); }
            }
            s.stats.invalidations += 1;
        }
        if let Some(disk) = &self.disk { if disk.remove(&decision.cache_key).is_err() { self.lock().stats.cache_warnings += 1; } }
    }
    /// Feed paired observations from a controlled replay of the SAME workload and load regime.
    /// Ordinary production request latency is NOT a comparable baseline measurement.
    pub fn record_comparison(&self, decision: &Decision, reference: Duration, selected: Duration, correctness_checked: bool) -> Result<bool, TuneFailure> {
        if !correctness_checked || reference.is_zero() || selected.is_zero() { return Err(TuneFailure::invalid("regression observations require nonzero, correctness-checked paired timings")); }
        let should_invalidate = {
            let mut s = self.lock();
            if !s.records.get(&decision.cache_key).is_some_and(|r| r.winner == decision.name) { return Ok(false); }
            let history = s.regressions.entry(decision.cache_key.clone()).or_default();
            if history.len() == self.policy.regression_pairs { history.pop_front(); }
            history.push_back(selected.as_secs_f64()/reference.as_secs_f64());
            if history.len() < self.policy.regression_pairs { false } else {
                let mut ratios: Vec<_> = history.iter().copied().collect(); median(&mut ratios).is_some_and(|r| r > self.policy.regression_ratio)
            }
        };
        if should_invalidate { self.invalidate(decision, true); }
        Ok(should_invalidate)
    }
}
