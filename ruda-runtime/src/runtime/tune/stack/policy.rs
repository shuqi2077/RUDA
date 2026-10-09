//! Shared policy and workload vocabulary. This file also builds in the standalone host harness.
use std::{fmt, string::{String, ToString}, time::Duration, vec::Vec};

/// Selection policy for adapters participating in the shared controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Reuse valid choices, otherwise validate and time eligible candidates.
    Explore,
    /// Reuse choices without fresh timing search; misses use the reference.
    /// Cold disk hits still require validation in the current process.
    CacheOnly,
    /// Use the declared reference without this controller's cache or trials.
    /// This does not restore the legacy LocalTuner route.
    Disabled
}
/// Completed-work timing requested from the trial adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timing {
    /// Use device profiling facilities and wait for the measured work to finish.
    Device,
    /// Include candidate-internal allocation, layout conversion, submission and waiting.
    EndToEnd
}
/// Whether the adapter can establish numerical agreement for this trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validation {
    /// The reference/candidate comparison completed and met the tolerance.
    Passed,
    /// No validator supports this output or its readback size.
    /// A numerical mismatch is an error, not this outcome.
    Unsupported
}

/// Absolute/relative comparison limits and bounded host readback size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Nonnegative finite absolute error limit; default `1e-4`.
    pub absolute: f64,
    /// Nonnegative finite relative error limit; default `1e-3`.
    pub relative: f64,
    /// Maximum combined reference/candidate readback bytes, not a GPU allocator quota.
    pub max_bytes: u64,
}
impl Default for Tolerance {
    fn default() -> Self { Self { absolute: 1e-4, relative: 1e-3, max_bytes: 64 << 20 } }
}

/// Immutable limits shared by operator, graph and application selection.
/// Defaults are selection rules, not measured performance guarantees.
#[derive(Debug, Clone)]
pub struct StackPolicy {
    /// Explore, cache-only or reference-only operation; defaults to Explore.
    pub mode: Mode,
    /// Completed timing scope; defaults to EndToEnd.
    pub timing: Timing,
    /// Require numerical validation before promoting a choice; defaults to true.
    pub require_validation: bool,
    /// Numerical limits and combined reference/candidate readback quota.
    pub tolerance: Tolerance,
    /// Positive warmup count in `1..=100`; default 2.
    pub warmups: usize,
    /// Odd sample count. Each non-reference sample is paired with a fresh reference measurement.
    pub samples: usize,
    /// Positive candidate trial limit up to 4096; default 32.
    pub max_candidates: usize,
    /// A soft deadline checked BETWEEN completed trials; running kernels are never interrupted.
    pub budget: Duration,
    /// Required reciprocal median timing ratio for promotion; finite and at least 1.
    /// Default 1.05 is a threshold, not an observed speedup.
    pub min_speedup: f64,
    /// Maximum median absolute deviation of paired ratios divided by their median.
    pub max_relative_mad: f64,
    /// Optional candidate scratch-byte ceiling. Unknown estimates do not fit
    /// an enabled limit, including for the declared reference. Default None.
    pub workspace_limit: Option<u64>,
    /// Maximum retained keys in `1..=65536`; default 1024.
    pub capacity: usize,
    /// Maximum simultaneous trials in `1..=256`; default 1, also per-device serialized.
    pub max_parallel_tunes: usize,
    /// Nonzero cache lifetime; default seven days. Future-dated entries expire.
    pub ttl: Duration,
    /// Odd nonzero regression window in `3..=101`; default 7.
    pub regression_pairs: usize,
    /// Median selected/reference threshold above 1; default 1.15.
    pub regression_ratio: f64,
}
impl Default for StackPolicy {
    fn default() -> Self {
        Self {
            mode: Mode::Explore, timing: Timing::EndToEnd, require_validation: true,
            tolerance: Tolerance::default(), warmups: 2, samples: 7,
            max_candidates: 32, budget: Duration::from_secs(30), min_speedup: 1.05,
            max_relative_mad: 0.15, workspace_limit: None, capacity: 1024,
            max_parallel_tunes: 1, ttl: Duration::from_secs(7 * 24 * 3600),
            regression_pairs: 7, regression_ratio: 1.15,
        }
    }
}
impl StackPolicy {
    /// Reject invalid counts, bounds, nonfinite numerical limits or zero durations.
    /// Returns `FailureKind::InvalidInput` without starting a trial or touching disk.
    pub fn validate(&self) -> Result<(), TuneFailure> {
        if self.warmups == 0 || self.warmups > 100 || self.samples < 3
            || self.samples > 101 || self.samples % 2 == 0
            || self.max_candidates == 0 || self.max_candidates > 4096
            || self.budget.is_zero() || self.capacity == 0 || self.capacity > 65536
            || self.max_parallel_tunes == 0 || self.max_parallel_tunes > 256
            || self.ttl.is_zero() || self.regression_pairs < 3 || self.regression_pairs > 101
            || self.regression_pairs % 2 == 0
            || !self.min_speedup.is_finite() || self.min_speedup < 1.0
            || !self.max_relative_mad.is_finite() || self.max_relative_mad < 0.0
            || !self.regression_ratio.is_finite() || self.regression_ratio <= 1.0
            || !self.tolerance.absolute.is_finite() || self.tolerance.absolute < 0.0
            || !self.tolerance.relative.is_finite() || self.tolerance.relative < 0.0
            || self.tolerance.max_bytes == 0 {
            return Err(TuneFailure::invalid("invalid stack autotune policy"));
        }
        Ok(())
    }
    /// Mode is intentionally excluded: an offline Explore cache is usable by CacheOnly.
    pub fn accuracy_key(&self) -> String {
        std::format!("timing={:?};checked={};atol={:016x};rtol={:016x};workspace={:?};min_speedup={:016x};mad={:016x}",
            self.timing, self.require_validation, self.tolerance.absolute.to_bits(),
            self.tolerance.relative.to_bits(), self.workspace_limit,
            self.min_speedup.to_bits(), self.max_relative_mad.to_bits())
    }
}

/// Workload level included in cache identity and dependency tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Scope {
    /// One complete operator, including its internal setup costs.
    Operator = 0,
    /// A fused graph with all externally visible outputs.
    Graph = 1,
    /// A complete application plan, such as greedy generation.
    Pipeline = 2
}

/// Exact operation/environment signature; shapes are not bucketed or guessed.
#[derive(Debug, Clone)]
pub struct Problem {
    /// Operator, graph or pipeline level.
    pub scope: Scope,
    /// Stable library/graph/model operation identity and semantic revision.
    pub operation: String,
    /// Backend + physical device + driver/runtime + compiler/build + capabilities.
    pub environment: String,
    /// Exact dtype, shape, strides, precision and operation parameters; no lossy bucketing.
    pub workload: String,
    /// Caller supplied load/topology/batching/power regime, not inferred from device id.
    pub execution_context: String,
    /// False when a reliable driver/build identity cannot be established.
    pub persistent: bool,
}
/// Stable candidate identity and its declared eligibility/scratch requirement.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Stable algorithm/configuration name, not an index from an old binary.
    pub name: String,
    /// Semantic/configuration revision invalidating old choices; default `"1"`.
    pub revision: String,
    /// Known temporary workspace bytes, or None when no reliable estimate exists.
    pub workspace_bytes: Option<u64>,
    /// Whether the existing operator eligibility rules admit this candidate.
    pub eligible: bool,
}
impl Candidate {
    /// Create an eligible revision-1 candidate without a workspace estimate.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let view = CandidateView::new(&name);
        let revision = view.revision.into();
        let workspace_bytes = view.workspace_bytes;
        let eligible = view.eligible;
        Self { name, revision, workspace_bytes, eligible }
    }
    /// Check eligibility and the optional workspace ceiling, without executing code.
    pub fn fits(&self, policy: &StackPolicy) -> bool {
        self.view().fits(policy)
    }
}

#[derive(Clone, Copy)]
pub(super) struct CandidateView<'a> {
    pub(super) name: &'a String,
    pub(super) revision: &'a str,
    pub(super) workspace_bytes: Option<u64>,
    pub(super) eligible: bool,
}

impl<'a> CandidateView<'a> {
    pub(super) fn new(name: &'a String) -> Self {
        Self { name, revision: "1", workspace_bytes: None, eligible: true }
    }

    pub(super) fn fits(&self, policy: &StackPolicy) -> bool {
        self.eligible && match policy.workspace_limit {
            Some(limit) => self.workspace_bytes.is_some_and(|n| n <= limit),
            None => true,
        }
    }
}

pub(super) trait CandidateSource {
    fn view(&self) -> CandidateView<'_>;
}

impl CandidateSource for Candidate {
    fn view(&self) -> CandidateView<'_> {
        CandidateView { name: &self.name, revision: &self.revision,
            workspace_bytes: self.workspace_bytes, eligible: self.eligible }
    }
}

impl CandidateSource for CandidateView<'_> {
    fn view(&self) -> CandidateView<'_> { *self }
}

/// Failure classification used to distinguish rejected trials from device faults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// Invalid policy, workload or trial metadata.
    InvalidInput,
    /// A candidate/reference failed correctness or a safely completed execution.
    Rejected,
    /// Validation/timing is unavailable; only the declared reference may be used unchecked.
    Unavailable,
    /// Device work could not be confirmed complete; faults the tuning lane.
    Device,
    /// This environment's lane was previously faulted and rejects further selection.
    Quarantined,
}
/// Structured selection failure; Display combines the kind and explanatory message.
#[derive(Debug, Clone)]
pub struct TuneFailure {
    /// Failure category controlling adapter error handling.
    pub kind: FailureKind,
    /// Human-readable failure reason.
    pub message: String
}
impl TuneFailure {
    /// Report an invalid caller-supplied policy or signature.
    pub fn invalid(message: impl Into<String>) -> Self { Self { kind: FailureKind::InvalidInput, message: message.into() } }
    /// Report incorrect output or a trial rejected after confirmed completion.
    pub fn rejected(message: impl Into<String>) -> Self { Self { kind: FailureKind::Rejected, message: message.into() } }
    /// Report a device completion failure, not an ordinary unsupported candidate.
    pub fn device(message: impl Into<String>) -> Self { Self { kind: FailureKind::Device, message: message.into() } }
}
impl fmt::Display for TuneFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}: {}", self.kind, self.message) }
}
impl std::error::Error for TuneFailure {}

/// Canonical, length-delimited encoding prevents concatenation aliases such as (ab,c)/(a,bc).
pub fn fields(parts: &[&str]) -> String {
    let mut out = String::new();
    for part in parts { append_field(&mut out, part); }
    out
}

fn append_field(out: &mut String, part: &str) {
    use std::fmt::Write;
    let _ = write!(out, "{}:", part.len());
    out.push_str(part);
}

fn append_fields(out: &mut String, parts: &[&str]) {
    use std::fmt::Write;
    let length = parts.iter().map(|part| {
        part.len() + 1 + part.len().checked_ilog10().map_or(1, |digits| digits as usize + 1)
    }).sum::<usize>();
    let _ = write!(out, "{length}:");
    for part in parts { append_field(out, part); }
}
/// Encode the full workload, ordered candidate manifest, reference and numerical policy.
/// This returns identity text, not a cryptographic hash or proof of correctness.
pub fn cache_key(problem: &Problem, candidates: &[Candidate], reference: usize, policy: &StackPolicy) -> String {
    cache_key_candidates(problem, candidates, reference, policy)
}

pub(super) fn cache_key_candidates<C: CandidateSource>(problem: &Problem, candidates: &[C], reference: usize, policy: &StackPolicy) -> String {
    let mut manifest = String::new();
    for c in candidates {
        let c = c.view();
        append_fields(&mut manifest, &[c.name, c.revision, &std::format!("{:?}/{}", c.workspace_bytes, c.eligible)]);
    }
    fields(&["ruda-stack-autotune-v1", &std::format!("{:?}", problem.scope), &problem.operation, &problem.environment, &problem.workload,
        &problem.execution_context, &reference.to_string(), &manifest,
        &policy.accuracy_key()])
}
/// Sort positive finite ratios in place and return their median.
/// Empty, zero, negative or nonfinite input returns None.
pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v <= 0.0) { return None; }
    values.sort_by(f64::total_cmp);
    let m = values.len() / 2;
    Some(if values.len() % 2 == 0 { values[m - 1] / 2.0 + values[m] / 2.0 } else { values[m] })
}
/// Return `(median_selected_over_reference, relative_mad)` for completed pairs.
/// Requires at least `policy.samples` nonzero pairs and acceptable relative MAD;
/// an unavailable/noisy score returns None, not a statistically proven speedup.
pub fn paired_score(pairs: &[(Duration, Duration)], policy: &StackPolicy) -> Option<(f64, f64)> {
    if pairs.len() < policy.samples || pairs.iter().any(|(a,b)| a.is_zero() || b.is_zero()) { return None; }
    let mut ratios: Vec<_> = pairs.iter().map(|(a,b)| b.as_secs_f64()/a.as_secs_f64()).collect();
    let center = median(&mut ratios)?;
    let mut deviations: Vec<_> = ratios.iter().map(|r| (r-center).abs()).collect();
    // Unlike timing ratios, zero deviations are valid and desirable.
    deviations.sort_by(f64::total_cmp);
    let mad = deviations[deviations.len()/2] / center;
    if !mad.is_finite() || mad > policy.max_relative_mad { None } else { Some((center, mad)) }
}
