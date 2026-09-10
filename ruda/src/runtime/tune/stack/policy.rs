//! Shared policy and workload vocabulary. This file also builds in the standalone host harness.
use std::{fmt, string::{String, ToString}, time::Duration, vec::Vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode { Explore, CacheOnly, Disabled }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timing { Device, EndToEnd }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validation { Passed, Unsupported }

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
    /// Maximum combined reference/candidate readback bytes, not a GPU allocator quota.
    pub max_bytes: u64,
}
impl Default for Tolerance {
    fn default() -> Self { Self { absolute: 1e-4, relative: 1e-3, max_bytes: 64 << 20 } }
}

#[derive(Debug, Clone)]
pub struct StackPolicy {
    pub mode: Mode,
    pub timing: Timing,
    pub require_validation: bool,
    pub tolerance: Tolerance,
    pub warmups: usize,
    /// Odd sample count. Each non-reference sample is paired with a fresh reference measurement.
    pub samples: usize,
    pub max_candidates: usize,
    /// A soft deadline checked BETWEEN completed trials; running kernels are never interrupted.
    pub budget: Duration,
    pub min_speedup: f64,
    /// Maximum median absolute deviation of paired ratios divided by their median.
    pub max_relative_mad: f64,
    pub workspace_limit: Option<u64>,
    pub capacity: usize,
    pub max_parallel_tunes: usize,
    pub ttl: Duration,
    pub regression_pairs: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Scope { Operator = 0, Graph = 1, Pipeline = 2 }

#[derive(Debug, Clone)]
pub struct Problem {
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
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Stable algorithm/configuration name, not an index from an old binary.
    pub name: String,
    pub revision: String,
    pub workspace_bytes: Option<u64>,
    pub eligible: bool,
}
impl Candidate {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), revision: "1".into(), workspace_bytes: None, eligible: true }
    }
    pub fn fits(&self, policy: &StackPolicy) -> bool {
        self.eligible && match policy.workspace_limit {
            Some(limit) => self.workspace_bytes.is_some_and(|n| n <= limit),
            None => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    InvalidInput,
    /// A candidate/reference failed correctness or a safely completed execution.
    Rejected,
    /// Validation/timing is unavailable; only the declared reference may be used unchecked.
    Unavailable,
    Device,
    Quarantined,
}
#[derive(Debug, Clone)]
pub struct TuneFailure { pub kind: FailureKind, pub message: String }
impl TuneFailure {
    pub fn invalid(message: impl Into<String>) -> Self { Self { kind: FailureKind::InvalidInput, message: message.into() } }
    pub fn rejected(message: impl Into<String>) -> Self { Self { kind: FailureKind::Rejected, message: message.into() } }
    pub fn device(message: impl Into<String>) -> Self { Self { kind: FailureKind::Device, message: message.into() } }
}
impl fmt::Display for TuneFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}: {}", self.kind, self.message) }
}
impl std::error::Error for TuneFailure {}

/// Canonical, length-delimited encoding prevents concatenation aliases such as (ab,c)/(a,bc).
pub fn fields(parts: &[&str]) -> String {
    let mut out = String::new();
    for part in parts { out.push_str(&part.len().to_string()); out.push(':'); out.push_str(part); }
    out
}
pub fn cache_key(problem: &Problem, candidates: &[Candidate], reference: usize, policy: &StackPolicy) -> String {
    let mut manifest = Vec::new();
    for c in candidates {
        manifest.push(fields(&[&c.name, &c.revision, &std::format!("{:?}/{}", c.workspace_bytes, c.eligible)]));
    }
    fields(&["ruda-stack-autotune-v1", &std::format!("{:?}", problem.scope), &problem.operation, &problem.environment, &problem.workload,
        &problem.execution_context, &reference.to_string(), &fields(&manifest.iter().map(String::as_str).collect::<Vec<_>>()),
        &policy.accuracy_key()])
}
pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v <= 0.0) { return None; }
    values.sort_by(f64::total_cmp);
    let m = values.len() / 2;
    Some(if values.len() % 2 == 0 { values[m - 1] / 2.0 + values[m] / 2.0 } else { values[m] })
}
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
