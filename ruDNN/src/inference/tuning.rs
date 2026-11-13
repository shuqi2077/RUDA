//! Inference uses the existing RUDA kernel/graph autotuner, not a second JIT
//! implementation or a vendor tactic cache. Creating a session launches nothing.
use ruda_kernel::dsl::tune::stack::StackTuner;
pub use ruda_kernel::dsl::tune::stack::{
    Candidate, Decision, Mode, Problem, StackPolicy as MeasurementPolicy, Stats, Timing,
    TrialRunner, TuneFailure, TuneReport,
};
use std::path::PathBuf;
/// Bounded native lookup/search diagnostics. Reports describe selection, not
/// observed GPU execution, and counters are read without launching device work.
#[derive(Clone, Debug)]
pub struct ReplayCoverage {
    pub hits: u64,
    pub misses: u64,
    pub skipped: u64,
    pub dropped_records: usize,
    pub records: Vec<TuneReport>,
}
pub struct Session {
    pub tuner: StackTuner,
    policy: MeasurementPolicy,
    root: Option<PathBuf>,
}
impl Session {
    pub fn new(policy: MeasurementPolicy, root: Option<PathBuf>) -> Result<Self, TuneFailure> {
        let tuner = StackTuner::new(policy.clone(), root.clone())?;
        Ok(Self {
            tuner,
            policy,
            root,
        })
    }
    pub fn select(
        &self,
        problem: &Problem,
        candidates: &[Candidate],
        reference: usize,
        runner: &mut impl TrialRunner,
    ) -> Result<Decision, TuneFailure> {
        self.tuner.select(problem, candidates, reference, runner)
    }
    pub fn coverage(&self, max_records: usize) -> ReplayCoverage {
        let stats = self.tuner.stats();
        let mut records = self.tuner.reports();
        let dropped_records = records.len().saturating_sub(max_records);
        records.truncate(max_records);
        ReplayCoverage {
            hits: stats.memory_hits + stats.disk_hits,
            misses: stats.misses,
            skipped: stats.busy_fallbacks,
            dropped_records,
            records,
        }
    }
    /// After the application's cross-rank barrier, discard in-process winners
    /// and reread managed persistent records on subsequent selections.
    pub fn reload(&mut self) -> Result<(), TuneFailure> {
        self.tuner = StackTuner::new(self.policy.clone(), self.root.clone())?;
        Ok(())
    }
}
pub fn autotune(policy: MeasurementPolicy, root: Option<PathBuf>) -> Result<Session, TuneFailure> {
    Session::new(policy, root)
}
pub fn autotune_v2(
    policy: MeasurementPolicy,
    root: Option<PathBuf>,
) -> Result<Session, TuneFailure> {
    Session::new(policy, root)
}
pub fn autotune_v2_reload(session: &mut Session) -> Result<(), TuneFailure> {
    session.reload()
}
