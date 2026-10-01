use super::*;
use crate::runtime::{backend::Runtime, client::ComputeClient, tune::{AutotuneKey, AutotuneOutput, TunableSet, TuneInputs, AutotuneError}};
use std::{any::type_name, path::PathBuf, string::{String, ToString}, sync::{Arc, OnceLock}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}, vec::Vec};

static GLOBAL: OnceLock<StackTuner> = OnceLock::new();
static SESSION: OnceLock<String> = OnceLock::new();
/// Install once, before worker threads/model loading. No implicit environment mutation or I/O.
pub fn enable_stack_autotune(policy: StackPolicy, cache_directory: Option<PathBuf>) -> Result<&'static StackTuner, TuneFailure> {
    let tuner = StackTuner::new(policy, cache_directory)?;
    GLOBAL.set(tuner).map_err(|_| TuneFailure::invalid("stack autotuning was already configured"))?;
    Ok(GLOBAL.get().expect("controller was initialized"))
}
pub fn stack_autotuner() -> Option<&'static StackTuner> { GLOBAL.get() }

/// Supply extra tags for a deployment's compiler flags and load/topology/power regime.
/// Set these BEFORE constructing devices/starting inference; changing them live is unsupported.
#[derive(Debug, Clone)]
pub struct RuntimeEnvironment { pub fingerprint: String, pub persistent: bool, pub execution_context: String }
pub fn runtime_environment<R: Runtime>(client: &ComputeClient<R>) -> RuntimeEnvironment {
    use std::{any::TypeId, collections::BTreeMap, sync::Mutex};
    type Key = (TypeId, u16, u16, u64);
    static ENVIRONMENTS: OnceLock<Mutex<BTreeMap<Key, RuntimeEnvironment>>> = OnceLock::new();
    let id = client.device_id();
    let key = (TypeId::of::<R>(), id.type_id, id.index_id, client.properties_fingerprint());
    let cache = ENVIRONMENTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(env) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() { return env; }
    let env = probe_environment(client);
    let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
    // Device/runtime metadata is immutable after initialization. Keep this memo bounded.
    if cache.len() < 256 { cache.insert(key, env.clone()); }
    env
}
fn probe_environment<R: Runtime>(client: &ComputeClient<R>) -> RuntimeEnvironment {
    let driver_override = std::env::var("RUDA_AUTOTUNE_DRIVER_TAG").ok().filter(|s| !s.is_empty());
    let driver = match (R::autotune_driver_fingerprint(client), driver_override) {
        (Some(probed), Some(extra)) => Some(fields(&[&probed, &extra])),
        (Some(probed), None) => Some(probed),
        (None, Some(asserted)) => Some(asserted),
        (None, None) => None,
    };
    let persistent = driver.is_some() && env!("RUDA_STACK_BUILD_ID") != "unavailable";
    let driver = driver.unwrap_or_else(|| SESSION.get_or_init(|| std::format!("session-only:{}:{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos())).clone());
    let build_extra = std::env::var("RUDA_AUTOTUNE_BUILD_TAG").unwrap_or_default();
    let execution_context = std::env::var("RUDA_AUTOTUNE_CONTEXT_TAG").unwrap_or_else(|_| "isolated-single-device".into());
    use crate::runtime::config::{RudaRuntimeConfig, RuntimeConfig};
    let config = RudaRuntimeConfig::get();
    let runtime_options = std::format!("compilation={:?};streaming={:?};memory={:?}", config.compilation, config.streaming, config.memory);
    let fingerprint = fields(&[R::name(client), &std::format!("{:?}", client.device_id()), &std::format!("{:016x}", client.properties_fingerprint()),
        &driver, type_name::<R::Compiler>(), env!("RUDA_STACK_BUILD_ID"), &build_extra,
        &std::format!("{:?}", client.info()), &runtime_options]);
    RuntimeEnvironment { fingerprint, persistent, execution_context }
}

fn complete<R: Runtime>(client: &ComputeClient<R>) -> Result<(), TuneFailure> {
    let submitted = client.flush();
    // Completion errors are never demoted to a failed candidate followed by another launch.
    ruda_core::future::block_on(client.sync()).map_err(|e| TuneFailure::device(std::format!("autotune synchronization failed: {e:?}")))?;
    submitted.map_err(|e| TuneFailure::rejected(std::format!("candidate submission failed after a successful completion fence: {e:?}")))
}
fn run_output<R: Runtime, K: AutotuneKey, I: TuneInputs, O: AutotuneOutput>(
    client: &ComputeClient<R>, set: &TunableSet<K,I,O>, index: usize, input: I::At<'_>,
) -> Result<O, TuneFailure> {
    let result = set.fastest(index).execute(input);
    complete(client)?;
    result.map_err(|e| TuneFailure::rejected(std::format!("candidate rejected: {e:?}")))
}
struct RuntimeTrials<'set, 'inp, R: Runtime, K: AutotuneKey, I: TuneInputs, O: AutotuneOutput> {
    client: ComputeClient<R>, set: &'set TunableSet<K,I,O>, key: &'set K,
    input: &'set I::At<'inp>, indices: &'set [usize], timing: Timing,
}
impl<R: Runtime, K: AutotuneKey, I: TuneInputs, O: AutotuneOutput> TrialRunner for RuntimeTrials<'_, '_, R,K,I,O> {
    fn validate(&mut self, reference: usize, candidate: usize, tolerance: Tolerance) -> Result<Validation, TuneFailure> {
        let a = self.set.generate_inputs(self.key, self.input);
        let b = self.set.generate_inputs(self.key, self.input);
        complete(&self.client)?;
        let client = self.client.clone(); let set = self.set;
        let ri = self.indices[reference]; let ci = self.indices[candidate];
        self.client.exclusive(move || {
            let _depth = super::engine::DepthGuard::enter();
            let expected = run_output(&client, set, ri, a)?;
            let actual = run_output(&client, set, ci, b)?;
            let result = expected.validate_for_tuning(&actual, tolerance.absolute, tolerance.relative, tolerance.max_bytes);
            // Readback/contiguous conversion can also launch kernels. Fence before accepting a
            // validation mismatch or freeing trial state.
            complete(&client)?;
            match result {
                Ok(true) => Ok(Validation::Passed), Ok(false) => Ok(Validation::Unsupported),
                Err(e) => Err(TuneFailure::rejected(e)),
            }
        }).map_err(|e| TuneFailure::device(std::format!("exclusive validation failed: {e:?}")))?
    }
    fn measure(&mut self, candidate: usize) -> Result<Duration, TuneFailure> {
        let input = self.set.generate_inputs(self.key, self.input);
        complete(&self.client)?;
        let client = self.client.clone(); let set = self.set;
        let index = self.indices[candidate]; let timing = self.timing;
        self.client.exclusive(move || {
            let _depth = super::engine::DepthGuard::enter();
            // Preparation/sandbox copies are outside the measured region. All allocations and
            // relayouts made INSIDE the actual candidate operation are part of EndToEnd timing.
            complete(&client)?;
            match timing {
                Timing::EndToEnd => {
                    let start = Instant::now();
                    let out = run_output(&client, set, index, input)?;
                    let elapsed = start.elapsed();
                    std::hint::black_box(&out); drop(out);
                    Ok(elapsed)
                }
                Timing::Device => {
                    let profiled = client.profile(move || set.fastest(index).execute(input), "stack-autotune");
                    complete(&client)?;
                    let (output, profile) = profiled.map_err(|e| TuneFailure::rejected(std::format!("device profile failed: {e:?}")))?;
                    let output = output.map_err(|e| TuneFailure::rejected(std::format!("candidate rejected: {e:?}")))?;
                    let elapsed = ruda_core::future::block_on(profile.resolve()).duration();
                    std::hint::black_box(&output); drop(output);
                    Ok(elapsed)
                }
            }
        }).map_err(|e| TuneFailure::device(std::format!("exclusive measurement failed: {e:?}")))?
    }
}

/// Fallible full-stack path. Actual request execution is performed ONCE after selection; an
/// execution error invalidates future choices but is never silently replayed on another kernel.
pub fn try_execute_stack<'a, R: Runtime, K: AutotuneKey, I: TuneInputs, O: AutotuneOutput>(
    name: &str, device_id: &str, client: &ComputeClient<R>, set: Arc<TunableSet<K,I,O>>, input: I::At<'a>,
) -> Result<O, AutotuneError> {
    let tuner = stack_autotuner().ok_or_else(|| autotune_error(name, "stack autotuning is not enabled"))?;
    let reference = set.stack_reference().ok_or_else(|| autotune_error(name, "no explicit reference/workload signature was registered"))?;
    if reference >= set.len() { return Err(autotune_error(name, "reference index out of range")); }
    let key = set.generate_key(&input);
    let env = runtime_environment(client);
    let problem = Problem { scope: if name.contains("fusion") { Scope::Graph } else { Scope::Operator }, operation: fields(&[name, device_id, &set.stack_checksum()]), environment: env.fingerprint,
        workload: set.stack_workload(&input).ok_or_else(|| autotune_error(name, "no exact workload signature"))?,
        execution_context: env.execution_context, persistent: env.persistent };
    // Enumerate EVERY eligible priority group. Group order remains a useful bounded-search
    // heuristic but a first valid group no longer terminates the search.
    let mut indices = std::vec![reference]; let mut plan = set.plan(&key);
    loop { let batch = plan.next(None); if batch.is_empty() { break; }
        for index in batch { if !indices.contains(&index) { indices.push(index); } }
    }
    let candidates: Vec<_> = indices.iter().map(|&i| Candidate::new(set.fastest(i).name.clone())).collect();
    // Cache hits must retain normal asynchronous execution. TrialRunner fences before every
    // validation/measurement; do NOT synchronize ordinary requests just to read this cache.
    let mut trials = RuntimeTrials { client: client.clone(), set: &set, key: &key, input: &input, indices: &indices, timing: tuner.policy().timing };
    let decision = tuner.select(&problem, &candidates, 0, &mut trials).map_err(|e| autotune_error(name, &e.to_string()))?;
    let output = set.fastest(indices[decision.index]).execute(input);
    if output.is_err() { tuner.invalidate(&decision, true); }
    output
}
fn autotune_error(name: &str, message: &str) -> AutotuneError {
    AutotuneError::Unknown { name: name.to_string(), err: message.to_string() }
}
