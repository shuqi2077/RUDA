//! Offline/startup calibration of COMPLETE packed generation plans.
//!
//! Call once for a representative workload, then apply the returned KernelMode to normal
//! generation. This function intentionally replays requests; it is not a per-token hot path.
use crate::{GreedyGenerationConfig, LlamaConfig, PackedLlamaForCausalLm, TokenGenerationOutput,
    GenerationError, generate_greedy_packed_with_mode, runtime::KernelMode};
use ruda::runtime::{server::ComputeServer, tune::stack::{self, Candidate, Decision, Mode, Problem,
    Scope, StackTuner, Timing, Tolerance, TrialRunner, TuneFailure, Validation}};
use ruda_tensor::{DeviceOps, api::backend::Backend};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use std::{any::type_name, time::{Duration, Instant}};

#[derive(Debug, Clone)]
pub struct GenerationPlan {
    pub mode: KernelMode,
    pub decision: Decision,
    /// Recalibrate after lower-level kernel decisions change. The snapshot is conservative:
    /// unrelated new workloads in the same controller can also invalidate it.
    pub lower_level_fingerprint: String,
}
impl GenerationPlan {
    pub fn lower_levels_unchanged(&self, tuner: &StackTuner) -> bool {
        self.lower_level_fingerprint == tuner.lower_level_fingerprint()
    }
}
struct PackedTrials<'a, R, F, I, BT>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    model: &'a PackedLlamaForCausalLm<DeviceBackend<R,F,I,BT>>,
    config: &'a LlamaConfig, prompt: &'a [i32], generation: &'a GreedyGenerationConfig,
    device: &'a R::Device,
}
impl<R,F,I,BT> PackedTrials<'_,R,F,I,BT>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    fn replay(&self, index: usize) -> Result<TokenGenerationOutput, TuneFailure> {
        let mode = match index { 0 => KernelMode::Portable, 1 => KernelMode::Automatic,
            _ => return Err(TuneFailure::invalid("unknown packed generation plan")) };
        let result = generate_greedy_packed_with_mode(self.model, self.config, self.prompt,
            self.generation, self.device, mode);
        // Generation creates a fresh KV cache for EVERY replay and synchronizes on success.
        // Fence again on errors before deciding whether another isolated trial is safe.
        DeviceBackend::<R,F,I,BT>::sync(self.device)
            .map_err(|e| TuneFailure::device(format!("generation replay did not complete: {e:?}")))?;
        result.map_err(|e| TuneFailure::rejected(e.to_string()))
    }
}
impl<R,F,I,BT> TrialRunner for PackedTrials<'_,R,F,I,BT>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    fn validate(&mut self, reference: usize, candidate: usize, _tolerance: Tolerance) -> Result<Validation, TuneFailure> {
        let expected = self.replay(reference)?;
        let actual = self.replay(candidate)?;
        if expected != actual {
            return Err(TuneFailure::rejected("complete generation plans disagree on token ids or stopping behavior"));
        }
        // This is exact OUTPUT-TOKEN equivalence on the calibration workload, not a proof that
        // all model logits equal a high precision oracle for all possible prompts.
        Ok(Validation::Passed)
    }
    fn measure(&mut self, index: usize) -> Result<Duration, TuneFailure> {
        DeviceBackend::<R,F,I,BT>::sync(self.device)
            .map_err(|e| TuneFailure::device(format!("pre-replay synchronization failed: {e:?}")))?;
        let start = Instant::now();
        let output = self.replay(index)?;
        let duration = start.elapsed(); std::hint::black_box(output);
        Ok(duration)
    }
}

/// Calibrate portable vs capability-gated packed generation using the shared stack controller.
/// `model_revision` must identify immutable WEIGHTS (e.g. a content digest), not just a path.
/// Calibration includes warmup replays even in CacheOnly; that mode skips timed search, not
/// explicitly requested model preparation/validation. Do not call this function per request.
#[allow(clippy::too_many_arguments)]
pub fn calibrate_greedy_packed<R,F,I,BT>(
    tuner: &StackTuner, model: &PackedLlamaForCausalLm<DeviceBackend<R,F,I,BT>>,
    config: &LlamaConfig, model_revision: &str, prompt: &[i32],
    generation: &GreedyGenerationConfig, device: &R::Device,
) -> Result<GenerationPlan, GenerationError>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    if model_revision.is_empty() || prompt.is_empty() || generation.max_new_tokens == 0 {
        return Err(GenerationError("calibration requires an immutable model revision, prompt and nonzero generation length".into()));
    }
    if tuner.policy().timing != Timing::EndToEnd {
        return Err(GenerationError("complete generation calibration requires EndToEnd timing".into()));
    }
    // One controller must govern the lower layers and the enclosing model plan.
    if !stack::stack_autotuner().is_some_and(|global| std::ptr::eq(global, tuner)) {
        return Err(GenerationError("use the controller returned by enable_stack_autotune so lower-level decisions are shared".into()));
    }
    let client = R::client(device);
    let env = stack::runtime_environment(&client);
    let mut trials = PackedTrials { model, config, prompt, generation, device };
    let mut candidates = vec![Candidate::new("portable-packed-v1"), Candidate::new("capability-packed-v1")];
    // Warm lower-level tuners OUTSIDE the enclosing tuning scope, then freeze child exploration
    // while timing full plans. Unknown device failures abort; only safely completed rejections
    // disqualify the optimized plan.
    if tuner.policy().mode != Mode::Disabled {
        trials.replay(0).map_err(|e| GenerationError(e.to_string()))?;
        if let Err(error) = trials.replay(1) {
            if error.kind == stack::FailureKind::Device { return Err(GenerationError(error.to_string())); }
            candidates[1].eligible = false;
        }
    }
    let dependency = tuner.lower_level_fingerprint();
    let prompt_bytes: Vec<_> = prompt.iter().flat_map(|token| token.to_le_bytes()).collect();
    let problem = Problem {
        scope: Scope::Pipeline, operation: "rullm-packed-greedy-plan-v1".into(), environment: env.fingerprint,
        workload: stack::fields(&[model_revision, &format!("{config:?}"), type_name::<F>(), type_name::<I>(), type_name::<BT>(),
            &prompt.len().to_string(), &stack::workload_digest(&prompt_bytes), &format!("{generation:?}")]),
        execution_context: stack::fields(&[&env.execution_context, &dependency]), persistent: env.persistent,
    };
    let decision = tuner.select(&problem, &candidates, 0, &mut trials).map_err(|e| GenerationError(e.to_string()))?;
    Ok(GenerationPlan { mode: if decision.index == 0 { KernelMode::Portable } else { KernelMode::Automatic },
        decision, lower_level_fingerprint: dependency })
}
