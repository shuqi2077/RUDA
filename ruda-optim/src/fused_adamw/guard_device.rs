// SPDX-License-Identifier: Apache-2.0
use super::{
    AdamWOptions, AdamWState, AdamWUpdate, FusedAdamWError, StepControl,
    device::{adamw_step_scaled, validate_step, validate_tensor},
    gradient_norm::{GradientGuardError, GradientGuardOptions, GradientStats},
    guard_kernel,
};
use ruda_core::{device::Device, future::block_on, tensor::DType};
use ruda_kernel::{
    dsl::{Runtime, prelude::{RudaCount, RudaDim}, server::CopyDescriptor},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype},
};

const THREADS: u32 = 256;
const MAX_BLOCKS: usize = 1024;
const ITEMS_PER_BLOCK: usize = 1024;
const SHARED_BYTES: usize = THREADS as usize * 12;

/// Statistics plus source-level execution accounting (not memory-controller counters).
#[derive(Debug, Clone, PartialEq)]
pub struct GradientStatsReport {
    /// The norm covers all entries in this local group, after FP32 unscale.
    pub stats: GradientStats,
    /// Number of reduction kernels submitted, excluding runtime copies.
    pub reduction_launches: usize,
    /// Logical host payload bytes (12 per nonempty tensor), not bus traffic.
    pub readback_bytes: usize,
    /// Sum of per-tensor peak logical scratch bytes, not allocator peak usage.
    pub scratch_bytes: usize,
}

/// A read-only parameter/gradient/state entry. Include tied parameters only once.
/// All entries must use one device/queue. Different shapes and gradient dtypes
/// are allowed across entries, not between a parameter and its own gradient.
pub struct AdamWEntry<'a, R: Runtime> {
    /// FP32 master parameters; never modified in place.
    pub parameters: &'a RudaTensor<R>,
    /// Accumulated, scaled F32/F16/BF16 gradient.
    pub gradients: &'a RudaTensor<R>,
    /// Prior moments; absent for the first update of this entry.
    pub state: Option<&'a AdamWState<R>>,
}

/// Reason no optimizer kernel was submitted for the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// `StepControl::skip_update` was supplied by the caller; no stats pass.
    Requested,
    /// No selected elements; no stats pass.
    Empty,
    /// At least one raw/unscaled gradient was NaN or Inf.
    NonFinite,
}

/// Group result. Host gradient inspection has completed, but successful optimizer
/// updates are still asynchronous; synchronize before committing a checkpoint.
pub struct GuardedAdamWUpdate<R: Runtime> {
    /// Results in input order. Skipped entries preserve old buffers and steps.
    pub updates: Vec<AdamWUpdate<R>>,
    /// Absent for caller-skipped or entirely empty groups.
    pub statistics: Option<GradientStatsReport>,
    /// Common multiplier applied after unscale, in every AdamW update kernel.
    pub clip_multiplier: f32,
    /// Whole-group skip reason. An individual empty tensor is not advanced either.
    pub skip_reason: Option<SkipReason>,
}

fn same_queue<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>) -> bool {
    a.device.to_id() == b.device.to_id() && a.client.same_execution_queue(&b.client)
}

fn preflight_stats<R: Runtime>(gradients: &[&RudaTensor<R>]) -> Result<Vec<usize>, GradientGuardError> {
    let mut sizes = Vec::with_capacity(gradients.len());
    for &g in gradients {
        let size = validate_tensor(g, true)?;
        if let Some(&first) = gradients.first() {
            if !same_queue(first, g) {
                return Err(FusedAdamWError::DifferentExecutionQueue("gradient group").into());
            }
        }
        if size > 0 {
            let h = &g.client.properties().hardware;
            if h.max_units_per_ruda < THREADS || h.max_ruda_dim.0 < THREADS
                || h.max_ruda_count.0 == 0 || h.max_shared_memory_size < SHARED_BYTES
            {
                return Err(GradientGuardError::UnsupportedDevice("requires 256 X threads and 3072 shared bytes"));
            }
        }
        sizes.push(size);
    }
    Ok(sizes)
}

/// Blocking diagnostic: scan gradients on the device, reduce to three FP32 values
/// per nonempty tensor, then explicitly read and combine only those summaries on
/// the host in FP64. This is NOT a hidden CPU fallback or a capture-safe API.
///
/// All inputs are validated before dispatch. One/two kernels are submitted per
/// nonempty tensor, all summaries queued before one batched readback API call.
/// Stage one uses <=1024 blocks and <=12 KiB scratch per tensor, stage two one
/// block. Gradients are not copied in full, clipped, unscaled in place or mutated.
///
/// Caller must finish gradient accumulation, use one consistent loss scale, and
/// not modify gradients (including aliases) until the dependent update completes.
/// This is a LOCAL norm. It neither communicates nor deduplicates FSDP/TP shards.
pub fn gradient_stats_sync<R: Runtime>(
    gradients: &[&RudaTensor<R>], gradient_scale: f32,
) -> Result<GradientStatsReport, GradientGuardError> {
    let mut stats = GradientStats::empty(gradient_scale)?;
    let sizes = preflight_stats(gradients)?;
    let mut pending = Vec::new();
    let mut launches = 0;
    let mut scratch_bytes = 0;
    for (&g, &size) in gradients.iter().zip(&sizes) {
        if size == 0 {
            stats.add_device_summary(&[0.0, 0.0, 0.0], 0)?;
            continue;
        }
        let blocks = size.div_ceil(ITEMS_PER_BLOCK).min(MAX_BLOCKS)
            .min(g.client.properties().hardware.max_ruda_count.0 as usize);
        let make = |elements| empty_device_contiguous_dtype(
            g.client.clone(), g.device.clone(), [elements].into(), DType::F32,
        );
        let partials = make(blocks * 3);
        let dim = RudaDim::new_1d(THREADS);
        guard_kernel::scaled_l2::launch::<R>(
            &g.client, RudaCount::new_1d(blocks as u32), dim,
            (*g).clone().into_array_arg(), partials.clone().into_array_arg(),
            gradient_scale.recip(), false, include_str!("guard_kernel.rs").to_owned(), g.dtype.into(),
        );
        launches += 1;
        scratch_bytes += blocks * 12;
        let summary = if blocks == 1 { partials } else {
            let summary = make(3);
            guard_kernel::scaled_l2::launch::<R>(
                &g.client, RudaCount::new_single(), dim,
                partials.into_array_arg(), summary.clone().into_array_arg(),
                1.0, true, include_str!("guard_kernel.rs").to_owned(), DType::F32.into(),
            );
            launches += 1;
            scratch_bytes += 12;
            summary
        };
        pending.push((summary, size));
    }
    if let Some((first, _)) = pending.first() {
        let descriptors = pending.iter().map(|(t, _)| CopyDescriptor::new(
            t.handle.clone().binding(), t.meta.shape().clone(), t.meta.strides().clone(), 4,
        )).collect();
        let bytes = block_on(first.client.read_tensor_async(descriptors))
            .map_err(|e| GradientGuardError::Readback(format!("{e}")))?;
        if bytes.len() != pending.len() {
            return Err(GradientGuardError::InvalidSummary("missing readback tensors"));
        }
        for (raw, (_, size)) in bytes.into_iter().zip(&pending) {
            if raw.len() != 12 { return Err(GradientGuardError::InvalidSummary("summary must be 12 bytes")); }
            let mut triple = [0.0f32; 3];
            for (dst, chunk) in triple.iter_mut().zip(raw.chunks_exact(4)) {
                *dst = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            stats.add_device_summary(&triple, *size)?;
        }
    }
    Ok(GradientStatsReport { stats, reduction_launches: launches, readback_bytes: pending.len() * 12, scratch_bytes })
}

/// Finite-check and clip a local group, folding the common clip coefficient into
/// the existing out-of-place AdamW kernel. There is NO full-size clipped-gradient
/// allocation or clipping kernel. Existing `adamw_step` remains unchanged.
///
/// This call BLOCKS for compact gradient statistics and then submits updates.
/// It is not CUDA-graph capture safe and does not implement a dynamic loss scaler.
/// On a bad gradient, policy Skip skips EVERY entry, including weight decay and
/// step counts; policy Error returns before submitting ANY update. Structural
/// errors and step overflows are checked for ALL entries before even statistics.
///
/// No transaction is promised for later runtime/allocator/device faults: inputs
/// are read-only, but any partially submitted work still must be drained or
/// cancelled according to the runtime contract. Commit returned parameters and
/// state only after successful device completion. This checks gradients, not old
/// parameters/moments or whether finite but huge gradients overflow AdamW's v.
pub fn guarded_adamw_step<R: Runtime>(
    entries: &[AdamWEntry<'_, R>], options: &AdamWOptions,
    control: StepControl, guard: GradientGuardOptions,
) -> Result<GuardedAdamWUpdate<R>, GradientGuardError> {
    options.validate()?;
    control.validate()?;
    guard.validate()?;
    let mut nonempty = false;
    for entry in entries {
        let size = validate_step(entry.parameters, entry.gradients, entry.state, options, control)?;
        if let Some(first) = entries.first() {
            if !same_queue(first.parameters, entry.parameters) {
                return Err(FusedAdamWError::DifferentExecutionQueue("optimizer group").into());
            }
        }
        nonempty |= size != 0;
    }
    let unchanged = |reason, statistics| GuardedAdamWUpdate {
        updates: entries.iter().map(|e| AdamWUpdate {
            parameters: e.parameters.clone(), state: e.state.cloned(), updated: false,
        }).collect(),
        statistics, clip_multiplier: 1.0, skip_reason: Some(reason),
    };
    if control.skip_update { return Ok(unchanged(SkipReason::Requested, None)); }
    if !nonempty { return Ok(unchanged(SkipReason::Empty, None)); }
    let gradients: Vec<_> = entries.iter().map(|e| e.gradients).collect();
    let statistics = gradient_stats_sync(&gradients, control.gradient_scale)?;
    let decision = statistics.stats.decision(guard)?;
    if decision.skip_update { return Ok(unchanged(SkipReason::NonFinite, Some(statistics))); }
    let updates = entries.iter().map(|e| adamw_step_scaled(
        e.parameters, e.gradients, e.state, options, control, decision.clip_multiplier,
    )).collect::<Result<Vec<_>, _>>()?;
    Ok(GuardedAdamWUpdate {
        updates, statistics: Some(statistics), clip_multiplier: decision.clip_multiplier, skip_reason: None,
    })
}
