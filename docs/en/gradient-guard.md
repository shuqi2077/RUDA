# Gradient guard and fused AdamW clipping (experimental)

**English** | [简体中文](../zh/gradient-guard.md) | [日本語](../ja/gradient-guard.md) | [Deutsch](../de/gradient-guard.md) | [Русский](../ru/gradient-guard.md)

Opt in explicitly. The existing
`adamw_step` public signature and the default model optimizer are unchanged.

## What it does

`gradient_stats_sync` scans accumulated gradients, checks raw and FP32-unscaled
values for NaN/Inf, and measures one L2 norm across the supplied local group.
`guarded_adamw_step` uses that common norm to clip inside the existing fused
AdamW/AMSGrad kernel, without allocating or writing full-size clipped gradients.

The order is storage-type conversion -> FP32 reciprocal loss-scale multiply ->
finite check/norm -> common FP32 clip coefficient -> maximize sign -> AdamW.
This is deliberately NOT an in-place FP16/BF16 clipping operation: the effective
gradient stays FP32, rather than rounding back to half storage before the update.

For finite gradients:

```
clip = min(max_norm / (L2_norm + epsilon), 1)
g_effective = (float32(g_stored) * reciprocal(loss_scale)) * clip
```

`max_norm=None` disables clipping, not finite checks. `max_norm=0` zeroes the
effective gradient but still advances moments, decay and step counts. NaN/Inf
with policy `Skip` skips the ENTIRE selected group including weight decay and
step counts. Policy `Error` fails before any optimizer-update kernel is launched.
All metadata and step-overflow checks happen before statistics are submitted.

## Features and use

- `gradient-guard`: host configuration, stats/decision types and CPU oracle.
- `gradient-guard-device`: generic device reduction and guarded optimizer.
- `gradient-guard-cuda`: CUDA integration tests and benchmark.

```rust,ignore
use ruda_optim::fused_adamw::{
    AdamWEntry, AdamWOptions, StepControl, guarded_adamw_step,
    gradient_norm::GradientGuardOptions,
};

// master1/2: contiguous FP32; grad1/2: F32/F16/BF16, same device and queue.
// Accumulation must already be finished, using ONE loss scale for this step.
let entries = [
    AdamWEntry { parameters: &master1, gradients: &grad1, state: state1.as_ref() },
    AdamWEntry { parameters: &master2, gradients: &grad2, state: state2.as_ref() },
];
let pending = guarded_adamw_step(
    &entries, &AdamWOptions::default(),
    StepControl { gradient_scale: 128.0, skip_update: false },
    GradientGuardOptions { max_norm: Some(1.0), ..Default::default() },
)?;
// Compact stats readback has completed, but the UPDATE is still asynchronous.
// Await/synchronize the runtime and check completion before replacing committed
// model/state or saving a checkpoint. Keep the existing committed state on failure.
```

No automatically updated GradScaler, model adaptor, FSDP reduction, tensor
flattening, deduplication of tied weights, or low-precision model recast is added.
Only the parameters explicitly supplied are counted. A local shard norm must NOT
be used as the global FSDP/TP norm. No inter-rank skip agreement is performed.

## Reduction implementation and resource cost

A grid-stride kernel retains a `(scale, sumsq, bad)` triple per lane and reduces
with a fixed shared-memory tree. It never directly squares a large raw FP32
value, so e.g. finite values near `1e30` do not create a spurious FP32 overflow in
norm computation. Up to 1024 partial triples are reduced by one additional block.
There are no float atomics; this is not an automatic device autotuner.

The configuration requires 256 X threads and 3072 bytes of shared memory per
block. Unsupported configurations are rejected, not silently sent to the CPU.
Empty tensors submit no work. Each nonempty tensor submits one or two reduction
kernels and returns one 12-byte summary. All tensor reductions are queued before
one batched host readback API call; the runtime may implement multiple DMA copies.
The host explicitly sums compact summaries in FP64 and decides the coefficient.
A tensor needs at most 12 KiB partial storage plus a 12-byte final summary, excluding
allocator alignment/metadata. `scratch_bytes` sums those per-tensor amounts and
is NOT a peak allocator measurement.

Reassociation, FMA and subnormal handling depend on the backend. Norms are tested
with tolerances, not promised bit-identical to PyTorch or across devices. This is
not the entire LAPACK LASSQ implementation and is not advertised as its numerical
compatibility. Large finite gradients can still overflow AdamW's second moment
when clipping is disabled. Old parameters/moments are not scanned for finiteness.

## Performance claim and explicit limitations

Compared to the included baseline (same norm computation, then a separate
unscale+clip kernel producing FP32 gradients, then fused AdamW), this removes one
clip launch and one `4*N` byte temporary per nonempty tensor, and avoids `8*N`
bytes of logical temporary writes/reads. These are source-level counts, NOT
measured device-memory traffic or acceleration. The old unguarded optimizer does
not pay the new statistics cost; adding diagnostics may make a step slower.

This first version blocks the host on compact readback every optimizer step. It
is not graph-capture safe, does not promise compute/communication overlap, and may
perform poorly on many tiny tensors. The stable reduction has extra arithmetic.
Benchmark before enabling; no claim against PyTorch fused/foreach AdamW is made.
The original AdamW output allocations are unchanged (out-of-place FP32 masters
and moments). Gradients must not be modified through another alias/queue while
statistics or dependent updates are in flight. Runtime failures are not a
transactional group commit; only the non-finite/validation skip decision is made
before update launch. Runtime allocation/launch APIs retain their error contract.

## Validation and benchmark

```bash
python tools/run_gradient_guard_regressions.py --suite oracle
python tools/run_gradient_guard_regressions.py --suite reference
python tools/run_gradient_guard_regressions.py --suite host
python tools/run_gradient_guard_regressions.py --suite build
python tools/run_gradient_guard_regressions.py --suite cuda --compiler both
python tools/run_gradient_guard_regressions.py --suite bench --compiler both --elements 65536 --tensors 4 --dtype bf16 --amsgrad
```

`oracle` needs NumPy and PyTorch CPU. It executes only a Python numerical model,
not RUDA code. `reference` compiles standalone Rust tests without Cargo registry
access. `build` checks both old and new device features. `cuda` executes both old
AdamW regressions and the new hardware tests. Missing tools are BLOCKED; timeout
and failure are recorded. `--dry-run` only prints planned commands. `--offline`
requires cached Cargo dependencies. No tool installation is performed.

Benchmark alternates paths after warmup, includes allocations, norm, host
readback/decision, submission and final device sync, and checks parameters and
all moments after every measured batch. Readback used only for the comparisons
is outside timing. Keep environment/source versions with the raw JSON samples.

## Numerical references

Conceptual references, not copied implementations:
- PyTorch AMP examples (unscale before gradient clipping):
  https://docs.pytorch.org/docs/main/notes/amp_examples.html
- PyTorch local concatenated-gradient norm contract:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html
- Scaled sum-of-squares representation:
  https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq.html
