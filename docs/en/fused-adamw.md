# Experimental fused device AdamW / AMSGrad

This is an **opt-in implementation awaiting RUDA Rust/GPU execution validation**.
It extends `ruda-optim`; it does not create another optimizer library or change the
existing `AdamW`, model optimizer adaptor, autograd graph, or checkpoint format.

## Why this operator

The existing generic optimizer builds the update from tensor operations. This
addition explicitly combines gradient unscale, moments, optional AMSGrad maximum,
bias correction, decoupled weight decay and parameter update into one device
kernel. It does not claim the existing fusion backend necessarily uses many
kernels, or that this implementation is faster than PyTorch's fused optimizer.

For each finite input element and one-based update `t`:

```text
g = stored_gradient / gradient_scale      # negate if maximize
m = beta1 * m_old + (1 - beta1) * g
v = beta2 * v_old + (1 - beta2) * g * g
v_used = max(v_max_old, v)                 # AMSGrad only; save uncorrected max
p_new = p * (1 - lr * weight_decay)
        - lr * (m / (1 - beta1^t)) / (sqrt(v_used / (1 - beta2^t)) + epsilon)
```

Without AMSGrad, `v_used = v`. Epsilon is outside the square root. Weight decay is
not added to the gradient or moments. The host computes bias coefficients once
per call, using FP64 integer exponentiation and an FP32 coefficient cast. Device
arithmetic uses FP32; the compiler may contract/reassociate operations. Numerical
comparisons use tolerances, not a guarantee of bit identity with other optimizers.

Formula reference: [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
Defaults intentionally match RUDA's existing `AdamWConfig` for beta, epsilon and
decay: beta=(0.9,0.999), epsilon=1e-5, weight_decay=1e-4. Set all options explicitly
when comparing with another framework.

## Supported contract

| Item | This implementation |
|---|---|
| Parameters and moments | FP32 master/state |
| Stored gradient | FP32, FP16 or BF16; promoted to FP32 in the update |
| Shape | Exact match, dense contiguous, no broadcasting, conservative <= u32 byte range |
| Mode | AdamW, AMSGrad, maximize, scalar learning rate, positive loss scale |
| Empty tensor | No launch, allocation or step advance |
| Caller-detected overflow | `skip_update=true`: no launch, allocation or step advance |
| Inputs/aliases | Read-only; outputs are new buffers |
| Initial state | Computed in the first update; no zero-fill launch required |
| Queue | Same device and submission queue for all inputs; mismatch is an error |
| Completion | Asynchronous, governed by the existing runtime |
| Half-precision model weights | Not updated/cast automatically; caller casts master output explicitly |

This is **not** FP8/FP4 optimizer state, a GradScaler, an automatic finite-gradient
check, cautious weight decay, a differentiable optimizer, a general strided kernel,
a variable-hyperparameter multi-tensor optimizer, automatic FSDP integration, or
a CUDA-graph-replay optimizer with a device-side step counter. Replaying captured
host bias coefficients without updating them is not supported.
A pre-flattened bucket works only when all elements share options and step count.
Sparse gradients, arbitrary host pointers and a silent CPU fallback are not added.

FP32 master state does not imply end-to-end mixed-precision training has been
validated. The caller manages model copies, scaling decisions and accumulated
or distributed gradients.

## Features

- `fused-adamw`: options and explicitly invoked CPU reference.
- `fused-adamw-device`: generic RudaTensor device launcher and kernel, no specific
  hardware driver enabled by this feature.
- `fused-adamw-cuda`: CUDA runtime, direct PTX capability and explicit CUDA tests/example.
  It still selects NVRTC or PTX using `RUDA_CUDA_COMPILER`.

No feature is enabled by default. No new dependency version is introduced.
`Cargo.lock` only gains the existing CUDA driver as an optional ruda-optim dependency.

## Low-level use

```rust
use ruda_optim::fused_adamw::{AdamWOptions, StepControl, adamw_step};

// master: dense FP32 RudaTensor<R>, gradient: same-shape F32/F16/BF16 tensor.
// state: Option<AdamWState<R>>, initially None.
let options = AdamWOptions {
    learning_rate: 1e-3,
    weight_decay: 0.01,
    amsgrad: true,
    ..Default::default()
};
let result = adamw_step(
    &master, &gradient, state.as_ref(), &options,
    StepControl { gradient_scale: 128.0, skip_update: found_inf },
)?;
master = result.parameters;
state = result.state;
```

`found_inf` is supplied by the caller, not computed here. Existing `AdamW::step`
continues using its original implementation. The low-level primitive does not
replace a module `Parameter` automatically or build a differentiable update graph.

Input validation leaves all inputs unchanged. A successful return means the
kernel was submitted; it does not prove execution completed. Before checkpointing,
check the runtime synchronization result. If the device fails, discard pending
outputs and restore an externally committed checkpoint. Do not increment the
training data cursor solely because `updated` is true.

`AdamWState::into_parts/from_parts` expose the step and moment buffers for explicit
checkpoint integration. They do not serialize or transfer by themselves. Preserve
master parameters, all moments, hyperparameters and step together; this module
does not retrofit the high-level `TrainingRecord` format.

## Allocation and performance model

This first implementation chooses out-of-place updates to preserve aliases and
avoid adding new unsafe ownership rules. Each active step allocates three FP32
outputs (four with AMSGrad). There is no intermediate delta tensor, but **no claim
of a zero-allocation step**. Old and new buffers can overlap in lifetime. Large
models therefore require a memory budget; in-place or arena/bucket reuse is future
work, not silently enabled here.

The benchmark-only staged baseline explicitly submits 4 kernels, or 5 with
AMSGrad. The fused path submits 1. For a *steady-state* FP32-gradient step:

| Model | Logical per-element bytes | Explicit update launches |
|---|---:|---:|
| Staged AdamW | 48 | 4 |
| Fused AdamW | 28 | 1 |
| Staged AMSGrad | 60 | 5 |
| Fused AMSGrad | 36 | 1 |

The 28-byte count is four FP32 reads (parameter, gradient, two moments) plus three
FP32 writes. The 48-byte baseline includes repeated gradient reads and the delta
buffer. This is source-level accounting, **not measured DRAM traffic or speedup**.
Caching, allocation, arithmetic and launch overhead may change the observed result.
A backend that already fuses generic AdamW may not benefit.

## Validation and benchmark commands

```sh
# Python/NumPy/PyTorch formula oracle ONLY, does not execute RUDA.
python tools/run_adamw_regressions.py --suite oracle

# Standalone Rust config/reference tests, without Cargo registry resolution.
python tools/run_adamw_regressions.py --suite reference

# Cargo unit tests, including a comparison with RUDA's existing Host AdamW.
python tools/run_adamw_regressions.py --suite host

# Type-check the opt-in generic device implementation.
python tools/run_adamw_regressions.py --suite build

# Explicit hardware regression, separately under both compilers.
python tools/run_adamw_regressions.py --suite cuda --compiler both

# Small controlled A/B run, then increase elements after checking resources.
python tools/run_adamw_regressions.py --suite bench --compiler both \
    --elements 65536 --dtype bf16 --iterations 20 --samples 7 --amsgrad
```

Use a `RUDA_PTX_VERSION` supported by the actual driver/GPU. `--offline` requires
cached Cargo dependencies. `--timeout` bounds each command; it is not a predicted
runtime. The runner preserves command lines, source hashes, logs and status.
Missing Rust/Cargo is `blocked`; an unavailable GPU fails the explicit CUDA suite.
No test silently switches backend or claims a measured result on a simulator.

The example excludes input creation/readback from timed sections, warms both
variants, starts both from identical materialized state, alternates run order,
and synchronizes before/after each measured batch. It reports median/min/max
**wall milliseconds per step**, including allocations and host submissions. It is
not a CUDA-event GPU-only timer. Every measured batch compares parameters and
moments. Save GPU, driver, clock/power settings and hardware load alongside JSON.

The committed `pytorch_fixtures.json` is generated with real **CPU** PyTorch AdamW;
12 cases cover 3 gradient dtypes x 2 AMSGrad modes x 2 maximize modes, 5 steps each.
`oracle.py --write-fixtures` regenerates it explicitly. The CUDA tests compare the
actual RUDA kernel outputs to this data, as well as the separate FP64 reference.
Passing the Python fixture generator is not passing those CUDA tests.

## Remaining gates

Rust type/macro expansion, CUDA execution and measured performance remain mandatory.
New tests must pass alongside the previous safety regression suites before release.
Further optimizations should start from measured profiler results, not this traffic
model. Do not enable a new default dispatcher until the comparison is complete.


See [experimental gradient checks and group clipping](gradient-guard.md) for an opt-in extension.
