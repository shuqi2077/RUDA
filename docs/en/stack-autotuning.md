# Shared stack autotuning

[Documentation](README.md) · [Runtime](runtime-api.md) · [中文](../zh/stack-autotuning.md)

## Scope and participating paths

The shared controller validates semantically equivalent implementations, measures paired execution times and caches a choice for the actual workload/environment. Operator, fusion and application adapters share these rules; this is not a global search of all kernels in a model.

| Layer | Adapter / search scope |
| --- | --- |
| Runtime | [`ruda-runtime/src/runtime/tune/stack`](../../ruda-runtime/src/runtime/tune/stack): policy, timing, cache and reports |
| Matrix multiplication | [`ruBLAS` tuner](../../ruBLAS/src/tensor_matmul/tune/base.rs): eligible candidate groups and actual layout/precision |
| Attention | [`ruDNN` attention tuner](../../ruDNN/src/attention/tensor/tune.rs): operator and mask/bias options |
| Forward convolution | [`ruDNN` convolution tuner](../../ruDNN/src/convolution/tensor/forward/tune.rs): complete forward operator |
| Fusion | [Fused matmul tuner](../../ruda-fusion/src/device/optim/matmul/tune.rs): all externally visible fused outputs |
| Generation | [`ruLLM` application adapter](../../ruLLM/src/autotune.rs): Portable versus Automatic packed greedy-generation plans |

Original candidate eligibility is retained, but all eligible priority groups can participate within the budget. FFT, sparse, solver, communication and unrelated fusion paths retain their existing behavior. Batch sizes, stream counts, model splitting and multi-GPU placement are not searched.

## Enable before execution

Install the controller once before loading a model, starting workers or running the first computation:

```rust
use ruda::runtime::tune::stack::{enable_stack_autotune, Mode, StackPolicy};
use std::path::PathBuf;

let tuner = enable_stack_autotune(
    StackPolicy { mode: Mode::Explore, ..StackPolicy::default() },
    Some(PathBuf::from(".ruda-stack-cache")),
)?;
```

The snippet belongs inside a fallible application function. `None` means memory-only caching. Disk reuse also requires reliable environment identity. The global controller cannot be installed twice or reconfigured during execution.

Enable `ruda-tensor-device/stack-autotune`, `ruda-fusion/device-stack-autotune` when using fusion, or `ruda-llm/stack-autotune` for generation. The controller uses native standard-library facilities; it does not cover no-std or WebAssembly configurations.

| Mode | Cache miss / execution |
| --- | --- |
| `Explore` | Validate and search within policy; reuse valid choices. |
| `CacheOnly` | Do not run new timing search; use the explicit reference on miss. Cold disk reuse still validates once. |
| `Disabled` | Participating adapters use their explicit reference without this cache or calibration. |

Not installing the controller retains the original LocalTuner route. `Disabled` is not equivalent to restoring that route. Operators without the new adapter metadata still use their old path.

## Policy and timing reference

| `StackPolicy` field | Default / meaning |
| --- | --- |
| `mode`, `timing` | Explore, EndToEnd |
| `require_validation` | True; output validation required |
| `tolerance` | Absolute `1e-4`, relative `1e-3`, combined reference/candidate readback limit 64 MiB |
| `warmups`, `samples` | 2 warmups, 7 paired reference/candidate samples; odd sample count |
| `max_candidates`, `budget` | At most 32 candidates; 30-second soft per-workload budget |
| `min_speedup`, `max_relative_mad` | 1.05 selection threshold; 0.15 relative paired-ratio noise limit |
| `workspace_limit` | None; hard candidate-workspace filtering is opt-in |
| `capacity`, `ttl` | 1024 cache records; 7 days |
| `max_parallel_tunes` | 1, additionally serialized per device in a process |
| `regression_pairs`, `regression_ratio` | 7 supplied comparison pairs; invalidate above ratio 1.15 |

See [policy definitions and validation ranges](../../ruda-runtime/src/runtime/tune/stack/policy.rs) before overriding defaults. The 1.05 threshold is a policy, not a measured speedup claim.

Each sample remeasures reference and candidate, alternates their order and compares candidate/reference ratios. Selection uses the median ratio with a noise criterion, not a single fastest run or a statistical significance claim.

EndToEnd timing includes candidate-internal allocation, layout preparation, dispatch and completion waiting. Isolated trial-input preparation is outside its timing interval. Device timing uses existing profiling/completion facilities; complete-generation calibration only accepts EndToEnd. The soft budget is checked **between completed trials** and cannot interrupt a long kernel or enforce a request timeout.

Memory-cache hits do not force readback, sync or retiming, though key/lock lookup still costs host work. Cold disk validation and exploration increase latency. Calibrate in controlled startup/offline conditions, not on every token or request.

## Validation and state isolation

Reference and candidate trials share numerical input, but writable output/state are isolated. Matmul trials preserve strides, view offsets and allocation spans. Fusion compares visible outputs. Each generation trial owns its KV cache and does not advance live request history.

Validation rejects NaN/Inf. Missing validators, readback-limit excess or unsupported quantized comparisons use the declared reference; a wrong numerical result or failed reference is an error. Setting `require_validation=False` explicitly permits unvalidated selection and reports `verified=False`. Relative comparison cannot prove independent mathematical correctness.

Generation calibration requires identical generated token IDs and termination on its calibration prompt; this is not an all-prompt/all-logit guarantee. There is no claim that every shape, dtype or hardware has been tuned by merely enabling the feature.

## Cache identity and lifecycle

Keys include operation/candidate versions, backend/device capabilities, driver identity, source/build fingerprint, options, real shape/stride/offset, precision, parameters and execution context. Fusion adds graph content; generation adds weight revision, model/prompt/configuration and lower-level selections.

Incomplete hardware/driver identity normally limits reuse to memory. A backend can implement `Runtime::autotune_driver_fingerprint`; deployment can supply immutable `RUDA_AUTOTUNE_DRIVER_TAG`, `RUDA_AUTOTUNE_BUILD_TAG`, `RUDA_AUTOTUNE_CONTEXT_TAG` before process startup. These tags augment detected identity and are the caller's responsibility. The default context name does not detect actual concurrency/topology.

Disk records use `stack-autotune-v1` under the selected directory. Full keys are checked; corrupt, expired, future-dated or version-mismatched entries become misses. Writes use temporary files and rename, preserving old records on failed replacement. Digests are noncryptographic checksums, not signatures, adversarial authentication or prompt anonymization. Keep caches in a trusted, permission-controlled directory.

`LocalTuner::clear` clears the old cache only. Use the shared controller's own invalidation APIs for this cache. Choices do not enumerate every possible dynamic compiler or external candidate dependency; immutable deployment tags are needed for dependencies not captured by the built-in fingerprint.

## Controller and cache API

These public APIs are exported by `ruda::runtime::tune::stack`; their definitions document each field and validation limit.

| API | Contract |
| --- | --- |
| `StackTuner::new(policy, cache_directory)` | Validate policy and create an independent controller without disk I/O; does not install it globally. |
| `enable_stack_autotune(policy, cache_directory)` | Install the process-global controller once; returns `Result<&'static StackTuner, TuneFailure>`. |
| `stack_autotuner()` | Borrow the installed controller, or `None`; no initialization or trials. |
| `runtime_environment(&client)` | Memoize backend/device/driver/build/options and caller-supplied execution context; returns `RuntimeEnvironment { fingerprint, persistent, execution_context }`. |
| `policy()`, `stats()`, `reports()` | Immutable policy, cloned counters and retained search reports; reports are oldest-first and bounded by `min(capacity,128)`. |
| `select(&problem, &candidates, reference, &mut runner)` | Return `Result<Decision,TuneFailure>` without executing the live request. `TrialRunner` performs isolated validation and completed-work timing. |
| `invalidate(&decision, ban_candidate)` | Remove future reuse; optionally ban a non-reference candidate for that key. Disk failures increment warnings; never replay current state. |
| `record_comparison(&decision, reference_time, selected_time, correctness_checked)` | Consume positive paired observations; return true only when the regression window invalidates the decision. Missing records/changed winners return false; this does not check TTL again. |
| `lower_level_fingerprint()` | Conservative digest of all retained non-pipeline choices/bypasses, including other devices/workloads. |
| `try_execute_stack(...)` | Runtime adapter requiring an installed controller, an explicit in-range reference and exact workload signature. Execute live input once after selection; errors do not trigger replay. |

`Problem` contains `scope`, `operation`, `environment`, `workload`, `execution_context` and `persistent`. Candidate count is 1–4096, names unique/nonempty and at most 4096 bytes each; operation/environment/workload must be nonempty, the full key at most 384 KiB, and the reference must fit policy. `Candidate` exposes `name`, semantic `revision`, optional `workspace_bytes` and `eligible`; `Candidate::new(name)` defaults to revision `"1"`, eligible and unknown workspace. Unknown estimates do not fit an enabled hard limit.

`Decision` exposes `index`, `reference_index`, `name`, `source`, `verified`, optional selected/reference `ratio` and `cache_key`. Indices belong to this call's candidate slice, not another build. `TuneReport` contains operation/environment/workload/context, winner, elapsed time, budget status and `CandidateReport` rows (`name`, `samples`, `ratio`, `relative_mad`, `verified`, `note`). Cache hits increment counters rather than adding new search reports.

Persistent records/`DiskCache` are internal, not supported public serialization APIs. Reads enforce format, full key, size, checksum and numeric validity; the controller checks TTL and current-process validation. `save` uses a new temporary file, fsync and rename before capacity pruning; a pruning error can occur after the new record was installed. See [policy types](../../ruda-runtime/src/runtime/tune/stack/policy.rs), [controller](../../ruda-runtime/src/runtime/tune/stack/engine.rs), [runtime adapter](../../ruda-runtime/src/runtime/tune/stack/runtime_adapter.rs) and [internal cache](../../ruda-runtime/src/runtime/tune/stack/cache.rs).

## Concurrency, failures and regressions

Contending calls do not wait for a tune; they use the reference. Nested tuning can only reuse cached choices and otherwise uses the reference, avoiding recursive exploration. There is no cross-process GPU exclusion lock, so other processes can perturb timings. This controller does not isolate unrelated old tuners or all device work.

Unknown completion faults the tuning lane. It does not reset the GPU, recover all physical resources or replay a failed live request. Synchronous failure invalidates future choices; asynchronous error attribution remains an application responsibility. Current operator adapters do not provide reliable temporary-workspace estimates: enabling a hard `workspace_limit` can reject the reference as well; it is not a total-VRAM estimate or OOM recovery mechanism.

`StackTuner::record_comparison` consumes caller-measured, correctness-checked reference/selected pairs. It does not execute shadow models automatically. After the configured regression criterion, future selection invalidates the choice. An already-held `GenerationPlan` is not modified automatically; explicitly recalibrate or choose the reference. `GenerationPlan::lower_levels_unchanged` checks lower-level choice identity, not whether the application plan itself has been revoked.

## Complete-generation example

Set `MODEL_DIR` to the existing local packed model and `WEIGHTS_REVISION` to an immutable identity of its actual weights. No model download or automatic directory-name identity is performed.

```bash
cargo run --release --locked -p ruda-llm \
  --features nvidia,stack-autotune --example qwen2_autotune -- \
  "$MODEL_DIR" "$WEIGHTS_REVISION" 'Explain matrix multiplication' 16 .ruda-stack-cache explore

cargo run --release --locked -p ruda-llm \
  --features amd,stack-autotune --example qwen2_autotune_amd -- \
  "$MODEL_DIR" "$WEIGHTS_REVISION" 'Explain matrix multiplication' 16 .ruda-stack-cache explore
```

Each example selects an explicit runtime type even in a dual-backend build. `cache-only` replaces the final `explore`, but invoking application calibration still performs lower-level warmup and necessary validation. Configure the actual [CUDA/PTX](ptx.md) or HIP environment first. Output reports the selected mode, cache source, verification, paired ratio/sample/noise information and actual request time; it is not a universal performance baseline.
