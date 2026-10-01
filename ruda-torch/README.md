# ruda-torch

Native PyTorch integration for RUDA on a single NVIDIA GPU, exposed as `ruda:0` through PyTorch's PrivateUse1 backend.

- Rust package: `ruda-torch-native` (`cdylib`, not published to crates.io).
- Python package: `ruda-torch`; import: `ruda_torch`.
- Rust/C++ bridge: **ABI 10**. Rebuild both components together after an ABI change.

## Build and install

Install the [NVIDIA environment](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md), Rust/Cargo, Python with PyTorch and setuptools, and a C++20 compiler. On Windows, use an x64 MSVC developer shell. Run the following from the RUDA repository root.

Select direct PTX in your shell, using a [PTX version](https://github.com/shuqi2077/RUDA/blob/main/docs/en/ptx.md) supported by your GPU and driver:

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

Or in Bash:

```sh
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

Build the Rust library, then the extension against your installed PyTorch:

```sh
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

The loader uses a packaged native library when present, otherwise `target/debug/ruda_torch_native.dll` on Windows or `target/debug/libruda_torch_native.so` on Linux. Set `RUDA_TORCH_LIBRARY` before import to select a release build or another location. On Windows, `RUDA_TORCH_DLL_DIR` accepts semicolon-separated dependency directories.

Prebuilt wheels are artifacts of successful [Windows build runs](https://github.com/shuqi2077/RUDA/actions/workflows/ruda-torch-windows.yml). They include the native DLL and target Windows x64, CPython 3.13 and PyTorch `2.13.0+cu130`. Install that matching PyTorch build first, extract the wheel artifact, and pass its `.whl` file to `python -m pip install --no-deps`. Artifacts are retained for seven days.

## Basic use

```python
import torch
import ruda_torch

x = torch.arange(4, dtype=torch.float32).to('ruda:0')
print((x + x).cpu())
```

Import registers `torch.ruda` and the device. Another PrivateUse1 backend cannot already be registered in the same process. RUDA tensors are distinct from `torch.cuda` tensors; use the `ruda` device explicitly. Unsupported operators raise an error rather than silently executing through a generic CPU fallback.

## Native operators

- Matrix operations including `mm`, `bmm` and `addmm` use ruBLAS or the selected scalar GPU path. `RUDA_TORCH_MATMUL` accepts `auto` (default), `rublas` or `scalar`.
- FP32/FP16/BF16 paths retain their storage dtype where supported. Fused last-axis LayerNorm/RMSNorm and storage-aware reductions avoid separate full-size FP32 intermediates on those paths.
- Softmax/log-softmax and their backward kernels support `RUDA_TORCH_SOFTMAX=auto|warp|scalar`. The default `auto` uses the warp path for axis lengths of at least 32.

Operator-specific shape, dtype and layout checks still apply. See the [operator registrations](https://github.com/shuqi2077/RUDA/blob/main/ruda-torch/python/ruda_torch/_ops.py) and [native kernels](https://github.com/shuqi2077/RUDA/tree/main/ruda-torch/src).

## Streams, events and asynchronous dispatch

Dispatch is synchronous by default. Set `RUDA_TORCH_ASYNC=1` before the first native submission to enable asynchronous dispatch; this setting is cached for the process. Explicit synchronization, scalar extraction and host readback still wait for completion.

`Stream`, `Event`, `stream`, `current_stream`, `default_stream` and `record_stream` use RUDA's native runtime. Only stream priority 0 and device `ruda:0` are supported. Reuse streams: the pool is bounded by the runtime's `streaming.max_streams` setting.

```python
import torch
import ruda_torch

x = torch.arange(4, dtype=torch.float32).to('ruda:0')
producer = ruda_torch.current_stream()
worker = ruda_torch.Stream()
worker.wait_stream(producer)
with ruda_torch.stream(worker):
    y = x + x
    ruda_torch.record_stream(x, worker)
    done = worker.record_event()
producer.wait_event(done)
print(y.cpu())
done.close()
```

`query()` reports readiness without waiting for GPU completion; `wait_event()` inserts a GPU-side dependency. `synchronize()` waits on the host. `record_stream(tensor, stream)` retains storage for already-submitted work but does not establish execution ordering. For elapsed milliseconds, record two `Event(enable_timing=True)` events, wait for completion, then call `start.elapsed_time(end)`.

## Paged GQA and MLA

`PagedAttentionPlan` holds immutable scheduling metadata for packed variable-length prefill/decode. Query layout is `[queries, query_heads, features]`; cache layout is `[physical_pages, page_size, KV_heads, features]`. Sequence IDs index `block_tables` and `kv_lengths`; positions are absolute, zero-based positions within each sequence. Recreate the plan when the schedule changes.

```python
import torch
import ruda_torch

plan = ruda_torch.PagedAttentionPlan(
    page_size=2, num_pages=1, block_tables=[[0]],
    kv_lengths=[2], sequence_ids=[0], positions=[1], splits=1,
)
q = torch.ones((1, 4, 8), dtype=torch.float32).to('ruda:0')
k = torch.ones((1, 2, 1, 8), dtype=torch.float32).to('ruda:0')
v = torch.arange(16, dtype=torch.float32).reshape(1, 2, 1, 8).to('ruda:0')
out = plan.attention(q, k, v, scale=8 ** -0.5, causal=True)
print(out.cpu())
```

Inputs must be contiguous FP32/FP16/BF16 tensors with the same dtype on the native `ruda` device and matching execution queue. Query-head count must be divisible by KV-head count. Supply finite Q/K/V and a finite positive scale; key/value feature dimensions are limited to 1024. Forward and first-order autograd are supported; arbitrary external masks, quantized KV caches and higher-order derivatives are not.

`splits=1` is the default unsplit path; `2..32` uses partial attention followed by FP32 merging. `plan.workspace_bytes(query_heads, value_dim)` reports planned scratch bytes, capped at 64 MiB per workspace, not total device memory. Workspaces are cached per native plan/stream and shape, not in a global pool. Splitting is explicit and does not guarantee a speedup.

`plan.mla(absorbed_query, position_query, latent_cache, position_cache, scale=..., causal=True)` returns compressed context `[queries, heads, rank]`. The latent cache is `[pages, page_size, 1, rank]`, the positional cache `[pages, page_size, 1, position_dim]`, and the query tensors are `[queries, heads, rank]` and `[queries, heads, position_dim]`; position dimension is limited to 256. Apply positional encoding first and value/output projections afterward. Use the model's original QK scale, not `1/sqrt(rank)`. The caller prepares cache contents and owns cache updates.

### First-order gradients and ordered history

Set `requires_grad=True` on the inputs to differentiate. Forward keeps the fused/split inference path. Backward recomputes probabilities and allocates only the requested Q/K/V gradients, or Q/Q-position/latent/K-position gradients for MLA. The latent derivative contains both key and value contributions. Frozen inputs remain saved when other derivatives need their values; do not modify saved tensors before backward.

`PagedAttentionPlan(..., backward_strategy="atomic")` is the default. Its requested history gradients use FP32 atomics and respect PyTorch's deterministic-algorithm error/warning policy. Choose `backward_strategy="ordered"` explicitly for single-writer, atomic-free history reduction; it is not automatic tuning or a promise of faster execution or cross-device bitwise equality. This requires paged-backward API 2 in both native components.

Ordered backward caches FP32 row statistics and inverse page metadata per native plan/stream and compatible shape. Query pruning is enabled and skips only causally invisible contributions while retaining summation order. Two further options are disabled by default:

| Environment variable | Effect when set to `1` |
| --- | --- |
| `RUDA_PAGED_ORDERED_CACHE_ROWS` | Reuse history-row values in thread-local storage; may increase register pressure. |
| `RUDA_PAGED_ORDERED_COMPACT_HISTORY` | Reduce only active physical pages and explicitly zero inactive history-gradient pages. |

Unset or `0` disables these options; other values are errors. They are read when an ordered workspace is constructed, so set them before the first ordered backward on a plan/stream, or create a fresh plan after changing them. They do not affect the atomic path. Rust callers can also use the workspace setters described in the ruDNN guide.

Compaction uses effective KV lengths of sequences with queries, not all allocated pages or table capacity. Its retained page index costs `num_pages * 4` bytes inside the 64 MiB ordered-workspace budget. Disabling an already-created index in Rust keeps its allocation for reuse. This is not KV-cache compression and does not shrink gradient tensor shapes. `plan.workspace_bytes(...)` reports only forward split scratch, not ordered-backward storage or peak VRAM.

See the [ruDNN guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rudnn.md) for the underlying Rust paged-attention and MoE interfaces, and the [ruLLM guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md) for model integration.

## Optional fixed-address inference graphs (v24)

`ruda_torch.StaticGraph` connects selected native PyTorch tensor operations to
RUDA's existing `CudaGraph`, rather than a separate CUDA extension/runtime.
The optional graph interface is version 2; the current base tensor ABI is 10.
Rebuild both Rust and C++ for this feature. Defaults and eager dispatch stay unchanged.

Supported explicit nodes: copy, same-shape add/mul, SiLU, last-axis RMSNorm and storage-rounded SiLU-mul.
Opt-in `optimize=True` removes unused nodes and fuses single-use left SiLU/mul;
`reuse_workspace=True` reuses only equal-spec scratch storage after its last read.
Returned outputs remain dedicated. Both options default to False pending GPU validation.
This is not arbitrary model/stream capture, an autograd implementation or a
replacement for `torch.cuda.graph`. See `docs/zh/static-pytorch-graphs.md` in the
repository root for fixed-pointer, output-reuse and synchronization contracts.

```sh
python ruda-torch/python/examples/static_residual_norm.py
python ruda-torch/tools/validate_static_graph.py --build --output ./v24-gpu
```

The validator requires direct PTX configuration and real GPU execution; absent
hardware or skipped tests do not count as success.


## v27: opt-in hierarchical optimizer statistics (training API 4)

`AdamW(..., fused_step=True, hierarchical_stats=True, max_grad_norm=1.0)`
reduces gradient-statistics rows in bounded GPU stages. This does not change
AdamW's update kernel, gradient storage, the global clipping policy, or the
explicit 12-byte readback before any update. `hierarchical_stats` defaults to
`False`. Each merge warp processes at most 1024 rows; at most two additional
kernels and 49,200 bytes of reusable scratch are used for 4096 parameters.
Small workloads (<=1024 rows) need no extra merge kernel. Different reduction
order can change FP32 rounding; bitwise equality is not promised.

Rebuild both Rust and C++ (base ABI 10 / graph API 2 / training API 4).
Do not load a v26 training library into this bridge. Checkpoints without the
hierarchical option restore with that option disabled; opt-in checkpoints carry
step-options version 2 and are rejected by older v26 readers.

Run `python ruda-torch/tools/rust_host_gradient_stats.py --output ./stats-host`
for the production Rust planner. Run `python ruda-torch/tools/validate_training.py
--build --dtypes float32,float16 --output ./v27-training-results` for real GPU
acceptance (no successful skips). Configure
`RUDA_CUDA_COMPILER=ptx` and a driver-supported `RUDA_PTX_VERSION` first.
The source includes a paired `python/examples/benchmark_gradient_stats.py`
benchmark: it measures gradient analysis, not whole-model training speed.


## v28: native LayerNorm training (training API 4)

The common last-axis training path now keeps LayerNorm mean/rstd in FP32 and
runs first-order `dx`, `dweight` and `dbias` with native RUDA kernels. Standard
`torch.nn.LayerNorm` automatically selects this path when gradients are needed,
tensors are contiguous, and the native training extension is available; inference
continues to use the existing single-kernel inference path. The explicit
`ruda_torch.LayerNorm`/`ruda_torch.layer_norm` entry points also support FP32
affine parameters with FP16/BF16 activations.

The backward path does not materialize full-size FP32 copies of activation and
gradient tensors. FP32 saved statistics and bounded FP32 affine-gradient partials
are used instead. This remains first-order only and does not claim higher-order
autograd, multi-axis LayerNorm fusion, or training graph capture.

Rebuild both Rust and C++ components because the additive training API is now 4.
Run `python ruda-torch/tools/validate_training.py --build --dtypes float32,float16
--output ./v28-training-results` for strict device validation.

## Optional router-weight training extension (v30)

`ruda_torch.selected_router_weights(logits, indices, scoring="softmax", renormalize=False, scale=1.0)`
uses public ruDNN GPU kernels and first-order autograd. Logits are contiguous
FP32/FP16/BF16 `[tokens,experts]`; indices are contiguous int32/int64
`[tokens,top_k]`, with `1 <= top_k <= min(experts,64)`. Weights are FP32.
Selection/grouping/correction bias remain model-owned. Invalid device indices
produce bounded NaN rows, not an index exception or a CPU fallback. Repeated
indices have gather semantics. No higher-order derivatives or auxiliary router
loss are implemented. New optional Router API 1 requires rebuilding both sides;
base tensor ABI 10, training API 4 and graph API 2 are unchanged.

Run `python ruda-torch/tools/validate_router.py --build --output ./v30-results`
from the repository root for strict Rust/public-operator and native-PyTorch GPU
validation.

## Native training and optimizer use

`ruda_torch.RMSNorm`, `LayerNorm`, `rms_norm`, `layer_norm` and `silu_mul` expose first-order training on native `ruda:0` storage. Normalization is last-axis only, with FP32 saved statistics; the explicit normalization APIs allow FP32 affine parameters with FP16/BF16 activations. Native `mean` reduces from storage in FP32 without first rounding a low-precision sum.

`ruda_torch.AdamW` uses FP32 master parameters and moments. Its default path unscales gradients in place and reads a 4-byte finite flag before updates. Explicit `fused_step=True` instead keeps gradient storage unchanged and reads a 12-byte statistics report; it still launches one update kernel per active parameter. `max_grad_norm` is optional global clipping across all active parameter groups and requires the fused path. `hierarchical_stats=True` also requires that path. Both boolean options default to `False`. Clear gradients before accumulating the next step.

On one `ruda_torch.GradScaler` instance, use `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, then `scaler.update()`. Save model, optimizer and scaler states together with data position and RNG state. See the [training guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md) for checkpoint options and the examples [train_native.py](python/examples/train_native.py), [train_paged_block.py](python/examples/train_paged_block.py) and [train_router.py](python/examples/train_router.py). These training paths do not add training support to StaticGraph.
