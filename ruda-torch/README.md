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

### Release wheels

[v0.3.0](https://github.com/shuqi2077/RUDA/releases/tag/v0.3.0) includes wheels with the Python package, C++ extension and optimized Rust `release` library (ABI 10). These do not require a local Rust/C++ build. Use CPython 3.13 and the matching PyTorch build:

| Platform | Wheel | Required PyTorch |
| --- | --- | --- |
| Linux x86_64, built on Ubuntu 24.04 | `ruda_torch-0.3.0-cp313-cp313-linux_x86_64.whl` | `2.10.0+cu126` |
| Windows x86_64 | `ruda_torch-0.3.0-cp313-cp313-win_amd64.whl` | `2.13.0+cu130` |

Download the matching wheel and `SHA256SUMS` from the release, and compare the wheel's SHA-256 digest with its entry (`sha256sum <wheel>` on Linux or `Get-FileHash <wheel> -Algorithm SHA256` on Windows). `manifest.json` records the source revision and each wheel's Python, PyTorch, platform, native ABI and build profile. Install from the download directory on Linux:

```sh
python -m pip install 'torch==2.10.0+cu126' --index-url https://download.pytorch.org/whl/cu126
python -m pip install ./ruda_torch-0.3.0-cp313-cp313-linux_x86_64.whl --index-url https://download.pytorch.org/whl/cu126
```

On Windows:

```powershell
python -m pip install 'torch==2.13.0+cu130' --index-url https://download.pytorch.org/whl/cu130
python -m pip install ./ruda_torch-0.3.0-cp313-cp313-win_amd64.whl --index-url https://download.pytorch.org/whl/cu130
```

The overall release is v0.3.0; the Python subpackage uses version 0.3.0 and other crates retain independent versions. A compatible NVIDIA GPU/driver and the runtime dependencies of the selected compiler path are still required. The Linux wheel is not tagged manylinux; Ubuntu 24.04 is its build environment, not a promise of compatibility with older distributions. Other Python/PyTorch versions require a source build. Do not mix these wheels with another revision's Python sources or a stale `RUDA_TORCH_LIBRARY` override. Select PTX/NVRTC before execution as above.

Wheel metadata requires Python `>=3.13,<3.14` and the exact PyTorch version, including its CUDA suffix. The compiler, C++ headers and Rust toolchain are not needed to install a wheel; NVIDIA driver and compiler-path runtime requirements still apply. License texts are included under `ruda_torch/licenses/`. Source builds record the Python minor version and PyTorch version used to compile the extension rather than claiming cross-version binary compatibility. To upgrade from the v0.1.95 wheels, use the matching 0.3.0 wheel in the commands above and start a new Python process before importing it.

The [build workflow](https://github.com/shuqi2077/RUDA/actions/workflows/ruda-torch-windows.yml) also provides CI wheel artifacts, retained for seven days; release assets are a separate download source.

### Linux/Colab precompiled bundle

The [Colab T4 compiled bundle workflow](https://github.com/shuqi2077/RUDA/actions/workflows/colab-t4-build.yml) publishes compiled bundles in [GitHub Releases](https://github.com/shuqi2077/RUDA/releases) under `colab-t4-<full-source-revision>` tags. The bundles target Linux x86_64, CPython 3.13, PyTorch `2.10.0+cu126` with CXX11 ABI enabled, and glibc 2.39 or newer. A compatible NVIDIA GPU and driver are still required; dtype and instruction support depend on the GPU.

1. Download the `.tar.gz`, `manifest.json` and `SHA256SUMS` from the same release. In their download directory, run `sha256sum --check SHA256SUMS` before extracting.
2. Check out the RUDA source revision recorded as `source_revision` in the manifest, and extract the archive into `ruda-colab-t4/` inside that checkout. Do not combine an older bundle with unrelated `main` Python sources.
3. Install the matching PyTorch build with `python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126` using CPython 3.13.
4. Copy `ruda-colab-t4/cpp-extension/_C.cpython-313-x86_64-linux-gnu.so` into `ruda-torch/python/ruda_torch/`. Before importing the package, run the following from the matching checkout's root, in the Python process that will use RUDA:

```python
import os
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / 'ruda-torch/python'))
os.environ['RUDA_TORCH_LIBRARY'] = str(root / 'ruda-colab-t4/native-library/libruda_torch_native.so')
os.environ['RUDA_CUDA_COMPILER'] = 'ptx'
os.environ['RUDA_PTX_VERSION'] = '8.0'
import torch
import ruda_torch
```

The archive contains compiled libraries and executables, not a pip wheel. This path loads both native components without running Cargo, a C++ build or an editable pip install. Select a driver-supported PTX version as above; ABI 10, graph API 3 and training API 4 must match between the loaded components.

## Basic use

```python
import torch
import ruda_torch

x = torch.arange(4, dtype=torch.float32).to('ruda:0')
print((x + x).cpu())
```

Import registers `torch.ruda` and the device. Another PrivateUse1 backend cannot already be registered in the same process. RUDA tensors are distinct from `torch.cuda` tensors; use the `ruda` device explicitly. Unsupported operators raise an error rather than silently executing through a generic CPU fallback.

## Native operators

- Matrix operations including `mm`, `bmm` and `addmm` use ruBLAS or the selected scalar GPU path. `RUDA_TORCH_MATMUL` accepts `auto` (default), `rublas` or `scalar`. Eligible CUDA FP16/BF16 GEMM/BMM uses ordered tiled Tensor Core kernels with FP32 accumulation. Tile selection is cached separately by input/output dtype, shape and layout; a new key can incur substantial first-call compilation and tuning work. See the [ruBLAS dispatch requirements](../docs/en/libraries/rublas.md#tensor-matrix-multiplication).
- FP32/FP16/BF16 paths retain their storage dtype where supported. Eligible last-axis LayerNorm/RMSNorm forward uses cached FP32 row values for widths 512–16384, with centered variance for LayerNorm and FP32 saved training statistics. Other widths, HF-specific RMSNorm rounding and backward retain their existing paths.
- Softmax/log-softmax and their backward kernels support `RUDA_TORCH_SOFTMAX=auto|warp|scalar`. Default `auto` selects cached-row forward for contiguous last axes of width 512–16384, otherwise the warp path for axis lengths of at least 32 or the scalar path. Explicit overrides and backward selection are unchanged; ordinary softmax retains accurate exponential evaluation.
- Standard `torch.nn.functional.scaled_dot_product_attention` dispatches eligible RUDA tensors to native attention with first-order autograd, without replacing the global Python function. Eligible FP16/BF16 prefill uses shared-memory reuse and fast exponential evaluation with FP32 accumulation; this is not bitwise-equivalent to the accurate-exp path. See [dense attention contracts](../docs/en/native-pytorch-api.md#dense-attention).

Operator-specific shape, dtype and layout checks still apply. See the [operator registrations](https://github.com/shuqi2077/RUDA/blob/main/ruda-torch/python/ruda_torch/_ops.py) and [native kernels](https://github.com/shuqi2077/RUDA/tree/main/ruda-torch/src).

## Scientific operations

Scientific GPU operations expose `ruda_torch.gemm`, `fused_gemm`, `grouped_gemm` and `triangular_solve`, together with standard `torch.linalg` and `torch.fft` dispatch on RUDA tensors. GEMM and FFT support FP32/FP64 and complex64/complex128; solver operations use real FP32/FP64. The scientific, BLAS, linalg and FFT bridges each negotiate API 1 in addition to base ABI 10; rebuild both native components to use these capabilities.

See the [scientific API reference](../docs/en/native-pytorch-api.md#scientific-operations) ([中文](../docs/zh/native-pytorch-api.md#科学计算算子)) for signatures, driver restrictions, gradients and explicit `ruda_torch.fft.FFTPlan` ownership. These dtype extensions do not extend every random, training or attention operator to FP64/complex storage. FFT does not silently promote FP16/BF16/complex32. FP64 `erf` requires the NVRTC compiler path.

## Streams, events and asynchronous dispatch

Dispatch is synchronous by default. Set `RUDA_TORCH_ASYNC=1` before the first native submission to enable asynchronous dispatch; this setting is cached for the process. Explicit synchronization, scalar extraction and host readback still wait for completion.

`Stream`, `Event`, `stream`, `current_stream`, `default_stream` and `record_stream` use RUDA's native runtime on device `ruda:0`. `get_stream_priority_range()` queries the driver-supported range; `Stream(priority=...)` negotiates priority support before requesting a nonzero priority, and `Stream.priority` reports the effective value. Reuse streams: the pool is bounded by the runtime's `streaming.max_streams` setting.

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

Floating-cache inputs must be contiguous FP32/FP16/BF16 tensors with the same dtype on the native `ruda` device and matching execution queue. Query-head count must be divisible by KV-head count. Supply finite Q/K/V and a finite positive scale; key/value feature dimensions are limited to 1024. Forward and first-order autograd are supported; arbitrary external masks and higher-order derivatives are not. Packed caches use the separate inference-only [quantized KV attention](../docs/en/native-pytorch-api.md#quantized-kv-attention) path.

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

## Optional fixed-address subgraphs (v24 and opt-in training)

`ruda_torch.StaticGraph` connects selected native PyTorch tensor operations to
RUDA's existing `CudaGraph`, rather than a separate CUDA extension/runtime.
The optional graph interface is version 3; the current base tensor ABI is 10.
Both Rust and C++ components must provide graph API 3. Defaults and eager dispatch stay unchanged.

Supported explicit nodes include copy, unary pointwise operations, same-shape
add/mul/div, scalar arithmetic, `mm`/`bmm`, keepdim sum/mean, softmax/log-softmax
and their backward operations, SiLU/sigmoid/tanh backward, last-axis RMSNorm and
storage-rounded SiLU-mul. See the operator codes and metadata checks in
[`_graph_spec.py`](python/ruda_torch/_graph_spec.py). Explicit RMSNorm nodes do
not imply automatic AOT recognition; eligible SiLU-mul expressions have a separate
[pre-AOT route](../docs/en/model-compiler.md#exact-native-operator-coverage) in `native='auto'`.
Opt-in `optimize=True` removes unused nodes and fuses single-use SiLU on either side of multiplication;
`reuse_workspace=True` reuses only equal-spec scratch storage after its last read.
Returned outputs remain dedicated. Both options default to False pending GPU validation.
Explicit `training=True` enables the first-order gradient bridge described in
the static-graph guide. This low-level API is not arbitrary model/stream capture
or a replacement for `torch.cuda.graph`. See the [static-graph guide](../docs/en/static-pytorch-graphs.md)
for fixed-pointer, output-reuse and synchronization contracts.

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

Rebuild both Rust and C++ (base ABI 10 / graph API 3 / training API 4).
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
base tensor ABI 10 and training API 4 are unchanged; the native-coverage extension
requires graph API 3 in both rebuilt native components.

Run `python ruda-torch/tools/validate_router.py --build --output ./v30-results`
from the repository root for strict Rust/public-operator and native-PyTorch GPU
validation.

## Native training and optimizer use

`ruda_torch.RMSNorm`, `LayerNorm`, `rms_norm`, `layer_norm` and `silu_mul` expose first-order training on native `ruda:0` storage. Normalization is last-axis only, with FP32 saved statistics; the explicit normalization APIs allow FP32 affine parameters with FP16/BF16 activations. Native `mean` reduces from storage in FP32 without first rounding a low-precision sum.

`ruda_torch.AdamW` uses FP32 master parameters and moments. Its default path unscales gradients in place and reads a 4-byte finite flag before updates. Explicit `fused_step=True` instead keeps gradient storage unchanged and reads a 12-byte statistics report; it still launches one update kernel per active parameter. `max_grad_norm` is optional global clipping across all active parameter groups and requires the fused path. `hierarchical_stats=True` also requires that path. Both boolean options default to `False`. Clear gradients before accumulating the next step.

On one `ruda_torch.GradScaler` instance, use `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, then `scaler.update()`. Save model, optimizer and scaler states together with data position and RNG state. See the [training guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/training.md) for checkpoint options and the examples [train_native.py](python/examples/train_native.py), [train_paged_block.py](python/examples/train_paged_block.py) and [train_router.py](python/examples/train_router.py). These optimizer and paged-training paths are independent of the explicit StaticGraph gradient bridge and general model compiler described above and below.


## General model and training-graph entry point

`ruda_torch.compile(model)` accepts ordinary `torch.nn.Module` objects or Python
callables. PyTorch AOTAutograd generates forward/backward graphs; supported pure
ATen regions use the native StaticGraph, while other operations keep their
original-device dispatch. `StaticGraph.from_model(model)` is an alias returning
a callable wrapper (not a fixed-input object with `replay()`).

```python
model = model.to('ruda:0')
compiled = ruda_torch.compile(model, native='auto', fullgraph=False)
loss = compiled(inputs).square().mean()
loss.backward()
print(compiled.info)
compiled.close()  # after every outstanding backward has completed
```

This removes the hand-written GraphOp/model derivative restriction. It does
**not** implement every missing device operator or promise full native training
capture, arbitrary optimizer capture, GPU validation, or a speedup. Native
regions recognize exact overloads for clone, unary pointwise operations,
tensor/scalar add/sub/mul/div, `mm`/`bmm`, keepdim sum/mean, softmax/log-softmax
and their backward operations, and SiLU/sigmoid/tanh backward. Runtime metadata
guards still apply; no CPU fallback or retry-after-dispatch is installed.
`native='off'` isolates AOT capture from native regions; `capture='eager'`
explicitly uses normal PyTorch execution rather than claiming compilation.
Parameter identities and top-level original checkpoint keys are retained. A wrapper saved as a child module retains its structural `_original` prefix for recursive checkpoint loading.

See the [model compiler guide](../docs/en/model-compiler.md) ([中文](../docs/zh/model-compiler.md))
for execution policies, cache ownership, custom-op decompositions and limitations.
The full native acceptance command is
`python ruda-torch/tools/validate_model_compile.py --output model-gpu.json`.

### Native coverage and learned-scale extension (graph API 3)

The graph API 3 extension maps matrix multiplication, keepdim reductions, softmax
and more activation forward/backward operations into native regions. Both native
components must provide graph API 3. `LearnedFakeQuantize` adds trainable FP32 scales,
but returns floating tensors and is not packed INT4 inference.
See the [model compiler guide](../docs/en/model-compiler.md#exact-native-operator-coverage) for exact
native-region overloads, supported metadata and execution policies.

## Packed quantization, GPTQ and GGUF

`PackedQuantizedTensor`, `PackedLinear`, `PackedEmbedding` and `quantize_packed` provide native packed INT2/3/4/5/6/8, FP4 E2M1 and FP8 E4M3/E5M2 storage and execution. Linear supports input gradients with frozen weights and can be wrapped by LoRA; the current packed matmul uses FP32 scalar accumulation, not Tensor Core GEMM. `scaled_float8_mm` returns FP16/BF16/FP32, not FP8.

`gptq_linear` / `load_gptq_safetensors` import standard 2/3/4/8-bit GPTQ v1/v2 with `g_idx`. `GGUFWeights` / `load_gguf` import GGUF v2/v3 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and Q2_K through Q6_K plus Q8_K. Import repacks integer codes without requantizing or materializing full floating weights. Construct the model explicitly and supply selected GPTQ modules or GGUF name/row-layout mappings; Marlin and GGUF IQ/TQ formats are not supported.

`calibrate_quantization` provides MinMax, Percentile and MSE calibration; `QuantizationObserver` provides streaming min/max or EMA statistics, and `affine_fake_quantize` provides affine QAT. `QuantizedKVCache` stores packed pages for direct MHA/GQA/MLA attention, without whole-history expansion; this path is inference-only. See the [API contracts and examples](../docs/en/native-pytorch-api.md#packed-quantization-and-checkpoint-import).

## LoRA, NF4 and causal fine-tuning

`inject_lora` freezes a model's base and creates adapters on exact selected projections. `quantize_nf4` prepares packed CPU weights before adapter injection; `load_nf4_safetensors` streams an explicitly constructed meta model, and `load_hf_nf4_model` constructs a local HF architecture without downloads or remote code.

`CausalLMFinetuner`, `SFTCollator` and `SFTTrainer` provide explicit backbone/head training, assistant-label handling, full-vocabulary token-chunk loss and token-weighted accumulation. `adapter_state_dict` / `load_adapter_state_dict` exchange adapters; `finetune_state_dict` / `load_finetune_state_dict` add optimizer/scaler/RNG and data-cursor recovery at cleared-gradient step boundaries. Packed NF4 is not learned fake quantization or a PEFT/bitsandbytes checkpoint reader.

See [fine-tuning and recovery](../docs/en/finetuning.md) ([中文](../docs/zh/finetuning.md)) for setup order, parameter/shape limits and the existing [CLI example](python/examples/finetune_causal_lm.py). The [native API reference](../docs/en/native-pytorch-api.md) ([中文](../docs/zh/native-pytorch-api.md)) covers normalization, optimizers, streams, attention, sequence training and component-version requirements.

The [architecture training guide](../docs/en/architecture-training.md) ([中文](../docs/zh/architecture-training.md)) covers mHC, DSA/CSA/HCA, compressed KV-cache ownership, hybrid model composition and Python Muon parameter groups.
