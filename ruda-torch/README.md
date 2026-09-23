# ruda-torch

Native PyTorch integration for RUDA on a single NVIDIA GPU, exposed as `ruda:0` through PyTorch's PrivateUse1 backend.

- Rust package: `ruda-torch-native` (`cdylib`, not published to crates.io).
- Python package: `ruda-torch`; import: `ruda_torch`.
- Rust/C++ bridge: **ABI 9**. Rebuild both components together after an ABI change.

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

Inputs must be contiguous FP32/FP16/BF16 tensors with the same dtype on the native `ruda` device and matching execution queue, without gradients. Query-head count must be divisible by KV-head count. Supply finite Q/K/V and a finite positive scale; key/value feature dimensions are limited to 1024. This is forward-only, without arbitrary external masks or quantized KV caches.

`splits=1` is the default unsplit path; `2..32` uses partial attention followed by FP32 merging. `plan.workspace_bytes(query_heads, value_dim)` reports planned scratch bytes, capped at 64 MiB per workspace, not total device memory. Workspaces are cached per native plan/stream and shape, not in a global pool. Splitting is explicit and does not guarantee a speedup.

`plan.mla(absorbed_query, position_query, latent_cache, position_cache, scale=..., causal=True)` returns compressed context `[queries, heads, rank]`. The latent cache is `[pages, page_size, 1, rank]`, the positional cache `[pages, page_size, 1, position_dim]`, and the query tensors are `[queries, heads, rank]` and `[queries, heads, position_dim]`; position dimension is limited to 256. Apply positional encoding first and value/output projections afterward. Use the model's original QK scale, not `1/sqrt(rank)`. The caller prepares cache contents and owns cache updates.

See the [ruDNN guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rudnn.md) for the underlying Rust paged-attention and MoE interfaces, and the [ruLLM guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/model-inference.md) for model integration.
