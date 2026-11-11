# Native PyTorch quickstart

[Documentation](README.md) · [Native API](native-pytorch-api.md) · [中文](../zh/native-pytorch-quickstart.md)

`ruda-torch` runs PyTorch operations on one NVIDIA GPU through the `ruda:0` device. Import `ruda_torch` before allocating RUDA tensors. This is a separate backend from `torch.cuda`; CUDA tensors are not interchangeable with RUDA tensors.

## 1. Build and install

To skip local Rust/C++ compilation, use the [v0.3.0 release wheels](../../ruda-torch/README.md#release-wheels): CPython 3.13, Linux x86_64 with PyTorch `2.10.0+cu126` (Ubuntu 24.04 build), or Windows x86_64 with `2.13.0+cu130`. Install the matching wheel/PyTorch pair, then set the compiler environment below. Otherwise, build from source:

Prepare the [NVIDIA build environment](getting-started.md), Python with PyTorch and setuptools, and a C++20 compiler. On Windows, run in an x64 MSVC developer shell. From the repository root:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

Before running Python, select the compiler. For direct PTX in PowerShell:

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

Or in Bash:

```sh
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

Choose a [PTX version](ptx.md) supported by your GPU and driver. `RUDA_CUDA_COMPILER=nvrtc` selects the CUDA C++/NVRTC path instead.

The loader searches for a packaged native library, then the repository's debug library. Set `RUDA_TORCH_LIBRARY` before import when using a separately built release library or a different path; wheels already include their library. The Rust library and C++ extension must both use ABI 10. For Linux/Windows wheels and the separate Linux/Colab bundles, follow the [package installation guide](../../ruda-torch/README.md#build-and-install); use its specified Python, PyTorch, and platform versions.

## 2. Transfer data and read a result

```python
import torch
import ruda_torch

device = torch.device("ruda:0")
host = torch.arange(4, dtype=torch.float32)
x = host.to(device)
y = x + x
ruda_torch.synchronize()
result = y.cpu()
torch.testing.assert_close(result, host + host)
print(result)
print(ruda_torch.execution_stats())
```

The result is `[0., 2., 4., 6.]`. `.to(device)` uploads the input; `.cpu()` reads the output back. `synchronize()` waits for device completion. `execution_stats()` returns cumulative operation/transfer counters, not elapsed time. `is_available()` reports integration initialization, not support for every operator or GPU instruction.

## 3. Run one FP32 training step

Initialize the model and data on the host, then move both explicitly to RUDA. This small example uses first-order autograd and RUDA AdamW:

```python
import torch
import ruda_torch

torch.manual_seed(7)
device = torch.device("ruda:0")
model = torch.nn.Sequential(
    torch.nn.Linear(8, 16),
    torch.nn.SiLU(),
    torch.nn.Linear(16, 4),
).to(device)
x = torch.ones(3, 8, dtype=torch.float32).to(device)
target = torch.zeros(3, 4, dtype=torch.float32).to(device)
optimizer = ruda_torch.AdamW(model.parameters(), lr=1e-3)

optimizer.zero_grad(set_to_none=True)
loss = (model(x) - target).square().mean()
loss.backward()
optimizer.step()
ruda_torch.synchronize()
print("loss:", loss.detach().cpu().item())
```

Clear gradients before the next accumulation window. Low-precision storage, FP32 optimizer masters, loss scaling, clipping, and optimizer state have their own contracts in the [native API reference](native-pytorch-api.md). Use that guide before changing this example to FP16/BF16 or restoring an optimizer checkpoint.

## 4. Compile a model or fine-tune local weights

- [Model compiler](model-compiler.md): wrap a module or callable with `ruda_torch.compile`, select native coverage policy, and release caches after outstanding backward work completes.
- [Static subgraphs](static-pytorch-graphs.md): build explicit fixed-address `GraphOp` plans and manage replay outputs.
- [LoRA/NF4 fine-tuning](finetuning.md): prepare local weights, select exact targets, accumulate token-weighted gradients, and save/restore restart state.
- [Architecture components](architecture-training.md): attention variants, compressed KV caches, mHC, and Python Muon.

## 5. Diagnose installation and execution errors

| Symptom | What to check |
| --- | --- |
| Native library cannot load | Check the selected `RUDA_TORCH_LIBRARY` path and dependency libraries. On Windows, `RUDA_TORCH_DLL_DIR` accepts semicolon-separated dependency directories. |
| ABI mismatch or missing optional API | Rebuild the Rust library and C++ extension from the same checkout; base ABI 10 alone does not supply every optional feature. |
| Another PrivateUse1 backend is registered | Use a Python process in which another backend has not already claimed PrivateUse1. |
| An operator is unsupported | Check the [native API](native-pytorch-api.md) and the operator's shape/dtype/device requirements. The eager backend does not install a generic CPU fallback. |
| Compiled model rejects an operation | Check [compiler options and coverage](model-compiler.md). `native='auto'` retains original-device dispatch for eligible fallback cases, not CPU execution. |

Device errors propagate; importing the backend or observing a launch counter does not by itself establish a correct result. For compiler and cache diagnostics, see [Debugging](debugging.md).
