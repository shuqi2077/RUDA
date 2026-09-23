# ruda_ptx 0.10.0 — PTX and explicit native ISA candidate backends

This package does not build CUDA C++ extensions. It retains the v9 PTX operators,
adds opt-in four-wide FP32 elementwise kernels, and adds explicit native-ISA
compilation/loading reference paths. It is NOT the integrated RUDA Rust runtime,
not a PyTorch native device, and not a validated full-model backend.

## Install

With PyTorch and NumPy already installed:

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m ruda_ptx isa-info
```

For AMD offline code generation install `llvmlite>=0.47,<0.48` and provide `ld.lld`
in PATH. This candidate requires LLVM 20 and is tested with LLVM 20.1.8 and
LLD 17.0.0 on Linux. GPU execution additionally requires the vendor's driver/runtime.

## PTX and NVIDIA native cubin

```python
import torch
from ruda_ptx import compile_exported, Executor
from ruda_ptx.nvidia_isa import NvidiaIsaRuntime

class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y

x, y = torch.randn(1025), torch.randn(1025)
plan = compile_exported(torch.export.export(Add().eval(), (x, y)),
                        vectorized_elementwise=True)
with NvidiaIsaRuntime(cache_dir="/tmp/my-private-ruda-isa-cache") as runtime:
    with Executor(plan, runtime) as executable:
        result = executable(x, y)
        torch.testing.assert_close(result, x + y)
```

This uses explicit host tensor I/O, NOT CPU computation fallback or zero-copy
PyTorch device registration. The existing device-buffer Executor path is retained.
The driver compiles PTX to native cubin on a cache miss. The image cache includes
exact SM and compiler/driver identity; it is not an offline SASS assembler.
Prepared calls reuse ctypes parameter storage, not a GPU execution graph.
Only load trusted binary artifacts; cache checksums are not digital signatures.

## AMD native ISA

```bash
python -m ruda_ptx emit-isa --backend amdgcn --target gfx90a \
  --op silu_mul --elements 1025 --vector-width 4 --emit-dir /tmp/ruda-native
python examples/export_isa.py --target gfx90a --emit-dir /tmp/ruda-small-graph
```

The first command creates real LLVM IR, AMD assembly, relocatable object, linked
`.hsaco`, and a checked `.risa` artifact. The second lowers a two-kernel PyTorch
activation graph. Add `--run` ONLY with the appropriate AMD device and HIP runtime.
This lowers operator semantics independently; it does NOT translate PTX text.

Scope: contiguous FP32 add/mul/SiLU/SiLU-times-up, gfx90a/gfx942/gfx1100. These are
compiler targets, not hardware-validated devices. Whole-graph preflight rejects
unsupported operations before compilation. There is no Intel native backend,
AMD GEMM/attention/MLA/MoE support or hidden vendor/CPU fallback.

`HipIsaRuntime(target)` explicitly loads AMD native images. `NativePlanRuntime`
adapts those images to the existing Executor and uses native launch geometry.
The target is explicitly supplied, not auto-probed; HIP loading enforces physical
compatibility. Four-wide AMD operands must be 16-byte aligned.

## Tests

```bash
python -m pytest -q -m 'not gpu and not amd_gpu' tests
RUDA_REQUIRE_GPU=1 python -m pytest -q -m gpu tests
RUDA_AMD_TARGET=gfx90a RUDA_REQUIRE_AMD_GPU=1 python -m pytest -q -m amd_gpu tests
```
