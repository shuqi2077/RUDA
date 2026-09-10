# Examples and Tutorials

[Documentation](README.md) · [Quickstart](getting-started.md) · [中文](../zh/samples.md)

## 1. NVIDIA runtime example

Entry point: [ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs).

The basic case performs FP32 addition twice at each length: 1, 63, 64, 65, and 257. It checks every result and 16 tail sentinel values. It demonstrates device selection, upload, argument binding, tail bounds, readback, and repeated execution.

See the [quickstart](getting-started.md) for building and selecting a backend.

## 2. Targeted cases

Place these arguments after the command's `--`. For example, run the tensor case with:

```powershell
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --tensor
```

| Argument | Behavior | Example source |
| --- | --- | --- |
| `--tensor` | Tensor metadata and layout checks | [tensor.rs](../../ruda-driver-cuda/examples/ptx_runtime/tensor.rs) |
| `--shared` | Shared memory checks | [shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs) |
| `--half` | FP16/BF16 checks | [half_precision.rs](../../ruda-driver-cuda/examples/ptx_runtime/half_precision.rs) |
| `--bitwise` | Bitwise operation checks | [bitwise.rs](../../ruda-driver-cuda/examples/ptx_runtime/bitwise.rs) |
| `--shared-over-limit` | Shared memory limit diagnostics | [shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs) |
| `--expect-cold` | Asserts compilation occurred without a disk cache hit | [Main entry point](../../ruda-driver-cuda/examples/ptx_runtime.rs) |
| `--expect-warm` | Asserts a disk cache hit without recompilation | [Main entry point](../../ruda-driver-cuda/examples/ptx_runtime.rs) |

`--shared-over-limit` takes a separate early-return branch. Do not combine it with cold/warm cache checks.

## 3. Cold and warm caches

Use a separate new cache directory for each compilation path. Reuse that directory for the cold and warm runs of the same path.

After selecting a compilation path as described in [Getting started](getting-started.md), run these commands in the same PowerShell session:

```powershell
$env:RUDA_PTX_TEST_CACHE = 'target/ptx-example-cache-' + [guid]::NewGuid().ToString('N')
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-cold
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-warm
```

Keep the compiler, input arguments, and cache path unchanged between runs.

## 4. Compute library examples

Prepare the [build environment](getting-started.md) before running examples.

| Task | Command | Output checks |
| --- | --- | --- |
| FP32 CSR matrix-vector multiplication | `cargo run --locked -p ruSPARSE --features cuda --example csrmv` | Reads back and checks `[7.0, 2.0, 18.5]` |
| CUDA Ring AllReduce | `cargo run --locked -p ruCCL --features cuda --example all_reduce` | Four logical ranks on GPU 0, 257 elements, Sum/Mean, and input preservation |

## 5. Training and model inference

- [Training and saving state](training.md): forward, backward, gradient accumulation, and training records.
- [Model loading and inference](model-inference.md): Qwen2/Qwen3.5 example commands, chat, sampling, AWQ, and image input.
