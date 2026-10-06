# General PyTorch model compilation

[Documentation](README.md) · [Native API](native-pytorch-api.md) · [中文](../zh/model-compiler.md)

## Model and function entry points

`ruda_torch.compile(model)` wraps an ordinary `torch.nn.Module` or callable. PyTorch AOTAutograd generates forward/backward graphs; eligible regions use RUDA StaticGraph and other operations retain their original-device dispatch. You do not write `GraphOp` nodes or model-specific derivatives for this entry point. Device kernels and custom-op tracing/autograd rules are still required for all operations the model executes.

`StaticGraph.from_model(model, **options)` is the same entry point. It returns a callable `CompiledModel`/`CompiledFunction`, **not** a fixed-input object with `replay()`.

```python
import torch
import ruda_torch

model = torch.nn.Sequential(
    torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 4),
).to('ruda:0')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
compiled = ruda_torch.compile(model, native='auto')
x = torch.ones(3, 8).to('ruda:0')
target = torch.zeros(3, 4).to('ruda:0')
optimizer.zero_grad(set_to_none=True)
loss = (compiled(x) - target).square().mean()
loss.backward()
optimizer.step()
print(compiled.info)
compiled.close()  # after every outstanding backward
```

Functions can also be wrapped or decorated with `@ruda_torch.compile(...)`. PyTorch retains supported nested inputs/outputs and non-tensor values. Move model parameters, buffers and call inputs explicitly to the selected device; the wrapper does not transfer them.

## Options and execution policies

| Option | Default | Contract |
| --- | --- | --- |
| `capture` | `'aot'` | `'aot'` captures forward/backward; `'eager'` explicitly runs normal uncompiled PyTorch. |
| `native` | `'auto'` | `'auto'` uses eligible regions; `'off'` keeps AOT but disables native regions; `'required'` rejects unsupported captured operators or runtime native guards. |
| `device_type` | `'ruda'` | Device type, not `'ruda:0'`; `'cpu'` is an explicit compiler-reference target, not native GPU execution. |
| `fullgraph` | `False` | Allow Python graph breaks; `True` rejects breaks, but does not require every operation to be a native node. |
| `dynamic` | `None` | Passed to `torch.compile`; otherwise a Python bool. |
| `min_native_ops` | `2` | Integer 1–256; minimum consecutive eligible operations per region in auto mode. Required mode uses a minimum of one. |
| `cache_size` | `4` | Integer 1–64, **per region**, not a whole-model memory limit. |
| `decompositions` | `None` | Mapping exact PyTorch operator overloads to caller-supplied mathematical decompositions for AOT. |

Eager mode cannot use `native='required'`, graph/dynamic options or nonempty decompositions. AOT wrapper calls reject `torch._dynamo.config.suppress_errors=True`; failed compilation must not become implicit eager re-execution. The wrapper does not modify global compiler settings.

`native='required'` examines captured operators, not uncaptured code outside them. Combine with `fullgraph=True` when reviewing an entire traceable call. Neither option automatically captures an arbitrary optimizer, `.backward()`, logs or Python side effects into one device graph. Explicitly wrapping a training-step callable can produce multiple PyTorch segments. Host-reading optimizers retain their synchronization.

## Exact native operator coverage

Native matching uses exact overloads, not similar names:

| Family | Supported overloads / restrictions |
| --- | --- |
| Copy / unary | `aten.clone.default`; `.default` unary operators from [`UNARY_CODES`](../../ruda-torch/python/ruda_torch/_graph_spec.py), including ReLU, SiLU, sigmoid, tanh, exp, log, sqrt and rsqrt. |
| Arithmetic | `add/sub/mul/div.Tensor` and `.Scalar`; tensor pairs have identical shapes/dtypes, without broadcasting or promotion; scalar values are finite FP32. |
| Matrix multiplication | `aten.mm.default`, `aten.bmm.default`; equal-dtype rank-2/rank-3 inputs, matching inner and batch dimensions, no batch broadcasting. |
| Reduction | `aten.sum.dim_IntList`, `aten.mean.dim`; `keepdim=True`, no explicit dtype conversion. |
| Softmax | `_softmax.default`, `_log_softmax.default`, `_softmax_backward_data.default`, `_log_softmax_backward_data.default`; no implicit half-to-FP32 output promotion, matching backward dtype. |
| Activation backward | `silu_backward.default`, `sigmoid_backward.default`, `tanh_backward.default`. |

Runtime inputs must be dense strided `ruda:0` tensors without unresolved conjugate/negative views. Supported specs are nonempty rank 1–8 FP32/FP16/BF16, with at most uint32 elements. Inputs are staged into contiguous storage; eligible operation outputs must have contiguous metadata. Each region allows at most 256 operations and 512 tensors. Explicit GraphOp RMSNorm/SiLU-mul nodes do not mean AOT automatically recognizes those composite operations.

Views, mutations, RNG and unsupported operations keep original-device execution in auto mode, or cause a required-mode error. Missing original-device kernels still fail; auto mode is not CPU fallback. Exact lowering and argument guards: [`_compile_native.py`](../../ruda-torch/python/ruda_torch/_compile_native.py).

## State, gradients, memory and streams

`CompiledModel` preserves parameter identities, so an optimizer may be created before wrapping. Direct wrapper `state_dict()` / `load_state_dict()` retain top-level original keys. When saved as a child module, its parent retains the `_original` structural prefix for recursive loading. `train()` / `eval()` affect the original module and `.original` returns it.

AOT mode supports first-order gradients, not higher-order backward. Explicit eager mode retains whatever derivatives the underlying device operators support. Activation checkpointing and shared/frozen parameters remain subject to PyTorch and the model's operator contracts; wrapping does not add missing kernels.

Tracing only uses tensor metadata and does not allocate native graphs or read FakeTensor data. First real invocation allocates fixed-address staging. Cache identity includes shapes, strides, dtypes, device, stream and inference-mode state. Each region uses bounded LRU eviction; failed handle closure is propagated rather than discarding an unreleased handle.

Before returning, region outputs are cloned into independent storage so later replay cannot overwrite saved activations or earlier results. Staging and copies cost device memory and time; native capture alone does not imply a speedup. A region serializes its own copy/replay/output operations. Caller-owned cross-stream dependencies must still be ordered explicitly.

Call `close()` only after all backward work finishes. Closed wrappers reject later calls or lazy backward compilation. Context-manager exit also closes the wrapper. Explicit closure can be retried if releasing native resources fails.

## Coverage information and errors

`compiled.info` and backend `.info` contain captured forward/backward operators, planned regions, build/replay/cache counters, executed native node counts, original-device reasons and errors. Plans and actual execution are different: inspect `native_replays`, `native_nodes_executed` and `reference_reasons`, not only candidate counts. This mapping is JSON-serializable, not a complete inventory of Python graph breaks.

- `NativeCoverageError`: required-native policy cannot support a captured operation or runtime region metadata.
- `GraphExecutionError`: captured execution failed on its original device; the model call is not retried and is not moved to CPU.
- Option/device/decomposition errors use `ValueError` or `TypeError`. Allocation, build and replay failures propagate with their cause.

Only known capability/metadata limitations can select original-device execution **before dispatch**. An execution failure never triggers replay through another path.

## Custom backend and decompositions

```python
backend = ruda_torch.make_backend(native='auto', cache_size=4)
compiled = torch.compile(model, backend=backend, fullgraph=False)
# Use compiled normally. After every backward is complete:
print(backend.info)
backend.close()
```

`make_backend()` is for external `torch.compile` calls; the convenience wrapper additionally checks public inputs and module devices. External compiler configuration remains the caller's responsibility. Custom operators need device kernels, FakeTensor/meta rules and training autograd rules. `decompositions={exact_overload: callable}` is an explicit caller-owned mathematical transformation, not permission to replace precise activations with approximations.

Related guides: [fixed-address graphs](static-pytorch-graphs.md), [fine-tuning](finetuning.md), [native components](../../ruda-torch/README.md). Implementation: [compiler.py](../../ruda-torch/python/ruda_torch/compiler.py).
