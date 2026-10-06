# Native fixed-address PyTorch subgraphs

[Documentation](README.md) · [Model compiler](model-compiler.md) · [中文](../zh/static-pytorch-graphs.md)

## Constructor and bindings

`StaticGraph(inputs, nodes, outputs=None, infer_dependencies=False, track_completion=False, optimize=False, reuse_workspace=False, training=False)` executes explicit native GraphOp plans through RUDA's existing CudaGraph and device runtime. It is not arbitrary model capture or a replacement for `torch.cuda.graph`; use [compile](model-compiler.md) for ordinary models.

`inputs` maps nonempty names to fixed-address, contiguous, dense `ruda:0` FP32/FP16/BF16 tensors. Shapes are nonempty rank 1–8 with at most uint32 elements. All options are explicit booleans. At most 256 nodes and 512 total tensors are allowed. `outputs` is a sequence of unique produced names; by default the last node is returned. A node cannot overwrite an existing name or depend on a later node.

| Node kind | Metadata / scalar contract |
| --- | --- |
| `copy` and unary pointwise operations | One input; no right tensor, canonical zero scalar; see [`UNARY_CODES`](../../ruda-torch/python/ruda_torch/_graph_spec.py). |
| `add`, `mul`, `div`, activation backward, `silu_mul` | Equal-shape/dtype tensor pair, no broadcasting. Add's scalar is alpha; other scalar fields are zero. |
| `add_scalar`, `mul_scalar`, `div_scalar` | One tensor and a finite FP32 scalar, no right tensor. Subtraction can be represented by an explicit negative add scalar/alpha. |
| `mm`, `bmm` | Equal-dtype rank-2/rank-3 matrices; matching inner and batch dimensions, no broadcast; zero scalar. |
| `rms_norm` | Optional same-dtype `[last_width]` weight; explicit positive finite FP32 epsilon. |
| `softmax`, `log_softmax` and their backward nodes | Scalar is a canonical nonnegative axis; backward uses a matching gradient/output tensor pair. |
| `sum_keepdim`, `mean_keepdim` | Scalar is a nonempty in-range dimension bitmask, not a result shape. Reduced dimensions remain size one. |

`GraphOp(kind, output, left, right=None, scalar=0.)` is the general constructor. Convenience constructors are `copy`, `add(alpha=...)`, `mul`, `silu`, `silu_mul` and `rms_norm(eps=...)`. Every scalar is fixed at construction; replace the plan to change geometry or scalar values. Complete codes and checks: [`_graph_spec.py`](../../ruda-torch/python/ruda_torch/_graph_spec.py).

```python
import torch
import ruda_torch as r

x = torch.ones(1, 4096, dtype=torch.float16).to('ruda:0')
up = torch.ones_like(x)
with r.StaticGraph(
    {'x': x, 'up': up},
    [r.GraphOp.silu('activation', 'x'), r.GraphOp.mul('y', 'activation', 'up')],
    optimize=True, reuse_workspace=True,
) as graph:
    y = graph.replay()['y']
    graph.synchronize()
    print(graph.info)
```

## Optimization and scratch lifetime

Both `optimize` and `reuse_workspace` default to False. Optimization validates the **whole** supplied plan before pruning unused branches; illegal unused nodes still fail. It fuses a single-use left-hand SiLU followed by multiply, unless the SiLU result is requested or has other consumers. It does not swap the right operand to create more matches.

Fused `silu_mul` rounds SiLU to the original storage dtype before multiplication, preserving that low-precision boundary. It removes the separate activation allocation; this is not a promise of end-to-end performance.

Workspace reuse only reuses equal-shape/dtype whole scratch buffers after their last consumer. It does not alias inputs or requested outputs, slice larger buffers, or overwrite a value within the kernel that still reads it. The bridge independently checks reuse lifetimes. `infer_dependencies=True` accounts for real reused addresses; the added dependencies can reduce parallelism.

`info.workspace_bytes` measures unique node-output allocations, `logical_workspace_bytes` the planned outputs without reuse, and `unoptimized_workspace_bytes` the original plan. Inputs, weights, driver graphs, parameter metadata, compiler cache, allocator residency and training snapshots are outside those counts. They are not peak VRAM measurements.

## Replay, completion and output ownership

- Update input **contents** in place; do not replace storage, shape, strides, dtype or device. Each replay rechecks bindings and requires the creation stream.
- `replay()` returns a mapping of output names to tensors. In inference mode, those tensors alias graph output buffers and the next run overwrites them. Clone and order work correctly when keeping earlier results.
- `run_eager()` enqueues the same planned kernels without graph launch. With optimization enabled, it is still the optimized plan; use a separate unoptimized plan for that comparison.
- `synchronize()` waits on the owning queue; `query()` checks readiness. `track_completion=True` enables per-run completion tracking with `query_completion()` / `wait_completion()`.
- Global asynchronous dispatch semantics are unchanged. External input writers and output readers on other streams need explicit dependencies.
- `close()` waits before releasing graph resources; retained output tensors remain valid afterward. Closed plans reject further execution. Context-manager exit calls close.

## Explicit first-order training

`training=False` rejects inputs with `requires_grad`. `training=True` adds an autograd edge to native forward output, saves per-forward input snapshots and returns independent result storage. Backward recomputes supported operations on the same device; neither backward nor optimizer updates become a captured native graph. This costs additional storage and only supports first-order gradients.

```python
x = torch.ones(2, 8).to('ruda:0').requires_grad_()
with r.StaticGraph(
    {'x': x}, [r.GraphOp.silu('y', 'x')], training=True,
) as graph:
    graph.replay()['y'].float().mean().backward()
    print(x.grad)
```

Do not modify original inputs/parameters before the corresponding backward finishes; snapshots do not disable PyTorch version checks. Finish backward before closing a training graph. Use the general AOT model entry point when PyTorch-generated derivatives and automatic native partitioning are wanted; `StaticGraph.from_model` returns that callable wrapper, not this fixed-input plan.

Current native components require base ABI 10 and graph API 3. Prepare both from the same source or a matching [precompiled bundle](../../ruda-torch/README.md#linuxcolab-precompiled-bundle). This API exposes one native NVIDIA device; it does not provide AMD/Intel graph adapters, hidden broadcasts, attention/MoE nodes or whole-optimizer capture. Implementation: [`_graph.py`](../../ruda-torch/python/ruda_torch/_graph.py), [autograd bridge](../../ruda-torch/python/ruda_torch/_graph_autograd.py).
