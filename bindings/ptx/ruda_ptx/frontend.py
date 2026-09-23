"""Strict torch.export -> static execution plan -> PTX, without eager fallback."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
from typing import Any
from .emitter import TensorSpec, Kernel, elementwise, rms_norm, matmul
from .reductions import softmax, rms_norm as fused_rms_norm
from .decode_linear import linear_decode, gated_linear
from .attention import attention_program


class UnsupportedGraph(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    inputs: tuple[str, ...]
    output: str
    kernel: Kernel


@dataclass
class Plan:
    specs: dict[str, TensorSpec]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    steps: tuple[Step, ...]
    constants: dict[str, Any] = field(repr=False)
    in_spec: Any = field(repr=False)
    out_spec: Any = field(repr=False)

    def workspace(self) -> tuple[dict[str, int], tuple[int, ...]]:
        """Best-fit reuse of nonoverlapping activation lifetimes on one stream.

        Inputs/weights are separate allocations. An output never aliases an input
        still consumed by the same launch. Final graph outputs remain live.
        """
        final = len(self.steps)
        last = {s.output: i for i, s in enumerate(self.steps)}
        for i, step in enumerate(self.steps):
            for name in step.inputs:
                if name in last:
                    last[name] = max(last[name], i)
        for name in self.outputs:
            if name in last:
                last[name] = final
        sizes, until, assignments = [], [], {}
        for i, step in enumerate(self.steps):
            size = self.specs[step.output].nbytes
            available = [j for j, capacity in enumerate(sizes) if until[j] < i and capacity >= size]
            if available:
                slot = min(available, key=lambda j: (sizes[j], j))
            else:
                slot = len(sizes)
                sizes.append(size)
                until.append(-1)
            until[slot] = last[step.output]
            assignments[step.output] = slot
        return assignments, tuple(sizes)

    def report(self):
        assignment, capacities = self.workspace()
        return {
            "format": "ruda-ptx-plan-v1",
            "compiler_version": "0.10.0",
            "executor_input_modes": ["host", "device"],
            "device_outputs_borrowed": True,
            "prefill_attention_tuned": False,
            "entry_boundary": "torch.export",
            "kernel_format": "ptx",
            "runtime": "injected_at_execution",
            "rust_runtime_connected": False,
            "cpu_compute_fallback": False,
            "inductor_fallback": False,
            "inputs": list(self.inputs), "outputs": list(self.outputs),
            "specs": {n: asdict(s) for n, s in self.specs.items()},
            "kernel_count": len(self.steps),
            "workspace_bytes": sum(capacities),
            "workspace_without_reuse_bytes": sum(self.specs[s.output].nbytes for s in self.steps),
            "input_buffer_bytes": sum(self.specs[n].nbytes for n in self.inputs),
            "constant_buffer_bytes": sum(self.specs[n].nbytes for n in self.constants),
            "workspace_slots": capacities,
            "workspace_assignment": assignment,
            "steps": [{"inputs": s.inputs, "output": s.output, "operation": s.kernel.operation,
                       "entry": s.kernel.name, "sha256": s.kernel.digest,
                       "parameters": s.kernel.parameters, "grid": s.kernel.grid,
                       "block": s.kernel.block, "target_sm": s.kernel.target_sm} for s in self.steps],
            "validation": {"ptx_assembly": "not_asserted", "gpu_execution": "not_asserted"},
        }

    def write(self, directory: str | Path):
        """Emit inspectable PTX and metadata only; never dump model weights implicitly."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Refusing to overwrite a nonempty output directory")
        directory.mkdir(parents=True, exist_ok=True)
        for step in self.steps:
            (directory / (step.kernel.name + ".ptx")).write_text(step.kernel.ptx, encoding="utf-8")
        (directory / "plan.json").write_text(json.dumps(self.report(), indent=2), encoding="utf-8")


def compile_exported(ep, *, decode_partitions: int = 1, allow_streaming_prefill: bool = False,
                     use_decode_linear: bool = True, fuse_gated_decode: bool = True,
                     decode_outputs_per_warp: int | None = None, vectorized_elementwise: bool = False) -> Plan:
    """Lower a frozen inference ExportedProgram. Unsupported nodes fail closed.

    Supports contiguous static FP32/FP16/BF16 tensors; add/mul (equal shapes),
    SiLU, last-axis affine RMSNorm, 2D mm/matmul, and linear (rank >= 2).
    Also lowers last-axis softmax and mask-free scaled-dot-product attention.
    Decode is enabled; streaming prefill is explicit opt-in, not a tuned kernel.
    No mutation, training, scalar inputs, dynamic shapes or alias ops.
    """
    import torch
    if type(vectorized_elementwise) is not bool:
        raise TypeError("vectorized_elementwise must be bool")
    def emit_elementwise(name, operation, spec):
        if vectorized_elementwise and spec.dtype == "float32":
            from .vectorized import elementwise4
            return elementwise4(name, operation, spec)
        return elementwise(name, operation, spec)
    from torch.fx import Node
    from torch.export.graph_signature import InputKind, OutputKind, TensorArgument
    if type(decode_partitions) is not int or not 1 <= decode_partitions <= 32:
        raise ValueError("decode_partitions must be an integer in [1, 32]")
    if any(type(flag) is not bool for flag in (allow_streaming_prefill, use_decode_linear, fuse_gated_decode)):
        raise TypeError("Compiler policy flags must be bool")
    if decode_outputs_per_warp is not None and (type(decode_outputs_per_warp) is not int or decode_outputs_per_warp not in (1, 2, 4)):
        raise ValueError("decode_outputs_per_warp must be None, 1, 2 or 4")
    if not isinstance(ep, torch.export.ExportedProgram):
        raise TypeError("Expected torch.export.ExportedProgram")
    if ep.range_constraints:
        raise UnsupportedGraph("Dynamic shapes are not implemented")
    if any(s.kind != OutputKind.USER_OUTPUT for s in ep.graph_signature.output_specs):
        raise UnsupportedGraph("State mutation or training outputs are not supported")
    specs, constants, inputs = {}, {}, []
    nodes = {n.name: n for n in ep.graph.nodes}

    def spec(node):
        if not isinstance(node, Node):
            raise UnsupportedGraph("This operation requires tensor inputs, not scalar constants")
        if node.name in specs:
            return specs[node.name]
        value = node.meta.get("val")
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or not value.is_contiguous() or value.device.type != "cpu"):
            raise UnsupportedGraph(f"{node.name}: this export adapter requires contiguous CPU boundary metadata")
        try:
            result = TensorSpec(tuple(value.shape), str(value.dtype).removeprefix("torch."))
        except ValueError as exc:
            raise UnsupportedGraph(f"{node.name}: {exc}") from exc
        specs[node.name] = result
        return result

    for item in ep.graph_signature.input_specs:
        if not isinstance(item.arg, TensorArgument):
            raise UnsupportedGraph("Only tensor inputs are supported")
        name = item.arg.name
        spec(nodes[name])
        if item.kind == InputKind.USER_INPUT:
            inputs.append(name)
        elif item.kind in {InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR}:
            state = ep.state_dict if item.target in ep.state_dict else ep.constants
            value = state[item.target]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.layout != torch.strided or not value.is_contiguous()):
                raise UnsupportedGraph(f"{name}: weights must already be contiguous CPU tensors at export")
            if TensorSpec(tuple(value.shape), str(value.dtype).removeprefix("torch.")) != specs[name]:
                raise UnsupportedGraph(f"{name}: constant metadata mismatch")
            constants[name] = value.detach().clone()  # Freeze: later model changes cannot alter this plan.
        else:
            raise UnsupportedGraph(f"Unsupported graph input kind: {item.kind}")

    def same(a, b):
        if spec(a) != spec(b):
            raise UnsupportedGraph("Broadcasting and mixed dtypes are not implemented")

    def arg(n, i, key, default=None):
        return n.args[i] if i < len(n.args) else n.kwargs.get(key, default)

    # Only remove a SiLU intermediate when it has a single equal-shape multiply consumer.
    fused, omit = {}, set()
    for node in ep.graph.nodes:
        if node.op == "call_function" and str(node.target) == "aten.mul.Tensor":
            a, b = arg(node, 0, "self"), arg(node, 1, "other")
            for activation, up in ((a, b), (b, a)):
                if (isinstance(activation, Node) and activation.op == "call_function"
                        and str(activation.target) == "aten.silu.default" and len(activation.users) == 1
                        and activation is not up):
                    gate = arg(activation, 0, "self")
                    same(gate, up)
                    fused[node.name] = (gate, up)
                    omit.add(activation.name)
                    break

    # A gated decode projection is only removed when neither projection escapes.
    # Shared projections, prefill, and unsupported patterns retain the old path.
    gated = {}
    if use_decode_linear and fuse_gated_decode:
        for name, (gate, up) in fused.items():
            if not all(isinstance(n, Node) and n.op == "call_function"
                       and str(n.target) == "aten.linear.default"
                       and len(n.users) == 1 for n in (gate, up)):
                continue
            ga, ua = arg(gate, 0, "input"), arg(up, 0, "input")
            if ga is not ua or gate is up:
                continue
            gw, uw = arg(gate, 1, "weight"), arg(up, 1, "weight")
            gb, ub = arg(gate, 2, "bias"), arg(up, 2, "bias")
            sa, sw = spec(ga), spec(gw)
            if (len(sa.shape) < 2 or math.prod(sa.shape[:-1]) > 4 or len(sw.shape) != 2
                    or sw != spec(uw) or sa.dtype != sw.dtype or sa.shape[-1] != sw.shape[1]):
                continue
            expected = TensorSpec(sa.shape[:-1] + (sw.shape[0],), sa.dtype)
            if spec(gate) != expected or spec(up) != expected or spec(nodes[name]) != expected:
                continue
            if any(b is not None and spec(b) != TensorSpec((sw.shape[0],), sw.dtype) for b in (gb, ub)):
                continue
            gated[name] = (ga, gw, uw, gb, ub)
            omit.update((gate.name, up.name))

    # Fuse residual addition only if the rounded residual tensor has no other user.
    residuals = {}
    for node in ep.graph.nodes:
        if node.op == "call_function" and str(node.target) == "aten.rms_norm.default":
            addition = arg(node, 0, "input")
            if (isinstance(addition, Node) and addition.op == "call_function"
                    and str(addition.target) == "aten.add.Tensor" and len(addition.users) == 1
                    and arg(addition, 2, "alpha", 1) == 1):
                left, right = arg(addition, 0, "self"), arg(addition, 1, "other")
                same(left, right)
                residuals[node.name] = (left, right)
                omit.add(addition.name)

    steps = []
    for node in ep.graph.nodes:
        if node.op in {"placeholder", "output"} or node.name in omit:
            continue
        if node.op != "call_function":
            raise UnsupportedGraph(f"{node.name}: unsupported FX node {node.op}")
        out = spec(node)
        target = str(node.target)
        name = f"ruda_{node.name}"
        refs = None
        if node.name in gated:
            a, gw, uw, gb, ub = gated[node.name]
            sa = spec(a)
            flat = TensorSpec((math.prod(sa.shape[:-1]), sa.shape[-1]), sa.dtype)
            kernel = gated_linear(name, flat, spec(gw), gate_bias=gb is not None, up_bias=ub is not None)
            refs = (a.name, gw.name, uw.name) + ((gb.name,) if gb is not None else ()) + ((ub.name,) if ub is not None else ())
        elif node.name in fused:
            gate, up = fused[node.name]
            same(gate, up)
            if out != spec(gate):
                raise UnsupportedGraph("Unexpected fused activation output shape")
            kernel = emit_elementwise(name, "silu_mul", out)
            refs = (gate.name, up.name)
        elif target in {"aten.add.Tensor", "aten.mul.Tensor"}:
            a, b = arg(node, 0, "self"), arg(node, 1, "other")
            same(a, b)
            if out != spec(a) or arg(node, 2, "alpha", 1) != 1:
                raise UnsupportedGraph("Only equal-shape add/mul and add alpha=1 are supported")
            kernel = emit_elementwise(name, "add" if target == "aten.add.Tensor" else "mul", out)
            refs = (a.name, b.name)
        elif target == "aten.silu.default":
            a = arg(node, 0, "self")
            if out != spec(a):
                raise UnsupportedGraph("SiLU output shape mismatch")
            kernel, refs = elementwise(name, "silu", out), (a.name,)
        elif target == "aten.rms_norm.default":
            a, shape, w = arg(node, 0, "input"), arg(node, 1, "normalized_shape"), arg(node, 2, "weight")
            sa, sw = spec(a), spec(w)
            if (tuple(shape) != (sa.shape[-1],) or sw.shape != (sa.shape[-1],)
                    or sw.dtype != sa.dtype or out != sa):
                raise UnsupportedGraph("RMSNorm needs last-axis normalization and a same-dtype vector weight")
            eps = arg(node, 3, "eps")
            if eps is None:
                eps = torch.finfo(getattr(torch, sa.dtype)).eps
            if node.name in residuals:
                left, right = residuals[node.name]
                kernel = fused_rms_norm(name, sa, eps, residual=True)
                refs = (left.name, right.name, w.name)
            else:
                kernel, refs = rms_norm(name, sa, eps), (a.name, w.name)
        elif target == "aten.linear.default":
            a, w, bias = arg(node, 0, "input"), arg(node, 1, "weight"), arg(node, 2, "bias")
            sa, sw = spec(a), spec(w)
            if len(sa.shape) < 2 or len(sw.shape) != 2 or sa.shape[-1] != sw.shape[1] or sa.dtype != sw.dtype:
                raise UnsupportedGraph("Linear needs compatible contiguous tensors and rank >= 2 input")
            expected = TensorSpec(sa.shape[:-1] + (sw.shape[0],), sa.dtype)
            if out != expected:
                raise UnsupportedGraph("Linear output shape mismatch")
            if bias is not None and spec(bias) != TensorSpec((sw.shape[0],), sw.dtype):
                raise UnsupportedGraph("Linear bias must match output width and dtype")
            flat = TensorSpec((math.prod(sa.shape[:-1]), sa.shape[-1]), sa.dtype)
            if use_decode_linear and flat.shape[0] <= 4:
                # Conservative shape policy, explicitly overridable for hardware A/B tests.
                tile = decode_outputs_per_warp or (2 if sw.shape[0] >= 1024 else 1)
                kernel = linear_decode(name, flat, sw, bias=bias is not None, outputs_per_warp=tile)
            else:
                kernel = matmul(name, flat, sw, transpose_b=True, bias=bias is not None)
            refs = (a.name, w.name) + ((bias.name,) if bias is not None else ())
        elif target in {"aten.softmax.int", "aten._softmax.default"}:
            a, dim = arg(node, 0, "self"), arg(node, 1, "dim")
            sa = spec(a)
            if dim not in {-1, len(sa.shape)-1} or out != sa:
                raise UnsupportedGraph("Softmax requires the last axis and unchanged dtype")
            option = arg(node, 2, "dtype" if target == "aten.softmax.int" else "half_to_float")
            if option not in (None, False):
                raise UnsupportedGraph("Softmax dtype overrides / half_to_float are not implemented")
            kernel, refs = softmax(name, sa), (a.name,)
        elif target == "aten.scaled_dot_product_attention.default":
            q, k, v = arg(node, 0, "query"), arg(node, 1, "key"), arg(node, 2, "value")
            if arg(node, 3, "attn_mask") is not None or arg(node, 4, "dropout_p", 0.0) != 0.0:
                raise UnsupportedGraph("PTX attention currently requires no external mask and dropout_p=0")
            sq, sk, sv = spec(q), spec(k), spec(v)
            if len(sq.shape) != 4 or len(sk.shape) != 4 or len(sv.shape) != 4:
                raise UnsupportedGraph("PTX attention requires contiguous 4D tensors")
            if sq.shape[2] != 1 and not allow_streaming_prefill:
                raise UnsupportedGraph("Streaming prefill is not tuned; set allow_streaming_prefill=True explicitly")
            if sq.shape[1] != sk.shape[1] and not arg(node, 7, "enable_gqa", False):
                raise UnsupportedGraph("Different query/KV head counts require enable_gqa=True")
            # PyTorch's rectangular is_causal is upper-left. Never silently replace
            # it with cached-decode lower-right semantics (provided by the cache API).
            try:
                program = attention_program(name, sq, sk, sv,
                    causal=arg(node, 5, "is_causal", False), alignment="upper_left",
                    scale=arg(node, 6, "scale"),
                    partitions=decode_partitions if sq.shape[2] == 1 else 1)
            except (TypeError, ValueError) as exc:
                raise UnsupportedGraph(str(exc)) from exc
            if out != program.output:
                raise UnsupportedGraph("Attention output shape mismatch")
            refs = (q.name, k.name, v.name)
            kernel = program.first
            if program.merge is not None:
                temporary = f"{node.name}_ptx_partials"
                if temporary in specs or temporary in nodes:
                    raise UnsupportedGraph("Internal attention temporary name collision")
                specs[temporary] = program.partial
                steps.append(Step(refs, temporary, program.first))
                refs, kernel = (temporary,), program.merge
        elif target in {"aten.mm.default", "aten.matmul.default"}:
            a, b = arg(node, 0, "self"), arg(node, 1, "other")
            sa, sb = spec(a), spec(b)
            kernel = matmul(name, sa, sb)
            if out != TensorSpec((sa.shape[0], sb.shape[1]), sa.dtype):
                raise UnsupportedGraph("Matmul output shape mismatch")
            refs = (a.name, b.name)
        else:
            raise UnsupportedGraph(f"{node.name}: {target} has no PTX lowering; no CUDA/CPU fallback is used")
        steps.append(Step(refs, node.name, kernel))

    if not steps:
        raise UnsupportedGraph("This graph contains zero PTX kernels")
    outputs = []
    for item in ep.graph_signature.output_specs:
        if not isinstance(item.arg, TensorArgument) or item.arg.name not in specs:
            raise UnsupportedGraph("Only tensor graph outputs are supported")
        if item.arg.name in inputs or item.arg.name in constants:
            raise UnsupportedGraph("Input/weight alias outputs are not implemented in this export adapter")
        outputs.append(item.arg.name)
    for omitted in omit:
        specs.pop(omitted, None)
    # No silently ignored operator kwargs. Exported ATen schemas are constrained above.
    return Plan(specs, tuple(inputs), tuple(outputs), tuple(steps), constants,
                ep.call_spec.in_spec, ep.call_spec.out_spec)
