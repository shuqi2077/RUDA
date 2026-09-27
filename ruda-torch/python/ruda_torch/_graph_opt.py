"""Pure graph/lifetime planning. Does not execute tensors or provide a CPU fallback.

Validation runs BEFORE pruning so unused invalid operations are still errors.
Only single-consumer SiLU on the LEFT of mul is fused. Public results never
share a workspace slot. All options are opt-in pending hardware validation.
"""
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from ._graph_spec import GraphOp, Layout, NO_WEIGHT, TensorSpec, plan_layout


@dataclass(frozen=True)
class ExecutionPlan:
    layout: Layout
    nodes: tuple[GraphOp, ...]
    storage_roots: tuple[int, ...]
    original_nodes: int
    original_workspace_bytes: int
    eliminated_outputs: tuple[str, ...]
    fused_activations: tuple[str, ...]

    @property
    def workspace_bytes(self) -> int:
        roots = set(self.storage_roots[self.layout.inputs:])
        return sum(_bytes(self.layout.specs[i]) for i in roots)

    @property
    def workspace_allocations(self) -> int:
        return len(set(self.storage_roots[self.layout.inputs:]))


def _bytes(spec: TensorSpec) -> int:
    return math.prod(spec.shape) * (4 if spec.dtype == 'float32' else 2)


def _reads(node: GraphOp) -> tuple[str, ...]:
    return (node.left,) if node.right is None else (node.left, node.right)


def _prune(nodes: tuple[GraphOp, ...], outputs: tuple[str, ...]):
    needed = set(outputs)
    retained = []
    eliminated = []
    for node in reversed(nodes):
        if node.output not in needed:
            eliminated.append(node.output)
            continue
        retained.append(node)
        needed.update(_reads(node))
    return tuple(reversed(retained)), tuple(reversed(eliminated))


def _fuse(nodes: tuple[GraphOp, ...], outputs: tuple[str, ...]):
    # Count edges, not just consumers: mul(a, a) has two uses and is not fused.
    uses = Counter(name for n in nodes for name in _reads(n))
    producers = {n.output: n for n in nodes}
    public = set(outputs)
    replacements = {}
    removed = set()
    for node in nodes:
        producer = producers.get(node.left)
        if (node.kind == 'mul' and producer is not None
                and producer.kind == 'silu' and uses[producer.output] == 1
                and producer.output not in public):
            replacements[node.output] = GraphOp.silu_mul(
                node.output, producer.left, node.right)
            removed.add(producer.output)
    result = tuple(replacements.get(n.output, n) for n in nodes if n.output not in removed)
    # Public intermediate outputs and fan-out retain their original storage cast.
    return result, tuple(n.output for n in nodes if n.output in removed)


def plan_storage(layout: Layout, *, reuse: bool = False) -> tuple[int, ...]:
    """One tensor per slot; only equal shape/dtype, whole-buffer reuse is allowed.

    Inclusive lifetimes forbid reuse at the node performing the previous value's
    final read. A requested result always receives its OWN allocation. Leaf
    values are also kept to the end, matching the native bridge's conservative
    validation when pruning is disabled.
    """
    if type(reuse) is not bool:
        raise TypeError('reuse must be bool')
    count = len(layout.names)
    roots = list(range(count))
    if not reuse:
        return tuple(roots)
    last = list(range(count))
    consumed = set()
    for i in range(len(layout.scalars)):
        out = layout.inputs + i
        _, a, b = layout.words[3*i:3*i+3]
        for value in (a, b):
            if value != NO_WEIGHT:
                last[value] = max(last[value], out)
                consumed.add(value)
    protected = set(layout.output_indices)
    for i in range(layout.inputs, count):
        if i not in consumed or i in protected:
            last[i] = count
    # root -> inclusive last use of the current occupant. Only scratch roots
    # are inserted; input and public result allocations never enter the pool.
    pool: dict[int, int] = {}
    for i in range(layout.inputs, count):
        if i in protected:
            continue
        selected = next((root for root, end in pool.items()
                         if end < i and layout.specs[root] == layout.specs[i]), None)
        if selected is None:
            selected = i
        roots[i] = selected
        pool[selected] = last[i]
    return tuple(roots)


def prepare_plan(inputs: Mapping[str, TensorSpec], nodes: Sequence[GraphOp], outputs=None,
                 *, optimize: bool = False, reuse_workspace: bool = False) -> ExecutionPlan:
    if type(optimize) is not bool or type(reuse_workspace) is not bool:
        raise TypeError('optimizer options must be booleans')
    nodes = tuple(nodes)
    original = plan_layout(inputs, nodes, outputs)
    public_names = tuple(original.names[i] for i in original.output_indices)
    original_count = len(nodes)
    eliminated = fused = ()
    if optimize:
        nodes, eliminated = _prune(nodes, public_names)
        nodes, fused = _fuse(nodes, public_names)
    layout = plan_layout(inputs, nodes, public_names)
    roots = plan_storage(layout, reuse=reuse_workspace)
    return ExecutionPlan(layout, nodes, roots, original_count,
                         original.workspace_bytes, eliminated, fused)
