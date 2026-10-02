"""Conservative ATen-to-StaticGraph regions used *after* AOTAutograd.

Tracing never dereferences FakeTensor data. Native graphs are allocated lazily
from real invocation tensors. No operation is retried after a dispatch failure.
"""
from collections import OrderedDict
import math
import operator
from threading import RLock

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg

from ._graph_spec import GraphOp, MAX_NODES, MAX_TENSORS, TensorSpec, plan_layout


class NativeCoverageError(RuntimeError):
    """A required-native graph contains unsupported operations or input metadata."""


# Deliberately exact overloads, not string/sub-string matching. Views, mutation,
# broadcasts, random ops and reductions retain their original ATen semantics.
_TARGETS = {
    torch.ops.aten.add.Tensor: 'add',
    torch.ops.aten.mul.Tensor: 'mul',
    torch.ops.aten.silu.default: 'silu',
    torch.ops.aten.clone.default: 'copy',
}


def _lower(node):
    if node.op != 'call_function' or node.target not in _TARGETS:
        return None
    kind = _TARGETS[node.target]
    args, kwargs = node.args, node.kwargs
    if kind in ('add', 'mul'):
        if len(args) != 2 or not all(isinstance(a, Node) for a in args):
            return None
        if set(kwargs) - ({'alpha'} if kind == 'add' else set()):
            return None
        alpha = kwargs.get('alpha', 1.)
        if type(alpha) not in (int, float) or not math.isfinite(alpha):
            return None
        return (GraphOp.add(node.name, args[0].name, args[1].name, alpha=alpha)
                if kind == 'add' else GraphOp.mul(node.name, args[0].name, args[1].name))
    if len(args) != 1 or not isinstance(args[0], Node):
        return None
    if kind == 'copy':
        if set(kwargs) - {'memory_format'}:
            return None
        if kwargs.get('memory_format') not in (None, torch.preserve_format, torch.contiguous_format):
            return None
        return GraphOp.copy(node.name, args[0].name)
    if kwargs:
        return None
    return GraphOp.silu(node.name, args[0].name)


def _native_available():
    from . import _graph_available
    return _graph_available


def _stream_id():
    from ._streams import current_stream
    return current_stream().stream_id


def _build_graph(inputs, nodes, outputs):
    from ._graph import StaticGraph
    # AOTAutograd owns differentiation. No hand-written backward bridge is used.
    return StaticGraph(inputs, nodes, outputs=outputs, reuse_workspace=True)


def _input_spec(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError('native regions require tensor inputs')
    if value.device.type != 'ruda' or value.device.index not in (None, 0):
        raise ValueError('native regions require ruda:0 (no implicit device transfer)')
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError('native regions require contiguous dense tensors')
    if value.is_conj() or value.is_neg():
        raise ValueError('native regions do not support unresolved conjugate/negative views')
    return TensorSpec(tuple(value.shape), str(value.dtype).removeprefix('torch.'))


class NativeRegion(torch.nn.Module):
    """A pure region, with bounded fixed-address staging per shape/stream.

    Outputs are cloned before returning: AOT's saved tensors and outstanding
    forwards must never refer to native replay workspace. The reference callable
    is selected BEFORE dispatch and only for known capability/metadata limits.
    """
    def __init__(self, reference, names, nodes, outputs, state, phase, options):
        super().__init__()
        self.reference = reference
        self.names = tuple(names)
        self.nodes = tuple(nodes)
        self.outputs = tuple(outputs)
        self.state = state
        self.phase = phase
        self.options = options
        self._cache = OrderedDict()
        self._lock = RLock()
        self._closed = False

    def forward(self, *args):
        with self._lock:
            self.state.check_open()
            if self._closed:
                raise RuntimeError('native region is closed')
            reason = None
            if not _native_available():
                reason = 'native static graph extension unavailable'
            else:
                try:
                    specs = {name: _input_spec(value) for name, value in zip(self.names, args, strict=True)}
                    plan_layout(specs, self.nodes, self.outputs)
                except (ValueError, TypeError, OverflowError) as exc:
                    reason = str(exc)
            if reason is not None:
                if self.options.native == 'required':
                    raise NativeCoverageError(f'{self.phase}: {reason}')
                self.state.add('region_reference_runs')
                self.state.reason(self.phase, reason)
                return self.reference(*args)

            stream = _stream_id()
            key = (stream, torch.is_inference_mode_enabled(),
                   tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in args))
            entry = self._cache.get(key)
            with torch.no_grad():
                if entry is None:
                    # Reject/evict before allocating: the number of cached native
                    # handles per region never exceeds the documented bound.
                    if len(self._cache) >= self.options.cache_size:
                        old_key = next(iter(self._cache))
                        old_graph, _ = self._cache[old_key]
                        old_graph.close()
                        del self._cache[old_key]
                        self.state.add('native_cache_evictions')
                    buffers = {name: torch.empty_like(t, memory_format=torch.contiguous_format)
                               for name, t in zip(self.names, args, strict=True)}
                    graph = _build_graph(buffers, self.nodes, self.outputs)
                    entry = graph, buffers
                    self._cache[key] = entry
                    self.state.add('native_builds')
                else:
                    self._cache.move_to_end(key)
                    self.state.add('native_cache_hits')
                graph, buffers = entry
                for buffer, value in zip(buffers.values(), args, strict=True):
                    buffer.copy_(value)
                result = graph.replay()
                # Run/clone failures propagate: never execute the region twice.
                owned = tuple(result[name].clone() for name in self.outputs)
                self.state.add('native_replays')
                self.state.add('native_nodes_executed', len(self.nodes))
                return owned

    def close(self):
        with self._lock:
            if not self._closed:
                first_error = None
                for key, (graph, _) in list(self._cache.items()):
                    try:
                        graph.close()
                    except Exception as exc:
                        first_error = first_error or exc
                    else:
                        del self._cache[key]
                if first_error is not None:
                    # Keep failed handles alive so explicit close() can retry.
                    raise first_error
                self._closed = True


def _make_region(gm, nodes, state, phase, options):
    selected = set(nodes)
    dependencies = []
    for node in nodes:
        def dependency(arg):
            if arg not in selected and arg not in dependencies:
                dependencies.append(arg)
            return arg
        map_arg((node.args, node.kwargs), dependency)
    outputs = [n for n in nodes if any(user not in selected for user in n.users)]
    if not outputs or len(dependencies) + len(nodes) > MAX_TENSORS:
        return None
    graph, env = Graph(), {}
    for node in dependencies:
        env[node] = graph.placeholder(node.name)
    for node in nodes:
        env[node] = graph.node_copy(node, lambda n: env[n])
    graph.output(tuple(env[node] for node in outputs))
    reference = GraphModule(gm, graph)
    region = NativeRegion(reference, [n.name for n in dependencies],
                          [_lower(n) for n in nodes], [n.name for n in outputs], state, phase, options)
    return region, dependencies, outputs


def partition(gm, state, phase, options):
    """Replace consecutive pure eligible operations; keep all others unchanged.

    Conservatively consecutive grouping avoids moving work across mutations or
    RNG calls. Runtime guards also check shape/layout/dtype, not just op names.
    """
    all_nodes = list(gm.graph.nodes)
    calls = [n for n in all_nodes if n.op in ('call_function', 'call_method', 'call_module')]
    candidates = {n for n in calls if _lower(n) is not None}
    if options.native == 'required':
        unsupported = [f'{n.name}: {n.target}' for n in calls if n not in candidates]
        if unsupported:
            raise NativeCoverageError(f'{phase} is not entirely native: ' + '; '.join(unsupported[:12]))
    groups, pending = [], []
    for node in all_nodes:
        if node in candidates and options.native != 'off':
            pending.append(node)
            if len(pending) == MAX_NODES:
                groups.append(pending)
                pending = []
        elif pending:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    minimum = 1 if options.native == 'required' else options.min_native_ops
    regions = []
    for nodes in groups:
        if len(nodes) < minimum:
            continue
        made = _make_region(gm, nodes, state, phase, options)
        if made is None:
            if options.native == 'required':
                raise NativeCoverageError(f'{phase}: region exceeds native tensor limits or has no outputs')
            continue
        regions.append((nodes, made))
    covered = {n for nodes, _ in regions for n in nodes}
    record = {
        'phase': phase,
        'operator_count': len(calls),
        'native_candidate_ops': len(candidates),
        'planned_native_ops': len(covered),
        'native_regions': len(regions),
        'device_dispatch_ops': [str(n.target) for n in calls if n not in covered],
        'operators': [{'node': n.name, 'target': str(n.target),
                       'execution': 'guarded-native-region' if n in covered else 'device-dispatch'}
                      for n in calls],
    }
    state.record(record)
    if not regions:
        return gm

    graph, env = Graph(), {}
    starts = {nodes[0]: (nodes, made) for nodes, made in regions}
    root = torch.nn.Module()
    # GraphModule copies referenced parameters/submodules from gm; adding region
    # modules to gm itself would mutate an AOT-owned graph. Use a separate root.
    for name, module in gm.named_children():
        root.add_module(name, module)
    for name, parameter in gm.named_parameters(recurse=False):
        root.register_parameter(name, parameter)
    for name, buffer in gm.named_buffers(recurse=False):
        root.register_buffer(name, buffer)
    for node in all_nodes:
        if node.op == 'get_attr' and '.' not in str(node.target) and not hasattr(root, node.target):
            setattr(root, node.target, getattr(gm, node.target))
    for node in all_nodes:
        if node in starts:
            nodes, (region, dependencies, outputs) = starts[node]
            name = f'_ruda_native_region_{len(state.regions)}'
            while hasattr(root, name):
                name += '_'
            root.add_module(name, region)
            state.register_region(region)
            call = graph.call_module(name, tuple(env[n] for n in dependencies))
            for index, output in enumerate(outputs):
                value = graph.call_function(operator.getitem, (call, index))
                value.meta = dict(output.meta)
                env[output] = value
        elif node not in covered:
            env[node] = graph.node_copy(node, lambda n: env[n])
    result = GraphModule(root, graph)
    result.graph.lint()
    result.recompile()
    return result
