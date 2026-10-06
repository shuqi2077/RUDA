"""Conservative ATen-to-StaticGraph regions used *after* AOTAutograd.

Tracing never dereferences FakeTensor data. Native graphs are allocated lazily
from real invocation tensors. No operation is retried after a dispatch failure.
"""
from collections import OrderedDict
import math
import operator
from threading import RLock
import sys

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg

from ._graph_spec import GraphOp, MAX_NODES, MAX_TENSORS, TensorSpec, plan_layout, UNARY_CODES


class NativeCoverageError(RuntimeError):
    """A required-native graph contains unsupported operations or input metadata."""


# Exact overloads only; no mutations, RNG, alias-changing views or implicit casts.
_TARGETS = {getattr(torch.ops.aten, name).default: name for name in UNARY_CODES
            if name not in ('copy',)}
_TARGETS.update({torch.ops.aten.clone.default: 'copy', torch.ops.aten.add.Tensor: 'add',
    torch.ops.aten.sub.Tensor: 'sub', torch.ops.aten.mul.Tensor: 'mul',
    torch.ops.aten.div.Tensor: 'div', torch.ops.aten.mm.default: 'mm',
    torch.ops.aten.bmm.default: 'bmm', torch.ops.aten.silu_backward.default: 'silu_backward',
    torch.ops.aten.sigmoid_backward.default: 'sigmoid_backward',
    torch.ops.aten.tanh_backward.default: 'tanh_backward',
    torch.ops.aten._softmax.default: 'softmax', torch.ops.aten._log_softmax.default: 'log_softmax',
    torch.ops.aten._softmax_backward_data.default: 'softmax_backward',
    torch.ops.aten._log_softmax_backward_data.default: 'log_softmax_backward',
    torch.ops.aten.sum.dim_IntList: 'sum_keepdim', torch.ops.aten.mean.dim: 'mean_keepdim',
    torch.ops.aten.add.Scalar: 'add', torch.ops.aten.sub.Scalar: 'sub',
    torch.ops.aten.mul.Scalar: 'mul', torch.ops.aten.div.Scalar: 'div'})
_TARGETS.update({torch.ops.aten.view_copy.default: 'reshape_copy',
    torch.ops.aten.permute_copy.default: 'permute_copy', torch.ops.aten._to_copy.default: 'cast',
    torch.ops.aten.expand_copy.default: 'expand_copy'})
_ALIASES={torch.ops.aten.view.default:'reshape_copy',torch.ops.aten.reshape.default:'reshape_copy',
    torch.ops.aten._unsafe_view.default:'reshape_copy',torch.ops.aten.permute.default:'permute_copy',
    torch.ops.aten.t.default:'transpose_copy',torch.ops.aten.transpose.int:'transpose_copy',
    torch.ops.aten.unsqueeze.default:'reshape_copy',torch.ops.aten.squeeze.dim:'reshape_copy',
    torch.ops.aten.squeeze.default:'reshape_copy',torch.ops.aten.expand.default:'expand_copy'}


def _layout_capable():
    return getattr(sys.modules.get(__package__),'_graph_layout_available',False)


def _internal_alias(node, seen=None):
    """Materialize a view only when all consumers are pure, non-aliasing math.

    Returning a view, mutating through it or inspecting its storage/strides stays
    on ordinary device dispatch. No user-observable alias is replaced by a copy.
    """
    seen=set() if seen is None else seen
    if node in seen:return False
    seen.add(node)
    for user in node.users:
        if user.op!='call_function':return False
        if user.target in _ALIASES:
            if not _internal_alias(user,seen.copy()):return False
        elif user.target not in _TARGETS:return False
    return True


def _rank(node):
    value = node.meta.get('val')
    return value.ndim if isinstance(value, torch.Tensor) else None


def _lower(node):
    if node.op != 'call_function' or node.target not in _TARGETS and node.target not in _ALIASES:
        return None
    alias=node.target in _ALIASES
    if alias and (not _layout_capable() or not _internal_alias(node)):
        return None
    value = node.meta.get('val')
    if isinstance(value, torch.Tensor) and not value.is_contiguous() and not alias:
        return None  # Output stride/clone memory-format is observable to users.
    kind, args, kwargs = (_ALIASES[node.target] if alias else _TARGETS[node.target]), node.args, dict(node.kwargs)
    if not args or not isinstance(args[0], Node): return None
    left = args[0].name
    if kind=='transpose_copy':
        rank=_rank(args[0])
        if rank is None or kwargs:return None
        axes=list(range(rank))
        if node.target==torch.ops.aten.t.default:
            if rank>2 or len(args)!=1:return None
            if rank==2:axes=[1,0]
        else:
            if len(args)!=3 or any(type(d) is not int or not -rank<=d<rank for d in args[1:]):return None
            a,b=args[1]%rank,args[2]%rank
            axes[a],axes[b]=axes[b],axes[a]
        return GraphOp('permute_copy',node.name,left,scalar=sum(d<<(3*i) for i,d in enumerate(axes)),shape=tuple(value.shape))
    if kind in ('reshape_copy','permute_copy','expand_copy','cast'):
        if not _layout_capable():return None
        if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            return None
        if kind == 'cast':
            if len(args) != 1 or set(kwargs)-{'dtype','device','layout','pin_memory','non_blocking','memory_format'}:
                return None
            source = args[0].meta.get('val')
            if not isinstance(source,torch.Tensor) or source.device != value.device or source.layout != value.layout:
                return None
            if kwargs.get('pin_memory',False) or kwargs.get('memory_format') not in (None,torch.preserve_format,torch.contiguous_format):
                return None
            return GraphOp(kind,node.name,left,dtype=str(value.dtype).removeprefix('torch.'))
        if alias and kind=='reshape_copy':
            if kwargs or len(args) not in (1,2):return None
            return GraphOp(kind,node.name,left,shape=tuple(value.shape))
        if kind=='expand_copy' and len(args)==3 and args[2] is False:args=args[:2]
        if len(args) != 2 or kwargs or not isinstance(args[1],(tuple,list)) or any(type(d) is not int for d in args[1]):
            return None
        scalar = 0
        if kind == 'permute_copy':
            rank = _rank(args[0])
            if rank is None or len(args[1]) != rank or any(not -rank<=d<rank for d in args[1]):
                return None
            axes = [d%rank for d in args[1]]
            if len(set(axes)) != rank:
                return None
            scalar = sum(d<<(3*i) for i,d in enumerate(axes))
        return GraphOp(kind,node.name,left,scalar=scalar,shape=tuple(value.shape))
    if kind in ('add', 'sub', 'mul', 'div'):
        if len(args) != 2 or set(kwargs) - ({'alpha'} if kind in ('add','sub') else set()):
            return None
        alpha = kwargs.get('alpha', 1.)
        if type(alpha) not in (int, float) or not math.isfinite(alpha): return None
        if kind == 'sub': alpha = -alpha
        if isinstance(args[1], Node):
            operation = 'add' if kind == 'sub' else kind
            lv,rv = args[0].meta.get('val'),args[1].meta.get('val')
            if isinstance(lv,torch.Tensor) and isinstance(rv,torch.Tensor) and lv.shape != rv.shape:
                if not _layout_capable():return None
                operation += '_broadcast'
            return GraphOp(operation, node.name, left, args[1].name,
                           alpha if kind in ('add','sub') else 0.)
        if type(args[1]) not in (int,float) or not math.isfinite(args[1]): return None
        scalar = args[1] * alpha if kind in ('add','sub') else args[1]
        return GraphOp(('add' if kind == 'sub' else kind)+'_scalar', node.name, left, scalar=scalar)
    if kind in ('mm', 'bmm', 'silu_backward', 'sigmoid_backward', 'tanh_backward'):
        if len(args) != 2 or not isinstance(args[1], Node) or kwargs: return None
        return GraphOp(kind, node.name, left, args[1].name)
    if kind.startswith(('softmax', 'log_softmax')):
        backward = kind.endswith('_backward')
        if kwargs or len(args) != (4 if backward else 3): return None
        axis = args[2 if backward else 1]; rank = _rank(args[0])
        if type(axis) is not int or rank is None or not -rank <= axis < rank: return None
        if not backward and args[2] is not False: return None  # no half-to-float cast
        if backward:
            value = args[0].meta.get('val')
            if not isinstance(args[1], Node) or args[3] != value.dtype: return None
        return GraphOp(kind, node.name, left, args[1].name if backward else None, axis % rank)
    if kind in ('sum_keepdim', 'mean_keepdim'):
        if not 2 <= len(args) <= 3 or set(kwargs) - {'keepdim','dtype'}: return None
        keepdim = args[2] if len(args) == 3 else kwargs.get('keepdim', False)
        if type(keepdim) is not bool: return None
        if not keepdim and not _layout_capable():return None
        if kwargs.get('dtype') is not None: return None
        rank = _rank(args[0]); dims = args[1]
        if rank is None or not isinstance(dims, (tuple,list)): return None
        dims = list(dims) if dims else list(range(rank))
        if any(type(d) is not int or not -rank <= d < rank for d in dims): return None
        dims = [d % rank for d in dims]
        if len(set(dims)) != len(dims): return None
        return GraphOp(kind if keepdim else kind.removesuffix('_keepdim'), node.name, left,
                       scalar=sum(1 << d for d in dims))
    if kind == 'copy':
        if len(args) != 1 or set(kwargs) - {'memory_format'}: return None
        if kwargs.get('memory_format') not in (None, torch.preserve_format, torch.contiguous_format): return None
    elif len(args) != 1 or kwargs:
        return None
    return GraphOp(kind, node.name, left)


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
    if value.layout != torch.strided:
        raise ValueError('native regions require dense strided tensors')
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
                    plan_layout(specs, self.nodes, self.outputs,layout_api=int(_layout_capable()))
                    if any(node.kind in ('reshape_copy','permute_copy','cast','expand_copy',
                           'add_broadcast','mul_broadcast','div_broadcast','sum','mean') for node in self.nodes):
                        from . import _graph_layout_available
                        if not _graph_layout_available:
                            raise ValueError('native graph layout extension unavailable')
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
