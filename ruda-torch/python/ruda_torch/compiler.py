"""General model/function integration, independent of the explicit GraphOp API.

AOT mode captures forward and backward via PyTorch, not hand-written derivative
rules. Native-capable regions use StaticGraph; everything else is dispatched on
the original device. A model still needs kernels for every executed operation.
Eager mode is explicit, preserves eager autograd (including higher derivatives
where its operators support them), and performs no capture or acceleration.
"""
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import copy
from functools import partial, update_wrapper
from threading import RLock
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from ._compile_native import NativeCoverageError, partition


class GraphExecutionError(RuntimeError):
    """A captured graph failed on its original device; the operation is not retried."""


@dataclass(frozen=True)
class CompileOptions:
    capture: str = 'aot'
    native: str = 'auto'
    device_type: str = 'ruda'
    fullgraph: bool = False
    dynamic: bool | None = None
    min_native_ops: int = 2
    cache_size: int = 4

    def __post_init__(self):
        if self.capture not in ('aot', 'eager'):
            raise ValueError("capture must be 'aot' or 'eager'")
        if self.native not in ('auto', 'required', 'off'):
            raise ValueError("native must be 'auto', 'required' or 'off'")
        if self.capture == 'eager' and self.native == 'required':
            raise ValueError("capture='eager' cannot require native graph capture")
        if not isinstance(self.device_type, str) or not self.device_type or ':' in self.device_type:
            raise ValueError('device_type must be a device type (for example ruda or cpu), not an index')
        if type(self.fullgraph) is not bool or (self.dynamic is not None and type(self.dynamic) is not bool):
            raise TypeError('fullgraph and dynamic must be bool (dynamic may also be None)')
        if type(self.min_native_ops) is not int or not 1 <= self.min_native_ops <= 256:
            raise ValueError('min_native_ops must be an integer in 1..256')
        if type(self.cache_size) is not int or not 1 <= self.cache_size <= 64:
            raise ValueError('cache_size must be an integer in 1..64, per native region')
        if self.capture == 'eager' and (self.fullgraph or self.dynamic is not None):
            raise ValueError('fullgraph/dynamic options do not apply to explicit eager execution')


class _State:
    def __init__(self, options):
        self.options = options
        self.lock = RLock()
        self.closed = False
        self.counters = Counter()
        self.reasons = Counter()
        self.graphs = []
        self.regions = []

    def check_open(self):
        if self.closed:
            raise RuntimeError('compiled model/backend is closed; finish backward before close()')

    def add(self, name, amount=1):
        with self.lock:
            self.counters[name] += amount

    def reason(self, phase, reason):
        with self.lock:
            self.reasons[(phase, reason)] += 1

    def record(self, graph):
        with self.lock:
            self.check_open()
            self.graphs.append(graph)

    def register_region(self, region):
        with self.lock:
            self.check_open()
            self.regions.append(region)

    def snapshot(self):
        with self.lock:
            counters = {key: self.counters[key] for key in (
                'dynamo_graphs', 'forward_calls', 'inference_calls', 'backward_calls',
                'eager_calls', 'native_builds', 'native_replays', 'native_cache_hits',
                'native_cache_evictions', 'native_nodes_executed', 'region_reference_runs',
                'execution_errors')}
            return {
                'capture': self.options.capture,
                'native_policy': self.options.native,
                'device_type': self.options.device_type,
                'fullgraph': self.options.fullgraph,
                'dynamic': self.options.dynamic,
                'cache_size_per_region': self.options.cache_size,
                'implicit_cpu_fallback': False,
                'optimizer_capture': 'only if explicitly included in the callable and supported by PyTorch',
                'higher_order': 'operator-dependent eager autograd' if self.options.capture == 'eager' else 'not supported',
                'closed': self.closed,
                **counters,
                'graphs': copy.deepcopy(self.graphs),
                'reference_reasons': [{'phase': phase, 'reason': reason, 'calls': count}
                                      for (phase, reason), count in sorted(self.reasons.items())],
            }

    def close(self):
        with self.lock:
            self.closed = True
            regions = list(self.regions)
        # Do not hold the state lock while waiting on a region/device stream.
        errors = []
        for region in regions:
            try:
                region.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('failed to release one or more native graph regions') from errors[0]


def _check_tensors(values, device_type, label):
    flat, _ = tree_flatten(values)
    for value in flat:
        if isinstance(value, torch.Tensor) and value.device.type != device_type:
            raise ValueError(f'{label}: tensor on {value.device}; expected {device_type}. '
                             'Move model and inputs explicitly; no CPU/device fallback is installed.')
        if isinstance(value, torch.Tensor) and device_type == 'ruda' and value.device.index not in (None, 0):
            raise ValueError(f'{label}: only ruda:0 is exposed by this runtime')


def _check_module(model, device_type):
    if isinstance(model, torch.nn.Module):
        for label, values in (('parameters', tuple(model.parameters())), ('buffers', tuple(model.buffers()))):
            _check_tensors(values, device_type, label)


def _validate_decompositions(decompositions):
    if decompositions is None:
        return {}
    if not isinstance(decompositions, Mapping):
        raise TypeError('decompositions must map torch operator overloads to callables')
    result = dict(decompositions)
    for op, fn in result.items():
        if not isinstance(op, torch._ops.OpOverload) or not callable(fn):
            raise TypeError('decompositions must map torch operator overloads to callables')
    return result


class RudaBackend:
    """A torch.compile backend that receives generated forward/backward graphs.

    Intended for torch.compile(..., backend=make_backend()). The compile()
    convenience wrapper additionally validates module parameters and all public
    inputs and keeps top-level original state_dict keys. No global Dynamo flags are changed.
    """
    def __init__(self, options=None, *, decompositions=None):
        self.options = CompileOptions() if options is None else options
        if not isinstance(self.options, CompileOptions):
            raise TypeError('options must be CompileOptions')
        self._state = _State(self.options)
        self._decompositions = _validate_decompositions(decompositions)
        self._aot = None
        self._init_lock = RLock()

    @property
    def info(self):
        return self._state.snapshot()

    def _compiler(self, phase, gm, example_inputs):
        self._state.check_open()
        # example_inputs can contain FakeTensors and symbolic sizes. Never run a
        # native kernel, read tensor data or allocate native workspace here.
        _check_tensors(example_inputs, self.options.device_type, f'{phase} graph inputs')
        _check_module(gm, self.options.device_type)
        lowered = partition(gm, self._state, phase, self.options)
        from functorch.compile import make_boxed_func

        def run(*args):
            self._state.check_open()
            _check_tensors(args, self.options.device_type, f'{phase} invocation')
            self._state.add(f'{phase}_calls')
            try:
                return lowered(*args)
            except NativeCoverageError:
                self._state.add('execution_errors')
                raise
            except Exception as exc:
                self._state.add('execution_errors')
                raise GraphExecutionError(
                    f'RUDA {phase} graph failed on {self.options.device_type}: {exc}. '
                    'No retry, host transfer or eager re-execution was performed. '
                    'Inspect .info["graphs"] for the captured operators.'
                ) from exc
        return make_boxed_func(run)

    def __call__(self, gm, example_inputs):
        self._state.check_open()
        self._state.add('dynamo_graphs')
        if self.options.capture != 'aot':
            raise ValueError("an RudaBackend passed to torch.compile requires capture='aot'; "
                             "use ruda_torch.compile(..., capture='eager') for eager execution")
        with self._init_lock:
            if self._aot is None:
                # Import lazily; incompatible PyTorch APIs fail visibly, without
                # silently changing to another compiler or replaying a model.
                from torch._dynamo.backends.common import aot_autograd
                self._aot = aot_autograd(
                    fw_compiler=partial(self._compiler, 'forward'),
                    bw_compiler=partial(self._compiler, 'backward'),
                    inference_compiler=partial(self._compiler, 'inference'),
                    decompositions=self._decompositions,
                )
        return self._aot(gm, example_inputs)

    def close(self):
        self._state.close()

    def __enter__(self):
        self._state.check_open()
        return self

    def __exit__(self, *_):
        self.close()


def make_backend(*, native='auto', device_type='ruda', min_native_ops=2,
                 cache_size=4, decompositions=None):
    """Build an independently owned backend with .info and .close()."""
    backend = RudaBackend(CompileOptions(native=native, device_type=device_type,
                                         min_native_ops=min_native_ops, cache_size=cache_size),
                          decompositions=decompositions)
    _check_compiler_policy(backend)
    return backend


class CompiledModel(torch.nn.Module):
    """Model facade preserving parameter identity and top-level checkpoint keys.

    train()/eval()/to() affect the original module; optimizers constructed before
    wrapping still own the same parameters. Tensor shapes and Python control flow
    are guarded by torch.compile. Finish every backward before calling close().
    Training/evaluation and grad mode are not forced by this wrapper.
    """
    def __init__(self, model, backend):
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise TypeError('CompiledModel requires torch.nn.Module')
        _check_module(model, backend.options.device_type)
        self._original = model
        self._backend = backend
        # Register the original only once; OptimizedModule must not add a second
        # copy of its parameters to this wrapper's module tree.
        compiled = _make_callable(model, backend)
        object.__setattr__(self, '_compiled_call', compiled)
        self.training = model.training

    @property
    def original(self):
        return self._original

    @property
    def info(self):
        return self._backend.info

    def forward(self, *args, **kwargs):
        self._backend._state.check_open()
        _check_compiler_policy(self._backend)
        _check_module(self._original, self._backend.options.device_type)
        _check_tensors((args, kwargs), self._backend.options.device_type, 'model inputs')
        if self._backend.options.capture == 'eager':
            self._backend._state.add('eager_calls')
        return self._compiled_call(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        # Top-level checkpoints keep original keys. As a child of another
        # module, preserve the structural _original prefix so the parent's
        # recursive loader and versioned module metadata round-trip correctly.
        prefix = kwargs.get('prefix', args[1] if len(args) > 1 else '')
        if prefix:
            return super().state_dict(*args, **kwargs)
        return self._original.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self._original.load_state_dict(state_dict, strict=strict, assign=assign)

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self._original.named_parameters(prefix, recurse, remove_duplicate)

    def named_buffers(self, prefix='', recurse=True, remove_duplicate=True):
        return self._original.named_buffers(prefix, recurse, remove_duplicate)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            original = super().__getattr__('_original')
            return getattr(original, name)

    def close(self):
        self._backend.close()

    def __enter__(self):
        self._backend._state.check_open()
        return self

    def __exit__(self, *_):
        self.close()


class CompiledFunction:
    """Callable counterpart; supports closures and caller-owned training steps."""
    def __init__(self, function, backend):
        self.original = function
        self._backend = backend
        self._compiled_call = _make_callable(function, backend)
        update_wrapper(self, function, updated=())

    def __call__(self, *args, **kwargs):
        self._backend._state.check_open()
        _check_compiler_policy(self._backend)
        _check_tensors((args, kwargs), self._backend.options.device_type, 'function inputs')
        if self._backend.options.capture == 'eager':
            self._backend._state.add('eager_calls')
        return self._compiled_call(*args, **kwargs)

    @property
    def info(self):
        return self._backend.info

    def close(self):
        self._backend.close()

    def __enter__(self):
        self._backend._state.check_open()
        return self

    def __exit__(self, *_):
        self.close()


def _check_compiler_policy(backend):
    if backend.options.capture == 'aot' and torch._dynamo.config.suppress_errors:
        raise ValueError('torch._dynamo.config.suppress_errors must be False: '
                         'implicit eager re-execution after compiler errors is not allowed')


def _make_callable(function, backend):
    _check_compiler_policy(backend)
    if backend.options.capture == 'eager':
        return function
    if backend.options.device_type == 'ruda':
        # Ensure FakeTensor/device context hooks exist before Dynamo tracing.
        from ._compile_device import register_device_interface
        register_device_interface()
    return torch.compile(function, backend=backend, fullgraph=backend.options.fullgraph,
                         dynamic=backend.options.dynamic)


def compile(model=None, *, capture='aot', native='auto', device_type='ruda',
            fullgraph=False, dynamic=None, min_native_ops=2, cache_size=4,
            decompositions=None):
    """Wrap an nn.Module or callable without rewriting model code.

    capture='aot': PyTorch generates forward/backward graphs. fullgraph=False
      permits Python graph breaks; fullgraph=True rejects them. Neither setting
      means that every operation is a native RUDA graph node.
    capture='eager': explicit uncompiled PyTorch execution, including otherwise
      untraceable model code; higher derivatives depend on individual operators.
    native='auto': use guarded native regions; other ops stay on their device.
    native='required': reject any non-native captured operator or region guard.
    native='off': disable native regions, but keep AOT forward/backward capture.
    device_type='cpu': explicit compiler-reference execution, not RUDA validation.
    decompositions: optional caller-owned operator decompositions for AOT capture.

    No mode installs CPU fallback or automatically retries a failed model call.
    Unsupported device kernels/custom-op metadata must be implemented separately.
    """
    options = CompileOptions(capture, native, device_type, fullgraph, dynamic, min_native_ops, cache_size)
    decompositions = _validate_decompositions(decompositions)
    if options.capture == 'eager' and decompositions:
        raise ValueError('decompositions are only applied during AOT capture')
    if model is None:
        return partial(compile, capture=capture, native=native, device_type=device_type,
                       fullgraph=fullgraph, dynamic=dynamic, min_native_ops=min_native_ops,
                       cache_size=cache_size, decompositions=decompositions)
    if not callable(model):
        raise TypeError('model must be an nn.Module or callable')
    backend = RudaBackend(options, decompositions=decompositions)
    return CompiledModel(model, backend) if isinstance(model, torch.nn.Module) else CompiledFunction(model, backend)
