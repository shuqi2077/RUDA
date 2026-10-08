"""Whole-update tuning on actual training windows, with reversible trial updates."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import random
import statistics
import sys
import time

import torch

from .finetuning import NF4Linear, _cpu_tree


def _geometry(batches, packed):
    result = []
    for batch in batches:
        tensors = tuple((name, tuple(value.shape), str(value.dtype))
                        for name, value in sorted(batch.items()) if isinstance(value, torch.Tensor))
        boundaries = tuple(batch['cu_seqlens'].tolist()) if packed else ()
        result.append((tensors, boundaries))
    return tuple(result)


def _split(batches, packed, maximum):
    if maximum == 0:
        return list(batches)
    result = []
    for batch in batches:
        boundaries = batch['cu_seqlens'].tolist() if packed else None
        count = len(boundaries) - 1 if packed else batch['input_ids'].shape[0]
        for begin in range(0, count, maximum):
            end = min(begin + maximum, count)
            if not packed:
                result.append({name: value[begin:end] for name, value in batch.items()})
                continue
            lo, hi = boundaries[begin], boundaries[end]
            part = {name: value[lo:hi] for name, value in batch.items()
                    if name not in ('cu_seqlens', 'max_seqlen')}
            part['cu_seqlens'] = batch['cu_seqlens'][begin:end + 1] - lo
            part['max_seqlen'] = max(boundaries[index + 1] - boundaries[index]
                                    for index in range(begin, end))
            result.append(part)
    return result


@dataclass(frozen=True)
class TrainingPlan:
    """Measured execution layout; zero examples_per_microbatch preserves the input layout."""
    examples_per_microbatch: int
    token_chunk_size: int | None
    packed: bool
    geometry: tuple

    def apply(self, trainer, batches):
        batches = list(batches)
        if self.packed != trainer.packed_input or _geometry(batches, self.packed) != self.geometry:
            raise ValueError('training plan belongs to a different actual batch geometry')
        original_count = sum(trainer._batch_counts(batches))
        selected = _split(batches, self.packed, self.examples_per_microbatch)
        if sum(trainer._batch_counts(selected)) != original_count:
            raise ValueError('training plan changed the effective supervised token count')
        return selected


def _modules(trainer):
    seen, result = set(), []
    def visit(module):
        if id(module) in seen:
            return
        seen.add(id(module))
        result.append(module)
        for child in module.children():
            visit(child)
        template = getattr(module, '_template', None)
        if isinstance(template, torch.nn.Module):
            visit(template)
    model = trainer.model if hasattr(trainer, 'model') else trainer.stage.module
    visit(model)
    visit(getattr(trainer, 'executable', model))
    return result


def _rng(devices):
    value = {'python': random.getstate(), 'cpu': torch.get_rng_state().clone(),
             'cuda': {device.index: torch.cuda.get_rng_state(device).cpu() for device in devices
                      if device.type == 'cuda'}}
    if any(device.type == 'ruda' for device in devices):
        from . import get_rng_state
        value['ruda'] = get_rng_state()
    if 'numpy' in sys.modules:
        value['numpy'] = copy.deepcopy(sys.modules['numpy'].random.get_state())
    return value


def _restore_rng(value):
    random.setstate(value['python'])
    torch.set_rng_state(value['cpu'])
    for device, state in value['cuda'].items():
        torch.cuda.set_rng_state(state, device)
    if 'ruda' in value:
        from . import set_rng_state
        set_rng_state(value['ruda'])
    if 'numpy' in value:
        sys.modules['numpy'].random.set_state(copy.deepcopy(value['numpy']))


def _equal(expected, actual, rtol=0., atol=0., exact=False):
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return False
        if expected.is_floating_point() or expected.is_complex():
            if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()):
                return False
            return bool(torch.equal(expected, actual) if exact else
                        torch.allclose(expected, actual, rtol=rtol, atol=atol))
        return bool(torch.equal(expected, actual))
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _equal(expected[key], actual[key], rtol, atol, exact or key in ('step', 'param_groups')) for key in expected)
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(
            _equal(left, right, rtol, atol, exact) for left, right in zip(expected, actual))
    if isinstance(expected, float):
        return math.isfinite(expected) and math.isfinite(actual) and (
            expected == actual if exact else abs(expected - actual) <= atol + rtol * abs(expected))
    if 'numpy' in sys.modules and isinstance(expected, sys.modules['numpy'].ndarray):
        return expected.shape == actual.shape and expected.dtype == actual.dtype and bool(
            sys.modules['numpy'].array_equal(expected, actual))
    return expected == actual


def _bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_bytes(item) for item in value)
    return 0


def _pending_gradients(trainer):
    model = trainer.model if hasattr(trainer, 'model') else trainer.stage.module
    parameters = list(model.parameters())
    if trainer.optimizer is not None:
        parameters += [value for options in trainer.optimizer.param_groups for value in options['params']]
    return any(value.grad is not None for value in parameters)


class _UpdateState:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model if hasattr(trainer, 'model') else trainer.stage.module
        self.modules = _modules(trainer)
        tensors = {id(value): value for value in self.model.parameters() if value.requires_grad}
        for options in [] if trainer.optimizer is None else trainer.optimizer.param_groups:
            for value in options['params']:
                tensors[id(value)] = value
        self.buffers = []
        self.parameters = []
        self.mutable_buffers = []
        self.extra_modules = []
        from .sharded_training import FullyShardedModule
        for module in self.modules:
            self.buffers.append((module, dict(module._buffers)))
            self.parameters.append((module, dict(module._parameters)))
            # These four buffers are the immutable original NF4 base, not trial state.
            # Subclasses are not assumed immutable.
            readonly = set()
            if type(module) is FullyShardedModule:
                readonly.update(name for name in module._shard_names if not name.startswith('shards.'))
                readonly.update(module._buffer_shards)
            if type(module) is not NF4Linear:
                self.mutable_buffers.append((module, readonly))
            if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
                self.extra_modules.append(module)
        self.tensors = list(tensors.values())
        self.devices = {value.device for value in list(self.model.parameters()) + list(self.model.buffers())}
        self.devices.update(value.device for value in self.tensors)
        if hasattr(trainer, 'stage'):
            self.devices.add(trainer.stage.device)
        self.modes = [(module, module.training) for module in self.modules]
        self.bookkeeping = {name: copy.deepcopy(getattr(trainer, name)) for name in
                            ('step', 'cursor', 'tokens', '_step_durations', 'started', 'started_at',
                             'elapsed_before_resume', 'last_checkpoint') if hasattr(trainer, name)}
        attached_sampler = getattr(trainer, 'sampler', None)
        sampler = getattr(attached_sampler, 'sampler', attached_sampler)
        self.sampler = sampler
        self.sampler_cursors = None if sampler is None else (sampler.committed, sampler.issued)
        self.optimizer_flags = []
        optimizer = trainer.optimizer
        seen = set()
        while optimizer is not None and id(optimizer) not in seen:
            seen.add(id(optimizer))
            for name in ('last_step_skipped', '_weight', '_mesh_grad_norm', '_distributed_grad_norm'):
                if hasattr(optimizer, name):
                    self.optimizer_flags.append((optimizer, name, copy.deepcopy(getattr(optimizer, name))))
            optimizer = getattr(optimizer, 'optimizer', None)

    def estimate_bytes(self):
        return (sum(_bytes(value) for value in self.tensors)
                + (0 if self.trainer.optimizer is None else _bytes(self.trainer.optimizer.state_dict()))
                + sum(_bytes(value) for values in self.buffer_values() for value in values.values())
                + sum(_bytes(module.get_extra_state()) for module in self.extra_modules))

    def buffer_values(self):
        return [{name: value for name, value in module._buffers.items() if name not in readonly
                 and (value is None or value.device.type != 'meta')}
                for module, readonly in self.mutable_buffers]

    def synchronize(self):
        for device in self.devices:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elif device.type == 'ruda':
                from . import synchronize
                synchronize(device)
            elif device.type != 'cpu':
                raise ValueError(f'training autotune synchronization is not defined for {device.type}')

    def capture(self):
        trainer = self.trainer
        return {'values': [value.detach().cpu().clone() for value in self.tensors],
                'buffers': _cpu_tree(self.buffer_values()),
                'buffer_keys': [tuple(module._buffers) for module in self.modules],
                'parameter_keys': [tuple((name, id(value)) for name, value in module._parameters.items())
                                   for module in self.modules],
                'optimizer': None if trainer.optimizer is None else _cpu_tree(trainer.optimizer.state_dict()),
                'scaler': None if trainer.scaler is None else _cpu_tree(trainer.scaler.state_dict()),
                'scheduler': None if trainer.scheduler is None else _cpu_tree(trainer.scheduler.state_dict()),
                'extra': [_cpu_tree(module.get_extra_state()) for module in self.extra_modules],
                'rng': _rng(self.devices)}

    def restore(self, saved):
        trainer = self.trainer
        if trainer.optimizer is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        for module, buffers in self.buffers:
            module._buffers.clear()
            module._buffers.update(buffers)
        for module, parameters in self.parameters:
            module._parameters.clear()
            module._parameters.update(parameters)
        with torch.no_grad():
            for value, original in zip(self.tensors, saved['values'], strict=True):
                value.copy_(original.to(device=value.device))
            for (module, _), values in zip(self.mutable_buffers, saved['buffers'], strict=True):
                for name in values:
                    if module._buffers[name] is not None:
                        module._buffers[name].copy_(values[name].to(device=module._buffers[name].device))
        if trainer.optimizer is not None:
            trainer.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        if trainer.scaler is not None:
            trainer.scaler.load_state_dict(copy.deepcopy(saved['scaler']))
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(copy.deepcopy(saved['scheduler']))
        for module, state in zip(self.extra_modules, saved['extra'], strict=True):
            module.set_extra_state(copy.deepcopy(state))
        for name, value in self.bookkeeping.items():
            setattr(trainer, name, copy.deepcopy(value))
        for module, training in self.modes:
            module.training = training
        if self.sampler is not None:
            self.sampler.committed, self.sampler.issued = self.sampler_cursors
        for optimizer, name, value in self.optimizer_flags:
            setattr(optimizer, name, copy.deepcopy(value))
        if trainer.optimizer is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        _restore_rng(saved['rng'])
        self.synchronize()


class TrainingAutotuner:
    """Search actual microbatch and LM-head chunk layouts without consuming an update.

    Trials run the trainer's complete forward/backward, collectives and optimizer
    update, restoring local parameters, optimizer, scaler, scheduler, buffers,
    RNG and sampler cursors afterward. Only existing packed-document boundaries
    or padded rows may be split. No contexts, precision, effective batch or mesh
    are changed. Numeric tolerance is explicit; the default requires exact state.
    Session-local plans never cross a process/device/driver boundary.
    """
    def __init__(self, *, repeats=3, max_candidates=12, max_seconds=300.,
                 rtol=0., atol=0., max_snapshot_bytes=None, progress=None):
        if type(repeats) is not int or repeats < 1 or type(max_candidates) is not int or max_candidates < 1:
            raise ValueError('positive repeat and candidate counts required')
        if any(not math.isfinite(value) or value < 0 for value in (rtol, atol)):
            raise ValueError('numeric tolerances must be finite and nonnegative')
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('positive finite tuning budget required')
        if max_snapshot_bytes is not None and (type(max_snapshot_bytes) is not int or max_snapshot_bytes <= 0):
            raise ValueError('snapshot byte limit must be a positive integer')
        if progress is not None and not callable(progress):
            raise TypeError('progress must be a callable receiving actual tuning state')
        self.repeats, self.max_candidates, self.max_seconds = repeats, max_candidates, max_seconds
        self.rtol, self.atol, self.max_snapshot_bytes = rtol, atol, max_snapshot_bytes
        self.progress_callback = progress
        self.progress, self.results = {}, []
        self._cache = {}

    def _publish(self, **values):
        self.progress.update(values)
        if self.progress_callback is not None:
            self.progress_callback(dict(self.progress))

    @staticmethod
    def _gather(trainer, value):
        return [value] if trainer.replica_group is None else trainer.replica_group.gather_metadata(value)

    @staticmethod
    def _template(trainer):
        return getattr(trainer.model, '_template', trainer.model)

    def _candidates(self, trainer, batches):
        geometry = _geometry(batches, trainer.packed_input)
        template = self._template(trainer)
        # A compiled graph may have captured chunk_size; do not pretend an attribute
        # change rebuilds that graph. Raw finetuners read this attribute per forward.
        chunk = getattr(template, 'token_chunk_size', None)
        chunk_search = trainer.executable is trainer.model and template is trainer.model
        maximum = max((batch['cu_seqlens'].numel() - 1 if trainer.packed_input else
                       batch['input_ids'].shape[0] for batch in batches), default=0)
        requests = self._gather(trainer, (maximum, chunk, chunk_search))
        maximum = min(item[0] for item in requests)
        chunk_search = all(item[2] for item in requests) and all(item[1] == chunk for item in requests)
        current = TrainingPlan(0, chunk, trainer.packed_input, geometry)
        splits, chunks = [], []
        size = 1
        while size < maximum:
            splits.append(size)
            size *= 2
        if chunk_search and type(chunk) is int:
            size = 1
            while size < chunk:
                chunks.append(size)
                size *= 2
        layouts = [(size, chunk) for size in reversed(splits)]
        layouts += [(0, size) for size in reversed(chunks)]
        layouts += [(size, block) for size in reversed(splits) for block in reversed(chunks)]
        result, seen = [current], set()
        for size, block in layouts:
            plan = TrainingPlan(size, block, trainer.packed_input, geometry)
            selected = plan.apply(trainer, batches)
            counts = self._gather(trainer, len(selected))
            # This keeps TP/FSDP collective-bearing forward/backward counts aligned.
            signature = (tuple(self._gather(trainer, _geometry(selected, trainer.packed_input))), block)
            if len(set(counts)) != 1 or signature in seen:
                continue
            seen.add(signature)
            result.append(plan)
            if len(result) >= self.max_candidates:
                break
        return result[:self.max_candidates]

    def _run(self, trainer, batches, plan, state):
        template = self._template(trainer)
        chunk = getattr(template, 'token_chunk_size', None)
        selected = plan.apply(trainer, batches)
        try:
            if plan.token_chunk_size is not None:
                template.token_chunk_size = plan.token_chunk_size
            state.synchronize()
            started = time.perf_counter()
            metrics = trainer.train_step(selected)
            state.synchronize()
            elapsed = time.perf_counter() - started
            elapsed = max(self._gather(trainer, elapsed))
            return metrics, elapsed
        finally:
            if chunk is not None:
                template.token_chunk_size = chunk

    def tune(self, trainer, microbatches):
        """Calibrate against real updates and return a plan; no live step is committed."""
        batches = list(microbatches)
        error = None
        try:
            trainer._batch_counts(batches)
            if _pending_gradients(trainer):
                raise ValueError('training autotune starts only at a cleared optimizer-step boundary')
            if not batches and trainer.replica_group is None:
                raise ValueError('training autotune needs an actual nonempty local window')
        except (TypeError, ValueError, AttributeError) as failure:
            error = str(failure)
        for failure in self._gather(trainer, error):
            if failure:
                raise ValueError(failure)
        options = (self.repeats, self.max_candidates, self.max_seconds, self.rtol, self.atol, self.max_snapshot_bytes)
        if any(other != options for other in self._gather(trainer, options)):
            raise ValueError('all ranks must use the same training autotune policy')
        candidates = self._candidates(trainer, batches)
        state = _UpdateState(trainer)
        estimated_bytes = state.estimate_bytes()
        estimates = self._gather(trainer, estimated_bytes)
        if self.max_snapshot_bytes is not None and any(3 * size > self.max_snapshot_bytes for size in estimates):
            raise ValueError('initial/reference/candidate snapshots exceed the requested host-memory budget')
        started, completed = time.monotonic(), 0
        total = len(candidates) * self.repeats
        self.results = []
        self.progress = {'stage': 'snapshot', 'completed_updates': 0, 'planned_updates': total,
                         'host_snapshot_bytes_before_optimizer_initialization': estimated_bytes,
                         'optimizer_new_state_bytes': None, 'remaining_seconds_range': None,
                         'checkpoint': 'in-memory original update boundary', 'checkpoint_age_seconds': 0.,
                         'device_peak_bytes': None}
        self._publish()
        state.synchronize()
        original = state.capture()
        winner, fastest, durations = candidates[0], math.inf, []
        reference, reference_metrics = None, None
        stochastic = False
        try:
            for index, plan in enumerate(candidates):
                if index and any(self._gather(trainer, stochastic)) and plan.examples_per_microbatch:
                    self.results.append({'plan': plan, 'eligible': False, 'reason': 'reference consumes RNG; preserve its microbatch random stream'})
                    continue
                times, valid = [], True
                for repeat in range(self.repeats):
                    spent = time.monotonic() - started
                    estimate = max(durations, default=0.)
                    stop = spent + estimate > self.max_seconds
                    if completed and any(self._gather(trainer, stop)):
                        break
                    state.restore(original)
                    self._publish(stage='reference' if index == 0 else 'candidate', candidate=index,
                                  repeat=repeat, completed_updates=completed, elapsed_seconds=spent,
                                  checkpoint_age_seconds=spent)
                    metrics, elapsed = self._run(trainer, batches, plan, state)
                    if self.max_snapshot_bytes is not None and any(self._gather(trainer,
                            3 * state.estimate_bytes() > self.max_snapshot_bytes)):
                        raise MemoryError('initialized optimizer/trial state exceeds the requested snapshot budget')
                    observed = state.capture()
                    if reference is None:
                        reference, reference_metrics = observed, metrics
                        stochastic = not _equal(original['rng'], observed['rng'], exact=True)
                        self.progress['optimizer_new_state_bytes'] = _bytes(observed['optimizer']) - _bytes(original['optimizer'])
                    else:
                        valid = all(self._gather(trainer,
                            _equal(reference['values'], observed['values'], self.rtol, self.atol)
                            and _equal(reference['buffers'], observed['buffers'], self.rtol, self.atol)
                            and _equal(reference['buffer_keys'], observed['buffer_keys'], exact=True)
                            and _equal(reference['parameter_keys'], observed['parameter_keys'], exact=True)
                            and _equal(reference['optimizer'], observed['optimizer'], self.rtol, self.atol)
                            and _equal(reference['scaler'], observed['scaler'], exact=True)
                            and _equal(reference['scheduler'], observed['scheduler'], exact=True)
                            and _equal(reference['extra'], observed['extra'], exact=True)
                            and _equal(reference['rng'], observed['rng'], exact=True)
                            and metrics['supervised_tokens'] == reference_metrics['supervised_tokens']
                            and metrics['optimizer_update_skipped'] == reference_metrics['optimizer_update_skipped']
                            and _equal(float(reference_metrics['loss']), float(metrics['loss']), self.rtol, self.atol)))
                    del observed
                    completed += 1
                    durations.append(elapsed)
                    times.append(elapsed)
                    remaining = total - completed
                    self._publish(completed_updates=completed, elapsed_seconds=time.monotonic() - started,
                                  recent_updates_per_second=1. / elapsed,
                                  remaining_seconds_range=(remaining * min(durations), remaining * max(durations)),
                                  eta_basis='measured synchronized whole-update durations; excludes snapshot transfer')
                    if not valid:
                        break
                result = {'plan': plan, 'eligible': valid and len(times) == self.repeats,
                          'update_seconds': times, 'median_seconds': statistics.median(times) if times else None}
                if not valid:
                    result['reason'] = 'post-update state or effective training semantics differ'
                elif len(times) != self.repeats:
                    result['reason'] = 'tuning budget exhausted before the complete measurement'
                self.results.append(result)
                if result['eligible'] and result['median_seconds'] < fastest:
                    winner, fastest = plan, result['median_seconds']
        finally:
            state.restore(original)
            self._publish(stage='restored', completed_updates=completed,
                          elapsed_seconds=time.monotonic() - started, checkpoint_age_seconds=time.monotonic() - started,
                          remaining_seconds_range=(0., 0.))
        self._publish(stage='selected', plan=winner, median_update_seconds=None if math.isinf(fastest) else fastest)
        return winner

    def train_step(self, trainer, microbatches):
        """Calibrate once per session geometry, then commit exactly one selected update."""
        batches = list(microbatches)
        template = self._template(trainer)
        key = (trainer, _geometry(batches, trainer.packed_input), trainer.packed_input,
               getattr(template, 'token_chunk_size', None), repr(trainer.run_config),
               repr(trainer._loss_options()), trainer.gradient_overlap, trainer.bucket_bytes,
               tuple((name, tuple(value.shape), str(value.dtype), str(value.device), value.requires_grad)
                     for name, value in trainer.model.named_parameters()),
               tuple((type(module), tuple((name, repr(getattr(module, name))) for name in
                     ('dropout', 'p', 'activation_checkpointing', 'preserve_rng_state', 'recompute', 'ensure_backward')
                     if hasattr(module, name))) for module in _modules(trainer)))
        hit = key in self._cache
        if not all(self._gather(trainer, hit)):
            self._cache[key] = self.tune(trainer, batches)
        plan = self._cache[key]
        selected = plan.apply(trainer, batches)
        cursor = trainer.cursor
        chunk = getattr(template, 'token_chunk_size', None)
        try:
            if plan.token_chunk_size is not None:
                template.token_chunk_size = plan.token_chunk_size
            result = trainer.train_step(selected)
        finally:
            if chunk is not None:
                template.token_chunk_size = chunk
        # Cursor tracks caller/loader windows, not internal split execution units.
        trainer.cursor = cursor + len(batches)
        result['microbatch_cursor'] = trainer.cursor
        result['autotune_plan'] = plan
        return result


class PipelineAutotuner(TrainingAutotuner):
    """Measure native 1F1B versus fill/drain schedules on the unchanged stage mesh.

    Stage partitions, TP/DP dimensions, boundary layouts, effective token weight
    and input order remain the caller's actual contracts. Every rank participates
    and a schedule is accepted only when all local post-update states match.
    Fill/drain retains all microbatch graphs until backward; unlike 1F1B it may
    require substantially more activation memory. No failed collective is retried.
    """
    @staticmethod
    def _gather(trainer, value):
        return trainer.mesh.world.gather_metadata(value)

    def tune(self, trainer, inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None):
        inputs, targets = list(inputs or []), list(targets or [])
        specs = None if microbatch_specs is None else tuple(microbatch_specs)
        contracts = (self.repeats, self.max_seconds, self.rtol, self.atol, self.max_snapshot_bytes)
        if any(value != contracts for value in self._gather(trainer, contracts)):
            raise ValueError('all pipeline ranks must use the same tuning policy')
        if any(self._gather(trainer, _pending_gradients(trainer))):
            raise ValueError('pipeline autotune starts at a cleared optimizer-step boundary')
        state = _UpdateState(trainer)
        estimate = state.estimate_bytes()
        if self.max_snapshot_bytes is not None and any(
                3 * size > self.max_snapshot_bytes for size in self._gather(trainer, estimate)):
            raise ValueError('pipeline trial snapshots exceed the requested host-memory budget')
        state.synchronize()
        original = state.capture()
        reference, reference_metrics = None, None
        started, completed, durations = time.monotonic(), 0, []
        schedules = ('1f1b', 'gpipe')[:self.max_candidates]
        winner, fastest = '1f1b', math.inf
        self.results = []
        self.progress = {'stage': 'snapshot', 'completed_updates': 0,
                         'planned_updates': len(schedules) * self.repeats,
                         'host_snapshot_bytes_before_optimizer_initialization': estimate,
                         'checkpoint': 'in-memory original pipeline update boundary',
                         'remaining_seconds_range': None}
        try:
            for schedule in schedules:
                times, valid = [], True
                for repeat in range(self.repeats):
                    spent = time.monotonic() - started
                    if completed and any(self._gather(trainer, spent + max(durations, default=0.) > self.max_seconds)):
                        break
                    state.restore(original)
                    self._publish(stage=schedule, repeat=repeat, completed_updates=completed,
                                  elapsed_seconds=spent, checkpoint_age_seconds=spent)
                    begun = time.perf_counter()
                    metrics = trainer.train_step(inputs, targets, local_weight=local_weight,
                        loss_sum=loss_sum, microbatch_specs=specs, schedule=schedule)
                    state.synchronize()
                    elapsed = max(self._gather(trainer, time.perf_counter() - begun))
                    if self.max_snapshot_bytes is not None and any(self._gather(trainer,
                            3 * state.estimate_bytes() > self.max_snapshot_bytes)):
                        raise MemoryError('initialized pipeline trial state exceeds the requested snapshot budget')
                    observed = state.capture()
                    if reference is None:
                        reference, reference_metrics = observed, metrics
                    else:
                        valid = all(self._gather(trainer,
                            all(_equal(reference[name], observed[name], self.rtol, self.atol)
                                for name in ('values', 'buffers', 'optimizer'))
                            and all(_equal(reference[name], observed[name], exact=True)
                                    for name in ('buffer_keys', 'parameter_keys', 'scaler', 'scheduler', 'extra', 'rng'))
                            and metrics['supervised_tokens'] == reference_metrics['supervised_tokens']
                            and metrics['optimizer_update_skipped'] == reference_metrics['optimizer_update_skipped']
                            and _equal(float(reference_metrics['loss']), float(metrics['loss']), self.rtol, self.atol)))
                    del observed
                    completed += 1
                    durations.append(elapsed)
                    times.append(elapsed)
                    remaining = self.progress['planned_updates'] - completed
                    self._publish(completed_updates=completed, elapsed_seconds=time.monotonic() - started,
                        recent_updates_per_second=1. / elapsed,
                        remaining_seconds_range=(remaining * min(durations), remaining * max(durations)),
                        eta_basis='measured synchronized slowest-rank update durations; excludes snapshot transfer')
                    if not valid:
                        break
                result = {'schedule': schedule, 'eligible': valid and len(times) == self.repeats,
                          'update_seconds': times, 'median_seconds': statistics.median(times) if times else None}
                self.results.append(result)
                if result['eligible'] and result['median_seconds'] < fastest:
                    winner, fastest = schedule, result['median_seconds']
        finally:
            state.restore(original)
            self._publish(stage='restored', elapsed_seconds=time.monotonic() - started,
                          checkpoint_age_seconds=time.monotonic() - started, remaining_seconds_range=(0., 0.))
        self._publish(stage='selected', schedule=winner,
                      median_update_seconds=None if math.isinf(fastest) else fastest)
        return winner

    def train_step(self, trainer, inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None):
        from torch.utils._pytree import tree_flatten
        inputs, targets = list(inputs or []), list(targets or [])
        specs = None if microbatch_specs is None else tuple(microbatch_specs)
        def geometry(values):
            leaves, tree = tree_flatten(values)
            return repr(tree), tuple((tuple(value.shape), str(value.dtype)) if isinstance(value, torch.Tensor)
                                    else repr(value) for value in leaves)
        key = (trainer, geometry(inputs), geometry(targets), repr(specs), id(loss_sum),
               trainer.mesh.shape, repr(trainer.run_config),
               tuple((type(module), tuple((name, repr(getattr(module, name))) for name in
                    ('dropout', 'p', 'recompute', 'ensure_backward') if hasattr(module, name)))
                     for module in _modules(trainer)))
        if not all(self._gather(trainer, key in self._cache)):
            self._cache[key] = self.tune(trainer, inputs, targets, local_weight=local_weight,
                                       loss_sum=loss_sum, microbatch_specs=specs)
        schedule = self._cache[key]
        result = trainer.train_step(inputs, targets, local_weight=local_weight, loss_sum=loss_sum,
                                    microbatch_specs=specs, schedule=schedule)
        result['autotune_schedule'] = schedule
        return result
