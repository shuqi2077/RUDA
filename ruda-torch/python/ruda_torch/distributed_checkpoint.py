"""Committed all-rank checkpoints and explicit topology-changing tensor resharding."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid
import torch


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _write_json(path, value):
    temporary = path.with_name(path.name+'.next')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def model_shard_layout(model):
    """Describe actual FSDP/TP storage, retaining full logical names and ties."""
    from .sharded_training import FullyShardedModule
    from .parallel_training import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
    layout = {}
    for path, module in model.named_modules():
        prefix = path+'.' if path else ''
        if isinstance(module, FullyShardedModule):
            for name, index, shape, dtype, trainable in module._schema:
                local = prefix+module._shard_names[index]
                if local in layout:
                    layout[local]['aliases'].append(prefix+name)
                else:
                    layout[local] = {'name': prefix+name, 'aliases': [], 'shape': list(shape), 'axis': None}
            for name, key in module._buffer_names:
                if key in module._buffer_shards:
                    spec={'name':prefix+name,'aliases':[],'shape':list(module._buffer_shards[key]),'axis':None}
                else:
                    spec={'name':prefix+name,'aliases':[],'shape':list(module.get_buffer(key).shape),'replicated':True}
                if prefix+key in layout:
                    layout[prefix+key]['aliases'].append(prefix+name)
                else:
                    layout[prefix+key]=spec
        elif isinstance(module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
            axis = 1 if isinstance(module, RowParallelLinear) else 0
            shape = list(module.weight.shape)
            shape[axis] *= module.group.world_size
            layout[prefix+'weight'] = {'name': prefix+'weight', 'aliases': [], 'shape': shape, 'axis': axis}
            if isinstance(module, ColumnParallelLinear) and module.bias is not None:
                layout[prefix+'bias'] = {'name': prefix+'bias', 'aliases': [], 'shape': [module.out_features], 'axis': 0}
    return layout


def _optimizer_description(model, optimizer, layout):
    from .sharded_training import Zero2Optimizer
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    if isinstance(optimizer, Zero2Optimizer):
        names = {id(shard): names[id(parameter)] for parameter, shard in optimizer.entries}
        optimizer_layout = {names[id(shard)]: {'name': names[id(shard)], 'aliases': [],
                            'shape': list(parameter.shape), 'axis': None}
                            for parameter, shard in optimizer.entries}
        actual = optimizer.optimizer
    else:
        optimizer_layout, actual = layout, optimizer
    state = actual.state_dict()
    groups, states = [], {}
    for live, saved in zip(actual.param_groups, state['param_groups'], strict=True):
        group_names = [names[id(parameter)] for parameter in live['params']]
        groups.append(dict((key, value) for key, value in saved.items() if key != 'params') | {'params': group_names})
        for name, identity in zip(group_names, saved['params'], strict=True):
            if identity in state['state']:
                states[name] = state['state'][identity]
    extra = {key: value for key, value in state.items() if key not in ('param_groups', 'state')}
    return {'groups': groups, 'states': states, 'extra': extra, 'layout': optimizer_layout,
            'class': type(actual).__module__+'.'+type(actual).__qualname__}


class DistributedCheckpoint:
    """Shared-filesystem, all-rank commit at a completed optimizer boundary.

    The latest pointer moves only after every rank has fsynced and reread its
    checkpoint and root has published a complete manifest. A failed write leaves
    the previous committed checkpoint intact. Application/sampler/RNG positions
    are caller-owned state, not inferred from the number of processed batches.
    """
    def __init__(self, group, directory):
        self.group, self.directory = group, Path(directory)

    def save(self, model, optimizer, *, step, application_state=None, scheduler=None, scaler=None):
        layout = model_shard_layout(model)
        self.group.validate_training_options((step, layout))
        if type(step) is not int or step < 0:
            raise ValueError('checkpoint step must be a nonnegative integer')
        generations = self.group.gather_metadata(uuid.uuid4().hex if self.group.rank == 0 else None)
        generation = f'step-{step}-{generations[0]}'
        folder = self.directory/generation
        error, path = None, folder/f'rank-{self.group.rank}.pt'
        started = time.monotonic()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            state = {'version': 1, 'step': step, 'rank': self.group.rank, 'world_size': self.group.world_size,
                     'model': model.state_dict(), 'layout': layout,
                     'optimizer': _optimizer_description(model, optimizer, layout),
                     'application': application_state,
                     'scheduler': None if scheduler is None else scheduler.state_dict(),
                     'scaler': None if scaler is None else scaler.state_dict(),
                     'torch_rng': torch.get_rng_state()}
            if self.group.device_type == 'ruda':
                state['cuda_rng'] = torch.cuda.get_rng_state(self.group.cuda_index)
                from . import get_rng_state
                state['ruda_rng'] = get_rng_state()
            temporary = path.with_suffix('.next')
            with temporary.open('wb') as stream:
                torch.save(_cpu(state), stream)
                stream.flush()
                os.fsync(stream.fileno())
            checked = torch.load(temporary, map_location='cpu', weights_only=True)
            if (checked['step'], checked['rank'], checked['world_size']) != (step, self.group.rank, self.group.world_size):
                raise RuntimeError('rank checkpoint readback differs')
            os.replace(temporary, path)
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        receipts = self.group.gather_metadata((str(path.resolve()), error))
        for saved, failure in receipts:
            if failure:
                raise RuntimeError(f'checkpoint not committed: {failure}')
        committed = None
        if self.group.rank == 0:
            try:
                for rank, (saved, _) in enumerate(receipts):
                    checked = torch.load(saved, map_location='cpu', weights_only=True)
                    if (checked['rank'], checked['step'], checked['world_size']) != (rank, step, self.group.world_size):
                        raise RuntimeError('shared filesystem rank files differ')
                manifest = {'version': 1, 'step': step, 'world_size': self.group.world_size,
                            'files': [Path(saved).name for saved, _ in receipts]}
                _write_json(folder/'manifest.json', manifest)
                _write_json(self.directory/'latest.json', {'generation': generation, 'step': step})
            except Exception as failure:
                committed = f'{type(failure).__name__}: {failure}'
        errors = self.group.gather_metadata(committed)
        if errors[0]:
            raise RuntimeError(f'checkpoint not committed: {errors[0]}')
        return {'directory': str(folder.resolve()), 'step': step, 'seconds': time.monotonic()-started}

    def load(self, model, optimizer, *, generation=None, scheduler=None, scaler=None,validate_application=None):
        folder,manifest,error=None,None,None
        try:
            folder,manifest=self._manifest(generation)
        except Exception as failure:
            error=f'{type(failure).__name__}: {failure}'
        for failure in self.group.gather_metadata(error):
            if failure:raise ValueError(failure)
        self.group.validate_training_options((str(folder.resolve()), manifest))
        if manifest['world_size'] != self.group.world_size:
            raise ValueError('world size changed; use consolidate() and load_consolidated() with an explicit data resharder')
        state, error = None, None
        try:
            state = torch.load(folder/manifest['files'][self.group.rank], map_location='cpu', weights_only=True)
            if (state['step'], state['rank'], state['world_size'], state['layout']) != (
                    manifest['step'], self.group.rank, self.group.world_size, model_shard_layout(model)):
                raise ValueError('rank checkpoint topology or layout differs')
            if (state['scheduler'] is None) != (scheduler is None) or (state['scaler'] is None) != (scaler is None):
                raise ValueError('scheduler/scaler presence differs')
            current=model.state_dict()
            if set(current)!=set(state['model']):raise ValueError('checkpoint model keys differ')
            for name,target in current.items():
                value=state['model'][name]
                if isinstance(target,torch.Tensor) and (not isinstance(value,torch.Tensor) or target.shape!=value.shape or target.dtype!=value.dtype):
                    raise ValueError(f'checkpoint model shape/dtype differs: {name}')
            live=_optimizer_description(model,optimizer,state['layout'])
            if live['class']!=state['optimizer']['class'] or [g['params'] for g in live['groups']]!=[g['params'] for g in state['optimizer']['groups']]:
                raise ValueError('checkpoint optimizer class/parameter layout differs')
            if validate_application is not None:
                validate_application(state['application'])
        except Exception as failure:
            error = str(failure)
        for failure in self.group.gather_metadata(error):
            if failure:
                raise ValueError(failure)
        model.load_state_dict(state['model'])
        _restore_optimizer(model, optimizer, state['optimizer'], state['layout'], self.group, consolidated=False)
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        if scaler is not None:
            scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        if 'cuda_rng' in state:
            torch.cuda.set_rng_state(state['cuda_rng'], self.group.cuda_index)
        if 'ruda_rng' in state:
            from . import set_rng_state
            set_rng_state(state['ruda_rng'])
        return state['step'], state['application']

    def _manifest(self, generation):
        if generation is None:
            generation = json.loads((self.directory/'latest.json').read_text(encoding='utf-8'))['generation']
        if Path(generation).name != generation:
            raise ValueError('generation must be a checkpoint directory name')
        folder = self.directory/generation
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['version'] != 1 or len(manifest['files']) != manifest['world_size']:
            raise ValueError('invalid checkpoint commit manifest')
        return folder, manifest

    def consolidate(self, *, generation=None):
        """CPU operation: reconstruct weights and elementwise optimizer states.

        Memory is the full checkpoint plus the rank-local input checkpoints.
        This is explicit, never automatically run inside a training step.
        """
        folder, manifest = self._manifest(generation)
        ranks = [torch.load(folder/name, map_location='cpu', weights_only=True) for name in manifest['files']]
        first = ranks[0]
        for rank, state in enumerate(ranks):
            if (state['version'], state['rank'], state['world_size'], state['step'], state['layout']) != (
                    1, rank, manifest['world_size'], manifest['step'], first['layout']):
                raise ValueError('rank checkpoint identities differ')
        model = {}
        for key in first['model']:
            spec = first['layout'].get(key)
            value = _join([state['model'][key] for state in ranks], spec)
            name = key if spec is None else spec['name']
            model[name] = value
            for alias in () if spec is None else spec['aliases']:
                model[alias] = value
        descriptions = [state['optimizer'] for state in ranks]
        base = descriptions[0]
        if any(not _equal((d['groups'], d['extra'], d['class'], d['layout']),
                          (base['groups'], base['extra'], base['class'], base['layout'])) for d in descriptions):
            raise ValueError('optimizer rank descriptions differ')
        states = {}
        for name in set().union(*(d['states'] for d in descriptions)):
            values = [d['states'].get(name) for d in descriptions]
            if any(value is None for value in values):
                raise ValueError('optimizer state missing on some shards')
            spec = base['layout'].get(name)
            global_name = name if spec is None else spec['name']
            local_parameter = first['model'].get(name)
            fields = {}
            for key, value in values[0].items():
                shape = None if spec is None else spec['shape']
                is_parameter_state = isinstance(value, torch.Tensor) and shape is not None and (
                    local_parameter is not None and value.shape == local_parameter.shape or
                    spec.get('axis') is None and value.ndim == 1 and value.numel() == math.ceil(math.prod(shape)/len(ranks)))
                fields[key] = _join([entry[key] for entry in values], spec if is_parameter_state else None)
            states[global_name] = fields
        groups = [dict(group, params=[base['layout'].get(name, {}).get('name', name) for name in group['params']])
                  for group in base['groups']]
        return {'version': 1, 'step': first['step'], 'model': model,
                'optimizer': dict(base, groups=groups, states=states, layout={}),
                'rank_states': [{'application': state['application'], 'torch_rng': state['torch_rng'],
                                 'cuda_rng': state.get('cuda_rng'),'ruda_rng':state.get('ruda_rng')} for state in ranks],
                'scheduler': first['scheduler'], 'scaler': first['scaler']}

    def load_consolidated(self, model, optimizer, state, *, reshard_application, scheduler=None, scaler=None):
        """Restore tensors on a new topology; caller explicitly repartitions data/RNG.

        reshard_application(old_rank_states, new_rank, new_world) returns the
        application state and must explicitly restore any run-specific RNG.
        """
        layout = model_shard_layout(model)
        local = {}
        for name, target in model.state_dict().items():
            spec = layout.get(name)
            logical = name if spec is None else spec['name']
            local[name] = _slice(state['model'][logical], spec, self.group.rank, self.group.world_size)
            if isinstance(target,torch.Tensor) and (not isinstance(local[name],torch.Tensor) or local[name].shape != target.shape):
                raise ValueError(f'resharded model shape differs: {name}')
        model.load_state_dict(local)
        _restore_optimizer(model, optimizer, state['optimizer'], layout, self.group, consolidated=True)
        if (scheduler is None) != (state['scheduler'] is None) or (scaler is None) != (state['scaler'] is None):
            raise ValueError('scheduler/scaler presence differs')
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        if scaler is not None:
            scaler.load_state_dict(state['scaler'])
        application = reshard_application(state['rank_states'], self.group.rank, self.group.world_size)
        return state['step'], application


def _join(values, spec):
    if spec is None or spec.get('replicated') or not isinstance(values[0], torch.Tensor):
        if any(not _equal(values[0], value) for value in values[1:]):
            raise ValueError('replicated checkpoint values differ')
        return values[0]
    axis = spec['axis']
    if axis is None:
        return torch.cat([value.reshape(-1) for value in values])[:math.prod(spec['shape'])].view(spec['shape'])
    return torch.cat(values, dim=axis)


def _slice(value, spec, rank, world):
    if spec is None or spec.get('replicated'):
        return value
    if spec['axis'] is None:
        size = math.ceil(math.prod(spec['shape'])/world)
        result = value.new_zeros(size)
        start, end = rank*size, min((rank+1)*size, value.numel())
        if end > start:
            result[:end-start].copy_(value.reshape(-1)[start:end])
        return result
    if value.shape[spec['axis']] % world:
        raise ValueError('TP checkpoint dimension does not divide the new world size')
    return value.chunk(world, dim=spec['axis'])[rank].contiguous()


def _restore_optimizer(model, optimizer, description, layout, group, *, consolidated):
    from .sharded_training import Zero2Optimizer
    live_description = _optimizer_description(model, optimizer, layout)
    if description['class'] != live_description['class']:
        raise ValueError('optimizer class differs')
    actual = optimizer.optimizer if isinstance(optimizer, Zero2Optimizer) else optimizer
    live = actual.state_dict()
    saved_groups = description['groups']
    groups, states = [], {}
    if len(saved_groups) != len(live['param_groups']):
        raise ValueError('optimizer parameter groups differ')
    for named, indexed, saved in zip(live_description['groups'], live['param_groups'], saved_groups, strict=True):
        specs = live_description['layout']
        logical = [specs.get(name, {}).get('name', name) for name in named['params']]
        if (logical if consolidated else named['params']) != saved['params']:
            raise ValueError('optimizer parameter paths differ')
        groups.append(dict(saved, params=indexed['params']))
        for name, global_name, identity, parameter in zip(named['params'], logical, indexed['params'],
                                                         actual.param_groups[len(groups)-1]['params'], strict=True):
            entry = description['states'].get(global_name if consolidated else name)
            if entry is None:
                continue
            fields = {}
            spec = specs.get(name)
            for key, value in entry.items():
                if consolidated and spec is not None and not spec.get('replicated') and isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(spec['shape']):
                    value = _slice(value, spec, group.rank, group.world_size)
                fields[key] = value
            states[identity] = fields
    actual.load_state_dict(dict(description['extra'], param_groups=groups, state=states))
    if isinstance(optimizer, Zero2Optimizer):
        with torch.no_grad():
            for parameter, shard in optimizer.entries:
                value = _slice(parameter.detach(), {'axis': None, 'shape': list(parameter.shape)}, group.rank, group.world_size)
                shard.copy_(value)
