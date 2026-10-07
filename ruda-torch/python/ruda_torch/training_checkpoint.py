"""Actual model/optimizer snapshots for full-parameter causal training."""
from __future__ import annotations

import copy
from collections import OrderedDict
import torch

from .finetuning import _cpu_tree


def _tensor_schema(model):
    schema, aliases = {}, {}
    for kind, tensors in (
        ('parameter', model.named_parameters(remove_duplicate=False)),
        ('buffer', model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in tensors:
            canonical = aliases.setdefault(id(tensor), name)
            schema[name] = {'kind': kind, 'shape': tuple(tensor.shape), 'dtype': str(tensor.dtype),
                            'requires_grad': tensor.requires_grad, 'alias': canonical}
    return schema


def _optimizer_layout(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        row = []
        for parameter in group['params']:
            if id(parameter) not in names:
                raise ValueError('full snapshot optimizer owns a tensor outside the model; use its explicit sharded checkpoint API')
            row.append(names[id(parameter)])
        groups.append(row)
    return groups


def _cuda_devices(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return sorted({tensor.device.index for tensor in tensors if tensor.device.type == 'cuda'})


def training_state_dict(model, optimizer, *, base_id, step, data_state, scaler=None):
    """Capture all persistent model tensors at a cleared optimizer-step boundary.

    Storage dtypes, parameter ties, actual optimizer groups and CPU/device RNG
    are retained. Unlike finetune_state_dict this contains the full model, not
    only adapters; capture memory and serialized size scale with that state.
    This does not perform filesystem writes or distributed synchronization.
    Use DistributedCheckpoint for an all-rank commit or ZeRO-owned optimizer.
    Nonpersistent buffers are reconstructed by the caller's same architecture.
    """
    if not isinstance(base_id, str) or not base_id or type(step) is not int or step < 0:
        raise ValueError('supply exact initial-model identity and a nonnegative completed step')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('clear gradients before full snapshot; partial accumulation is not saved')
    layout = _optimizer_layout(model, optimizer)
    schema = _tensor_schema(model)
    model_state = model.state_dict()
    state = {'format': 'ruda-training', 'version': 1, 'base_id': base_id, 'step': step,
             'model': _cpu_tree(model_state), 'model_schema': schema,
             'model_metadata': _cpu_tree(getattr(model_state, '_metadata', None)),
             'optimizer': _cpu_tree(optimizer.state_dict()),
             'optimizer_type': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
             'optimizer_layout': layout,
             'data_state': _cpu_tree(data_state),
             'scaler': None if scaler is None else _cpu_tree(scaler.state_dict()),
             'scaler_type': None if scaler is None else type(scaler).__module__ + '.' + type(scaler).__qualname__,
             'cpu_rng': torch.get_rng_state().clone(),
             'cuda_rng': {index: torch.cuda.get_rng_state(index).cpu() for index in _cuda_devices(model)}}
    if any(tensor.device.type == 'ruda' for tensor in list(model.parameters()) + list(model.buffers())):
        from . import get_rng_state
        state['ruda_rng'] = get_rng_state()
    return state


def validate_training_state_dict(model, optimizer, state, *, base_id, scaler=None):
    """Check model, actual ties, optimizer/scaler and device-RNG contracts before load."""
    if not isinstance(state, dict) or state.get('format') != 'ruda-training' or state.get('version') != 1 or state.get('base_id') != base_id:
        raise ValueError('full training checkpoint format or initial-model identity differs')
    if type(state['step']) is not int or state['step'] < 0 or state['model_schema'] != _tensor_schema(model):
        raise ValueError('full training model geometry, dtype, frozen flags or tied aliases differ')
    name = type(optimizer).__module__ + '.' + type(optimizer).__qualname__
    if state['optimizer_type'] != name or state['optimizer_layout'] != _optimizer_layout(model, optimizer):
        raise ValueError('full training optimizer class or actual parameter layout differs')
    scaler_name = None if scaler is None else type(scaler).__module__ + '.' + type(scaler).__qualname__
    if state['scaler_type'] != scaler_name or (state['scaler'] is None) != (scaler is None):
        raise ValueError('full training scaler contract differs')
    current = model.state_dict()
    if current.keys() != state['model'].keys():
        raise ValueError('full training persistent state keys differ')
    for name, tensor in current.items():
        saved = state['model'][name]
        if isinstance(tensor, torch.Tensor):
            if not isinstance(saved, torch.Tensor) or saved.shape != tensor.shape or saved.dtype != tensor.dtype:
                raise ValueError(f'full training persistent shape/dtype differs: {name}')
        elif type(saved) is not type(tensor):
            raise ValueError(f'full training extra-state type differs: {name}')
    if sorted(state['cuda_rng']) != _cuda_devices(model):
        raise ValueError('full training CUDA RNG device set differs')
    has_ruda = any(tensor.device.type == 'ruda' for tensor in list(model.parameters()) + list(model.buffers()))
    if ('ruda_rng' in state) != has_ruda:
        raise ValueError('full training native RUDA RNG presence differs')


def load_training_state_dict(model, optimizer, state, *, base_id, scaler=None):
    """Restore actual full parameters/buffers, optimizer, scaler, RNG and data cursor.

    The caller recreates matching architecture and actual parameter aliases before
    restoration. Model loading copies into those existing tensors; it does not
    replace registered Parameters or change their optimizer identities.
    """
    validate_training_state_dict(model, optimizer, state, base_id=base_id, scaler=scaler)
    model_state = OrderedDict(state['model'])
    if state.get('model_metadata') is not None:
        model_state._metadata = copy.deepcopy(state['model_metadata'])
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if scaler is not None:
        scaler.load_state_dict(copy.deepcopy(state['scaler']))
    torch.set_rng_state(state['cpu_rng'])
    for index, rng in state['cuda_rng'].items():
        torch.cuda.set_rng_state(rng, index)
    if 'ruda_rng' in state:
        from . import set_rng_state
        set_rng_state(state['ruda_rng'])
    optimizer.zero_grad(set_to_none=True)
    return state['step'], copy.deepcopy(state['data_state'])
