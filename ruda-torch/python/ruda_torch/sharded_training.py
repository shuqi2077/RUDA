"""Element-sharded data parallelism with RUDA computation and NCCL tensor storage."""
from __future__ import annotations

from collections import OrderedDict
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.autograd.function import once_differentiable


class _GatherUnit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, elements, *shards):
        ctx.group, ctx.elements = group, elements
        ctx.specs = tuple((s.numel(), s.dtype, s.device, s.requires_grad) for s in shards)
        ctx.set_materialize_grads(False)
        outputs = tuple(group.all_gather(shard)[:count] for shard, count in zip(shards, elements, strict=True))
        ctx.mark_non_differentiable(*(output for output, shard in zip(outputs, shards) if not shard.requires_grad))
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        # One autograd node per unit fixes collective order even when ranks use
        # different parameters inside that unit (for example expert routing).
        flags = ctx.group.gather_metadata(tuple(g is not None for g in gradients))
        results = []
        for index, (gradient, count, spec) in enumerate(zip(gradients, ctx.elements, ctx.specs, strict=True)):
            size, dtype, device, trainable = spec
            if not trainable or not any(rank[index] for rank in flags):
                results.append(None)
                continue
            padded = torch.zeros(size*ctx.group.world_size, dtype=torch.float32, device=device)
            if gradient is not None:
                padded[:count].copy_(gradient.reshape(-1).float())
            results.append(ctx.group.reduce_scatter(padded).to(dtype))
        return None, None, *results


def _initial_shard(value, group, device, root):
    size = math.ceil(value.numel()/group.world_size)
    shard = torch.empty(size, dtype=value.dtype, device=device)
    flat = value.detach().reshape(-1)
    # Broadcasting one slice at a time also permits CPU checkpoint loading
    # without materializing a full parameter on any GPU.
    for owner in range(group.world_size):
        work = torch.zeros_like(shard)
        begin, end = owner*size, min((owner+1)*size, value.numel())
        if group.rank == root and end > begin:
            work[:end-begin].copy_(flat[begin:end].to(device))
        group.broadcast_(work, root)
        if owner == group.rank:
            shard.copy_(work)
    return shard


class FullyShardedModule(nn.Module):
    """ZeRO-3/FSDP unit: local parameter, gradient and optimizer-state slices.

    Wrap transformer blocks separately to bound gathered parameters per unit.
    Non-reentrant activation recomputation releases saved full weights after
    forward and gathers them again for backward. Every rank must execute the
    same collective-bearing unit order, including checkpoint recomputation.
    A tied parameter belongs to one unit; use the root as a unit for cross-block
    ties, or keep tied modules together. No local-shard Muon approximation.
    """
    def __init__(self, module, group, *, device=None, root=0, recompute=True, shard_buffers=None):
        super().__init__()
        device = torch.device(device or ('ruda:0' if group.device_type == 'ruda' else 'cpu'))
        if device.type != group.device_type or type(recompute) is not bool:
            raise ValueError('select the group device and a boolean recompute policy')
        named = list(module.named_parameters(remove_duplicate=False))
        schema, aliases, unique = [], {}, []
        error = None
        if type(root) is not int or not 0 <= root < group.world_size:
            error = 'invalid FSDP root'
        for name, parameter in named:
            identity = id(parameter)
            if identity not in aliases:
                aliases[identity] = len(unique)
                unique.append(parameter)
            schema.append((name, aliases[identity], tuple(parameter.shape), str(parameter.dtype), parameter.requires_grad))
            if parameter.device.type == 'meta' or not parameter.numel() or parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                error = 'load floating nonempty parameters on CPU or the selected device before sharding'
        buffers = list(module.named_buffers(remove_duplicate=False))
        if shard_buffers is None:
            from .finetuning import NF4Linear
            shard_buffers={prefix+field for path,layer in module.named_modules() if isinstance(layer,NF4Linear)
                           for prefix in (path+'.' if path else '',) for field in ('packed','scales')}
        else:
            shard_buffers=set(shard_buffers)
        if not shard_buffers<={name for name,value in buffers}:
            error='sharded buffer names must refer to existing, read-only model buffers'
        buffer_aliases={}
        buffer_schema=[]
        for name,value in buffers:
            identity=id(value)
            if identity in buffer_aliases and buffer_aliases[identity]!=(name in shard_buffers):
                error='tied buffers must have the same sharding policy'
            buffer_aliases[identity]=name in shard_buffers
            buffer_schema.append((name,tuple(value.shape),str(value.dtype),name in shard_buffers))
        requests = group.gather_metadata((schema, buffer_schema, root, recompute, error))
        for other_schema, other_buffers, other_root, other_recompute, failure in requests:
            if failure:
                raise ValueError(failure)
            if (other_schema, other_buffers, other_root, other_recompute) != (schema, buffer_schema, root, recompute):
                raise ValueError('FSDP structure, aliases or policy differs across ranks')
        self.group, self.recompute = group, recompute
        self.shards = nn.ParameterList()
        self._schema = tuple(schema)
        self._element_counts = tuple(p.numel() for p in unique)
        self._shard_names = []
        self._buffer_names = []
        self._buffer_shards = {}
        for index, parameter in enumerate(unique):
            local = _initial_shard(parameter, group, device, root)
            key = f'frozen_{index}'
            if parameter.requires_grad:
                key = f'shards.{len(self.shards)}'
                shard=nn.Parameter(local)
                shard._ruda_full_shape=tuple(parameter.shape)
                shard._ruda_tp_sharded=getattr(parameter,'_ruda_tp_sharded',False)
                self.shards.append(shard)
            else:
                self.register_buffer(key, local)
            self._shard_names.append(key)
        buffer_aliases = {}
        for name, value in buffers:
            if id(value) not in buffer_aliases:
                key = f'buffer_{len(buffer_aliases)}'
                if name in shard_buffers:
                    transferred=_initial_shard(value,group,device,root)
                    self._buffer_shards[key]=tuple(value.shape)
                else:
                    transferred = value.detach().to(device).contiguous().clone()
                    group.broadcast_(transferred, root)
                self.register_buffer(key, transferred)
                buffer_aliases[id(value)] = key
            self._buffer_names.append((name, buffer_aliases[id(value)]))
        # The functional template retains structure, not a duplicate full model.
        metas = {id(p): nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device='meta'),
                                     requires_grad=p.requires_grad) for p in unique}
        for name, parameter in named:
            parent, _, leaf = name.rpartition('.')
            setattr(module.get_submodule(parent), leaf, metas[id(parameter)])
        buffer_metas={id(value):torch.empty(value.shape,dtype=value.dtype,device='meta') for name,value in buffers}
        for name,value in buffers:
            parent,_,leaf=name.rpartition('.')
            setattr(module.get_submodule(parent),leaf,buffer_metas[id(value)])
        object.__setattr__(self, '_template', module)
        self.train(module.training)

    def _local_shards(self):
        return tuple(self.get_parameter(name) if name.startswith('shards.') else self.get_buffer(name)
                     for name in self._shard_names)

    def _invoke(self, shards, args, kwargs):
        values = {}
        gathered = _GatherUnit.apply(self.group, self._element_counts, *shards) if shards else ()
        for name, index, shape, dtype, trainable in self._schema:
            values[name] = gathered[index].view(shape)
        materialized={}
        for name, key in self._buffer_names:
            if key not in materialized:
                value=self.get_buffer(key)
                if key in self._buffer_shards:
                    shape=self._buffer_shards[key]
                    value=self.group.all_gather(value)[:math.prod(shape)].view(shape)
                materialized[key]=value
            values[name] = materialized[key]
        return torch.func.functional_call(self._template, values, args, kwargs, tie_weights=True, strict=True)

    def forward(self, *args, **kwargs):
        shards = self._local_shards()
        if self.training and torch.is_grad_enabled() and self.recompute:
            def invoke(*local):
                return self._invoke(local, args, kwargs)
            return checkpoint(invoke, *shards, use_reentrant=False, preserve_rng_state=True)
        return self._invoke(shards, args, kwargs)

    def train(self, mode=True):
        super().train(mode)
        template = getattr(self, '_template', None)
        if template is not None:
            template.train(mode)
        return self

    def extra_repr(self):
        return f'rank={self.group.rank}, world_size={self.group.world_size}, recompute={self.recompute}'

    def get_extra_state(self):
        extras={}
        for name,module in self._template.named_modules():
            if type(module).get_extra_state is not nn.Module.get_extra_state:
                extras[name]=module.get_extra_state()
        return {'version':1,'schema':self._schema,'buffer_shards':self._buffer_shards,'module_extra':extras}

    def set_extra_state(self,state):
        if state['version']!=1 or state['schema']!=self._schema or state['buffer_shards']!=self._buffer_shards:
            raise ValueError('FSDP template configuration differs')
        for name,extra in state['module_extra'].items():
            self._template.get_submodule(name).set_extra_state(extra)

    def full_state_dict(self):
        """Collective, ordinary CPU weights; every rank participates."""
        result = OrderedDict()
        with torch.no_grad():
            gathered = {}
            for name, index, shape, dtype, trainable in self._schema:
                if index not in gathered:
                    gathered[index] = self.group.all_gather(self._local_shards()[index])[:math.prod(shape)].view(shape).cpu()
                result[name] = gathered[index]
            for name, key in self._buffer_names:
                value=self.get_buffer(key)
                if key in self._buffer_shards:
                    shape=self._buffer_shards[key]
                    value=self.group.all_gather(value)[:math.prod(shape)].view(shape)
                result[name] = value.detach().cpu().clone()
        return result

    def load_full_state_dict(self, state, *, strict=True):
        """Load ordinary weights into slices for the current world size."""
        expected = {entry[0] for entry in self._schema} | {name for name, key in self._buffer_names}
        if strict and set(state) != expected:
            raise ValueError('full FSDP checkpoint keys differ')
        with torch.no_grad():
            loaded = {}
            for name, index, shape, dtype, trainable in self._schema:
                if name not in state:
                    continue
                value = state[name]
                if tuple(value.shape) != shape or str(value.dtype) != dtype:
                    raise ValueError(f'full FSDP tensor metadata differs: {name}')
                if index in loaded:
                    if not torch.equal(loaded[index], value):
                        raise ValueError('tied checkpoint weights differ')
                    continue
                loaded[index] = value
                target = self._local_shards()[index]
                begin = self.group.rank*target.numel()
                end = min(begin+target.numel(), value.numel())
                target.zero_()
                if end > begin:
                    target[:end-begin].copy_(value.reshape(-1)[begin:end].to(target.device))
            for name, key in self._buffer_names:
                if name in state:
                    target = self.get_buffer(key)
                    expected_shape=self._buffer_shards.get(key,tuple(target.shape))
                    if tuple(state[name].shape) != expected_shape or state[name].dtype != target.dtype:
                        raise ValueError(f'buffer metadata differs: {name}')
                    if key in self._buffer_shards:
                        begin=target.numel()*self.group.rank
                        end=min(begin+target.numel(),state[name].numel())
                        target.zero_()
                        if end>begin:target[:end-begin].copy_(state[name].reshape(-1)[begin:end].to(target))
                    else:
                        target.copy_(state[name])


def fully_shard(model, group, *, unit_paths=None, device=None, root=0, recompute=True, shard_buffers=None):
    """Shard the root or explicit nonoverlapping units without guessing model families."""
    if unit_paths is None:
        return FullyShardedModule(model, group, device=device, root=root, recompute=recompute,shard_buffers=shard_buffers)
    paths = tuple(unit_paths)
    modules = dict(model.named_modules(remove_duplicate=False))
    error = None
    if not paths or len(set(paths)) != len(paths) or any(not p or p not in modules for p in paths):
        error = 'supply distinct exact FSDP unit paths'
    elif any(a != b and b.startswith(a+'.') for a in paths for b in paths):
        error = 'FSDP units must not overlap'
    owners = {}
    selected = {id(modules[p]): p for p in paths if p in modules}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        candidates = [p for p in paths if name.startswith(p+'.')]
        owner = candidates[0] if candidates else None
        if id(parameter) in owners and owners[id(parameter)] != owner:
            error = 'a tied parameter crosses FSDP units; put its consumers in one unit'
        owners[id(parameter)] = owner
    requests = group.gather_metadata((paths, error))
    for other_paths, failure in requests:
        if failure:
            raise ValueError(failure)
        if other_paths != paths:
            raise ValueError('FSDP unit paths differ across ranks')
    replacements = {identity: FullyShardedModule(modules[path], group, device=device, root=root,
                                                recompute=recompute,shard_buffers=(shard_buffers.get(path)
                                                if isinstance(shard_buffers,dict) else shard_buffers)) for identity, path in selected.items()}
    for path, module in modules.items():
        if path and id(module) in replacements:
            parent, _, leaf = path.rpartition('.')
            setattr(model.get_submodule(parent), leaf, replacements[id(module)])
    return model


class Zero2Optimizer:
    """Replicated parameters, reduce-scattered gradients and local optimizer state.

    optimizer_factory receives groups of local 1-D Parameter slices. Use an
    elementwise optimizer (AdamW/SGD); matrix-dependent optimizers need a separate
    complete-matrix update and are not equivalent on these slices.
    """
    def __init__(self, parameter_groups, group, optimizer_factory):
        entries = list(parameter_groups)
        entries = entries if entries and isinstance(entries[0], dict) else [{'params': entries}]
        self.group, self.entries, groups = group, [], []
        seen, schema = set(), []
        for options in entries:
            local = {k: v for k, v in options.items() if k != 'params'}
            local['params'] = []
            for parameter in options['params']:
                if id(parameter) in seen or not parameter.numel() or not parameter.requires_grad or not parameter.is_contiguous() or parameter.device.type != group.device_type:
                    raise ValueError('provide unique contiguous trainable group-device parameters')
                seen.add(id(parameter))
                size = math.ceil(parameter.numel()/group.world_size)
                flat = parameter.detach().reshape(-1)
                shard = torch.zeros(size, dtype=parameter.dtype, device=parameter.device)
                begin, end = size*group.rank, min(size*(group.rank+1), flat.numel())
                if end > begin:
                    shard[:end-begin].copy_(flat[begin:end])
                shard = nn.Parameter(shard)
                local['params'].append(shard)
                self.entries.append((parameter, shard))
                schema.append((tuple(parameter.shape), str(parameter.dtype), size))
            groups.append(local)
        group.validate_training_options(schema)
        if not self.entries:
            raise ValueError('ZeRO-2 requires trainable parameters')
        self.optimizer = optimizer_factory(groups)
        if 'Muon' in type(self.optimizer).__name__:
            raise ValueError('complete-matrix Muon cannot update element-sharded parameters')
        self.param_groups = self.optimizer.param_groups
        self.last_step_skipped = False
        self._weight = None

    def synchronize_gradients(self, *, local_weight, normalized=False, missing='zero'):
        flags = [p.grad is not None for p, shard in self.entries]
        requests = self.group.gather_metadata((local_weight, normalized, missing, flags))
        total, active = 0, [False]*len(flags)
        for weight, norm, policy, present in requests:
            if type(weight) is not int or weight < 0 or norm != normalized or policy != missing or len(present) != len(flags):
                raise ValueError('ZeRO-2 rank reduction contracts differ')
            if missing not in ('zero', 'error') or (weight and missing == 'error' and not all(present)):
                raise ValueError('ZeRO-2 missing gradient policy failed')
            total += weight
            if weight:
                active = [a or b for a, b in zip(active, present)]
        if not total:
            raise ValueError('global effective weight must be positive')
        with torch.no_grad():
            for (parameter, shard), present in zip(self.entries, active):
                if not present:
                    shard.grad = None
                    parameter.grad = None
                    continue
                padded = torch.zeros(shard.numel()*self.group.world_size, device=shard.device, dtype=torch.float32)
                if local_weight and parameter.grad is not None:
                    padded[:parameter.numel()].copy_(parameter.grad.reshape(-1).float())
                reduced = self.group.reduce_scatter(padded)
                if not normalized:
                    reduced.div_(total)
                shard.grad = reduced.to(shard.dtype)
                parameter.grad = None
        self._weight = total
        return total

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        if self._weight is None:
            raise RuntimeError('reduce-scatter accumulated gradients before the ZeRO-2 update')
        if closure is not None:
            raise ValueError('ZeRO-2 closures cannot implicitly rerun distributed backward')
        # A nonfinite shard must skip the SAME update on every rank.
        bad = torch.zeros(1, dtype=torch.float32, device=self.entries[0][1].device)
        for parameter, shard in self.entries:
            if shard.grad is not None:
                bad.add_((~torch.isfinite(shard.grad.float()/kwargs.get('loss_scale',1.))).any().float())
        self.group.sum_(bad)
        self.last_step_skipped = bool(bad.item())
        if not self.last_step_skipped:
            if getattr(self.optimizer,'max_grad_norm',None) is not None and hasattr(self.optimizer,'_distributed_grad_norm'):
                from .sharded_optim import distributed_grad_norm
                self.optimizer._distributed_grad_norm=distributed_grad_norm([shard for parameter,shard in self.entries],
                    self.group,loss_scale=kwargs.get('loss_scale',1.))
            self.optimizer.step(**kwargs)
            self.last_step_skipped = bool(getattr(self.optimizer, 'last_step_skipped', False))
            statuses = self.group.gather_metadata(self.last_step_skipped)
            if any(status != statuses[0] for status in statuses):
                raise RuntimeError('shard optimizers disagreed on update; restore a consistent checkpoint')
            if not self.last_step_skipped:
                for parameter, shard in self.entries:
                    parameter.copy_(self.group.all_gather(shard)[:parameter.numel()].view_as(parameter))
        self._weight = None

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter, shard in self.entries:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def state_dict(self):
        return {'version': 1, 'rank': self.group.rank, 'world_size': self.group.world_size,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if (state['version'], state['rank'], state['world_size']) != (1, self.group.rank, self.group.world_size):
            raise ValueError('ZeRO-2 checkpoint topology differs; consolidate and reshard explicitly')
        self.optimizer.load_state_dict(state['optimizer'])
        for parameter, shard in self.entries:
            size, begin = shard.numel(), shard.numel()*self.group.rank
            end = min(begin+size, parameter.numel())
            with torch.no_grad():
                shard.zero_()
                if end > begin:
                    shard[:end-begin].copy_(parameter.detach().reshape(-1)[begin:end])
        self._weight = None


class ShardedReplicaGroup:
    """SFT/data-parallel coordinator for a model containing FSDP units.

    Units reduce their gradients during backward; unsharded parameters use the
    ordinary replica reduction. Execute the same unit/microbatch order on every
    rank. This coordinator never all-reduces an already reduced shard twice.
    """
    def __init__(self, model, group):
        self.transport = group
        self.rank, self.world_size, self.device_type = group.rank, group.world_size, group.device_type
        self.group, self.cuda_index = group.group, group.cuda_index
        self.model = model
        self._parameters = list(model.named_parameters())
        self.sharded_ids = {id(p) for unit in model.modules() if isinstance(unit, FullyShardedModule) for p in unit.shards}
        if not self.sharded_ids and not any(isinstance(unit, FullyShardedModule) for unit in model.modules()):
            raise ValueError('fully_shard the model before attaching its training coordinator')
        self._specs = [(name, id(p), tuple(p.shape), p.dtype, p.device, p.requires_grad) for name, p in self._parameters]
        group.validate_training_options([(name, tuple(p.shape), str(p.dtype), p.requires_grad, id(p) in self.sharded_ids)
                                         for name, p in self._parameters])

    def gather_metadata(self, value):
        return self.transport.gather_metadata(value)

    def validate_training_options(self, value):
        return self.transport.validate_training_options(value)

    def total_weight(self, value):
        return self.transport.total_weight(value)

    def sum_(self, tensor):
        return self.transport.sum_(tensor)

    def validate_model(self, model):
        current = [(name, id(p), tuple(p.shape), p.dtype, p.device, p.requires_grad) for name, p in model.named_parameters()]
        if model is not self.model or current != self._specs:
            raise ValueError('sharded training model changed after coordinator creation')

    def validate_microbatches(self, batches):
        signatures = [tuple((name, tuple(value.shape), str(value.dtype)) for name, value in sorted(batch.items()))
                      for batch in batches]
        self.transport.validate_training_options(signatures)
        if not batches:
            raise ValueError('every FSDP rank must execute the collective-bearing microbatches')

    def synchronize_gradients(self, *, local_weight, normalized=False, missing='zero'):
        count = self.total_weight(local_weight)
        original = self.transport._parameters
        schema = self.transport._schema
        unsharded = [(name, p) for name, p in self._parameters if id(p) not in self.sharded_ids]
        try:
            self.transport._parameters, self.transport._schema = unsharded, []
            self.transport.synchronize_gradients(local_weight=local_weight, normalized=normalized, missing=missing)
        finally:
            self.transport._parameters, self.transport._schema = original, schema
        if not normalized:
            with torch.no_grad():
                for name, parameter in self._parameters:
                    if id(parameter) in self.sharded_ids and parameter.grad is not None:
                        parameter.grad.div_(count)
        return count

    def prepare_optimizer_step(self, optimizer, *, loss_scale=1.):
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        if 'Muon' in type(optimizer).__name__ and not getattr(optimizer,'complete_matrix_shards',False):
            raise ValueError('element-sharded matrices require a complete-matrix Muon algorithm')
        if getattr(optimizer,'complete_matrix_shards',False):
            return
        if not hasattr(optimizer,'last_step_skipped'):
            return
        bad = torch.zeros(1, dtype=torch.float32, device=parameters[0].device)
        for parameter in parameters:
            if parameter.grad is not None:
                bad.add_((~torch.isfinite(parameter.grad.float()/loss_scale)).any().float())
        self.sum_(bad)
        if bad.item():
            # Existing RUDA optimizers/scalers perform their usual overflow skip
            # when every rank sees a nonfinite gradient. No new skip policy.
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.fill_(float('inf'))
        elif getattr(optimizer,'max_grad_norm',None) is not None and hasattr(optimizer,'_distributed_grad_norm'):
            from .sharded_optim import distributed_grad_norm
            optimizer._distributed_grad_norm=distributed_grad_norm(parameters,self.transport,loss_scale=loss_scale,
                replicated_ids={id(p) for p in parameters if id(p) not in self.sharded_ids})
