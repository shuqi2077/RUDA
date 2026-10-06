"""Portable LoRA weights over FSDP storage without gathering frozen base tensors."""
from __future__ import annotations

import math
import torch
from .sharded_training import FullyShardedModule
from .finetuning import LoRALinear
from .parallel_adapters import _ParallelLoRA,adapter_base_kind,local_adapter_matrix


def has_sharded_units(model):
    return any(isinstance(module,FullyShardedModule) for module in model.modules())


def adapter_layers(model,*,remove_duplicate=True):
    """Actual adapter templates and their live storage owner, by logical path."""
    result={}
    for path,module in model.named_modules(remove_duplicate=remove_duplicate):
        if isinstance(module,(LoRALinear,_ParallelLoRA)):
            result[path]=(module,None,path)
        if isinstance(module,FullyShardedModule):
            for local,layer in module._template.named_modules(remove_duplicate=remove_duplicate):
                if isinstance(layer,(LoRALinear,_ParallelLoRA)):
                    name='.'.join(part for part in (path,local) if part)
                    result[name]=(layer,module,local)
    return result


def _storage(layer,unit,path,field):
    if unit is None:
        parameter=getattr(layer,field)
        return parameter,tuple(parameter.shape)
    logical=path+'.'+field if path else field
    entry=next((item for item in unit._schema if item[0]==logical),None)
    if entry is None:raise ValueError(f'adapter is missing from its FSDP parameter schema: {logical}')
    name,index,shape,dtype,trainable=entry
    return unit._local_shards()[index],tuple(shape)


def _metadata(layer,version):
    metadata={'rank':layer.rank,'alpha':layer.alpha,'in_features':layer.in_features,
              'out_features':layer.out_features,'base':adapter_base_kind(layer)}
    if version==2:metadata.update(dropout=layer.dropout,use_rslora=layer.use_rslora)
    elif layer.dropout or layer.use_rslora:raise ValueError('legacy checkpoint lacks dropout/RSLoRA semantics')
    return metadata


def _axis(layer,field):
    if not isinstance(layer,_ParallelLoRA):return None
    return 1 if field=='lora_A' and layer.axis==1 else 0 if field=='lora_B' and layer.axis==0 else None


@torch.no_grad()
def sharded_adapter_state_dict(model):
    """Collective adapter-only export, including TP reconstruction when needed.

    Each FSDP/TP participant enters the same layer/field order. Only trainable
    A/B matrices are materialized; frozen dense or packed NF4 bases stay sharded.
    The ordinary ruda-lora record is independent of both data and tensor topology.
    """
    layers=adapter_layers(model)
    if not layers:raise ValueError('model has no LoRA adapters')
    version=2 if any(layer.dropout or layer.use_rslora for layer,unit,path in layers.values()) else 1
    values,cache={},{}
    for name,(layer,unit,path) in layers.items():
        entry=_metadata(layer,version)
        for field in ('lora_A','lora_B'):
            parameter,shape=_storage(layer,unit,path,field)
            axis=_axis(layer,field)
            key=(id(parameter),None if unit is None else id(unit.group.group),
                 None if not isinstance(layer,_ParallelLoRA) else id(layer.group.group),axis)
            if key not in cache:
                value=parameter.detach()
                if unit is not None:value=unit.group.all_gather(value)[:math.prod(shape)].view(shape)
                if axis is not None:value=layer.group.all_gather(value,axis=axis)
                cache[key]=value.cpu().clone()
            entry[field]=cache[key]
        values[name]=entry
    return {'format':'ruda-lora','version':version,'layers':values}


@torch.no_grad()
def load_sharded_adapter_state_dict(model,state):
    """Validate full logical adapters, then split TP axes and data elements.

    Existing parameter objects, ties, storage dtype and trainable flags remain
    unchanged. Optimizer masters are not silently rewritten by a weights loader.
    """
    if not isinstance(state,dict) or set(state)!={'format','version','layers'} or state['format']!='ruda-lora' or state['version'] not in (1,2):
        raise ValueError('unsupported adapter checkpoint')
    layers=adapter_layers(model)
    if not layers or not isinstance(state['layers'],dict) or set(layers)!=set(state['layers']):
        raise ValueError('adapter target names mismatch')
    copies={}
    for name,(layer,unit,path) in layers.items():
        metadata=_metadata(layer,state['version'])
        entry=state['layers'][name]
        if not isinstance(entry,dict) or set(entry)!=set(metadata)|{'lora_A','lora_B'} or any(entry[key]!=value for key,value in metadata.items()):
            raise ValueError(f'adapter configuration mismatch: {name}')
        for field in ('lora_A','lora_B'):
            source=entry[field]
            shape=(layer.rank,layer.in_features) if field=='lora_A' else (layer.out_features,layer.rank)
            if not isinstance(source,torch.Tensor) or tuple(source.shape)!=shape or source.dtype not in (torch.float32,torch.float16,torch.bfloat16):
                raise ValueError(f'adapter tensor mismatch: {name}.{field}')
            if source.device.type!='cpu':source=source.detach().cpu()
            value=local_adapter_matrix(layer,field,source) if isinstance(layer,_ParallelLoRA) else source
            parameter,local_shape=_storage(layer,unit,path,field)
            if tuple(value.shape)!=local_shape:raise ValueError('FSDP/TP adapter shape differs from template storage')
            if unit is not None:
                size=parameter.numel()
                begin=unit.group.rank*size
                end=min(begin+size,value.numel())
                local=value.new_zeros(parameter.shape)
                if end>begin:local[:end-begin].copy_(value.reshape(-1)[begin:end])
                value=local
            if id(parameter) in copies and not torch.equal(copies[id(parameter)][1],value):
                raise ValueError('shared adapter storage has different checkpoint values')
            copies[id(parameter)]=(parameter,value)
    for parameter,value in copies.values():parameter.copy_(value)


def load_sharded_peft_adapter(model,pairs,config,*,module_map=None):
    """Import PEFT matrices into adapters injected before FSDP construction."""
    from .adapter_interop import _pattern
    layers=adapter_layers(model)
    aliases=adapter_layers(model,remove_duplicate=False)
    canonical={(id(layer),id(unit)):name for name,(layer,unit,path) in layers.items()}
    entries={}
    for source,pair in pairs.items():
        target=source if module_map is None else module_map[source]
        if target not in aliases or set(pair)!={'A','B'}:raise ValueError('inject matching LoRA targets before sharding their base model')
        layer,unit,path=aliases[target]
        target=canonical[(id(layer),id(unit))]
        expected=(_pattern(config.get('rank_pattern',{}),source,config['r']),
                  float(_pattern(config.get('alpha_pattern',{}),source,config['lora_alpha'])),
                  float(config.get('lora_dropout',0.)),config.get('use_rslora',False))
        if (layer.rank,layer.alpha,layer.dropout,layer.use_rslora)!=expected:
            raise ValueError(f'existing sharded adapter configuration differs: {target}')
        if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in pair.values()):
            raise ValueError('PEFT adapters must contain finite floating matrices')
        entry=dict(_metadata(layer,2),lora_A=pair['A'],lora_B=pair['B'])
        if target in entries and any(not torch.equal(entries[target][field],entry[field]) for field in ('lora_A','lora_B')):
            raise ValueError('shared PEFT target paths have different adapter matrices')
        entries[target]=entry
    load_sharded_adapter_state_dict(model,{'format':'ruda-lora','version':2,'layers':entries})
    return model
