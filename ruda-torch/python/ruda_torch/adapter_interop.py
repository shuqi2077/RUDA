"""PEFT safetensors interchange for dense/NF4 LoRA and rank-stabilized LoRA."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
import torch
from .finetuning import LoRALinear, NF4Linear, _replace,adapter_state_dict


def _pattern(patterns,name,default):
    for pattern,value in patterns.items():
        if re.match(r'(.*\.)?'+pattern+r'$',name):
            return value
    return default


def load_peft_adapter(model,directory,*,module_map=None,prefix='base_model.model.',adapter_dtype=torch.float32):
    """Install explicit checkpoint A/B matrices without invoking model-family heuristics.

    module_map maps checkpoint module paths (after prefix removal) to exact model
    paths. Adapter target paths come from the actual safetensors, not a guessed
    target_modules mapping. Base-model identity remains the caller's selection.
    """
    from safetensors.torch import load_file
    from .parallel_adapters import _ParallelLoRA,_ParallelNF4,local_adapter_matrix,lora_from_parallel_base
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    parallel_types=(ColumnParallelLinear,RowParallelLinear,_ParallelNF4)
    directory=Path(directory)
    config=json.loads((directory/'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('peft_type')!='LORA' or config.get('bias','none')!='none':
        raise ValueError('this adapter loader requires LoRA with bias=none')
    unsupported=('use_dora','lora_bias','modules_to_save','layer_replication','target_parameters',
                 'trainable_token_indices','alora_invocation_tokens','arrow_config','kasa_config')
    if any(config.get(key) for key in unsupported) or config.get('fan_in_fan_out',False):
        raise ValueError('checkpoint requests adapter/base transformations outside linear LoRA semantics')
    tensors=load_file(str(directory/'adapter_model.safetensors'),device='cpu')
    pairs={}
    for key,value in tensors.items():
        if not key.startswith(prefix):
            raise ValueError(f'adapter key lacks the explicit prefix: {key}')
        match=re.fullmatch(r'(.+)\.lora_([AB])(?:\.[^.]+)?\.weight',key[len(prefix):])
        if match is None:
            raise ValueError(f'unsupported adapter tensor: {key}')
        source,side=match.groups()
        entry=pairs.setdefault(source,{})
        if side in entry:
            raise ValueError('multiple named adapters are not one adapter checkpoint')
        entry[side]=value
    if not pairs:
        raise ValueError('adapter checkpoint has no linear A/B matrices')
    from .sharded_adapter_interop import has_sharded_units,load_sharded_peft_adapter
    if has_sharded_units(model):return load_sharded_peft_adapter(model,pairs,config,module_map=module_map)
    replacements,aliases={},{}
    plans={}
    modules=dict(model.named_modules(remove_duplicate=False))
    for source,pair in pairs.items():
        path=source if module_map is None else module_map[source]
        base=modules[path]
        if set(pair)!={'A','B'} or not isinstance(base,(torch.nn.Linear,NF4Linear,LoRALinear,_ParallelLoRA)+parallel_types):
            raise ValueError('adapter target or A/B matrices are missing')
        rank=_pattern(config.get('rank_pattern',{}),source,config['r'])
        alpha=_pattern(config.get('alpha_pattern',{}),source,config['lora_alpha'])
        dropout=config.get('lora_dropout',0.)
        rslora=config.get('use_rslora',False)
        if type(rank) is not int or rank<=0 or type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<=0:
            raise ValueError('adapter rank/alpha must be positive and finite')
        if type(dropout) not in (int,float) or not 0<=dropout<=1 or type(rslora) is not bool:
            raise ValueError('invalid adapter dropout/RSLoRA policy')
        if adapter_dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError('adapter dtype must be FP32/FP16/BF16')
        if any(not tensor.is_floating_point() or not torch.isfinite(tensor).all() for tensor in pair.values()):
            raise ValueError('adapter matrices must contain finite floating values')
        if pair['A'].shape!=(rank,base.in_features) or pair['B'].shape!=(base.out_features,rank):
            raise ValueError(f'adapter dimensions differ: {source}')
        if isinstance(base,(LoRALinear,_ParallelLoRA)):
            expected=(rank,float(alpha),float(dropout),rslora)
            if (base.rank,base.alpha,base.dropout,base.use_rslora)!=expected:
                raise ValueError('existing adapter configuration differs')
        identity=id(base)
        if identity in plans:
            old_base,old_pair,old_options=plans[identity]
            if not all(torch.equal(old_pair[side],pair[side]) for side in ('A','B')):
                raise ValueError('shared target paths have different adapter matrices')
            if old_options!=(rank,alpha,dropout,rslora):
                raise ValueError('shared adapter paths have different configurations')
        plans[identity]=(base,pair,(rank,alpha,dropout,rslora))
    flags=[(parameter,parameter.requires_grad,parameter.grad) for parameter in model.parameters()]
    try:
        for identity,(base,pair,(rank,alpha,dropout,rslora)) in plans.items():
            factory=lora_from_parallel_base if isinstance(base,parallel_types) else LoRALinear
            layer=base if isinstance(base,(LoRALinear,_ParallelLoRA)) else factory(base,rank=rank,alpha=alpha,
                adapter_dtype=adapter_dtype,dropout=dropout,use_rslora=rslora)
            aliases[identity]=(pair,layer)
            replacements[identity]=layer
    except Exception:
        for parameter,trainable,gradient in flags:
            parameter.requires_grad_(trainable)
            parameter.grad=gradient
        raise
    # Freeze the selected base model only after the entire checkpoint preflight.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad=None
    with torch.no_grad():
        for pair,layer in aliases.values():
            a=local_adapter_matrix(layer,'lora_A',pair['A']) if isinstance(layer,_ParallelLoRA) else pair['A']
            b=local_adapter_matrix(layer,'lora_B',pair['B']) if isinstance(layer,_ParallelLoRA) else pair['B']
            layer.lora_A.copy_(a.to(layer.lora_A))
            layer.lora_B.copy_(b.to(layer.lora_B))
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)
    _replace(model,[(name,replacements[id(module)]) for name,module in modules.items()
                    if name and id(module) in replacements])
    return model


def save_peft_adapter(model,directory,*,base_model_name_or_path,prefix='base_model.model.',
                      task_type=None,module_map=None,replica_group=None):
    """Write full, PEFT-compatible matrices into a new directory.

    All DP/TP ranks participate in reconstruction; only the coordinator root
    writes files. FSDP storage requires its actual replica_group coordinator.
    An export spans one model/stage group, not separate pipeline namespaces.
    """
    from safetensors.torch import save_file
    from .parallel_adapters import _ParallelLoRA
    from .sharded_adapter_interop import adapter_layers,has_sharded_units
    layers={name:entry[0] for name,entry in adapter_layers(model).items()}
    if has_sharded_units(model) and replica_group is None:
        raise ValueError('FSDP PEFT export requires the actual training coordinator')
    if replica_group is not None:replica_group.validate_model(model)
    if not layers:
        raise ValueError('model contains no LoRA layers')
    first=next(iter(layers.values()))
    if any((layer.dropout,layer.use_rslora)!=(first.dropout,first.use_rslora) for layer in layers.values()):
        raise ValueError('PEFT config requires one dropout and RSLoRA policy')
    if not isinstance(base_model_name_or_path,str) or not base_model_name_or_path:
        raise ValueError('identify the actual base checkpoint')
    rank_pattern,alpha_pattern,tensors={},{},{}
    groups={id(layer.group):layer.group for layer in layers.values() if isinstance(layer,_ParallelLoRA)}
    if len(groups)>1:raise ValueError('PEFT export requires a single tensor-parallel group')
    group=next(iter(groups.values()),None)
    if replica_group is not None:group=replica_group
    names=[]
    paths={}
    for name,layer in layers.items():
        target=name if module_map is None else module_map[name]
        if target in names:
            raise ValueError('adapter export path mapping is not one-to-one')
        names.append(target)
        paths[name]=target
        if layer.rank!=first.rank:rank_pattern[target]=layer.rank
        if layer.alpha!=first.alpha:alpha_pattern[target]=layer.alpha
    config={'peft_type':'LORA','base_model_name_or_path':base_model_name_or_path,'task_type':task_type,
            'r':first.rank,'lora_alpha':first.alpha,'lora_dropout':first.dropout,'use_rslora':first.use_rslora,
            'bias':'none','fan_in_fan_out':False,'inference_mode':True,'target_modules':names,
            'rank_pattern':rank_pattern,'alpha_pattern':alpha_pattern}
    if group is not None:
        group.validate_training_options((str(Path(directory).resolve()),prefix,paths,config))
    state=adapter_state_dict(model)
    for name,entry in state['layers'].items():
        target=paths[name]
        tensors[prefix+target+'.lora_A.weight']=entry['lora_A'].contiguous().clone()
        tensors[prefix+target+'.lora_B.weight']=entry['lora_B'].contiguous().clone()
    directory=Path(directory)
    error=None
    if group is None or group.rank==0:
        try:
            directory.mkdir(parents=True,exist_ok=False)
            save_file(tensors,str(directory/'adapter_model.safetensors'),metadata={'format':'pt'})
            (directory/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        except Exception as failure:
            if group is None:raise
            error=f'{type(failure).__name__}: {failure}'
    if group is not None:
        for failure in group.gather_metadata(error):
            if failure:raise RuntimeError(f'PEFT export failed: {failure}')
    return directory
