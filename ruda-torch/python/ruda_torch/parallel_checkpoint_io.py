"""Stream full safetensors checkpoints into explicit meta TP layouts."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn
from .finetuning import NF4Linear,_replace
from .parallel_training import ColumnParallelLinear,RowParallelLinear,VocabParallelEmbedding
from .parallel_adapters import ColumnParallelNF4Linear,RowParallelNF4Linear,_geometry
from .distributed_checkpoint import model_shard_layout


def load_tensor_parallel_safetensors(model,directory,group,*,device,dtype=torch.bfloat16,
                                    nf4_targets=(),block_size=64,tile_rows=128,
                                    parameter_dtypes=None,buffer_dtypes=None):
    """Materialize an already partitioned META model from ordinary full weights.

    Dense TP tensors read only their local slice. Explicit NF4 targets quantize
    one full CPU matrix at a time before slicing packed codes/scales, preserving
    global quantization blocks without ever uploading a full dense base. Inject
    LoRA AFTER loading. Names, architecture, sharding and device are caller-owned.
    This loader does not decode bitsandbytes/AWQ/GPTQ checkpoint formats.
    A materialization failure requires retrying with a fresh meta model.
    """
    from safetensors import safe_open
    directory=Path(directory).resolve()
    device=torch.device(device)
    floats=(torch.float32,torch.float16,torch.bfloat16)
    if dtype not in floats or device.type!=group.device_type:
        raise ValueError('select a floating dtype and the process-group execution device')
    if isinstance(nf4_targets,str) or len(set(nf4_targets))!=len(nf4_targets):
        raise ValueError('NF4 targets must be unique, explicit projection paths')
    if type(block_size) is not int or block_size<=0 or block_size%2 or type(tile_rows) is not int or tile_rows<=0:
        raise ValueError('NF4 requires positive even blocks and positive row tiles')
    modules=dict(model.named_modules(remove_duplicate=False))
    for module in modules.values():
        if isinstance(module,(ColumnParallelLinear,RowParallelLinear,VocabParallelEmbedding)) and (
                module.group.rank!=group.rank or module.group.world_size!=group.world_size or module.group.group is not group.group):
            raise ValueError('projection shards and loader must use the same process group')
    selected={name:modules[name] for name in nf4_targets}
    if any(not name or not isinstance(module,(ColumnParallelLinear,RowParallelLinear)) for name,module in selected.items()):
        raise ValueError('NF4 targets must be complete dense tensor-parallel projections')
    selected_ids={id(module) for module in selected.values()}
    selected={name:module for name,module in modules.items() if id(module) in selected_ids}
    parameters=list(model.named_parameters(remove_duplicate=False))
    buffers=list(model.named_buffers(remove_duplicate=False))
    if any(parameter.device.type!='meta' for _,parameter in parameters):
        raise ValueError('construct and tensor-parallelize the base on meta before streaming load')
    layout=model_shard_layout(model)
    parameter_dtypes={} if parameter_dtypes is None else dict(parameter_dtypes)
    buffer_dtypes={} if buffer_dtypes is None else dict(buffer_dtypes)
    aliases={}
    for name,tensor in parameters+buffers:aliases.setdefault(id(tensor),[]).append(name)
    for overrides,tensors in ((parameter_dtypes,parameters),(buffer_dtypes,buffers)):
        known=dict(tensors)
        if any(name not in known or value not in floats or not known[name].is_floating_point() for name,value in overrides.items()):
            raise ValueError('dtype overrides must select existing floating tensors')
        tied={}
        for name,tensor in tensors:
            if name in overrides:
                if id(tensor) in tied and tied[id(tensor)]!=overrides[name]:
                    raise ValueError('tied tensors have conflicting dtype overrides')
                tied[id(tensor)]=overrides[name]
        for name,tensor in tensors:
            if id(tensor) in tied:overrides[name]=tied[id(tensor)]
    for name,module in selected.items():
        _geometry(module.out_features,module.in_features,block_size,1 if isinstance(module,RowParallelLinear) else 0,group.world_size)
        weight=name+'.weight'
        if weight in parameter_dtypes:raise ValueError('exclude dtype-preserved weights from NF4 targets')
        consumers={path+'.weight' for path,other in selected.items() if other is module}
        if set(aliases[id(module.weight)])!=consumers:
            raise ValueError('NF4 conversion cannot break shared parameter consumers')
    index_path=directory/'model.safetensors.index.json'
    if index_path.is_file():
        index=json.loads(index_path.read_text(encoding='utf-8'))['weight_map']
        if not isinstance(index,dict) or not all(isinstance(key,str) and isinstance(value,str) for key,value in index.items()):
            raise ValueError('invalid safetensors weight map')
    else:
        with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as source:
            index={name:'model.safetensors' for name in source.keys()}
    files={}
    for name,relative in index.items():
        path=(directory/relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('checkpoint shards must be files inside the checkpoint directory')
        files[name]=path
    def key_for(name,tensor):
        return name if name in files else next((key for key in aliases[id(tensor)] if key in files),None)
    persistent=set(model.state_dict())
    for name,tensor in parameters+buffers:
        key=key_for(name,tensor)
        if key is None:
            if name in persistent or tensor.device.type=='meta':
                raise ValueError(f'checkpoint tensor or initialized nonpersistent buffer missing: {name}')
            continue
        spec=layout.get(name,{})
        if spec.get('transform') or (spec.get('axis') is None and spec and not spec.get('replicated')):
            raise ValueError('this loader expects dense TP storage, not FSDP or prepacked transforms')
        expected=tuple(spec.get('shape',tensor.shape))
        with safe_open(str(files[key]),framework='pt',device='cpu') as source:
            if tuple(source.get_slice(key).get_shape())!=expected:
                raise ValueError(f'full checkpoint shape differs: {name}')
    group.validate_training_options(([(name,tuple(tensor.shape),str(tensor.dtype)) for name,tensor in parameters+buffers],
                                    layout,list(selected),dtype,block_size,tile_rows,parameter_dtypes,buffer_dtypes,index))
    def read(name,tensor,*,full=False):
        key=key_for(name,tensor)
        if key is None:return tensor.detach().cpu()
        with safe_open(str(files[key]),framework='pt',device='cpu') as source:
            axis=layout.get(name,{}).get('axis')
            if full or axis is None:value=source.get_tensor(key)
            else:
                selection=[slice(None)]*tensor.ndim
                selection[axis]=slice(group.rank*tensor.shape[axis],(group.rank+1)*tensor.shape[axis])
                value=source.get_slice(key)[tuple(selection)]
        if value.is_floating_point()!=tensor.is_floating_point():
            raise ValueError(f'checkpoint tensor kind differs: {name}')
        return value
    replacements={}
    for name,base in selected.items():
        if id(base) not in replacements:
            dense=nn.Linear(base.in_features,base.out_features,bias=base.bias is not None,device='meta')
            dense.weight=nn.Parameter(read(name+'.weight',base.weight,full=True).contiguous(),requires_grad=False)
            if base.bias is not None:
                dense.bias=nn.Parameter(read(name+'.bias',base.bias,full=True).to(parameter_dtypes.get(name+'.bias',dtype)),requires_grad=False)
            packed=NF4Linear.from_linear(dense,block_size=block_size,tile_rows=tile_rows)
            del dense
            if isinstance(base,ColumnParallelLinear):
                replacement=ColumnParallelNF4Linear(packed,group,gather_output=base.gather_output)
            else:replacement=RowParallelNF4Linear(packed,group,input_is_parallel=base.input_is_parallel)
            del packed
            replacements[id(base)]=replacement.to(device=device,dtype=dtype).train(base.training)
        _replace(model,[(name,replacements[id(base)])])
    loaded={}
    for name,original in parameters+buffers:
        if any(name in (target+'.weight',target+'.bias') for target in selected):continue
        parent_name,_,leaf=name.rpartition('.')
        parent=model.get_submodule(parent_name) if parent_name else model
        if id(original) not in loaded:
            storage_dtype=parameter_dtypes.get(name,dtype) if isinstance(original,nn.Parameter) else buffer_dtypes.get(name,original.dtype)
            value=read(name,original).to(device=device,dtype=storage_dtype if original.is_floating_point() else original.dtype)
            if isinstance(original,nn.Parameter):
                value=nn.Parameter(value,requires_grad=original.requires_grad)
                if getattr(original,'_ruda_tp_sharded',False):value._ruda_tp_sharded=True
            loaded[id(original)]=value
        setattr(parent,leaf,loaded[id(original)])
    return model
