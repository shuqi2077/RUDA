"""Read bitsandbytes NF4 checkpoints without loading bitsandbytes or requantizing."""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch
from torch import nn
from .finetuning import NF4Linear, _selected, _replace, _FLOATS


def bnb_nf4_linear(state,weight_key,*,device='ruda:0',dtype=torch.float16,bias=None,tile_rows=128):
    """Decode quantization metadata, preserving packed NF4 codes exactly.

    Nested/double-quantized absolute maxima are expanded to FP32 scales on CPU.
    Weight codes are never dequantized into a full dense matrix or requantized.
    Float quant_storage is interpreted as raw packed bytes, not floating values.
    """
    tag=weight_key+'.quant_state.bitsandbytes__nf4'
    metadata=state[tag]
    if metadata.device.type!='cpu' or metadata.dtype!=torch.uint8:
        raise ValueError('packed bitsandbytes quantization metadata must be CPU bytes')
    config=json.loads(bytes(metadata.reshape(-1).tolist()).decode('utf-8'))
    if config.get('quant_type')!='nf4' or config.get('dtype') not in ('float16','bfloat16','float32'):
        raise ValueError('checkpoint is not a supported NF4 floating-weight format')
    shape=tuple(config['shape'])
    if len(shape)!=2 or any(type(size) is not int or size<1 for size in shape):
        raise ValueError('NF4 linear weight must have [out,in] shape')
    block_size=config['blocksize']
    if type(block_size) is not int or block_size<1 or block_size%2:
        raise ValueError('NF4 blocksize must be positive and even')
    packed=state[weight_key]
    if packed.device.type!='cpu' or not packed.is_contiguous():
        raise ValueError('packed checkpoint weight must be contiguous on CPU')
    packed=packed.view(torch.uint8).reshape(-1)
    bytes_required=(math.prod(shape)+1)//2
    if packed.numel()<bytes_required:
        raise ValueError('packed NF4 weight is truncated')
    packed=packed[:bytes_required].clone()
    scales=state[weight_key+'.absmax'].reshape(-1)
    if scales.device.type!='cpu':
        raise ValueError('NF4 scale metadata must be on CPU')
    nested_key=weight_key+'.nested_absmax'
    if nested_key in state:
        if scales.dtype!=torch.uint8:
            raise ValueError('double-quantized scale codes must be uint8')
        nested_block=config['nested_blocksize']
        nested=state[nested_key].reshape(-1).float()
        mapping=state[weight_key+'.nested_quant_map'].reshape(-1).float()
        if type(nested_block) is not int or nested_block<1 or mapping.numel()!=256:
            raise ValueError('invalid nested block quantization metadata')
        index=torch.arange(scales.numel(),device='cpu')//nested_block
        if nested.numel()<(scales.numel()+nested_block-1)//nested_block:
            raise ValueError('nested absolute maxima are truncated')
        scales=mapping[scales.to(torch.int64)]*nested[index]+float(config['nested_offset'])
    else:
        if scales.dtype!=torch.float32:
            raise ValueError('uncompressed absolute maxima must be FP32')
        scales=scales.clone()
    expected_scales=(math.prod(shape)+block_size-1)//block_size
    if scales.numel()!=expected_scales or not torch.isfinite(scales).all() or (scales<0).any():
        raise ValueError('NF4 absolute maxima do not match the weight geometry')
    table=state[weight_key+'.quant_map'].reshape(-1)
    if table.device.type!='cpu' or table.dtype!=torch.float32 or table.shape!=(16,) or not torch.isfinite(table).all():
        raise ValueError('NF4 codebook must contain sixteen finite FP32 values')
    if dtype not in _FLOATS:
        raise ValueError('NF4 compute storage must be FP32/FP16/BF16')
    layer=NF4Linear(shape[1],shape[0],packed,scales,block_size=block_size,tile_rows=tile_rows,
                    bias=None if bias is None else bias.to(dtype=dtype))
    layer.codebook=table.clone()
    return layer.to(device=device)


def _safetensor_index(directory):
    from safetensors import safe_open
    directory=Path(directory).resolve()
    index=directory/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text(encoding='utf-8'))['weight_map']
        result={}
        for key,name in mapping.items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError('checkpoint shard lies outside the supplied directory or is missing')
            result[key]=path
        return result
    files=sorted(directory.glob('*.safetensors'))
    result={}
    for path in files:
        with safe_open(str(path),framework='pt',device='cpu') as source:
            for key in source.keys():
                if key in result:raise ValueError(f'duplicate checkpoint tensor: {key}')
                result[key]=path
    if not result:raise ValueError('no local safetensors checkpoint was found')
    return result


def load_bnb_nf4_safetensors(model,directory,*,target_modules,device='ruda:0',dtype=torch.float16,tile_rows=128):
    """Stream local HF-style NF4 shards into a caller-constructed model.

    Other parameters/buffers retain source values and explicit storage dtype.
    No model family, network download, remote-code permission or key renaming is
    inferred. Instantiate a large model on meta with CPU-generated fixed buffers.
    """
    from safetensors import safe_open
    files=_safetensor_index(directory)
    selected=_selected(model,target_modules,(nn.Linear,))
    parameters=list(model.named_parameters(remove_duplicate=False))
    buffers=list(model.named_buffers(remove_duplicate=False))
    targets={name for name,layer in selected}
    quantized_ids={id(layer.weight) for name,layer in selected}
    for name,parameter in parameters:
        if id(parameter) in quantized_ids and name.rsplit('.',1)[0] not in targets:
            raise ValueError('a quantized linear weight is tied to an unselected consumer')
    def read(key,original=None):
        if key not in files:
            if original is not None and not isinstance(original,nn.Parameter) and original.device.type!='meta':
                return original.detach().cpu()
            raise ValueError(f'checkpoint tensor is missing: {key}')
        with safe_open(str(files[key]),framework='pt',device='cpu') as source:
            return source.get_tensor(key)
    replacements={}
    for name,layer in selected:
        if id(layer) not in replacements:
            key=name+'.weight'
            keys=[key]+[item for item in files if item.startswith(key+'.')]
            state={item:read(item) for item in keys}
            bias=None if layer.bias is None else read(name+'.bias')
            replacement=bnb_nf4_linear(state,key,device=device,dtype=dtype,bias=bias,tile_rows=tile_rows)
            if (replacement.in_features,replacement.out_features)!=(layer.in_features,layer.out_features):
                raise ValueError('quantized checkpoint/model linear geometry differs')
            replacement.train(layer.training)
            replacements[id(layer)]=replacement
        _replace(model,[(name,replacements[id(layer)])])
    loaded={}
    for name,original in parameters+buffers:
        if any(name==target+'.weight' or name==target+'.bias' for target in targets):continue
        if id(original) not in loaded:
            value=read(name,original)
            if value.shape!=original.shape:raise ValueError(f'checkpoint tensor shape differs: {name}')
            value=value.to(device=device,dtype=dtype if value.is_floating_point() and isinstance(original,nn.Parameter) else value.dtype)
            loaded[id(original)]=nn.Parameter(value,requires_grad=False) if isinstance(original,nn.Parameter) else value
        parent,_,leaf=name.rpartition('.')
        setattr(model.get_submodule(parent),leaf,loaded[id(original)])
    return model
