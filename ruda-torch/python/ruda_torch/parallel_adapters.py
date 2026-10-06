"""LoRA/NF4 tensor-parallel projections without dequantizing or requantizing bases."""
from __future__ import annotations

import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .finetuning import LoRALinear,NF4Linear
from .parallel_training import (ColumnParallelLinear,RowParallelLinear,copy_to_tensor_parallel,
    reduce_from_tensor_parallel,scatter_to_tensor_parallel,gather_from_tensor_parallel)


def _geometry(outputs,width,block,axis,world):
    if axis not in (0,1) or type(world) is not int or world<=0:
        raise ValueError('select a valid matrix axis and positive shard count')
    size=(outputs,width)[axis]
    if size%world:raise ValueError('quantized matrix shard dimension must divide the group size')
    local=(outputs//world,width) if axis==0 else (outputs,width//world)
    smaller=block if world==1 else math.gcd(block,math.prod(local)) if axis==0 else math.gcd(block,width,local[1])
    if smaller%2:
        raise ValueError('NF4 TP shards must align to even quantization sub-blocks; no requantization is performed')
    return local,smaller


def _scale_interval(scales,start,count,old_block,new_block):
    first=start//old_block
    last=(start+count+old_block-1)//old_block
    expanded=scales[first:last].repeat_interleave(old_block//new_block)
    offset=(start%old_block)//new_block
    return expanded[offset:offset+(count+new_block-1)//new_block]


@torch.no_grad()
def shard_nf4_linear(layer,group,*,axis):
    """Partition packed codes and copy the ORIGINAL scales into smaller blocks.

    Column shards are contiguous flat ranges; row shards extract byte-aligned
    input columns. Metadata expansion is bounded by the layer's row tile size.
    Odd-aligned layouts outside this packed format are rejected, not silently
    decoded into dense weights or quantized a second time.
    """
    if not isinstance(layer,NF4Linear):raise TypeError('expected an NF4Linear')
    local,block=_geometry(layer.out_features,layer.in_features,layer.block_size,axis,group.world_size)
    rows,width=local
    if group.world_size==1:
        packed,scales=layer.packed.clone(),layer.scales.clone()
    elif axis==0:
        start=group.rank*rows*width
        count=rows*width
        packed=layer.packed[start//2:(start+count)//2].contiguous().clone()
        chunks=[]
        chunk=((layer.tile_rows*width+block-1)//block)*block
        for begin in range(0,count,chunk):
            chunks.append(_scale_interval(layer.scales,start+begin,min(chunk,count-begin),layer.block_size,block).clone())
        scales=torch.cat(chunks).contiguous()
    else:
        start=group.rank*width
        packed=layer.packed.view(rows,layer.in_features//2)[:,start//2:(start+width)//2].contiguous().reshape(-1).clone()
        chunks=[]
        for row in range(0,rows,layer.tile_rows):
            count=min(layer.tile_rows,rows-row)
            expanded=_scale_interval(layer.scales,row*layer.in_features,count*layer.in_features,layer.block_size,block)
            selected=expanded.view(count,layer.in_features//block)[:,start//block:(start+width)//block]
            chunks.append(selected.contiguous().reshape(-1))
        scales=torch.cat(chunks).contiguous()
    bias=layer.bias
    if bias is not None:
        bias=bias[group.rank*rows:(group.rank+1)*rows].clone() if axis==0 else None
    result=NF4Linear(width,rows,packed,scales,block_size=block,tile_rows=layer.tile_rows,bias=bias)
    result.codebook=layer.codebook.clone()
    return result.train(layer.training)


class _ParallelNF4(nn.Module):
    axis=None
    def _setup(self,layer,group):
        if not isinstance(layer,NF4Linear):raise TypeError('expected a frozen NF4 base')
        self.group=group
        self.in_features,self.out_features=layer.in_features,layer.out_features
        self.original_block_size=layer.block_size
        self.local=shard_nf4_linear(layer,group,axis=self.axis)

    def get_extra_state(self):
        return {'version':1,'format':'ruda-nf4-tp','axis':self.axis,'in_features':self.in_features,
                'out_features':self.out_features,'block_size':self.original_block_size}

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('NF4 TP checkpoint geometry differs')


class ColumnParallelNF4Linear(_ParallelNF4):
    """Frozen packed output-row shard; input derivatives sum across TP ranks."""
    axis=0
    def __init__(self,layer,group,*,gather_output=True):
        super().__init__()
        self._setup(layer,group)
        self.gather_output=bool(gather_output)
        self.local_out_features=self.local.out_features

    def forward(self,value):
        output=self.local(copy_to_tensor_parallel(value,self.group))
        return gather_from_tensor_parallel(output,self.group) if self.gather_output else output


class RowParallelNF4Linear(_ParallelNF4):
    """Frozen packed input-column shard; replicated bias is added after reduction."""
    axis=1
    def __init__(self,layer,group,*,input_is_parallel=False):
        super().__init__()
        self._setup(layer,group)
        self.input_is_parallel=bool(input_is_parallel)
        self.local_in_features=self.local.in_features
        self.register_buffer('bias',None if layer.bias is None else layer.bias.clone())

    def forward(self,value):
        local=value if self.input_is_parallel else scatter_to_tensor_parallel(value,self.group)
        output=reduce_from_tensor_parallel(self.local(local),self.group)
        return output if self.bias is None else output+self.bias.to(output.dtype)


def _dropout(value,p,group,*,sharded=False):
    if not p:return value
    if p==1:return torch.where(torch.zeros_like(value,dtype=torch.bool),value,torch.zeros_like(value))
    shape=list(value.shape)
    if sharded:shape[-1]*=group.world_size
    random=torch.empty(shape,device=value.device,dtype=torch.float32)
    if not random.numel():return value
    if group.rank==0:
        if value.device.type=='ruda':
            from .random import uniform_
            uniform_(random)
        else:random.uniform_()
    group.broadcast_(random)
    if sharded:random=random.chunk(group.world_size,dim=-1)[group.rank]
    # A single logical dropout mask, not different replicated A inputs on each rank.
    return value*(random>=p).to(value.dtype)/(1-p)


class _ParallelLoRA(nn.Module):
    axis=None
    def _setup(self,layer,group):
        if not isinstance(layer,LoRALinear):raise TypeError('expected a LoRALinear')
        self.group=group
        self.rank,self.alpha=layer.rank,layer.alpha
        self.dropout,self.use_rslora=layer.dropout,layer.use_rslora
        self.in_features,self.out_features=layer.in_features,layer.out_features
        a,b=layer.lora_A.detach(),layer.lora_B.detach()
        if self.axis==0:b=b.chunk(group.world_size,dim=0)[group.rank]
        else:a=a.chunk(group.world_size,dim=1)[group.rank]
        self.lora_A=nn.Parameter(a.contiguous().clone(),requires_grad=layer.lora_A.requires_grad)
        self.lora_B=nn.Parameter(b.contiguous().clone(),requires_grad=layer.lora_B.requires_grad)
        self.lora_A._ruda_tp_sharded=self.axis==1
        self.lora_B._ruda_tp_sharded=self.axis==0
        self.train(layer.training)

    @property
    def scaling(self):return self.alpha/(math.sqrt(self.rank) if self.use_rslora else self.rank)

    def get_extra_state(self):
        state={'version':1,'rank':self.rank,'alpha':self.alpha}
        if self.dropout or self.use_rslora:state.update(version=2,dropout=self.dropout,use_rslora=self.use_rslora)
        return state

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('parallel LoRA configuration differs')


class ColumnParallelLoRALinear(_ParallelLoRA):
    """Local base/B output rows, replicated A with SUM parameter derivatives."""
    axis=0
    def __init__(self,layer,group,*,gather_output=True):
        super().__init__()
        if not isinstance(layer,LoRALinear):raise TypeError('expected a LoRALinear')
        constructor=ColumnParallelNF4Linear if isinstance(layer.base,NF4Linear) else ColumnParallelLinear
        self.base=constructor(layer.base,group,gather_output=False)
        self.gather_output=bool(gather_output)
        self._setup(layer,group)
        self.local_out_features=self.out_features//group.world_size

    def forward(self,value):
        result=self.base(value)
        adapted=copy_to_tensor_parallel(value.to(self.lora_A.dtype),self.group)
        if self.training:adapted=_dropout(adapted,self.dropout,self.group)
        a=copy_to_tensor_parallel(self.lora_A,self.group)
        update=F.linear(F.linear(adapted,a),self.lora_B)
        output=result+(update*self.scaling).to(result.dtype)
        return gather_from_tensor_parallel(output,self.group) if self.gather_output else output


class RowParallelLoRALinear(_ParallelLoRA):
    """Local base/A input columns; reduce rank activations before replicated B."""
    axis=1
    def __init__(self,layer,group,*,input_is_parallel=False):
        super().__init__()
        if not isinstance(layer,LoRALinear):raise TypeError('expected a LoRALinear')
        constructor=RowParallelNF4Linear if isinstance(layer.base,NF4Linear) else RowParallelLinear
        self.base=constructor(layer.base,group,input_is_parallel=True)
        self.input_is_parallel=bool(input_is_parallel)
        self._setup(layer,group)
        self.local_in_features=self.in_features//group.world_size

    def forward(self,value):
        local=value if self.input_is_parallel else scatter_to_tensor_parallel(value,self.group)
        result=self.base(local)
        adapted=local.to(self.lora_A.dtype)
        if self.training:adapted=_dropout(adapted,self.dropout,self.group,sharded=True)
        hidden=reduce_from_tensor_parallel(F.linear(adapted,self.lora_A),self.group)
        update=F.linear(hidden,self.lora_B)
        return result+(update*self.scaling).to(result.dtype)


def lora_from_parallel_base(base,*,rank=16,alpha=16.,adapter_dtype=torch.float32,dropout=0.,use_rslora=False):
    """Collectively inject adapters into an already partitioned dense/NF4 base.

    Only adapter matrices are initialized in full; existing packed/dense base
    storage is retained. Replicated A initialization comes from TP rank zero.
    """
    column=isinstance(base,(ColumnParallelLinear,ColumnParallelNF4Linear))
    row=isinstance(base,(RowParallelLinear,RowParallelNF4Linear))
    if not (column or row):raise TypeError('expected a tensor-parallel linear base')
    if type(rank) is not int or rank<=0 or type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<=0:
        raise ValueError('adapter rank/alpha must be positive and finite')
    if adapter_dtype not in (torch.float32,torch.float16,torch.bfloat16):
        raise ValueError('adapter_dtype must be FP32, FP16 or BF16')
    if type(dropout) not in (int,float) or not 0<=dropout<=1 or type(use_rslora) is not bool:
        raise ValueError('invalid adapter dropout/RSLoRA policy')
    group=base.group
    group.validate_training_options((type(base).__name__,base.in_features,base.out_features,
                                     rank,alpha,str(adapter_dtype),dropout,use_rslora))
    device=base.local.packed.device if isinstance(base,_ParallelNF4) else base.weight.device
    a=torch.empty((rank,base.in_features),device='cpu',dtype=adapter_dtype)
    nn.init.kaiming_uniform_(a,a=math.sqrt(5))
    a=a.to(device)
    group.broadcast_(a)
    b=torch.zeros((base.out_features,rank),device='cpu',dtype=adapter_dtype).to(device)
    if column:b=b.chunk(group.world_size,dim=0)[group.rank]
    else:a=a.chunk(group.world_size,dim=1)[group.rank]
    cls=ColumnParallelLoRALinear if column else RowParallelLoRALinear
    result=cls.__new__(cls)
    nn.Module.__init__(result)
    result.base=copy.copy(base)
    result.group=group
    result.in_features,result.out_features=base.in_features,base.out_features
    result.rank,result.alpha=rank,float(alpha)
    result.dropout,result.use_rslora=float(dropout),use_rslora
    result.lora_A=nn.Parameter(a.contiguous().clone())
    result.lora_B=nn.Parameter(b.contiguous().clone())
    result.lora_A._ruda_tp_sharded=row
    result.lora_B._ruda_tp_sharded=column
    if column:
        result.gather_output=base.gather_output
        result.local_out_features=base.local_out_features
        result.base.gather_output=False
    else:
        result.input_is_parallel=base.input_is_parallel
        result.local_in_features=base.local_in_features
        result.base.input_is_parallel=True
    for parameter in base.parameters():
        parameter.requires_grad_(False)
        parameter.grad=None
    return result.train(base.training)


@torch.no_grad()
def merge_parallel_lora(layer):
    """Merge a dense adapter locally without gathering a full base weight."""
    if not isinstance(layer,_ParallelLoRA) or layer.training or isinstance(layer.base,_ParallelNF4):
        raise ValueError('merge requires an eval-mode dense TP adapter')
    base=layer.base
    result=copy.copy(base)
    result._parameters=base._parameters.copy()
    result.weight=nn.Parameter((base.weight.float()+(layer.lora_B.float()@layer.lora_A.float())*layer.scaling).to(base.weight.dtype),requires_grad=False)
    result.weight._ruda_tp_sharded=True
    if base.bias is not None:
        result.bias=nn.Parameter(base.bias.detach().clone(),requires_grad=False)
        result.bias._ruda_tp_sharded=layer.axis==0
    if layer.axis==0:result.gather_output=layer.gather_output
    else:result.input_is_parallel=layer.input_is_parallel
    return result.eval()


def parallel_linear(source,group,*,axis,**options):
    """Choose a projection from the ACTUAL dense/NF4/LoRA layer type."""
    if axis not in (0,1):raise ValueError('select input or output matrix shards')
    if isinstance(source,LoRALinear):
        constructor=ColumnParallelLoRALinear if axis==0 else RowParallelLoRALinear
    elif isinstance(source,NF4Linear):
        constructor=ColumnParallelNF4Linear if axis==0 else RowParallelNF4Linear
    else:constructor=ColumnParallelLinear if axis==0 else RowParallelLinear
    return constructor(source,group,**options)


def parameter_shards(source,replacement,axis):
    """Original Parameter identity, destination owner/name, and logical shard axis."""
    entries=[]
    if isinstance(source,LoRALinear):
        entries.extend(((source.lora_A,replacement,'lora_A',1 if axis==1 else None),
                        (source.lora_B,replacement,'lora_B',0 if axis==0 else None)))
        source,replacement=source.base,replacement.base
    if isinstance(source,nn.Linear):
        entries.append((source.weight,replacement,'weight',axis))
        if source.bias is not None:entries.append((source.bias,replacement,'bias',0 if axis==0 else None))
    return entries


def adapter_base_kind(layer):
    return 'nf4' if isinstance(layer.base,(NF4Linear,_ParallelNF4)) else 'dense'


def full_adapter_matrix(layer,name):
    value=getattr(layer,name)
    axis=1 if name=='lora_A' and layer.axis==1 else 0 if name=='lora_B' and layer.axis==0 else None
    return value.detach().cpu().clone() if axis is None else layer.group.all_gather(value,axis=axis).cpu()


def local_adapter_matrix(layer,name,value):
    shape=(layer.rank,layer.in_features) if name=='lora_A' else (layer.out_features,layer.rank)
    if name not in ('lora_A','lora_B') or not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or not value.is_floating_point():
        raise ValueError(f'full adapter tensor differs: {name}')
    axis=1 if name=='lora_A' and layer.axis==1 else 0 if name=='lora_B' and layer.axis==0 else None
    return value if axis is None else value.chunk(layer.group.world_size,dim=axis)[layer.group.rank].contiguous()


def nf4_layout(module,prefix):
    """Typed logical layout; packed bytes are NOT ordinary axis-zero matrices."""
    shape=[module.out_features,module.in_features]
    common={'aliases':[],'shape':shape,'axis':module.axis,'block_size':module.original_block_size}
    result={prefix+'local.packed':dict(common,name=prefix+'packed',transform='nf4-packed'),
            prefix+'local.scales':dict(common,name=prefix+'scales',transform='nf4-scales'),
            prefix+'local._extra_state':dict(common,name=prefix+'_extra_state',transform='nf4-extra'),
            prefix+'_extra_state':dict(common,name=prefix+'_extra_state',transform='nf4-wrapper-extra'),
            prefix+'local.codebook':{'name':prefix+'codebook','aliases':[],'shape':[16],'replicated':True}}
    if module.axis==0 and module.local.bias is not None:
        result[prefix+'local.bias']={'name':prefix+'bias','aliases':[],'shape':[module.out_features],'axis':0}
    return result


def join_nf4_field(values,spec):
    """CPU checkpoint joining, preserving byte codes and bit-identical FP32 scales."""
    outputs,width=spec['shape']
    local,block=_geometry(outputs,width,spec['block_size'],spec['axis'],len(values))
    rows,columns=local
    kind=spec['transform']
    if kind in ('nf4-extra','nf4-wrapper-extra'):
        expected=({'version':1,'format':'ruda-nf4','out_features':rows,'in_features':columns,'block_size':block}
                  if kind=='nf4-extra' else
                  {'version':1,'format':'ruda-nf4-tp','axis':spec['axis'],'out_features':outputs,'in_features':width,'block_size':spec['block_size']})
        if any(value!=expected for value in values):raise ValueError('NF4 shard metadata differs')
        return {'version':1,'format':'ruda-nf4','out_features':outputs,'in_features':width,'block_size':spec['block_size']}
    dtype=torch.uint8 if kind=='nf4-packed' else torch.float32
    count=(rows*columns+1)//2 if kind=='nf4-packed' else (rows*columns+block-1)//block
    if any(not isinstance(value,torch.Tensor) or value.dtype!=dtype or value.shape!=(count,) for value in values):
        raise ValueError('NF4 checkpoint buffer shape/dtype differs')
    if len(values)==1:return values[0]
    if kind=='nf4-packed':
        if spec['axis']==0:return torch.cat(values)
        return torch.cat([v.reshape(rows,columns//2) for v in values],dim=1).reshape(-1)
    if kind!='nf4-scales':raise ValueError('unsupported NF4 checkpoint field')
    expanded=torch.cat(values) if spec['axis']==0 else torch.cat([v.reshape(rows,columns//block) for v in values],dim=1).reshape(-1)
    ratio=spec['block_size']//block
    original=expanded[::ratio].clone()
    for begin in range(0,original.numel(),4096):
        reference=original[begin:begin+4096].repeat_interleave(ratio)
        current=expanded[begin*ratio:min((begin+4096)*ratio,expanded.numel())]
        if not torch.equal(reference[:current.numel()].view(torch.int32),current.view(torch.int32)):
            raise ValueError('NF4 sub-block scales no longer represent the original quantization blocks')
    return original


def slice_nf4_field(value,spec,rank,world):
    outputs,width=spec['shape']
    local,block=_geometry(outputs,width,spec['block_size'],spec['axis'],world)
    rows,columns=local
    kind=spec['transform']
    if kind in ('nf4-extra','nf4-wrapper-extra'):
        expected={'version':1,'format':'ruda-nf4','out_features':outputs,'in_features':width,'block_size':spec['block_size']}
        if value!=expected:raise ValueError('full NF4 checkpoint metadata differs')
        if kind=='nf4-extra':return dict(expected,out_features=rows,in_features=columns,block_size=block)
        return dict(expected,format='ruda-nf4-tp',axis=spec['axis'])
    dtype=torch.uint8 if kind=='nf4-packed' else torch.float32
    count=(outputs*width+1)//2 if kind=='nf4-packed' else (outputs*width+spec['block_size']-1)//spec['block_size']
    if not isinstance(value,torch.Tensor) or value.dtype!=dtype or value.shape!=(count,):
        raise ValueError('full NF4 checkpoint buffer shape/dtype differs')
    if world==1:return value
    if kind=='nf4-packed':
        if spec['axis']==0:
            length=rows*columns//2
            return value[rank*length:(rank+1)*length].clone()
        return value.view(rows,width//2)[:,rank*columns//2:(rank+1)*columns//2].contiguous().reshape(-1)
    if kind!='nf4-scales':raise ValueError('unsupported NF4 checkpoint field')
    if spec['axis']==0:return _scale_interval(value,rank*rows*columns,rows*columns,spec['block_size'],block).clone()
    chunks=[]
    for row in range(0,rows,128):
        count=min(128,rows-row)
        expanded=_scale_interval(value,row*width,count*width,spec['block_size'],block)
        chunks.append(expanded.view(count,width//block)[:,rank*columns//block:(rank+1)*columns//block].contiguous().reshape(-1))
    return torch.cat(chunks)


def full_nf4_state(module):
    result={}
    for key,spec in nf4_layout(module,'').items():
        if spec.get('transform') in ('nf4-extra','nf4-wrapper-extra'):
            continue
        if spec.get('replicated'):
            result[spec['name']]=module.local.codebook.detach().cpu().clone()
        elif 'transform' in spec:
            value=module.local.packed if spec['transform']=='nf4-packed' else module.local.scales
            gathered=module.group.all_gather(value).cpu()
            pieces=list(gathered.chunk(module.group.world_size))
            result[spec['name']]=join_nf4_field(pieces,spec)
        else:result[spec['name']]=module.group.all_gather(module.local.bias).cpu()
    if module.axis==1 and module.bias is not None:result['bias']=module.bias.detach().cpu().clone()
    result['_extra_state']={'version':1,'format':'ruda-nf4','in_features':module.in_features,
                           'out_features':module.out_features,'block_size':module.original_block_size}
    return result


def local_nf4_state(module,state,prefix):
    result={}
    for key,spec in nf4_layout(module,prefix).items():
        logical=spec['name']
        if logical not in state:continue
        value=state[logical]
        if spec.get('transform')=='nf4-extra':
            expected={'version':1,'format':'ruda-nf4','in_features':module.in_features,
                      'out_features':module.out_features,'block_size':module.original_block_size}
            if value!=expected:raise ValueError('full NF4 checkpoint geometry differs')
        if 'transform' in spec:value=slice_nf4_field(value,spec,module.group.rank,module.group.world_size)
        elif not spec.get('replicated'):value=value.chunk(module.group.world_size,dim=spec['axis'])[module.group.rank].contiguous()
        result[key]=value
    if module.axis==1 and prefix+'bias' in state:result[prefix+'bias']=state[prefix+'bias']
    result[prefix+'_extra_state']=module.get_extra_state()
    return result
