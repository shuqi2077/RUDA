"""Online-softmax attention with tile recomputation, not a quadratic saved graph."""
from __future__ import annotations

import math
import torch
from torch.autograd.function import once_differentiable
from ._architecture_ops import positions, precision_context, work_dtype


def _rng(device, blocks, generator):
    if device.type=='ruda':
        from .random import Generator, default_generator
        source=default_generator if generator is None else generator
        if not isinstance(source,Generator):raise TypeError('RUDA dropout requires a RUDA Generator')
        with source.lock:
            if source.counter+blocks>=1<<64:raise OverflowError('RUDA random counter exhausted')
            result=Generator(source.seed_value)
            result.counter=source.counter
            source.counter+=blocks
        return result
    if device.type not in ('cpu','cuda'):
        raise ValueError('block dropout requires a CPU, CUDA or RUDA RNG')
    seed=torch.empty((),device=device,dtype=torch.int64).random_(generator=generator).item()
    return torch.Generator(device=device).manual_seed(seed)


def _restore_rng(device,state):
    if device.type=='ruda':
        from .random import Generator
        return Generator(0).set_state(state)
    return torch.Generator(device=device).set_state(state)


def _dropout(shape,device,dtype,p,generator):
    random=torch.empty(shape,device=device,dtype=torch.float32)
    if device.type=='ruda':
        from .random import uniform_
        uniform_(random,generator=generator)
    else:
        random.uniform_(generator=generator)
    return (random>=p).to(dtype)/(1-p)


def _scores(query,key,mask,causal,scale,row,col):
    scores=query@key.transpose(-1,-2)*scale
    if causal:
        allowed=positions(key.shape[-2],key.device,col).unsqueeze(0)<=positions(query.shape[-2],query.device,row).unsqueeze(-1)
        scores=scores.masked_fill(~allowed,float('-inf'))
    elif mask is not None:
        tile=mask[...,row:row+query.shape[-2],col:col+key.shape[-2]]
        scores=scores.masked_fill(~tile,float('-inf')) if tile.dtype==torch.bool else scores+tile.to(scores.dtype)
    return scores


def _exponential(scores,maximum):
    # Avoid -inf - -inf for fully masked rows. NaNs/+inf in unmasked input
    # remain NaNs, rather than being silently accepted as masked data.
    safe=torch.where(maximum==float('-inf'),torch.zeros_like(maximum),maximum)
    return (scores-safe).exp()


class _BlockAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,query,key,value,mask,causal,scale,p,qblock,kblock,generator):
        dtype=work_dtype(query)
        prefix=query.shape[:-2]
        rows,columns=query.shape[-2],key.shape[-2]
        original_mask=mask
        if mask is not None:mask=mask.expand(*prefix,rows,columns)
        out=torch.zeros((*prefix,rows,value.shape[-1]),device=query.device,dtype=dtype)
        maxima=torch.full((*prefix,rows,1),float('-inf'),device=query.device,dtype=dtype)
        sums=torch.zeros_like(maxima)
        ctx.rng_state=None
        if 0<p<1:
            heads=math.prod(prefix)
            blocks=sum((heads*min(qblock,rows-r)*min(kblock,columns-c)+3)//4
                       for r in range(0,rows,qblock) for c in range(0,columns,kblock))
            generator=_rng(query.device,blocks,generator)
            ctx.rng_state=generator.get_state()
        with precision_context(query):
            for row in range(0,rows,qblock):
                q=query[...,row:row+qblock,:].to(dtype)
                maximum=maxima[...,row:row+qblock,:]
                total=sums[...,row:row+qblock,:]
                result=out[...,row:row+qblock,:]
                for col in range(0,columns,kblock):
                    k=key[...,col:col+kblock,:].to(dtype)
                    v=value[...,col:col+kblock,:].to(dtype)
                    scores=_scores(q,k,mask,causal,scale,row,col)
                    next_max=torch.maximum(maximum,scores.amax(-1,keepdim=True))
                    old_safe=torch.where(maximum==float('-inf'),torch.zeros_like(maximum),maximum)
                    next_safe=torch.where(next_max==float('-inf'),torch.zeros_like(next_max),next_max)
                    correction=torch.where(maximum==float('-inf'),torch.zeros_like(maximum),(old_safe-next_safe).exp())
                    probability=_exponential(scores,next_max)
                    total.mul_(correction).add_(probability.sum(-1,keepdim=True))
                    if p:
                        probability=probability*_dropout(probability.shape,query.device,dtype,p,generator)
                    result.mul_(correction).add_(probability@v)
                    maximum.copy_(next_max)
                result.div_(torch.where(total==0,torch.ones_like(total),total))
        ctx.save_for_backward(query,key,value,original_mask if original_mask is not None else query.new_empty(0),out,maxima,sums)
        ctx.has_mask=original_mask is not None
        ctx.mask_shape=None if original_mask is None else tuple(original_mask.shape)
        ctx.options=causal,scale,p,qblock,kblock
        return out.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        query,key,value,saved_mask,out,maxima,sums=ctx.saved_tensors
        mask=saved_mask if ctx.has_mask else None
        causal,scale,p,qblock,kblock=ctx.options
        dtype=work_dtype(query)
        dq=torch.zeros_like(query,dtype=dtype)
        dk=torch.zeros_like(key,dtype=dtype)
        dv=torch.zeros_like(value,dtype=dtype)
        # Only a trainable additive mask has a gradient. Its gradient is the
        # caller-requested mask size; no full broadcasted score gradient is kept.
        dm=torch.zeros(ctx.mask_shape,device=query.device,dtype=dtype) if ctx.needs_input_grad[3] else None
        generator=_restore_rng(query.device,ctx.rng_state) if ctx.rng_state is not None else None
        rows,columns=query.shape[-2],key.shape[-2]
        if mask is not None:mask=mask.expand(*query.shape[:-2],rows,columns)
        with precision_context(query):
            for row in range(0,rows,qblock):
                q=query[...,row:row+qblock,:].to(dtype)
                do=gradient[...,row:row+qblock,:].to(dtype)
                delta=(do*out[...,row:row+qblock,:]).sum(-1,keepdim=True)
                maximum=maxima[...,row:row+qblock,:]
                total=sums[...,row:row+qblock,:]
                denominator=torch.where(total==0,torch.ones_like(total),total)
                for col in range(0,columns,kblock):
                    k=key[...,col:col+kblock,:].to(dtype)
                    v=value[...,col:col+kblock,:].to(dtype)
                    probability=_exponential(_scores(q,k,mask,causal,scale,row,col),maximum)/denominator
                    drop=_dropout(probability.shape,query.device,dtype,p,generator) if p else None
                    dp=do@v.transpose(-1,-2)
                    if drop is not None:dp=dp*drop
                    ds=probability*(dp-delta)
                    dq[...,row:row+qblock,:].add_(ds@k*scale)
                    dk[...,col:col+kblock,:].add_(ds.transpose(-1,-2)@q*scale)
                    dv[...,col:col+kblock,:].add_((probability if drop is None else probability*drop).transpose(-1,-2)@do)
                    if dm is not None:
                        padded=(1,)*(ds.ndim-dm.ndim)+tuple(dm.shape)
                        reduction=tuple(axis for axis,size in enumerate(padded[:-2]) if size==1 and ds.shape[axis]!=1)
                        item=ds.sum(reduction,keepdim=True) if reduction else ds
                        if padded[-2]==1:item=item.sum(-2,keepdim=True)
                        if padded[-1]==1:item=item.sum(-1,keepdim=True)
                        # Slice original (possibly scalar/1-D) mask by viewing
                        # it with leading singleton axes, never by expanding it.
                        row_start=0 if padded[-2]==1 else row
                        col_start=0 if padded[-1]==1 else col
                        dm.view(padded)[...,row_start:row_start+item.shape[-2],col_start:col_start+item.shape[-1]].add_(item)
        return dq.to(query.dtype),dk.to(key.dtype),dv.to(value.dtype),None if dm is None else dm.to(mask.dtype),None,None,None,None,None,None


def block_scaled_dot_product_attention(query,key,value,*,attn_mask=None,dropout_p=0.,is_causal=False,
                                       scale=None,enable_gqa=False,query_block_size=128,key_block_size=256,
                                       generator=None):
    """Exact softmax MHA/GQA with bounded score tiles and recomputed backward.

    Intermediate probabilities/masks are not retained as an N-by-N tensor.
    This is a device composition, not a fused FlashAttention kernel. Different
    tile reductions may differ by floating-point rounding. Backward is first
    order. A dropout call consumes its own reserved RNG stream; backward does
    not rewind or advance the application's generator.
    """
    if query.ndim<3 or key.ndim!=query.ndim or value.ndim!=query.ndim:
        raise ValueError('attention requires [...,heads,tokens,features] operands')
    if query.device!=key.device or query.device!=value.device or query.dtype!=key.dtype or query.dtype!=value.dtype:
        raise ValueError('attention operands must share floating dtype and device')
    work_dtype(query)
    if query.shape[-1]<=0 or query.shape[-1]!=key.shape[-1] or key.shape[-2]!=value.shape[-2]:
        raise ValueError('attention key/query/value dimensions differ')
    if type(is_causal) is not bool or type(enable_gqa) is not bool or not 0<=dropout_p<=1:
        raise ValueError('invalid attention dropout/causal/GQA policy')
    if any(type(size) is not int or size<=0 for size in (query_block_size,key_block_size)):
        raise ValueError('attention tile sizes must be positive integers')
    if is_causal and attn_mask is not None:raise ValueError('supply either a mask or causal attention')
    if enable_gqa:
        if key.shape[-3]<=0 or key.shape[-3]!=value.shape[-3] or query.shape[-3]%key.shape[-3]:
            raise ValueError('GQA requires positive common KV heads dividing query heads')
        copies=query.shape[-3]//key.shape[-3]
        key=key.repeat_interleave(copies,dim=-3)
        value=value.repeat_interleave(copies,dim=-3)
    prefix=torch.broadcast_shapes(query.shape[:-2],key.shape[:-2],value.shape[:-2])
    query=query.expand(*prefix,*query.shape[-2:])
    key=key.expand(*prefix,*key.shape[-2:])
    value=value.expand(*prefix,*value.shape[-2:])
    factor=query.shape[-1]**-.5 if scale is None else float(scale)
    if not math.isfinite(factor):raise ValueError('attention scale must be finite')
    mask_shape=None
    if attn_mask is not None:
        if attn_mask.device!=query.device or attn_mask.dtype!=torch.bool and not attn_mask.is_floating_point():
            raise ValueError('attention mask must be same-device boolean or floating storage')
        mask_shape=tuple(attn_mask.shape)
        scores_shape=(*prefix,query.shape[-2],key.shape[-2])
        if torch.broadcast_shapes(scores_shape,mask_shape)!=scores_shape:
            raise ValueError('attention mask does not broadcast to scores')
    if not query.shape[-2] or not key.shape[-2] or dropout_p==1 or math.prod(prefix)==0:
        result=query.new_zeros((*prefix,query.shape[-2],value.shape[-1]))
        result=result+query.sum()*0+key.sum()*0+value.sum()*0
        if attn_mask is not None and attn_mask.is_floating_point():
            result=result+attn_mask.masked_fill(~torch.isfinite(attn_mask),0).sum()*0
        return result
    return _BlockAttention.apply(query,key,value,attn_mask,is_causal,factor,float(dropout_p),
                                 query_block_size,key_block_size,generator)
