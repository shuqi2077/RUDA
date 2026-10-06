"""Exact vocabulary-sharded cross entropy without gathering vocabulary logits."""
from __future__ import annotations

import math
import torch
from torch.autograd.function import once_differentiable
from ._architecture_ops import positions, precision_context, work_dtype


class _VocabLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,targets,group,start,vocabulary,ignore,smoothing):
        dtype=work_dtype(logits)
        local=logits.shape[-1]
        flat=logits.reshape(-1,local).to(dtype)
        labels=targets.reshape(-1).to(torch.int64)
        valid=labels!=ignore
        invalid=(valid&((labels<0)|(labels>=vocabulary))).any().to(torch.float32).reshape(1)
        group.sum_(invalid)
        if invalid.item():raise ValueError('target lies outside the global vocabulary')
        classes=positions(local,logits.device,start).reshape(1,-1)
        real=classes<vocabulary
        owned=valid&(labels>=start)&(labels<min(start+local,vocabulary))
        indices=torch.where(owned,labels-start,torch.zeros_like(labels))
        with precision_context(logits):
            maximum=flat.masked_fill(~real,float('-inf')).amax(-1,keepdim=True).contiguous()
            group.max_(maximum)
            probability=(flat.masked_fill(~real,float('-inf'))-maximum).exp()
            total=probability.sum(-1,keepdim=True).contiguous()
            group.sum_(total)
            probability.div_(total)
            predicted=torch.where(owned,flat.gather(-1,indices.unsqueeze(-1)).squeeze(-1),torch.zeros_like(labels,dtype=dtype)).contiguous()
            group.sum_(predicted)
            normalizer=maximum.squeeze(-1)+total.squeeze(-1).log()
            loss=normalizer-predicted
            if smoothing:
                uniform=flat.masked_fill(~real,0).sum(-1).contiguous()
                group.sum_(uniform)
                loss=(1-smoothing)*loss+smoothing*(normalizer-uniform/vocabulary)
            loss=torch.where(valid,loss,torch.zeros_like(loss))
        ctx.save_for_backward(probability,labels,valid,classes,real)
        ctx.options=start,vocabulary,smoothing,tuple(logits.shape),logits.dtype
        return loss.view(targets.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        probability,labels,valid,classes,real=ctx.saved_tensors
        start,vocabulary,smoothing,shape,dtype=ctx.options
        result=probability-smoothing/vocabulary*real.to(probability.dtype)
        result=result-(classes==labels.unsqueeze(-1)).to(probability.dtype)*(1-smoothing)
        result.mul_(torch.where(valid,gradient.reshape(-1),torch.zeros_like(gradient.reshape(-1))).unsqueeze(-1))
        return result.view(shape).to(dtype),None,None,None,None,None,None


def vocab_parallel_cross_entropy(logits,targets,group,*,vocab_start=None,global_vocab_size=None,
                                 ignore_index=-100,label_smoothing=0.,reduction='mean'):
    """Token-wise replicated loss with local vocabulary gradients.

    Logits are contiguous vocabulary intervals in rank order, with optional
    padding beyond global_vocab_size. Targets and token order must be identical
    across this tensor-parallel group. Unequal local vocabulary widths are
    allowed; every rank must own at least one storage column. The returned loss
    is one logical loss, not a rank-summed data-parallel loss.
    """
    error=None
    try:
        work_dtype(logits)
        if logits.ndim<1 or logits.shape[-1]<=0 or targets.shape!=logits.shape[:-1]:
            raise ValueError('targets must match logits token dimensions with a positive local vocabulary')
        if targets.device!=logits.device or targets.dtype not in (torch.int32,torch.int64):
            raise ValueError('targets must be same-device integer storage')
        if reduction not in ('none','sum','mean') or type(ignore_index) is not int:
            raise ValueError('invalid cross-entropy reduction or ignored target')
        if isinstance(label_smoothing,bool) or not math.isfinite(label_smoothing) or not 0<=label_smoothing<=1:
            raise ValueError('label smoothing must be finite in [0,1]')
        if vocab_start is not None and (type(vocab_start) is not int or vocab_start<0):
            raise ValueError('vocabulary start must be a nonnegative integer')
        if global_vocab_size is not None and (type(global_vocab_size) is not int or global_vocab_size<=0):
            raise ValueError('global vocabulary must be a positive integer')
        if logits.device.type!=group.device_type:raise ValueError('logits and process group devices differ')
    except (TypeError,ValueError) as failure:
        error=str(failure)
    contracts=group.gather_metadata((tuple(logits.shape),str(logits.dtype),reduction,ignore_index,
                                     label_smoothing,vocab_start,global_vocab_size,error))
    for shape,dtype,mode,ignore,smoothing,start,total,failure in contracts:
        if failure:raise ValueError(failure)
        if (shape[:-1],dtype,mode,ignore,smoothing,total)!=(tuple(logits.shape[:-1]),str(logits.dtype),reduction,ignore_index,label_smoothing,global_vocab_size):
            raise ValueError('vocabulary loss token shapes/dtypes/options differ across ranks')
    widths=[item[0][-1] for item in contracts]
    starts=[sum(widths[:rank]) for rank in range(group.world_size)]
    if any(item[5] is not None and item[5]!=starts[rank] for rank,item in enumerate(contracts)):
        raise ValueError('vocabulary intervals must be contiguous and ordered by group rank')
    total=sum(widths) if global_vocab_size is None else global_vocab_size
    if total>sum(widths):raise ValueError('global vocabulary exceeds shard storage')
    if targets.numel()==0:
        return logits.sum(-1)*0 if reduction=='none' else logits.sum()*0
    loss=_VocabLoss.apply(logits,targets,group,starts[group.rank],total,ignore_index,float(label_smoothing))
    if reduction=='none':return loss
    result=loss.sum()
    if reduction=='sum':return result
    count=(targets!=ignore_index).sum().to(result.dtype)
    return result/torch.where(count==0,torch.ones_like(count),count)
