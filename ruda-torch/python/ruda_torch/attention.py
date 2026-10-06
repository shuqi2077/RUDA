"""Device-preserving differentiable attention with explicit GQA and mask semantics."""
from __future__ import annotations

import math
import torch
from ._architecture_ops import masked_softmax,positions,precision_context,work_dtype


def scaled_dot_product_attention(query,key,value,*,attn_mask=None,dropout_p=0.,is_causal=False,
                                 scale=None,enable_gqa=False):
    """First-order MHA/GQA composition; tensor operations stay on the source device.

    This path materializes attention scores. Paged native attention is a separate
    interface for applications requiring that execution/storage contract.
    Boolean masks use True=allowed; floating masks are added before softmax.
    """
    if query.ndim<3 or key.ndim!=query.ndim or value.ndim!=query.ndim:
        raise ValueError('attention requires [...,heads,tokens,features] operands')
    if query.device!=key.device or query.device!=value.device or query.dtype!=key.dtype or query.dtype!=value.dtype:
        raise ValueError('attention operands must share floating dtype and device')
    if query.shape[-1]!=key.shape[-1] or key.shape[-2]!=value.shape[-2]:
        raise ValueError('attention key/query/value dimensions differ')
    if not 0<=dropout_p<=1 or type(is_causal) is not bool or type(enable_gqa) is not bool:
        raise ValueError('invalid attention dropout/causal/GQA policy')
    if is_causal and attn_mask is not None:
        raise ValueError('supply either an attention mask or causal masking, not both')
    if enable_gqa:
        if key.shape[-3]!=value.shape[-3] or query.shape[-3]%key.shape[-3]:
            raise ValueError('GQA query heads must divide the common KV head count')
        repeats=query.shape[-3]//key.shape[-3]
        key=key.repeat_interleave(repeats,dim=-3)
        value=value.repeat_interleave(repeats,dim=-3)
    factor=query.shape[-1]**-.5 if scale is None else float(scale)
    if not math.isfinite(factor):
        raise ValueError('attention scale must be finite')
    if not key.shape[-2]:
        return query.new_zeros((*query.shape[:-1],value.shape[-1]))+query.sum()*0+value.sum()*0
    with precision_context(query):
        dtype=work_dtype(query)
        scores=query.to(dtype)@key.to(dtype).transpose(-1,-2)*factor
        allowed=torch.ones_like(scores,dtype=torch.bool)
        if is_causal:
            row=positions(query.shape[-2],query.device).unsqueeze(-1)
            col=positions(key.shape[-2],key.device).unsqueeze(0)
            allowed=col<=row
        elif attn_mask is not None:
            if attn_mask.device!=query.device:
                raise ValueError('attention mask must be on the input device')
            if attn_mask.dtype==torch.bool:
                allowed=attn_mask
            elif attn_mask.is_floating_point():
                scores=scores+attn_mask.to(dtype)
                allowed=scores!=float('-inf')
            else:
                raise ValueError('attention masks must be boolean or floating point')
        probability=masked_softmax(scores,allowed)
        if dropout_p:
            if query.device.type=='ruda':
                from .random import native_dropout
                probability=native_dropout(probability,dropout_p)[0]
            else:
                probability=torch.nn.functional.dropout(probability,p=dropout_p,training=True)
        return (probability@value.to(dtype)).to(query.dtype)
