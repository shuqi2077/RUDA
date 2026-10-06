"""Packed-document attention using same-device online-softmax tiles."""
from __future__ import annotations

import torch
from .block_attention import block_scaled_dot_product_attention


def _boundaries(values,length):
    if not isinstance(values,torch.Tensor) or values.device.type!='cpu' or values.dtype not in (torch.int32,torch.int64) or values.ndim!=1:
        raise ValueError('cu_seqlens must be explicit CPU integer metadata')
    result=values.tolist()
    if len(result)<2 or result[0]!=0 or result[-1]!=length or any(a>b for a,b in zip(result,result[1:])):
        raise ValueError('cu_seqlens must partition exactly the actual token payload')
    return result


def varlen_scaled_dot_product_attention(query,key,value,cu_seqlens_q,cu_seqlens_k=None,*,
        dropout_p=0.,is_causal=False,scale=None,enable_gqa=False,query_block_size=128,key_block_size=256,
        generator=None,window_size=None,causal_alignment='upper_left'):
    """Attend independently within each real document, without an N-by-N mask.

    Payloads are [total_tokens,heads,features]. CPU boundaries are input metadata,
    never downloaded GPU activations. Forward/backward use the existing bounded
    block attention on the SAME device. This is not a fused varlen kernel.
    Causal alignment is explicit for unequal query/key lengths (including cache).
    """
    if query.ndim!=3 or key.ndim!=3 or value.ndim!=3 or key.shape[0]!=value.shape[0]:
        raise ValueError('varlen attention requires [tokens,heads,features] operands')
    if causal_alignment not in ('upper_left','lower_right'):raise ValueError('select an explicit causal alignment')
    queries=_boundaries(cu_seqlens_q,query.shape[0])
    keys=_boundaries(cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k,key.shape[0])
    if len(queries)!=len(keys):raise ValueError('query/key document counts differ')
    outputs=[]
    for index in range(len(queries)-1):
        qa,qb=queries[index:index+2]
        ka,kb=keys[index:index+2]
        q=query[qa:qb].transpose(0,1)
        k=key[ka:kb].transpose(0,1)
        v=value[ka:kb].transpose(0,1)
        offset=0 if causal_alignment=='upper_left' else (kb-ka)-(qb-qa)
        outputs.append(block_scaled_dot_product_attention(q,k,v,dropout_p=dropout_p,is_causal=is_causal,
            scale=scale,enable_gqa=enable_gqa,query_block_size=query_block_size,key_block_size=key_block_size,
            generator=generator,causal_offset=offset,window_size=window_size).transpose(0,1))
    return torch.cat(outputs,dim=0)
