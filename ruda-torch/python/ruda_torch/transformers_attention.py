"""Opt-in Transformers AttentionInterface registration for RUDA block attention."""
from __future__ import annotations

from .block_attention import block_scaled_dot_product_attention


def register_transformers_attention(name='ruda_block',*,query_block_size=128,key_block_size=256):
    """Register attention AND the standard HF padding/causal-mask adapter.

    Select the returned name through Transformers' attn_implementation option.
    No model is patched and no built-in implementation is overridden. Models
    must use AttentionInterface. HF's standard mask construction may itself
    allocate a dense mask; score/probability storage remains block-bounded.
    """
    from transformers.modeling_utils import AttentionInterface,ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import AttentionMaskInterface,ALL_MASK_ATTENTION_FUNCTIONS
    if not isinstance(name,str) or not name.startswith('ruda_'):
        raise ValueError('use a distinct ruda_ attention registration name')
    if any(type(n) is not int or n<=0 for n in (query_block_size,key_block_size)):
        raise ValueError('attention tile sizes must be positive integers')
    if name in ALL_ATTENTION_FUNCTIONS or name in ALL_MASK_ATTENTION_FUNCTIONS:
        raise ValueError('attention registration name already exists')

    def attention(module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
        if query.ndim!=4 or key.ndim!=4 or value.ndim!=4:
            raise ValueError('Transformers attention requires [batch,heads,tokens,features] tensors')
        if kwargs.get('output_attentions',False) or kwargs.get('head_mask') is not None:
            raise ValueError('block attention does not return quadratic weights or implement head masking')
        if kwargs.get('softcap') is not None:
            raise ValueError('soft-capped logits require a separately defined attention transform')
        if attention_mask is not None:
            if attention_mask.ndim!=4:
                raise ValueError('HF must construct a 4-D boolean/additive attention mask')
            attention_mask=attention_mask[...,:key.shape[-2]]
            if is_causal:
                raise ValueError('do not add causal masking to an already prepared HF mask')
            causal=False
        elif is_causal is not None:
            if type(is_causal) is not bool:raise ValueError('is_causal must be explicit boolean metadata')
            causal=is_causal
        elif hasattr(module,'is_causal'):
            causal=bool(module.is_causal) and query.shape[-2]>1
        else:
            raise ValueError('supply is_causal or a module-declared causal policy when no mask exists')
        window=None
        requested_window=kwargs.get('sliding_window')
        if requested_window is not None and attention_mask is None:
            if type(requested_window) is not int or requested_window<=0 or not causal:
                raise ValueError('unmasked sliding attention requires a positive causal window')
            window=(requested_window-1,0)
        output=block_scaled_dot_product_attention(query,key,value,attn_mask=attention_mask,
            dropout_p=dropout,is_causal=causal,scale=scaling,enable_gqa=query.shape[1]!=key.shape[1],
            query_block_size=query_block_size,key_block_size=key_block_size,
            causal_offset=key.shape[-2]-query.shape[-2],window_size=window)
        return output.transpose(1,2).contiguous(),None

    # An unregistered mask backend would make HF discard padding/packed masks.
    # Keep its standard sdpa mask policy, including cached/sliding/packed inputs.
    mask=ALL_MASK_ATTENTION_FUNCTIONS['sdpa']
    AttentionMaskInterface.register(name,mask)
    AttentionInterface.register(name,attention)
    return name
