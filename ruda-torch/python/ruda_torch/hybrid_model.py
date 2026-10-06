"""Small configurable training model connecting mHC, CSA/HCA, DSA and Muon.

This is a reusable architecture integration example, NOT a DeepSeek-V4 model
replica or a loader for its pretrained checkpoints. No MoE, MTP, YaRN or official
quantized attention kernels are implied by the component names.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .mhc import MHC
from .sparse_attention import CSA, HCA, AttentionOutput, CompressedAttentionCache, _valid_mask
from ._architecture_ops import positive_int, positive_float, positions, rms, work_dtype


def _cache_signature(module):
    tensors=tuple((name,id(value),value._version,tuple(value.shape),value.dtype,value.device)
                  for name,value in (*module.named_parameters(),*module.named_buffers()))
    configuration=tuple((name,id(child),tuple(sorted((key,value) for key,value in vars(child).items()
        if not key.startswith('_') and isinstance(value,(bool,int,float,str)))))
        for name,child in module.named_modules())
    return tensors,configuration


@dataclass(frozen=True)
class HybridAttentionCache:
    """Functional, model-owned CSA/HCA histories for the same model revision."""
    owner: int
    signature: tuple
    seen: int
    layers: tuple[CompressedAttentionCache,...]

    def reorder(self,batch_indices):
        """Select/fork all layer histories without modifying the source cache."""
        if not self.layers or not isinstance(batch_indices,torch.Tensor) or batch_indices.numel()==0:
            raise ValueError('beam selection requires nonempty layer histories and integer indices')
        return HybridAttentionCache(self.owner,self.signature,self.seen,
                                    tuple(layer.reorder(batch_indices) for layer in self.layers))

    @property
    def tensor_bytes(self):
        """Logical retained bytes across layers, not allocator peak memory."""
        return sum(layer.tensor_bytes for layer in self.layers)


def _hybrid_hidden(module,tokens,valid_mask,return_aux,indexer_warmup):
    if tokens.ndim!=2 or min(tokens.shape)<1 or tokens.dtype not in (torch.int32,torch.int64):
        raise ValueError('tokens must be a nonempty [batch, length] integer tensor')
    hidden=module.embedding(tokens)
    valid=_valid_mask(hidden,valid_mask)
    hidden=hidden*valid.unsqueeze(-1).to(hidden.dtype)
    state=module.layers[0].attention_connection.expand(hidden)
    losses=[]
    for layer in module.layers:
        result=layer(state,valid_mask=valid,return_aux=return_aux,indexer_warmup=indexer_warmup)
        if return_aux:
            state=result.output
            losses.append(result.indexer_loss)
        else:state=result
    hidden=rms(module.layers[-1].ffn_connection.reduce(state),module.eps,module.final_norm)
    return AttentionOutput(hidden,torch.stack(losses).sum()) if return_aux else hidden


class MHCTransformerBlock(nn.Module):
    """Two independent mHC residual sublayers around attention and a gated FFN.

    Input/output state is [batch, tokens, streams, width]. return_aux=True
    returns the independent DSA distillation loss alongside the state.
    """
    def __init__(self, width: int, num_heads: int, *, streams=4, attention_kind='csa',
                 feedforward_width=None, sinkhorn_iterations=20, eps=1e-6, **attention_kwargs):
        super().__init__()
        self.width = positive_int(width, 'width')
        self.streams = positive_int(streams, 'streams')
        self.eps = positive_float(eps, 'eps')
        hidden = positive_int(feedforward_width if feedforward_width is not None else 4*width, 'feedforward_width')
        if attention_kind not in ('csa', 'hca'):
            raise ValueError('attention_kind must be csa or hca')
        opts = {k: attention_kwargs[k] for k in ('device', 'dtype') if k in attention_kwargs}
        self.attention_connection = MHC(width, streams, sinkhorn_iterations=sinkhorn_iterations, eps=eps, **opts)
        self.ffn_connection = MHC(width, streams, sinkhorn_iterations=sinkhorn_iterations, eps=eps, **opts)
        self.attention = (CSA if attention_kind == 'csa' else HCA)(width, num_heads, eps=eps, **attention_kwargs)
        self.attention_norm = nn.Parameter(torch.ones(width, **opts))
        self.ffn_norm = nn.Parameter(torch.ones(width, **opts))
        self.gate = nn.Linear(width, hidden, bias=False, **opts)
        self.up = nn.Linear(width, hidden, bias=False, **opts)
        self.down = nn.Linear(hidden, width, bias=False, **opts)

    def forward(self, state, *, valid_mask=None, return_aux=False, indexer_warmup=False):
        query, mappings = self.attention_connection.pre(state)
        normalized = rms(query, self.eps, self.attention_norm)
        result = self.attention(normalized, valid_mask=valid_mask, return_aux=return_aux, indexer_warmup=indexer_warmup)
        attention = result.output if return_aux else result
        state = self.attention_connection.post(state, attention, mappings)
        merged, mappings = self.ffn_connection.pre(state)
        normalized = rms(merged, self.eps, self.ffn_norm)
        update = self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        if valid_mask is not None:
            update = update * valid_mask.unsqueeze(-1).to(update.dtype)
        state = self.ffn_connection.post(state, update, mappings)
        return AttentionOutput(state, result.indexer_loss) if return_aux else state

    def forward_cached(self,state,cache=None,*,valid_mask=None):
        """Inference chunk with the same mHC mappings and FFN as full forward."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('cached hybrid blocks require eval() and no_grad()/inference_mode()')
        query,mappings=self.attention_connection.pre(state)
        normalized=rms(query,self.eps,self.attention_norm)
        valid=_valid_mask(normalized,valid_mask)
        attention,new_cache=self.attention.forward_cached(normalized,cache,valid_mask=valid)
        state=self.attention_connection.post(state,attention,mappings)
        merged,mappings=self.ffn_connection.pre(state)
        normalized=rms(merged,self.eps,self.ffn_norm)
        update=self.down(F.silu(self.gate(normalized))*self.up(normalized))
        update=update*valid.unsqueeze(-1).to(update.dtype)
        return self.ffn_connection.post(state,update,mappings),new_cache


class HybridAttentionLanguageModel(nn.Module):
    """Trainable causal language model with alternating CSA/HCA and mHC.

    Architecture sizes and compression ratios are caller choices. Defaults are
    component defaults, not published large-model hyperparameters. Parameters
    should normally be constructed on CPU and then moved to the target device,
    since RUDA does not provide all random initialization operators.
    """
    def __init__(self, vocab_size: int, width: int, num_heads: int, num_layers: int, *,
                 streams=4, csa_ratio=4, hca_ratio=128, tie_embeddings=False,
                 sinkhorn_iterations=20, eps=1e-6, **attention_kwargs):
        super().__init__()
        self.vocab_size = positive_int(vocab_size, 'vocab_size')
        self.width = positive_int(width, 'width')
        positive_int(num_layers, 'num_layers')
        self.eps = positive_float(eps, 'eps')
        if 'compress_ratio' in attention_kwargs or 'attention_kind' in attention_kwargs:
            raise ValueError('set csa_ratio and hca_ratio, not per-layer attention arguments')
        opts = {k: attention_kwargs[k] for k in ('device', 'dtype') if k in attention_kwargs}
        self.embedding = nn.Embedding(vocab_size, width, **opts)
        self.layers = nn.ModuleList([MHCTransformerBlock(width, num_heads, streams=streams,
            attention_kind='csa' if index % 2 == 0 else 'hca',
            compress_ratio=csa_ratio if index % 2 == 0 else hca_ratio,
            sinkhorn_iterations=sinkhorn_iterations, eps=eps, **attention_kwargs)
            for index in range(num_layers)])
        self.final_norm = nn.Parameter(torch.ones(width, **opts))
        self.head = nn.Linear(width, vocab_size, bias=False, **opts)
        if tie_embeddings:
            self.head.weight = self.embedding.weight

    def forward(self, tokens, *, valid_mask=None, return_aux=False, indexer_warmup=False):
        result=self.forward_hidden(tokens,valid_mask=valid_mask,return_aux=return_aux,indexer_warmup=indexer_warmup)
        return AttentionOutput(self.head(result.output),result.indexer_loss) if return_aux else self.head(result)

    def forward_hidden(self,tokens,*,valid_mask=None,return_aux=False,indexer_warmup=False):
        """Normalized backbone features without allocating full vocabulary logits."""
        return _hybrid_hidden(self,tokens,valid_mask,return_aux,indexer_warmup)

    def hidden_backbone(self):
        """Share existing embedding/blocks/norm, excluding the vocabulary head."""
        return HybridAttentionBackbone(self)

    def forward_hidden_cached(self,tokens,cache=None,*,valid_mask=None):
        """Prefill/decode actual chunks, with one compressed history per block.

        Calls are eval-only and functional. Parameter/buffer versions, module
        identities, scalar architecture settings and tensor devices/dtypes must
        match the source cache. No full prompt hidden states are retained.
        """
        if any(module.training for module in self.modules()) or torch.is_grad_enabled():
            raise RuntimeError('cached hybrid models require eval() and no_grad()/inference_mode()')
        if tokens.ndim!=2 or min(tokens.shape)<1 or tokens.dtype not in (torch.int32,torch.int64):
            raise ValueError('tokens must be a nonempty [batch, length] integer tensor')
        signature=_cache_signature(self)
        if cache is not None:
            if not isinstance(cache,HybridAttentionCache) or cache.owner!=id(self) or cache.signature!=signature:
                raise ValueError('hybrid cache belongs to another model or model state changed; reset it')
            if len(cache.layers)!=len(self.layers) or any(layer.seen!=cache.seen for layer in cache.layers):
                raise ValueError('hybrid cache layer positions differ')
            for layer,history in zip(self.layers,cache.layers,strict=True):
                if history.owner!=id(layer.attention) or history.local.shape[0]!=tokens.shape[0] or history.local.device!=tokens.device:
                    raise ValueError('hybrid layer cache owner, batch or device differs')
        hidden=self.embedding(tokens)
        valid=_valid_mask(hidden,valid_mask)
        state=self.layers[0].attention_connection.expand(hidden*valid.unsqueeze(-1).to(hidden.dtype))
        histories=[]
        for index,layer in enumerate(self.layers):
            state,history=layer.forward_cached(state,None if cache is None else cache.layers[index],valid_mask=valid)
            histories.append(history)
        hidden=rms(self.layers[-1].ffn_connection.reduce(state),self.eps,self.final_norm)
        seen=(0 if cache is None else cache.seen)+tokens.shape[1]
        return hidden,HybridAttentionCache(id(self),signature,seen,tuple(histories))

    def forward_cached(self,tokens,cache=None,*,valid_mask=None):
        hidden,cache=self.forward_hidden_cached(tokens,cache,valid_mask=valid_mask)
        return self.head(hidden),cache

    def forward_cached_last(self,tokens,cache=None,*,valid_mask=None):
        """Project only the last physical token: [batch,vocabulary] logits."""
        hidden,cache=self.forward_hidden_cached(tokens,cache,valid_mask=valid_mask)
        return self.head(hidden[:,-1]),cache


class HybridAttentionBackbone(nn.Module):
    """Shared hidden-state view for CausalLMFinetuner/PackedCausalLMFinetuner.

    Dense masks use zero/nonzero validity. Packed documents run independent CSA/
    HCA/mHC histories and reset RoPE/compressor positions; no cross-document
    attention or partial compression block is shared. Packed position_ids must
    be the actual zero-based positions supplied by the packing collator.
    """
    def __init__(self,model):
        super().__init__()
        if not isinstance(model,HybridAttentionLanguageModel):raise TypeError('expected an existing hybrid model')
        self.embedding,self.layers=model.embedding,model.layers
        self.final_norm=model.final_norm
        self.eps,self.width=model.eps,model.width
        self.training=model.training

    def forward(self,input_ids,attention_mask=None,*,position_ids=None,cu_seqlens=None,max_seqlen=None):
        if attention_mask is not None:
            if attention_mask.shape!=input_ids.shape or attention_mask.device!=input_ids.device:
                raise ValueError('attention mask must match tokens on the token device')
            valid=attention_mask.to(torch.bool)
        else:valid=None
        if cu_seqlens is None:
            if position_ids is not None or max_seqlen is not None:
                raise ValueError('packed position metadata requires cu_seqlens')
            return _hybrid_hidden(self,input_ids,valid,False,False)
        from .varlen_attention import _boundaries
        if input_ids.ndim!=1 or input_ids.dtype not in (torch.int32,torch.int64):
            raise ValueError('packed input_ids must be a flat integer token tensor')
        boundaries=_boundaries(cu_seqlens,input_ids.numel())
        maximum=max(b-a for a,b in zip(boundaries,boundaries[1:]))
        if type(max_seqlen) is not int or max_seqlen!=maximum:
            raise ValueError('max_seqlen must equal the actual longest document')
        if position_ids is None or position_ids.shape!=input_ids.shape or position_ids.device!=input_ids.device or position_ids.dtype not in (torch.int32,torch.int64):
            raise ValueError('packed positions must match tokens as same-device integers')
        lengths=[b-a for a,b in zip(boundaries,boundaries[1:])]
        expected=torch.cat([positions(length,input_ids.device) for length in lengths])
        if not (position_ids.to(torch.int64)==expected).all().item():
            raise ValueError('hybrid packed positions must restart at zero within each document')
        outputs=[]
        for begin,end in zip(boundaries,boundaries[1:]):
            if end==begin:continue
            outputs.append(_hybrid_hidden(self,input_ids[begin:end].unsqueeze(0),
                None if valid is None else valid[begin:end].unsqueeze(0),False,False).squeeze(0))
        return torch.cat(outputs,0) if outputs else self.embedding(input_ids)


def next_token_loss(logits: torch.Tensor, tokens: torch.Tensor, *, valid_mask=None):
    """Mean next-token cross entropy, excluding padded source/target pairs.

    Returns a connected zero loss for length-one/all-padded batches. Targets
    must be valid vocabulary IDs, including masked locations, because device
    gather bounds checking is never disabled to accommodate invalid sentinels.
    """
    if logits.ndim != 3 or tokens.shape != logits.shape[:2] or tokens.device != logits.device:
        raise ValueError('logits [B,T,V] and targets [B,T] must match on the same device')
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError('tokens must be int32 or int64')
    valid = _valid_mask(logits, valid_mask)
    if logits.shape[1] < 2:
        return logits.to(work_dtype(logits)).sum() * 0
    logp = F.log_softmax(logits[:, :-1].to(work_dtype(logits)), dim=-1)
    values = -logp.gather(-1, tokens[:, 1:].to(torch.int64).unsqueeze(-1)).squeeze(-1)
    keep = (valid[:, :-1] & valid[:, 1:]).to(values.dtype)
    count = keep.sum()
    return (values * keep).sum() / torch.where(count > 0, count, torch.ones_like(count))
