"""Small configurable training model connecting mHC, CSA/HCA, DSA and Muon.

This is a reusable architecture integration example, NOT a DeepSeek-V4 model
replica or a loader for its pretrained checkpoints. No MoE, MTP, YaRN or official
quantized attention kernels are implied by the component names.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .mhc import MHC
from .sparse_attention import CSA, HCA, AttentionOutput, _valid_mask
from ._architecture_ops import positive_int, positive_float, rms, work_dtype


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
        if tokens.ndim != 2 or min(tokens.shape) < 1 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError('tokens must be a nonempty [batch, length] integer tensor')
        hidden = self.embedding(tokens)
        valid = _valid_mask(hidden, valid_mask)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        state = self.layers[0].attention_connection.expand(hidden)
        losses = []
        for layer in self.layers:
            result = layer(state, valid_mask=valid, return_aux=return_aux, indexer_warmup=indexer_warmup)
            if return_aux:
                state = result.output
                losses.append(result.indexer_loss)
            else:
                state = result
        hidden = self.layers[-1].ffn_connection.reduce(state)
        logits = self.head(rms(hidden, self.eps, self.final_norm))
        if not return_aux:
            return logits
        return AttentionOutput(logits, torch.stack(losses).sum())


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
