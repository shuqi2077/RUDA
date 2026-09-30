"""Immutable packed-query scheduling metadata; tensors execute in native ruDNN.

Cache tensors: [physical_pages,page_size,KV_heads,features].
Queries: [total_queries,query_heads,features]. No padding between requests.
A plan can mix single-token decode with multi-token prefill in the SAME batch.
"""
import operator
import torch
from . import _C

def _u32(value):
    if isinstance(value,bool):
        raise TypeError("boolean is not a paged index")
    value=operator.index(value)
    if not 0<=value<2**32:
        raise ValueError("paged index exceeds uint32 range")
    return value


class _PagedGQA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native, q, k, v, scale, causal, ordered=False):
        ctx.native=native;ctx.scale=float(scale);ctx.causal=bool(causal);ctx.ordered=bool(ordered)
        ctx.save_for_backward(q,k,v);ctx.set_materialize_grads(False)
        return native.run(q,k,v,None,None,ctx.scale,ctx.causal)
    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("RUDA paged attention supports first-order gradients only")
        if grad is None:return (None,)*len(ctx.needs_input_grad)
        q,k,v=ctx.saved_tensors
        needs=tuple(ctx.needs_input_grad[1:4])
        dq,dk,dv=ctx.native.backward_selected(q,k,v,None,None,grad.contiguous(),ctx.scale,ctx.causal,needs,*((True,) if ctx.ordered else ()))
        return (None,dq,dk,dv,None,None,None)[:len(ctx.needs_input_grad)]

class _PagedMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,native,q,qp,latent,kp,scale,causal,ordered=False):
        ctx.native=native;ctx.scale=float(scale);ctx.causal=bool(causal);ctx.ordered=bool(ordered)
        ctx.save_for_backward(q,qp,latent,kp);ctx.set_materialize_grads(False)
        return native.run(q,latent,latent,qp,kp,ctx.scale,ctx.causal)
    @staticmethod
    def backward(ctx,grad):
        if torch.is_grad_enabled():
            raise RuntimeError("RUDA paged MLA supports first-order gradients only")
        if grad is None:return (None,)*len(ctx.needs_input_grad)
        q,qp,latent,kp=ctx.saved_tensors
        needs=tuple(ctx.needs_input_grad[1:5])
        dq,dqp,dlatent,dkp=ctx.native.backward_selected(q,latent,latent,qp,kp,grad.contiguous(),ctx.scale,ctx.causal,needs,*((True,) if ctx.ordered else ()))
        return (None,dq,dqp,dlatent,dkp,None,None,None)[:len(ctx.needs_input_grad)]

class PagedAttentionPlan:
    def __init__(self, *, page_size, num_pages, block_tables, kv_lengths, sequence_ids, positions, splits=1, backward_strategy="atomic"):
        if backward_strategy not in ("atomic", "ordered"):
            raise ValueError("backward_strategy must be 'atomic' or 'ordered'")
        self._backward_strategy=backward_strategy
        page_size=_u32(page_size); num_pages=_u32(num_pages)
        splits=_u32(splits)
        if not 1<=splits<=32:
            raise ValueError("splits must be 1..32; 1 preserves the single-kernel path")
        self._splits=splits
        tables=tuple(tuple(_u32(p) for p in row) for row in block_tables)
        lengths=tuple(_u32(x) for x in kv_lengths)
        seq=tuple(_u32(x) for x in sequence_ids)
        pos=tuple(_u32(x) for x in positions)
        if not page_size or not num_pages or not tables or len(tables)!=len(lengths) or len(seq)!=len(pos):
            raise ValueError("invalid paged plan dimensions")
        width=max(1,max(map(len,tables)))
        words_count=2*len(seq)+len(tables)*(1+width)
        if words_count>=2**32:
            raise ValueError("paged metadata exceeds uint32 indexing")
        for table,length in zip(tables,lengths):
            required=(length+page_size-1)//page_size
            if len(table)<required or any(p>=num_pages for p in table) or len(set(table[:required]))!=required:
                raise ValueError("missing, repeated or out-of-range physical page")
        for request,position in zip(seq,pos):
            if request>=len(tables) or (lengths[request] and position>=lengths[request]):
                raise ValueError("invalid query request or absolute position")
        self._spec=(page_size,num_pages,len(tables),len(seq),width,splits)
        self._words=seq+pos+lengths+tuple(p for table in tables for p in table+(2**32-1,)*(width-len(table)))
        self._bound={}

    @property
    def backward_strategy(self):
        return self._backward_strategy

    @property
    def splits(self):
        return self._splits

    def workspace_bytes(self, query_heads, value_dim):
        """Planned FP32 scratch bytes, NOT measured peak device memory.

        Cached once per native plan/stream and shape. Newly created schedules
        own their own workspaces; this is not a cross-request global pool.
        """
        heads=_u32(query_heads); dim=_u32(value_dim)
        if not heads or not 1<=dim<=1024:
            raise ValueError("invalid attention heads/value dimension")
        size=0 if self.splits==1 else self._spec[3]*heads*self.splits*(dim+2)*4
        if size>64*1024*1024:
            raise ValueError("split workspace exceeds 64 MiB per-plan limit")
        return size

    def _native(self, like):
        if like.device.type!="ruda":
            raise ValueError("paged attention requires native ruda tensors, not CPU/CUDA tensors")
        key=(like.device.index,_C.stream_command(0))
        if key not in self._bound:
            self._bound[key]=_C.NativePagedPlan(like,self._spec,self._words)
        return self._bound[key]

    def attention(self, q, k, v, *, scale, causal=True):
        """Paged GQA/MHA with optional first-order autograd.

        Forward keeps the existing fused/split inference path. When any Q/K/V
        input requires gradients, backward recomputes attention probabilities
        from Q/K instead of saving a full score matrix. Frozen inputs do not
        allocate gradient outputs or FP32 history-gradient scratch. All inputs
        remain saved: freezing an input does not make its value unnecessary to
        the other derivatives. History derivatives use FP32 atomic accumulation
        and respect PyTorch's deterministic error/warning policy. Explicit
        backward_strategy="ordered" uses single-writer history reduction instead,
        with cached per-query statistics and inverse page metadata. It is opt-in,
        not an automatically tuned faster path; it may be slower on some shapes.
        """
        native=self._native(q)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
            return _PagedGQA.apply(native,q,k,v,float(scale),bool(causal),self._backward_strategy=="ordered")
        return native.run(q,k,v,None,None,float(scale),bool(causal))

    def mla(self, absorbed_query, position_query, latent_cache, position_cache, *, scale, causal=True):
        """Return compressed context; model applies value/output projection.

        RoPE must already be applied; scale is the model's QK scale, NOT 1/sqrt(R).
        FP8/INT4 checkpoints must first use a supported model dequantization path.
        """
        if latent_cache.ndim!=4 or latent_cache.shape[2]!=1:
            raise ValueError("MLA requires a single shared compressed cache head")
        native=self._native(absorbed_query)
        tensors=(absorbed_query,position_query,latent_cache,position_cache)
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            return _PagedMLA.apply(native,*tensors,float(scale),bool(causal),self._backward_strategy=="ordered")
        return native.run(absorbed_query,latent_cache,latent_cache,
            position_query,position_cache,float(scale),bool(causal))
