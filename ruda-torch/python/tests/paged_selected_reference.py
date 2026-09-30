"""TEST ONLY dense autograd reference and independent selected analytical VJP.

Uses host tensors; NEVER imported by production ruda_torch or its runtime.
"""
import torch

SPEC=dict(page_size=3,num_pages=4,block_tables=[[2,0],[2,3],[]],kv_lengths=[5,4,0],
          sequence_ids=[0,1,0,2],positions=[1,3,4,0])

def data(mla=False,*,dtype=torch.float64,queries=4,seed=817):
    gen=torch.Generator().manual_seed(seed)
    def r(*shape):return (torch.randn(shape,generator=gen,dtype=torch.float64)*.3).to(dtype)
    q=r(queries,4,5);k=r(4,3,1 if mla else 2,5)
    return (q,r(queries,4,3),k,r(4,3,1,3)) if mla else (q,k,r(4,3,2,7))

def dense(tensors,mla=False,*,scale=.37,causal=True,spec=SPEC):
    if mla:q,qp,k,kp=tensors;v=k
    else:q,k,v=tensors;qp=kp=None
    # Keep empty outputs connected to every input for empty-history cases.
    zero=sum(t.sum()*0 for t in tensors)
    result=q.new_zeros((q.shape[0],q.shape[1],v.shape[-1]))+zero
    rows=[]
    for row,seq in enumerate(spec['sequence_ids']):
        end=min(spec['kv_lengths'][seq],spec['positions'][row]+1) if causal else spec['kv_lengths'][seq]
        if not end:rows.append(result[row]);continue
        slots=[(spec['block_tables'][seq][t//spec['page_size']],t%spec['page_size']) for t in range(end)]
        heads=[]
        for h in range(q.shape[1]):
            kh=h//(q.shape[1]//k.shape[2]);kk=torch.stack([k[p,o,kh] for p,o in slots]);vv=torch.stack([v[p,o,kh] for p,o in slots])
            scores=kk@q[row,h]
            if mla:scores=scores+torch.stack([kp[p,o,0] for p,o in slots])@qp[row,h]
            probs=torch.softmax(scores*scale,dim=0);heads.append(probs@vv)
        rows.append(torch.stack(heads))
    return torch.stack(rows) if rows else result

def analytical(tensors,go,needs,mla=False,*,scale=.37,causal=True,spec=SPEC):
    """Separate analytical reference: no autograd inside this function."""
    vals=[t.detach().double() for t in tensors];go=go.detach().double()
    if mla:q,qp,k,kp=vals;v=k
    else:q,k,v=vals;qp=kp=None
    out=[torch.zeros_like(t) if n else None for t,n in zip(vals,needs)]
    for row,seq in enumerate(spec['sequence_ids']):
        end=min(spec['kv_lengths'][seq],spec['positions'][row]+1) if causal else spec['kv_lengths'][seq]
        if not end:continue
        slots=[(spec['block_tables'][seq][t//spec['page_size']],t%spec['page_size']) for t in range(end)]
        for h in range(q.shape[1]):
            kh=h//(q.shape[1]//k.shape[2]);kk=torch.stack([k[p,o,kh] for p,o in slots]);vv=torch.stack([v[p,o,kh] for p,o in slots])
            score=kk@q[row,h]
            if mla:pp=torch.stack([kp[p,o,0] for p,o in slots]);score=score+pp@qp[row,h]
            prob=torch.softmax(score*scale,dim=0)
            dp=vv@go[row,h];ds=prob*(dp-(prob*dp).sum())*scale
            if needs[0]:out[0][row,h]=ds@kk
            if mla and needs[1]:out[1][row,h]=ds@pp
            for j,(p,o) in enumerate(slots):
                if mla:
                    if needs[2]:out[2][p,o,0]+=ds[j]*q[row,h]+prob[j]*go[row,h]
                    if needs[3]:out[3][p,o,0]+=ds[j]*qp[row,h]
                else:
                    if needs[1]:out[1][p,o,kh]+=ds[j]*q[row,h]
                    if needs[2]:out[2][p,o,kh]+=prob[j]*go[row,h]
    return tuple(None if x is None else x.to(t.dtype) for x,t in zip(out,tensors))
