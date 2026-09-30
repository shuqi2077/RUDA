"""TEST ONLY: independent visibility-index model, not a CPU execution backend.

It models integer control flow separately from the dense/autograd oracle. It
never runs in production and cannot certify the Rust DSL or GPU execution.
"""
from collections import Counter
import math
BLOCK_ROWS = 32


def make_spec(mode="monotonic", count=97):
    # Page 0 is shared at DIFFERENT logical offsets. Sequence 2 owns a page but
    # has no query; the last physical page is unused by every sequence.
    spec=dict(page_size=16,num_pages=18,
              block_tables=[list(range(8)),[9,10,0,11,12,13,14,15],[16],[]],
              kv_lengths=[113,109,16,0],sequence_ids=[],positions=[])
    if mode=="empty":return spec
    if mode=="duplicate": pos=[min(108,i//3) for i in range(count)]
    elif mode=="unsorted":pos=[(i*37)%109 for i in range(count)]
    elif mode=="blocked":pos=[(31-i%32)%17 if i<64 else 100+i%9 for i in range(count)]
    elif mode=="monotonic":pos=[min(108,i) for i in range(count)]
    else:raise ValueError(mode)
    spec["sequence_ids"]=[i%2 for i in range(count)]+[3]
    spec["positions"]=pos+[0xffffffff] # zero-length request, no reads allowed
    return spec


def encode(spec, max_words=16*1024*1024):
    p=spec['num_pages'];s=len(spec['kv_lengths']);q=len(spec['positions']);size=spec['page_size']
    rows=sorted((seq,row) for row,seq in enumerate(spec['sequence_ids']))
    counts=Counter(spec['sequence_ids'])
    fixed=p+6+4*s+q
    if fixed>max_words:raise ValueError('budget')
    links=sorted((page,seq,logical) for seq,table in enumerate(spec['block_tables']) if counts[seq]
                 for logical,page in enumerate(table[:math.ceil(spec['kv_lengths'][seq]/size)]))
    blocks=sum((counts[i]+BLOCK_ROWS-1)//BLOCK_ROWS for i in range(s))
    total=fixed+2*len(links)+blocks
    if total>max_words:raise ValueError('budget')
    e=p+4;sb=e+2*len(links);qb=sb+s+1;vb=qb+q;ob=vb+2*s;mb=ob+s+1
    out=[0]*total;out[:3]=[e,sb,qb]
    for page,_,_ in links:out[3+page+1]+=1
    for page in range(p):out[4+page]+=out[3+page]
    for i,(_,seq,logical) in enumerate(links):out[e+2*i:e+2*i+2]=[seq,logical]
    for seq,_ in rows:out[sb+seq+1]+=1
    for seq in range(s):out[sb+seq+1]+=out[sb+seq]
    out[qb:qb+q]=[row for _,row in rows]
    cursor=0
    for seq in range(s):
        positions=[spec['positions'][row] for group,row in rows if group==seq]
        out[vb+seq]=int(all(a<=b for a,b in zip(positions,positions[1:])))
        out[vb+s+seq]=max(positions,default=0);out[ob+seq]=cursor
        for start in range(0,len(positions),BLOCK_ROWS):
            out[mb+cursor]=max(positions[start:start+BLOCK_ROWS]);cursor+=1
    out[ob+s]=cursor
    assert mb+cursor==total
    return out


def visible_rows(spec,index,sequence,token,causal=True,enabled=True,counters=None):
    c=Counter() if counters is None else counters
    q=len(spec['positions']);s=len(spec['kv_lengths']);sb=index[1];qb=index[2]
    vb=qb+q;ob=vb+2*s;mb=ob+s+1
    begin=index[sb+sequence];end=index[sb+sequence+1];cursor=begin;mono=False
    if causal and enabled:
        mono=bool(index[vb+sequence])
        if token>index[vb+s+sequence]:cursor=end
        if mono and cursor<end and token>spec['positions'][index[qb+cursor]]:
            right=end
            while cursor<right:
                mid=cursor+(right-cursor)//2;c['binary_probes']+=1
                if spec['positions'][index[qb+mid]]<token:cursor=mid+1
                else:right=mid
    result=[]
    while cursor<end:
        stop=end;scan=True
        if causal and enabled and not mono:
            block=(cursor-begin)//BLOCK_ROWS;stop=min(end,begin+(block+1)*BLOCK_ROWS)
            c['block_probes']+=1;scan=token<=index[mb+index[ob+sequence]+block]
        if not scan:c['skipped_blocks']+=1;cursor=stop;continue
        while cursor<stop:
            row=index[qb+cursor];c['visited_rows']+=1
            if not causal or mono or token<=spec['positions'][row]:result.append(row)
            cursor+=1
    return result


def tensors(spec,mla=False,dtype=None,seed=34):
    import torch
    dtype=torch.float64 if dtype is None else dtype
    g=torch.Generator().manual_seed(seed);q=len(spec['positions']);p=spec['num_pages'];ps=spec['page_size']
    def r(*shape):return (torch.randn(shape,generator=g,dtype=torch.float64)*.15).to(dtype)
    # Deliberately unaligned features and more than one query head per KV head.
    x=r(q,2,5);k=r(p,ps,1,5)
    return (x,r(q,2,3),k,r(p,ps,1,3)) if mla else (x,k,r(p,ps,1,7))


import torch

def pruned_vjp(tensors, grad, needs, *, mla, spec, causal=True, scale=.37):
    """Physical-position ownership using cached row statistics, no autograd."""
    vals=[x.detach().double() for x in tensors];g=grad.double()
    if mla:q,qp,k,kp=vals;v=k
    else:q,k,v=vals;qp=kp=None
    outputs=[torch.zeros_like(x) if need else None for x,need in zip(vals,needs)]
    index=encode(spec);ebase,sbase,qbase=index[:3];p=spec['page_size']
    stats=q.new_zeros(q.shape[0],q.shape[1],3)
    for row,seq in enumerate(spec['sequence_ids']):
        end=spec['kv_lengths'][seq]
        if causal:end=min(end,spec['positions'][row]+1)
        if not end:continue
        for h in range(q.shape[1]):
            kh=h//(q.shape[1]//k.shape[2]);slots=[(spec['block_tables'][seq][i//p],i%p) for i in range(end)]
            kk=torch.stack([k[a,b,kh] for a,b in slots]);vv=torch.stack([v[a,b,kh] for a,b in slots]);scores=kk@q[row,h]
            if mla:pp=torch.stack([kp[a,b,0] for a,b in slots]);scores=scores+pp@qp[row,h]
            scores=scores*scale;maximum=scores.max();ex=(scores-maximum).exp();den=ex.sum();prob=ex/den
            dp=vv@g[row,h];expected=(prob*dp).sum();stats[row,h]=torch.stack([maximum,den,expected])
            ds=prob*(dp-expected)*scale
            if needs[0]:outputs[0][row,h]=ds@kk
            if mla and needs[1]:outputs[1][row,h]=ds@pp
    for page in range(spec['num_pages']):
        for off in range(p):
            for kh in range(k.shape[2]):
                for link in range(index[3+page],index[4+page]):
                    seq,logical=index[ebase+2*link:ebase+2*link+2];token=logical*p+off
                    if token>=spec['kv_lengths'][seq]:continue
                    for row in visible_rows(spec,index,seq,token,causal):
                        group=q.shape[1]//k.shape[2]
                        for h in range(kh*group,(kh+1)*group):
                            maximum,den,expected=stats[row,h]
                            if not den:continue
                            score=q[row,h]@k[page,off,kh]
                            if mla:score=score+qp[row,h]@kp[page,off,0]
                            prob=(score*scale-maximum).exp()/den
                            dp=v[page,off,kh]@g[row,h];ds=prob*(dp-expected)*scale
                            if mla:
                                if needs[2]:outputs[2][page,off,0]+=ds*q[row,h]+prob*g[row,h]
                                if needs[3]:outputs[3][page,off,0]+=ds*qp[row,h]
                            else:
                                if needs[1]:outputs[1][page,off,kh]+=ds*q[row,h]
                                if needs[2]:outputs[2][page,off,kh]+=prob*g[row,h]
    return tuple(None if o is None else o.to(t.dtype) for o,t in zip(outputs,tensors))
