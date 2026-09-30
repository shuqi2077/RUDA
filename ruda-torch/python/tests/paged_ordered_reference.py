"""TEST ONLY inverse scheduling and single-writer gradient reference.

No production imports: this is a CPU mathematical oracle, not a backend.
"""
import torch


def inverse_index(spec):
    pages=spec['num_pages'];size=spec['page_size'];s=len(spec['kv_lengths'])
    links=sorted((page,seq,logical) for seq,table in enumerate(spec['block_tables'])
                 for logical,page in enumerate(table[:(spec['kv_lengths'][seq]+size-1)//size]))
    rows=sorted((seq,row) for row,seq in enumerate(spec['sequence_ids']))
    ebase=3+pages+1;sbase=ebase+2*len(links);qbase=sbase+s+1
    out=[0]*(qbase+len(rows));out[:3]=[ebase,sbase,qbase]
    for page,_,_ in links:out[3+page+1]+=1
    for page in range(pages):out[3+page+1]+=out[3+page]
    for i,(_,seq,logical) in enumerate(links):out[ebase+2*i:ebase+2*i+2]=[seq,logical]
    for seq,_ in rows:out[sbase+seq+1]+=1
    for seq in range(s):out[sbase+seq+1]+=out[sbase+seq]
    out[qbase:]=[row for _,row in rows]
    return out


def ordered_vjp(tensors, grad, needs, *, mla, spec, causal=True, scale=.37):
    """Physical-position ownership using cached row statistics, no autograd."""
    vals=[x.detach().double() for x in tensors];g=grad.double()
    if mla:q,qp,k,kp=vals;v=k
    else:q,k,v=vals;qp=kp=None
    outputs=[torch.zeros_like(x) if need else None for x,need in zip(vals,needs)]
    index=inverse_index(spec);ebase,sbase,qbase=index[:3];p=spec['page_size']
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
                    for ri in range(index[sbase+seq],index[sbase+seq+1]):
                        row=index[qbase+ri]
                        if causal and token>spec['positions'][row]:continue
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
