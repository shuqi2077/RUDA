"""Same-device forward+backward samples. Only valid indices are compared.

The decomposed gather can have strict bounds-check overhead; the native path's
invalid-index contract is bounded NaNs. Thus this is NOT an equivalence claim
for invalid inputs or an end-to-end MoE/model throughput benchmark.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import ruda_torch as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=128);p.add_argument('--experts',type=int,default=256)
    p.add_argument('--top-k',type=int,default=8);p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--samples',type=int,default=7);p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
    a=p.parse_args()
    if a.tokens<1 or not 1<=a.top_k<=min(a.experts,64) or a.iterations<1 or a.samples<2:p.error('invalid benchmark dimensions')
    g=torch.Generator().manual_seed(309);dtype=getattr(torch,a.dtype)
    host=torch.randn(a.tokens,a.experts,generator=g).to(dtype)
    ids=torch.stack([torch.randperm(a.experts,generator=g)[:a.top_k] for _ in range(a.tokens)]).to('ruda')
    upstream=torch.randn(a.tokens,a.top_k,generator=g).to('ruda')
    x=host.to('ruda').requires_grad_()
    def native():return r.selected_router_weights(x,ids,scoring='softmax',renormalize=True,scale=2.5)
    def decomposed():
        prob=x.float().softmax(-1);selected=prob.gather(1,ids)
        return selected/selected.sum(-1,keepdim=True)*2.5
    def execute(fn):
        y=fn();dx=torch.autograd.grad(y,x,upstream)[0]
        return y,dx
    # Compile and validate outside timing.
    for _ in range(5):execute(native);execute(decomposed)
    r.synchronize();yn,dn=execute(native);yd,dd=execute(decomposed);r.synchronize()
    tol=3e-4 if dtype==torch.float32 else .05
    torch.testing.assert_close(yn.cpu(),yd.cpu(),atol=tol,rtol=tol)
    torch.testing.assert_close(dn.cpu(),dd.cpu(),atol=tol,rtol=tol)
    samples={'native':[],'decomposed':[]}
    for repeat in range(a.samples):
        order=[('native',native),('decomposed',decomposed)]
        if repeat%2:order.reverse()
        for label,fn in order:
            r.synchronize();start=time.perf_counter()
            for _ in range(a.iterations):execute(fn)
            r.synchronize();samples[label].append((time.perf_counter()-start)/a.iterations)
    config=vars(a).copy();config['output']=str(a.output)
    result={'scope':'valid-index router forward+backward, not model throughput','configuration':config,
            'seconds_samples':samples,'median_seconds':{k:statistics.median(v) for k,v in samples.items()},
            'peak_memory_measured':False,'torch_version':torch.__version__}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
