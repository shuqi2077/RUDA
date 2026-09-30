"""Paired real-device forward+backward timings; never reports CPU surrogate speed."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch
import ruda_torch as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
    p.add_argument('--width',type=int,default=4096);p.add_argument('--rows',type=int,default=8)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--pairs',type=int,default=7)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.width,a.rows,a.iterations,a.pairs,a.warmup)<1:p.error('all dimensions/counts must be positive')
    if not r._training_available or os.environ.get('RUDA_CUDA_COMPILER')!='ptx':
        raise RuntimeError('real RUDA training API 4 and direct PTX are required')
    torch.manual_seed(925);dtype=getattr(torch,a.dtype)
    h=torch.randn(a.rows,a.width,dtype=dtype)
    x=h.to('ruda').requires_grad_();u=torch.randn_like(h).to('ruda').requires_grad_()
    w=torch.ones(a.width).to('ruda').requires_grad_()
    def ref_norm():
        f=x.float();return (f*(f.square().mean(-1,keepdim=True)+1e-5).rsqrt()*w).to(dtype)
    def ref_gate():return torch.nn.functional.silu(x.float()).to(dtype)*u
    cases={'rms_norm':(ref_norm,lambda:r.rms_norm(x,w,eps=1e-5)),
           'silu_mul':(ref_gate,lambda:r.silu_mul(x,u))}
    report={'training_api':4,'base_abi':r._C.abi_version,'torch':torch.__version__,
            'dtype':a.dtype,'rows':a.rows,'width':a.width,'iterations':a.iterations,
            'pairs':a.pairs,'warmup':a.warmup,'timing':'host wall clock including Python autograd and allocations',
            'peak_vram_measured':False,'cases':{}}
    def call(fn):
        x.grad=None;u.grad=None;w.grad=None
        y=fn();y.float().sum().backward();return y
    tol={'float32':5e-4,'float16':6e-3,'bfloat16':5e-2}[a.dtype]
    for name,(before,after) in cases.items():
        # Numerical comparison and downloads are outside timed regions.
        y=call(before);r.synchronize();expected=[y.detach().cpu()]+[None if z.grad is None else z.grad.detach().cpu() for z in (x,u,w)]
        y=call(after);r.synchronize();actual=[y.detach().cpu()]+[None if z.grad is None else z.grad.detach().cpu() for z in (x,u,w)]
        for left,right in zip(actual,expected):
            if left is None or right is None:assert left is right
            else:torch.testing.assert_close(left.float(),right.float(),atol=tol,rtol=tol)
        for _ in range(a.warmup):call(before);call(after)
        r.synchronize();samples={'unfused':[],'fused':[]}
        for i in range(a.pairs):
            order=[('unfused',before),('fused',after)]
            if i%2:order.reverse()
            for label,fn in order:
                r.synchronize();start=time.perf_counter()
                for _ in range(a.iterations):call(fn)
                r.synchronize();samples[label].append((time.perf_counter()-start)*1000/a.iterations)
        report['cases'][name]={'milliseconds_per_forward_backward':samples,
            'median_ms':{k:statistics.median(v) for k,v in samples.items()},'correctness_checked':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
