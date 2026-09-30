#!/usr/bin/env python3
"""Paired RUDA LayerNorm training benchmark; no CPU timing fallback.

Compares the v28 fused first-order path against a device-resident decomposition
using the same input/affine tensors. Results are wall-clock after explicit RUDA
synchronization; correctness is checked outside timed regions.
"""
import argparse, json, statistics, time
import torch
import ruda_torch as r


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',type=int,default=1024);p.add_argument('--width',type=int,default=4096)
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16');p.add_argument('--samples',type=int,default=7);p.add_argument('--warmup',type=int,default=3);p.add_argument('--out')
    a=p.parse_args();
    if a.rows<=0 or a.width<=0 or a.samples<3 or a.warmup<1:raise SystemExit('invalid dimensions/samples/warmup')
    if not r._training_available or r._C.training_api_version!=4:raise RuntimeError('RUDA training API 4 required')
    dtype=getattr(torch,a.dtype);torch.manual_seed(128)
    base=torch.randn(a.rows,a.width,dtype=dtype).to('ruda');weight=torch.randn(a.width,dtype=dtype).to('ruda').requires_grad_();bias=torch.randn(a.width,dtype=dtype).to('ruda').requires_grad_();grad=torch.randn_like(base)

    def fused():
        x=base.detach().requires_grad_();y=r.layer_norm(x,weight,bias,eps=1e-5);y.backward(grad);return y,x.grad
    def staged():
        x=base.detach().requires_grad_();xf=x.float();mu=xf.mean(-1,keepdim=True);center=xf-mu;inv=(center.square().mean(-1,keepdim=True)+1e-5).rsqrt();y=(center*inv*weight.float()+bias.float()).to(dtype);y.backward(grad);return y,x.grad
    for _ in range(a.warmup):fused();r.synchronize();staged();r.synchronize()
    fy,fg=fused();r.synchronize();sy,sg=staged();r.synchronize();torch.testing.assert_close(fy.cpu().float(),sy.cpu().float(),rtol=.05 if dtype==torch.bfloat16 else .008,atol=.05 if dtype==torch.bfloat16 else .008);torch.testing.assert_close(fg.cpu().float(),sg.cpu().float(),rtol=.06 if dtype==torch.bfloat16 else .01,atol=.06 if dtype==torch.bfloat16 else .01)
    samples={'fused':[],'staged':[]}
    for i in range(a.samples):
        order=('fused','staged') if i%2==0 else ('staged','fused')
        for name in order:
            start=time.perf_counter_ns();(fused if name=='fused' else staged)();r.synchronize();samples[name].append((time.perf_counter_ns()-start)/1e6)
    result={'training_api':4,'rows':a.rows,'width':a.width,'dtype':a.dtype,'samples':samples,'median_ms':{k:statistics.median(v) for k,v in samples.items()},'note':'whole forward+backward LayerNorm only; not model throughput'}
    text=json.dumps(result,indent=2);print(text)
    if a.out:open(a.out,'w').write(text+'\n')
if __name__=='__main__':main()
