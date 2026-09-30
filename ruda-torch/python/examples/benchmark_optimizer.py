"""Paired native AdamW timings. Same gradients, no clipping, loss_scale=1.

Only compares the API-1 baseline with read-only/batched API-2; initial states,
updates and warmup counts match. Not an end-to-end model/VRAM benchmark.
"""
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
    p.add_argument('--parameters',type=int,default=32);p.add_argument('--elements',type=int,default=4096)
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--pairs',type=int,default=7)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.parameters,a.elements,a.iterations,a.pairs,a.warmup)<=0 or a.parameters>4096:p.error('positive counts and <=4096 parameters required')
    if not r._training_available or r._C.training_api_version!=4 or os.environ.get('RUDA_CUDA_COMPILER')!='ptx':
        raise RuntimeError('native training API 4 with direct PTX required')
    torch.manual_seed(426);dtype=getattr(torch,a.dtype)
    inputs=[torch.randn(a.elements+(i%3),dtype=dtype) for i in range(a.parameters)]
    grads=[torch.randn_like(x)*.1 for x in inputs]
    params={name:[torch.nn.Parameter(x.to('ruda')) for x in inputs] for name in ('legacy','batched')}
    opts={name:r.AdamW(ps,fused_step=name=='batched') for name,ps in params.items()}
    for name,ps in params.items():
        for x,g in zip(ps,grads):x.grad=g.to('ruda')
    def compare():
        tolerance=5e-4 if dtype==torch.float32 else .01 if dtype==torch.float16 else .06
        for p,q in zip(params['legacy'],params['batched']):
            torch.testing.assert_close(p.cpu().float(),q.cpu().float(),atol=tolerance,rtol=tolerance)
            for key in ('exp_avg','exp_avg_sq'):
                torch.testing.assert_close(opts['legacy'].state[p][key].cpu(),opts['batched'].state[q][key].cpu(),atol=1e-6,rtol=1e-4)
    for opt in opts.values():opt.step()
    r.synchronize();compare()
    for _ in range(a.warmup):
        for opt in opts.values():opt.step()
    r.synchronize();samples={name:[] for name in opts}
    for pair in range(a.pairs):
        order=list(opts)
        if pair%2:order.reverse()
        for name in order:
            r.synchronize();start=time.perf_counter()
            for _ in range(a.iterations):opts[name].step()
            r.synchronize();samples[name].append((time.perf_counter()-start)*1000/a.iterations)
    compare()
    result={'training_api':4,'torch':torch.__version__,'dtype':a.dtype,'parameters':a.parameters,
            'elements_base':a.elements,'pairs':a.pairs,'iterations':a.iterations,'warmup':a.warmup,
            'async':os.environ.get('RUDA_TORCH_ASYNC','0'),'ptx':os.environ.get('RUDA_PTX_VERSION'),
            'loss_scale':1,'clipping':False,'correctness_checked':True,'peak_vram_measured':False,
            'timing':'host wallclock incl Python validation, readback and native submit; initialization excluded',
            'milliseconds_per_step':samples,'median_ms':{k:statistics.median(v) for k,v in samples.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
