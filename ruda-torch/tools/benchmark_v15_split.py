#!/usr/bin/env python3
"""Same-device unsplit/split attention comparison. No CPU operator fallback.
No results are emitted before numerical consistency checks succeed.
"""
import argparse
import json
import math
import os
import statistics
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag,default in [('context',4096),('queries',1),('heads',32),('kv-heads',8),('dim',128),('page-size',16),('iterations',20)]:
        p.add_argument('--'+flag,type=int,default=default)
    p.add_argument('--splits',type=int,nargs='+',default=[1,2,4,8,16])
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='bfloat16')
    p.add_argument('--mla',action='store_true',help='compressed rank = --dim, positional dim 64, cache head count 1')
    a=p.parse_args()
    if min(a.context,a.queries,a.heads,a.kv_heads,a.dim,a.page_size,a.iterations)<=0 or a.queries>a.context or a.heads%a.kv_heads:
        p.error('invalid dimensions')
    if a.dim>1024 or any(not 1<=x<=32 for x in a.splits):p.error('unsupported dimension/split count')
    if os.environ.get('RUDA_CUDA_COMPILER')!='ptx' or not os.environ.get('RUDA_PTX_VERSION'):
        p.error('set RUDA_CUDA_COMPILER=ptx and RUDA_PTX_VERSION')
    import torch
    import ruda_torch as r
    assert r._C.abi_version==9
    dtype=getattr(torch,a.dtype);g=torch.Generator().manual_seed(315)
    lengths=[a.context,max(a.queries,a.context//3)];counts=[math.ceil(n/a.page_size) for n in lengths]
    pages=sum(counts);order=torch.randperm(pages,generator=g).tolist();tables=[order[:counts[0]],order[counts[0]:]]
    ids=[s for s in range(2) for _ in range(a.queries)];pos=[j for n in lengths for j in range(n-a.queries,n)]
    def rand(shape):return torch.randn(shape,generator=g).to(dtype).to('ruda')
    q=rand((len(ids),a.heads,a.dim));kh=1 if a.mla else a.kv_heads
    k=rand((pages,a.page_size,kh,a.dim));v=k if a.mla else rand((pages,a.page_size,kh,a.dim))
    qp=rand((len(ids),a.heads,64)) if a.mla else None;kp=rand((pages,a.page_size,1,64)) if a.mla else None
    scale=192**-.5 if a.mla else a.dim**-.5
    splits=list(dict.fromkeys([1]+a.splits))
    plans={s:r.PagedAttentionPlan(page_size=a.page_size,num_pages=pages,block_tables=tables,
        kv_lengths=lengths,sequence_ids=ids,positions=pos,splits=s) for s in splits}
    def call(s):
        return plans[s].mla(q,qp,k,kp,scale=scale) if a.mla else plans[s].attention(q,k,v,scale=scale)
    first={};outputs={}
    for s in splits:
        # Reject oversized workspaces before compilation/timing.
        plans[s].workspace_bytes(a.heads,a.dim)
        r.synchronize();start=time.perf_counter();outputs[s]=call(s);r.synchronize();first[s]=(time.perf_counter()-start)*1000
    reference=outputs[1].cpu();tol={'float32':5e-5,'float16':5e-3,'bfloat16':3e-2}[a.dtype]
    for s in splits:
        actual=outputs[s].cpu()
        if not bool(torch.isfinite(actual).all()):raise RuntimeError('non-finite GPU output; refusing performance report')
        torch.testing.assert_close(actual,reference,rtol=tol,atol=tol)
        for _ in range(3):outputs[s]=call(s)
    r.synchronize();before=r.execution_stats();gpu={s:[] for s in splits};wall={s:[] for s in splits}
    begin=r.Event(enable_timing=True);end=r.Event(enable_timing=True)
    try:
        for iteration in range(a.iterations):
            # Rotate order so a fixed path is not always measured first.
            order=splits[iteration%len(splits):]+splits[:iteration%len(splits)]
            for s in order:
                start=time.perf_counter();begin.record();outputs[s]=call(s);end.record();end.synchronize()
                gpu[s].append(begin.elapsed_time(end));wall[s].append((time.perf_counter()-start)*1000)
    finally:begin.close();end.close()
    after=r.execution_stats();base=statistics.median(gpu[1])
    rows=[{'splits':s,'first_call_ms':first[s],'median_gpu_ms':statistics.median(gpu[s]),
           'median_wall_ms':statistics.median(wall[s]),'unsplit_over_split_gpu_ratio':base/statistics.median(gpu[s]),
           'workspace_bytes':plans[s].workspace_bytes(a.heads,a.dim),'gpu_ms_samples':gpu[s]} for s in splits]
    cache_bytes=k.numel()*k.element_size()+(kp.numel()*kp.element_size() if a.mla else v.numel()*v.element_size())
    print(json.dumps({'parameters':vars(a),'backend':'native RUDA/PTX ABI9','async_requested':os.environ.get('RUDA_TORCH_ASYNC','0'),
        'rows':rows,'cache_tensor_bytes':cache_bytes,'peak_device_memory_measured':False,
        'measured_loop_counter_deltas':{key:after[key]-before[key] for key in after},
        'comparison':'GPU split vs GPU unsplit; run the GPU acceptance suite for independent dense correctness'},indent=2))

if __name__=='__main__':main()
