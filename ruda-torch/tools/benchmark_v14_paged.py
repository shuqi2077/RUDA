#!/usr/bin/env python3
"""Hardware-only paged attention microbenchmark. No CPU operator fallback.
First call includes native compilation/metadata; warm samples use GPU events.
Reported cache bytes are tensor sizes, NOT measured device peak memory.
"""
import argparse,json,math,os,statistics,time
p=argparse.ArgumentParser()
for flag,default in [('context',4096),('queries',1),('heads',32),('kv-heads',8),('dim',128),('page-size',16),('iterations',30)]:p.add_argument('--'+flag,type=int,default=default)
p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='bfloat16')
a=p.parse_args()
if any(x<=0 for x in (a.context,a.queries,a.heads,a.kv_heads,a.dim,a.page_size,a.iterations)) or a.queries>a.context or a.heads%a.kv_heads:p.error('invalid attention dimensions')
if os.environ.get('RUDA_CUDA_COMPILER')!='ptx' or not os.environ.get('RUDA_PTX_VERSION'):p.error('set RUDA_CUDA_COMPILER=ptx and a supported RUDA_PTX_VERSION')
import torch
import ruda_torch as r
assert r._C.abi_version==9
lengths=[a.context,max(a.queries,a.context//3)]
page_counts=[math.ceil(n/a.page_size) for n in lengths];pages=sum(page_counts)
g=torch.Generator().manual_seed(41);order=torch.randperm(pages,generator=g).tolist()
tables=[order[:page_counts[0]],order[page_counts[0]:]]
ids=[seq for seq in range(2) for _ in range(a.queries)];positions=[pos for length in lengths for pos in range(length-a.queries,length)]
dtype=getattr(torch,a.dtype)
q=torch.randn((len(ids),a.heads,a.dim),generator=g).to(dtype).to('ruda')
k=torch.randn((pages,a.page_size,a.kv_heads,a.dim),generator=g).to(dtype).to('ruda');v=k.clone()
plan=r.PagedAttentionPlan(page_size=a.page_size,num_pages=pages,block_tables=tables,kv_lengths=lengths,sequence_ids=ids,positions=positions)
r.synchronize();start=time.perf_counter();out=plan.attention(q,k,v,scale=a.dim**-.5);r.synchronize();first=time.perf_counter()-start
for _ in range(3):out=plan.attention(q,k,v,scale=a.dim**-.5)
r.synchronize();samples=[];host=[];begin=r.Event(enable_timing=True);end=r.Event(enable_timing=True)
for _ in range(a.iterations):
    t=time.perf_counter();begin.record();out=plan.attention(q,k,v,scale=a.dim**-.5);end.record();end.synchronize()
    samples.append(begin.elapsed_time(end));host.append((time.perf_counter()-t)*1000)
begin.close();end.close()
print(json.dumps({'parameters':vars(a),'backend':'native RUDA/PTX ABI9','async_requested':os.environ.get('RUDA_TORCH_ASYNC','0'),
 'first_call_seconds':first,'median_gpu_ms':statistics.median(samples),'median_wall_ms':statistics.median(host),
 'gpu_ms_samples':samples,'cache_tensor_bytes':2*k.numel()*k.element_size(),'peak_device_memory_measured':False,
 'execution_stats':r.execution_stats(),'finite_output':bool(torch.isfinite(out.cpu()).all())},indent=2))
