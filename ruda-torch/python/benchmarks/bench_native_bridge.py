"""Native PTX bridge comparisons. Measures synchronized end-to-end wall time.
The scalar path is this revision's storage-aware GPU baseline, NOT an old
binary. Peak device memory is intentionally null until measured externally.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ptx-version',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
    args=parser.parse_args()
    if args.repeats<1 or args.warmup<0: parser.error('invalid repetition count')
    if args.output.exists(): parser.error('output already exists; use a fresh result filename')
    os.environ['RUDA_CUDA_COMPILER']='ptx'
    os.environ['RUDA_PTX_VERSION']=args.ptx_version
    import torch
    import ruda_torch
    dtype=getattr(torch,args.dtype)
    generator=torch.Generator().manual_seed(717)
    records=[]
    def measure(name,route,fn):
        for _ in range(args.warmup): fn()
        ruda_torch.synchronize()
        before=ruda_torch.execution_stats()
        samples=[]
        for _ in range(args.repeats):
            start=time.perf_counter_ns()
            result=fn()
            ruda_torch.synchronize()
            samples.append((time.perf_counter_ns()-start)/1e6)
            del result
        after=ruda_torch.execution_stats()
        delta={key:after[key]-before[key] for key in before}
        for key in ['host_to_device_bytes','device_to_host_bytes']:
            if delta[key]: raise RuntimeError(f'{name}: unexpected tensor transfer {key}={delta[key]}')
        records.append({'case':name,'route':route,'median_wall_ms':statistics.median(samples),
                        'min_wall_ms':min(samples),'max_wall_ms':max(samples),
                        'samples_ms':samples,'counters_total_over_repeats':delta,
                        'peak_device_memory_bytes':None})
    for m,k,n in [(1,4096,4096),(64,1024,1024),(17,33,19)]:
        a=torch.randn(m,k,generator=generator).to(dtype).to('ruda')
        b=torch.randn(k,n,generator=generator).to(dtype).to('ruda')
        bias=torch.randn(n,generator=generator).to(dtype).to('ruda')
        for route in ['scalar','auto']:
            os.environ['RUDA_TORCH_MATMUL']=route
            measure(f'mm_{m}_{k}_{n}',route,lambda: torch.mm(a,b))
            measure(f'addmm_{m}_{k}_{n}',route,lambda: torch.addmm(bias,a,b))
        del a,b,bias
    for width in [31,33,1024,8192]:
        a=torch.randn(32,width,generator=generator).to(dtype).to('ruda')
        for route in ['scalar','warp']:
            os.environ['RUDA_TORCH_SOFTMAX']=route
            measure(f'softmax_32_{width}',route,lambda: torch.softmax(a,-1))
        del a
    for width in [128,1024,4096,8192]:
        a=torch.randn(32,width,generator=generator).to(dtype).to('ruda')
        weight=torch.randn(width,generator=generator).to(dtype).to('ruda')
        bias=torch.randn(width,generator=generator).to(dtype).to('ruda')
        measure(f'layer_norm_32_{width}','fused-last-axis',
                lambda: torch.nn.functional.layer_norm(a,(width,),weight,bias))
        measure(f'mean_last_32_{width}','warp-reduction',lambda: a.mean(-1,keepdim=True))
        del a,weight,bias
    # Embedding lookup can opt out of synchronous bounds validation only when
    # the caller guarantees tokenizer/model indices are valid. Measure both.
    vocab,width=32000,4096
    table=torch.randn(vocab,width,generator=generator).to(dtype).to('ruda')
    indices=torch.randint(0,vocab,(128,),generator=generator,dtype=torch.int64).to('ruda')
    for route in ['strict','trusted']:
        os.environ['RUDA_TORCH_INDEX_CHECK']=route
        measure(f'embedding_128_{vocab}_{width}',route,lambda: torch.nn.functional.embedding(indices,table))
    del table,indices
    result={'torch':torch.__version__,'dtype':args.dtype,'compiler':'ptx',
            'ptx_version':args.ptx_version,'warmup':args.warmup,'repeats':args.repeats,
            'timer':'perf_counter_ns plus explicit synchronization (host and GPU overhead)',
            'memory_note':'allocation counters are cumulative requested bytes, not peak device memory',
            'results':records}
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(args.output)

if __name__=='__main__':
    main()
