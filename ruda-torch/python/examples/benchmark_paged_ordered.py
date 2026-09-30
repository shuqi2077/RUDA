"""Paired REAL GPU backward benchmark. Includes output allocation and synchronization.

Both strategies use the same resident input tensors; forward is outside timing.
This is not whole-model throughput, not a peak-memory measurement, and never
falls back to CPU. Outputs/gradients are checked separately before timing.
"""
import argparse,json,statistics,time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries',type=int,default=32);p.add_argument('--length',type=int,default=1024)
    p.add_argument('--heads',type=int,default=8);p.add_argument('--kv-heads',type=int,default=2)
    p.add_argument('--dim',type=int,default=64);p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
    p.add_argument('--samples',type=int,default=7);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--mla',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not (0<a.queries<=a.length and a.length%16==0 and a.heads>0 and a.kv_heads>0 and a.heads%a.kv_heads==0
            and 0<a.dim<=1024 and a.samples>=3 and a.warmup>=1):p.error('invalid dimensions/repetition counts')
    if a.output.exists():p.error('refuse to overwrite benchmark evidence')
    import ruda_torch as r
    if torch.are_deterministic_algorithms_enabled():raise RuntimeError('atomic comparison requires deterministic mode disabled')
    dtype=getattr(torch,a.dtype);torch.manual_seed(803)
    def rand(*shape):return (torch.randn(shape)*.15).to(dtype).to('ruda').requires_grad_()
    pages=a.length//16;kh=1 if a.mla else a.kv_heads
    q=rand(a.queries,a.heads,a.dim);k=rand(pages,16,kh,a.dim)
    tensors=(q,rand(a.queries,a.heads,32),k,rand(pages,16,1,32)) if a.mla else (q,k,rand(pages,16,kh,a.dim))
    spec=dict(page_size=16,num_pages=pages,block_tables=[list(range(pages))],kv_lengths=[a.length],
              sequence_ids=[0]*a.queries,positions=list(range(a.length-a.queries,a.length)))
    plans={mode:r.PagedAttentionPlan(**spec,backward_strategy=mode) for mode in ['atomic','ordered']}
    grad=torch.full((a.queries,a.heads,a.dim),.2,dtype=dtype).to('ruda')
    def once(mode,read=False):
        for t in tensors:t.grad=None
        plan=plans[mode];y=(plan.mla if a.mla else plan.attention)(*tensors,scale=a.dim**-.5)
        r.synchronize();start=time.perf_counter();y.backward(grad);r.synchronize();elapsed=time.perf_counter()-start
        result=[t.grad.detach().cpu() for t in tensors] if read else None
        return elapsed,result
    _,ref=once('atomic',True);_,candidate=once('ordered',True)
    tolerance=8e-4 if dtype==torch.float32 else (.01 if dtype==torch.float16 else .05)
    for x,y in zip(candidate,ref):torch.testing.assert_close(x.float(),y.float(),rtol=tolerance,atol=tolerance)
    for i in range(a.warmup):
        for mode in ('atomic','ordered') if i%2==0 else ('ordered','atomic'):once(mode)
    times={mode:[] for mode in plans};before=r.execution_stats()
    for i in range(a.samples):
        for mode in ('atomic','ordered') if i%2==0 else ('ordered','atomic'):times[mode].append(once(mode)[0])
    after=r.execution_stats();result={'version':'v33','scope':'backward incl. outputs and sync; forward excluded; not full training',
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'samples_seconds':times,
        'median_seconds':{k:statistics.median(v) for k,v in times.items()},
        'counter_deltas':{k:after[k]-before[k] for k in after},'reference_checked':True,'peak_memory_measured':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
