#!/usr/bin/env python3
"""Paired native static graph vs preallocated eager: same kernels and final sync.

Reports host wall latency, not isolated kernel time or full-model tokens/s.
Build time reported separately. No weights/model downloads; no CPU fallback.
"""
import argparse,json,os,statistics,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('static-graph-benchmark.json'))
    p.add_argument('--dtype',choices=('float32','float16','bfloat16'),default='float32')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--pairs',type=int,default=7)
    a=p.parse_args()
    if a.iterations<1 or a.pairs<3:p.error('positive iterations and at least 3 pairs required')
    if os.environ.get('RUDA_CUDA_COMPILER')!='ptx':p.error('explicit direct PTX configuration required')
    import torch
    import ruda_torch as r
    dtype=getattr(torch,a.dtype);results=[]
    for rows,width in [(1,128),(1,4096),(32,4096)]:
        torch.manual_seed(41)
        x=torch.randn(rows,width).to(dtype);res=torch.randn_like(x);w=torch.ones(width,dtype=dtype)
        xd,rd,wd=(v.to('ruda') for v in (x,res,w))
        r.synchronize();start=time.perf_counter()
        graph=r.StaticGraph({'x':xd,'r':rd,'w':wd},[r.GraphOp.add('s','x','r'),r.GraphOp.rms_norm('y','s','w')])
        graph.synchronize();setup=time.perf_counter()-start
        try:
            for _ in range(10):graph.run_eager();graph.replay()
            graph.synchronize()
            expected=graph.run_eager()['y'].cpu();actual=graph.replay()['y'].cpu()
            tolerance={'float32':3e-5,'float16':5e-3,'bfloat16':4e-2}[a.dtype]
            torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
            samples={'graph':[],'preallocated_eager':[]}
            for pair in range(a.pairs):
                order=('graph','preallocated_eager') if pair%2==0 else ('preallocated_eager','graph')
                for name in order:
                    fn=graph.replay if name=='graph' else graph.run_eager
                    graph.synchronize();t=time.perf_counter()
                    for _ in range(a.iterations):fn()
                    graph.synchronize();samples[name].append((time.perf_counter()-t)*1e6/a.iterations)
            results.append({'rows':rows,'width':width,'setup_seconds':setup,'plan':graph.info,
                'samples_us':samples,'median_us':{k:statistics.median(v) for k,v in samples.items()}})
        finally:graph.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'torch':torch.__version__,'dtype':a.dtype,
        'async':os.getenv('RUDA_TORCH_ASYNC','0'),'ptx_version':os.getenv('RUDA_PTX_VERSION'),
        'iterations':a.iterations,'pairs':a.pairs,'results':results,
        'metric':'host wall time per subgraph invocation, including host checks and selected sync policy',
        'not_measured':['whole model throughput','peak VRAM','isolated GPU kernel duration']},indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
