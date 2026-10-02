#!/usr/bin/env python3
"""Compare equivalent native subgraphs, separating fusion from workspace reuse.

No kernel/whole-model timing claims: this measures synchronized host wall time
per call. Every variant is warmed. Order rotates; setup is reported separately.
Workspace bytes are exact plan accounting, not measured peak device memory.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('v24-optimizer-benchmark.json'))
    p.add_argument('--dtype',choices=('float32','float16','bfloat16'),default='float32')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--pairs',type=int,default=7)
    a=p.parse_args()
    if a.iterations<1 or a.pairs<3:p.error('positive iterations and >=3 repetitions required')
    if a.output.exists():p.error('use a new output file; existing evidence is not overwritten')
    if os.environ.get('RUDA_CUDA_COMPILER')!='ptx':p.error('explicit direct PTX configuration required')
    import torch
    import ruda_torch as r
    if r._C.graph_api_version!=3:raise RuntimeError('rebuild graph API 3 Rust and C++ bridge')
    dtype=getattr(torch,a.dtype)
    results=[]
    configurations={'baseline':(False,False),'fusion':(True,False),
                    'reuse':(False,True),'fusion_and_reuse':(True,True)}
    for workload in ('gate','repeated_gate_norm'):
        for rows,width in ((1,128),(1,4096),(32,4096)):
            torch.manual_seed(246)
            host={'x':torch.randn(rows,width).to(dtype),'up':torch.randn(rows,width).to(dtype),
                  'res':torch.randn(rows,width).to(dtype),'w':torch.ones(width,dtype=dtype)}
            inputs={k:v.to('ruda') for k,v in host.items()}
            nodes=[];previous='x'
            for i in range(1 if workload=='gate' else 6):
                nodes.extend([r.GraphOp.silu(f'a{i}',previous),r.GraphOp.mul(f'p{i}',f'a{i}','up')])
                previous=f'p{i}'
                if workload!='gate':
                    nodes.extend([r.GraphOp.add(f's{i}',previous,'res'),
                                  r.GraphOp.rms_norm(f'y{i}',f's{i}','w')])
                    previous=f'y{i}'
            graphs={};setup={}
            try:
                for name,(opt,reuse) in configurations.items():
                    r.synchronize();start=time.perf_counter()
                    graphs[name]=r.StaticGraph(inputs,nodes,optimize=opt,reuse_workspace=reuse)
                    graphs[name].synchronize();setup[name]=time.perf_counter()-start
                expected=graphs['baseline'].run_eager()[previous].cpu()
                tolerance={'float32':3e-5,'float16':5e-3,'bfloat16':4e-2}[a.dtype]
                for graph in graphs.values():
                    torch.testing.assert_close(graph.replay()[previous].cpu(),expected,
                                               rtol=tolerance,atol=tolerance,equal_nan=True)
                    for _ in range(10):graph.replay()
                    graph.synchronize()
                samples={name:[] for name in graphs}
                names=list(graphs)
                for repeat in range(a.pairs):
                    offset=repeat%len(names)
                    for name in names[offset:]+names[:offset]:
                        graph=graphs[name];graph.synchronize();start=time.perf_counter()
                        for _ in range(a.iterations):graph.replay()
                        graph.synchronize()
                        samples[name].append((time.perf_counter()-start)*1e6/a.iterations)
                results.append({'workload':workload,'rows':rows,'width':width,
                    'setup_seconds':setup,'plans':{k:g.info for k,g in graphs.items()},
                    'samples_us':samples,'median_us':{k:statistics.median(v) for k,v in samples.items()}})
            finally:
                for graph in graphs.values():graph.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'torch':torch.__version__,'dtype':a.dtype,
        'async':os.getenv('RUDA_TORCH_ASYNC','0'),'ptx_version':os.getenv('RUDA_PTX_VERSION'),
        'pairs':a.pairs,'iterations':a.iterations,'results':results,
        'metric':'host wall microseconds per call, including validation and selected sync policy',
        'not_measured':['whole model throughput','peak VRAM','isolated GPU kernel duration']},indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
