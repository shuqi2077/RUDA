"""Paired GPU gradient-statistics timings, NOT whole-step or model timings.

Same preallocated inputs and outputs, equal warmup, alternating run order, raw
samples. Requires native API 4 with direct PTX; no CPU fallback. Allocation,
state initialization and CPU reference checks are outside timed regions.
"""
import argparse,json,math,os,statistics,time
from pathlib import Path
import torch
import ruda_torch as r
from ruda_torch._gradient_stats import statistics_plan

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
    p.add_argument('--parameters',type=int,default=64);p.add_argument('--elements',type=int,default=32768)
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--pairs',type=int,default=7)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--without-norm',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.parameters,a.elements,a.iterations,a.pairs,a.warmup)<=0 or a.parameters>4096:p.error('positive counts; <=4096 parameters')
    if not r._training_available or r._C.training_api_version!=4 or os.environ.get('RUDA_CUDA_COMPILER')!='ptx':
        raise RuntimeError('native training API 4/direct PTX required')
    torch.manual_seed(527);dtype=getattr(torch,a.dtype)
    hs=[torch.randn(a.elements+i%3,dtype=dtype) for i in range(a.parameters)]
    gs=[h.to('ruda') for h in hs]
    rows=sum(max(1,min(1024,(g.numel()+31)//32)) for g in gs);plan=statistics_plan(rows)
    workspace=torch.empty(rows*3,device='ruda');scratch=torch.empty(plan.scratch_elements,device='ruda')
    reports={name:torch.empty(3,device='ruda') for name in ('serial_merge','hierarchical_merge')}
    def run(name):
        if name=='serial_merge':r._C.training_analyze(gs,.125,workspace,reports[name],not a.without_norm)
        else:r._C.training_analyze_hierarchical(gs,.125,workspace,scratch,reports[name],not a.without_norm)
    for _ in range(a.warmup):
        for name in reports:run(name)
    r.synchronize()
    expected=math.sqrt(sum(float((h.float().double()*.125).square().sum()) for h in hs))
    for report in reports.values():
        bad,scale,squares=report.cpu().tolist()
        if bad or (not a.without_norm and not math.isclose(scale*math.sqrt(squares),expected,rel_tol=4e-5)):
            raise RuntimeError('GPU statistics mismatch before timing')
    samples={name:[] for name in reports}
    for pair in range(a.pairs):
        order=list(reports)
        if pair%2:order.reverse()
        for name in order:
            r.synchronize();start=time.perf_counter()
            for _ in range(a.iterations):run(name)
            r.synchronize();samples[name].append((time.perf_counter()-start)*1000/a.iterations)
    torch.testing.assert_close(reports['serial_merge'].cpu(),reports['hierarchical_merge'].cpu(),rtol=4e-5,atol=2e-5)
    data={'version':'v27','dtype':a.dtype,'parameters':a.parameters,'elements_base':a.elements,
        'statistics_rows':rows,'hierarchy_stages':len(plan.stages),'scratch_bytes':plan.scratch_elements*4,
        'async':os.environ.get('RUDA_TORCH_ASYNC','0'),'ptx':os.environ.get('RUDA_PTX_VERSION'),
        'torch':torch.__version__,'with_norm':not a.without_norm,'correctness_checked':True,
        'warmup':a.warmup,'iterations':a.iterations,'pairs':a.pairs,
        'timing_scope':'gradient analysis only, host wallclock incl native validation and submit',
        'peak_vram_measured':False,'milliseconds_per_call':samples,
        'median_ms':{k:statistics.median(v) for k,v in samples.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
