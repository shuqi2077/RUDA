"""Real GPU acceptance for API 4; missing hardware fails, never skips.

The two-level boundary test uses about 150 MiB on FP32 devices; it deliberately
covers 1025 distinct gradients. Run in both sync and async modes with sanitizers.
"""
import math
import pytest
import torch
from test_training_gpu import r, DTYPES, close


def buffers(gs):
    from ruda_torch._gradient_stats import statistics_plan
    rows=sum(max(1,min(1024,(g.numel()+31)//32)) for g in gs)
    plan=statistics_plan(rows)
    return (torch.empty(rows*3,device='ruda'),
            torch.empty(plan.scratch_elements,device='ruda'),
            torch.empty(3,device='ruda'))

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('sizes',[(0,33),(32768,),(32768,1),(32768,32769)])
@pytest.mark.parametrize('with_norm',[False,True])
def test_hierarchical_analysis(r,dtype,sizes,with_norm):
    torch.manual_seed(527)
    hs=[torch.randn(n,dtype=dtype) for n in sizes]
    gs=[h.to('ruda') for h in hs];w,s,report=buffers(gs)
    old=torch.empty(3,device='ruda');versions=[g._version for g in gs]
    before=r.execution_stats()
    with torch.no_grad():
        r._C.training_analyze_hierarchical(gs,.125,w,s,report,with_norm)
        r._C.training_analyze(gs,.125,w,old,with_norm)
    r.synchronize();after=r.execution_stats()
    for key in ('host_to_device_bytes','device_to_host_bytes'):
        assert before[key]==after[key]
    torch.testing.assert_close(report.cpu(),old.cpu(),atol=2e-5,rtol=4e-5)
    bad,scale,squares=report.cpu().tolist();assert bad==0
    expected=torch.linalg.vector_norm(torch.cat([h.float().flatten().double()*.125 for h in hs])).item()
    if with_norm:assert scale*math.sqrt(squares)==pytest.approx(expected,rel=4e-5)
    else:assert scale==squares==0
    for g,h,v in zip(gs,hs,versions):
        torch.testing.assert_close(g.cpu(),h,atol=0,rtol=0);assert g._version==v

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_hierarchical_invalid_tail(r,dtype,bad):
    ps=[torch.nn.Parameter(torch.ones(n,dtype=dtype).to('ruda')) for n in (32768,33)]
    for p in ps:p.grad=torch.ones(p.shape,dtype=dtype).to('ruda')
    h=torch.ones(33,dtype=dtype);h[-1]=bad;ps[-1].grad=h.to('ruda')
    opt=r.AdamW(ps,fused_step=True,hierarchical_stats=True,max_grad_norm=1.)
    opt.step();r.synchronize()
    assert opt.last_step_skipped and not opt.state
    for p in ps:torch.testing.assert_close(p.cpu(),torch.ones(p.shape,dtype=dtype),atol=0,rtol=0)

@pytest.mark.parametrize('dtype',DTYPES)
def test_hierarchical_optimizer_matches_baseline(r,dtype):
    hs=[torch.ones(n,dtype=dtype) for n in (32768,33)]
    ps=[torch.nn.Parameter(h.to('ruda')) for h in hs]
    qs=[torch.nn.Parameter(h.to('ruda')) for h in hs]
    a=r.AdamW(ps,fused_step=True,hierarchical_stats=True,max_grad_norm=.5)
    b=r.AdamW(qs,fused_step=True,max_grad_norm=.5)
    for i in range(3):
        for p,q,h in zip(ps,qs,hs):
            grad=torch.arange(h.numel()).remainder(13).to(dtype)*.125
            p.grad=grad.to('ruda');q.grad=grad.to('ruda')
        a.step(loss_scale=8);b.step(loss_scale=8);r.synchronize()
        assert a.last_grad_norm==pytest.approx(b.last_grad_norm,rel=4e-5)
        for p,q in zip(ps,qs):
            close(p,q,dtype)
            for key in ('exp_avg','exp_avg_sq'):
                torch.testing.assert_close(a.state[p][key].cpu(),b.state[q][key].cpu(),atol=1e-7,rtol=1e-4)

@pytest.mark.parametrize('dtype',DTYPES)
def test_hierarchy_workspace_reuse_and_resume(r,dtype):
    ps=[torch.nn.Parameter(torch.ones(n,dtype=dtype).to('ruda')) for n in (32768,33)]
    opt=r.AdamW(ps,fused_step=True,hierarchical_stats=True)
    for p in ps:p.grad=torch.ones(p.shape,dtype=dtype).to('ruda')
    opt.step();ptr=opt._reduction_scratch.data_ptr();ps[-1].grad=None;opt.step()
    assert ptr==opt._reduction_scratch.data_ptr()
    saved=opt.state_dict();opt.load_state_dict(saved)
    assert opt.hierarchical_stats and opt._reduction_scratch is None
    opt.step();r.synchronize()


def test_hierarchical_stream_lifetime(r):
    stream=r.Stream()
    with r.stream(stream):
        gs=[torch.ones(n).to('ruda') for n in (32768,1)];w,s,report=buffers(gs)
        with torch.no_grad():r._C.training_analyze_hierarchical(gs,1.,w,s,report,True)
        del gs,w,s
        pressure=[torch.empty(32768,device='ruda') for _ in range(8)]
        stream.synchronize()
        bad,magnitude,squares=report.cpu().tolist()
        assert bad==0 and magnitude*math.sqrt(squares)==pytest.approx(math.sqrt(32769),rel=4e-5)


def test_two_hierarchy_levels(r):
    # Distinct read buffers satisfy the native no-overlap contract.
    h=torch.ones(32768)
    gs=[h.to('ruda') for _ in range(1025)]
    w,s,report=buffers(gs)
    assert s.numel()==3081
    with torch.no_grad():r._C.training_analyze_hierarchical(gs,1.,w,s,report,True)
    r.synchronize();bad,magnitude,squares=report.cpu().tolist()
    assert bad==0 and magnitude*math.sqrt(squares)==pytest.approx(math.sqrt(1025*32768),rel=4e-5)

@pytest.mark.parametrize('magnitude',[1e-30,1e30])
def test_hierarchical_extreme_finite(r,magnitude):
    gs=[torch.full((n,),magnitude).to('ruda') for n in (32768,33)]
    w,s,report=buffers(gs)
    with torch.no_grad():r._C.training_analyze_hierarchical(gs,1.,w,s,report,True)
    r.synchronize();bad,scale,squares=report.cpu().tolist()
    assert bad==0 and math.isfinite(squares)
    assert scale*math.sqrt(squares)==pytest.approx(magnitude*math.sqrt(32801),rel=4e-5)
