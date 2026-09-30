"""New API-2 kernels on real RUDA GPUs; no skips or CPU execution substitute.

CPU AdamW below is an independent reference only. Every tested candidate update
uses native ruda:0 tensors. Run with the strict validate_training.py wrapper.
"""
import copy
import math
import pytest
import torch
from test_training_gpu import r, DTYPES, close

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('limit',[None,0.,.5])
def test_fused_optimizer_fp32_reference(r,dtype,limit):
    torch.manual_seed(126)
    originals=[torch.randn(n,dtype=dtype) for n in (1,33,4097)]
    ps=[torch.nn.Parameter(x.to('ruda')) for x in originals]
    qs=[torch.nn.Parameter(x.float().clone()) for x in originals]
    groups=lambda xs:[{'params':xs[:2],'lr':.001},{'params':xs[2:],'lr':.002}]
    opt=r.AdamW(groups(ps),fused_step=True,max_grad_norm=limit)
    ref=torch.optim.AdamW(groups(qs),foreach=False)
    for _ in range(4):
        grads=[torch.randn_like(x)*16 for x in originals]
        for p,q,g in zip(ps,qs,grads):p.grad=g.to('ruda');q.grad=g.float()/16
        norm=torch.linalg.vector_norm(torch.cat([q.grad.double() for q in qs])).item()
        clip=min(1.,limit/(norm+1e-6)) if limit is not None else 1.
        for q in qs:q.grad.mul_(clip)
        versions=[p.grad._version for p in ps]
        opt.step(loss_scale=16);ref.step();r.synchronize()
        assert not opt.last_step_skipped
        if limit is not None:assert opt.last_grad_norm==pytest.approx(norm,rel=3e-5)
        for p,q,g,version in zip(ps,qs,grads,versions):
            close(p,q,dtype);torch.testing.assert_close(p.grad.cpu(),g,atol=0,rtol=0)
            assert p.grad._version==version
        for p,q in zip(ps,qs):
            torch.testing.assert_close(opt.state[p]['exp_avg'].cpu(),ref.state[q]['exp_avg'],atol=2e-6,rtol=4e-4)
            torch.testing.assert_close(opt.state[p]['exp_avg_sq'].cpu(),ref.state[q]['exp_avg_sq'],atol=2e-7,rtol=4e-4)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('initialized',[False,True])
def test_fused_optimizer_whole_step_overflow(r,dtype,bad,initialized):
    ps=[torch.nn.Parameter(torch.ones(n,dtype=dtype).to('ruda')) for n in (33,65)]
    opt=r.AdamW(ps,fused_step=True,max_grad_norm=1.)
    if initialized:
        for p in ps:p.grad=torch.ones(p.shape,dtype=dtype).to('ruda')
        opt.step()
    before=[p.detach().cpu() for p in ps]
    saved=[(opt.state[p]['step'],opt.state[p]['exp_avg'].cpu(),opt.state[p]['exp_avg_sq'].cpu()) for p in ps] if initialized else []
    ps[0].grad=torch.ones(33,dtype=dtype).to('ruda');ps[1].grad=torch.full((65,),bad,dtype=dtype).to('ruda')
    versions=[p.grad._version for p in ps];opt.step(loss_scale=8)
    assert opt.last_step_skipped
    for i,p in enumerate(ps):
        torch.testing.assert_close(p.cpu(),before[i],rtol=0,atol=0)
        assert p.grad._version==versions[i]
        if initialized:
            assert opt.state[p]['step']==saved[i][0]
            torch.testing.assert_close(opt.state[p]['exp_avg'].cpu(),saved[i][1],rtol=0,atol=0)
            torch.testing.assert_close(opt.state[p]['exp_avg_sq'].cpu(),saved[i][2],rtol=0,atol=0)
    if not initialized:assert not opt.state

@pytest.mark.parametrize('dtype',DTYPES)
def test_fused_optimizer_accumulation_and_checkpoint(r,dtype):
    p=torch.nn.Parameter(torch.ones(31,dtype=dtype).to('ruda'))
    opt=r.AdamW([p],fused_step=True,max_grad_norm=.3);scaler=r.GradScaler(init_scale=8,growth_interval=2)
    for _ in range(2):scaler.scale(p.float().sum()/2).backward()
    gradient=p.grad.cpu();scaler.step(opt);scaler.update()
    assert opt.last_grad_norm==pytest.approx(math.sqrt(31),rel=3e-5)
    torch.testing.assert_close(p.grad.cpu(),gradient,atol=0,rtol=0)
    saved=copy.deepcopy(opt.state_dict());q=torch.nn.Parameter(p.detach().clone());other=r.AdamW([q]);other.load_state_dict(saved)
    assert other.fused_step and other.max_grad_norm==.3
    for x in (p,q):x.grad=torch.full(x.shape,2.,dtype=dtype).to('ruda')
    opt.step();other.step();r.synchronize();torch.testing.assert_close(p.cpu(),q.cpu(),rtol=0,atol=0)

@pytest.mark.parametrize('dtype',DTYPES)
def test_fused_optimizer_missing_grad_reuses_workspace(r,dtype):
    ps=[torch.nn.Parameter(torch.ones(n,dtype=dtype).to('ruda')) for n in (7,2048)]
    opt=r.AdamW(ps,fused_step=True)
    for p in ps:p.grad=torch.ones(p.shape,dtype=dtype).to('ruda')
    opt.step();scratch=opt._analysis_scratch[0].data_ptr();ps[1].grad=None;opt.step()
    assert scratch==opt._analysis_scratch[0].data_ptr() and opt.state[ps[1]]['step']==1


def test_fused_optimizer_extreme_finite_norm_not_square_overflow(r):
    for value in (1e-30,1e20,3e38):
        p=torch.nn.Parameter(torch.ones(4).to('ruda'));p.grad=torch.full((4,),value).to('ruda')
        opt=r.AdamW([p],fused_step=True,max_grad_norm=1.);opt.step()
        assert not opt.last_step_skipped and opt.last_grad_norm==pytest.approx(2*value,rel=3e-5)
        assert torch.isfinite(p.cpu()).all()


def test_fused_optimizer_half_underflow_keeps_fp32_moments(r):
    p=torch.nn.Parameter(torch.ones(5,dtype=torch.float16).to('ruda'))
    p.grad=torch.full((5,),2**-14,dtype=torch.float16).to('ruda')
    opt=r.AdamW([p],fused_step=True);opt.step(loss_scale=65536.)
    assert torch.all(opt.state[p]['exp_avg'].cpu()>0)
    torch.testing.assert_close(p.grad.cpu(),torch.full((5,),2**-14,dtype=torch.float16),rtol=0,atol=0)


def test_fused_optimizer_stream_and_pending_buffers(r):
    stream=r.Stream()
    with r.stream(stream):
        ps=[torch.nn.Parameter(torch.ones(257).to('ruda')) for _ in range(4)]
        for p in ps:p.grad=torch.ones(257).to('ruda')
        opt=r.AdamW(ps,fused_step=True,max_grad_norm=1.);opt.step()
        result=ps[0];del opt,ps
        pressure=[torch.empty(4096,device='ruda') for _ in range(8)]
        stream.synchronize();assert torch.isfinite(result.cpu()).all()
