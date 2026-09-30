"""Explicit CPU references for Python optimizer contracts, NOT GPU/Rust tests."""
import copy
import math
import types
import pytest
import torch
from test_training_host import mod

@pytest.fixture
def fused(mod):
    m, ref = mod
    ref.batch_calls = 0
    ref.analysis_calls = 0
    def analyze(gs, inverse, workspace, report, with_norm):
        ref.analysis_calls += 1
        values = [g.float() * inverse for g in gs]
        bad = any(not torch.isfinite(g).all() or not torch.isfinite(v).all() for g,v in zip(gs,values))
        if bad:
            report.copy_(torch.tensor([1.,0.,0.])); return
        if with_norm:
            flat = torch.cat([v.reshape(-1).double() for v in values])
            scale = flat.abs().max().item() if flat.numel() else 0.
            squares = ((flat/scale)**2).sum().item() if scale else 0.
            report.copy_(torch.tensor([0.,scale,squares]))
        else: report.zero_()
    def batch(ps,gs,masters,ms,vs,hypers,inverse,clip):
        ref.batch_calls += 1
        if ref.fail: raise RuntimeError('explicit host batch failure')
        for p,g,master,mt,vt,h in zip(ps,gs,masters,ms,vs,hypers):
            lr,b1,b2,eps,decay,c1,c2=h
            grad=(g.float()*inverse)*clip
            mt.mul_(b1).add_(grad,alpha=1-b1)
            vt.mul_(b2).addcmul_(grad,grad,value=1-b2)
            master.mul_(1-lr*decay).addcdiv_(mt/c1,(vt/c2).sqrt()+eps,value=-lr)
            p.copy_(master.to(p.dtype))
    ref.training_analyze = analyze
    ref.training_adamw_batch_ = batch
    return m,ref

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('limit',[None,0.,.5,100.])
@pytest.mark.parametrize('scale',[1.,128.])
def test_clipped_update_against_fp32_adamw(fused,dtype,limit,scale):
    m,ref=fused;torch.manual_seed(772)
    ps=[torch.nn.Parameter(torch.randn(n,dtype=dtype)) for n in (7,33,257)]
    qs=[torch.nn.Parameter(p.detach().float().clone()) for p in ps]
    opt=m.AdamW([{'params':ps[:2],'lr':.01},{'params':ps[2:],'lr':.02}],betas=(.8,.9),fused_step=True,max_grad_norm=limit)
    control=torch.optim.AdamW([{'params':qs[:2],'lr':.01},{'params':qs[2:],'lr':.02}],betas=(.8,.9),foreach=False)
    for _ in range(4):
        gs=[(torch.randn_like(p)*scale) for p in ps]
        for p,g,q in zip(ps,gs,qs):p.grad=g.clone();q.grad=g.float()/scale
        if limit is not None:
            norm=torch.linalg.vector_norm(torch.cat([q.grad.double() for q in qs])).item()
            coefficient=float(torch.tensor(min(1.,limit/(norm+1e-6))))
            for q in qs:q.grad.mul_(coefficient)
        before=[p.grad.clone() for p in ps];versions=[p.grad._version for p in ps]
        opt.step(loss_scale=scale);control.step()
        for p,q,g,version in zip(ps,qs,before,versions):
            torch.testing.assert_close(p,q.to(dtype),atol=2e-6 if dtype==torch.float32 else .008,rtol=2e-6 if dtype==torch.float32 else .008)
            torch.testing.assert_close(p.grad,g,atol=0,rtol=0)
            assert p.grad._version==version
        if limit is not None: assert opt.last_grad_norm==pytest.approx(norm,rel=2e-7)
        else:assert opt.last_grad_norm is None
    assert ref.batch_calls==4 and ref.analysis_calls==4

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('initialized',[False,True])
def test_overflow_changes_nothing_and_allocates_no_initial_state(fused,bad,initialized):
    m,ref=fused;ps=[torch.nn.Parameter(torch.ones(n)) for n in (5,17)]
    opt=m.AdamW(ps,fused_step=True,max_grad_norm=1.)
    if initialized:
        for p in ps:p.grad=torch.ones_like(p)
        opt.step()
    saved=copy.deepcopy(opt.state_dict());params=[p.detach().clone() for p in ps]
    calls=ref.batch_calls
    ps[0].grad=torch.ones_like(ps[0]);ps[1].grad=torch.full_like(ps[1],bad)
    grads=[p.grad.clone() for p in ps];versions=[p.grad._version for p in ps]
    opt.step(loss_scale=8.)
    assert opt.last_step_skipped and calls==ref.batch_calls
    for p,expected,g,version in zip(ps,params,grads,versions):
        torch.testing.assert_close(p,expected,rtol=0,atol=0)
        torch.testing.assert_close(p.grad,g,equal_nan=True,rtol=0,atol=0)
        assert p.grad._version==version
    for key,value in saved['state'].items():
        now=opt.state_dict()['state'][key]
        for k,v in value.items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(now[k],v,rtol=0,atol=0)
            else:assert now[k]==v
    if not initialized:assert not opt.state

@pytest.mark.parametrize('magnitude',[1e-30,1.,1e20,3e38])
def test_stable_norm_large_small_and_zero(fused,magnitude):
    m,_=fused;p=torch.nn.Parameter(torch.ones(4));p.grad=torch.full_like(p,magnitude)
    opt=m.AdamW([p],fused_step=True,max_grad_norm=1.);opt.step()
    assert opt.last_grad_norm==pytest.approx(2*float(p.grad[0]),rel=1e-7)
    assert not opt.last_step_skipped and torch.isfinite(p).all()


def test_avoids_low_precision_unscale_underflow(fused):
    m,_=fused;p=torch.nn.Parameter(torch.ones(5,dtype=torch.float16));p.grad=torch.full_like(p,2**-14)
    opt=m.AdamW([p],fused_step=True);opt.step(loss_scale=65536.)
    assert torch.all(opt.state[p]['exp_avg']>0)
    assert torch.all((p.grad/65536.)==0)  # API 1 would store zero in FP16.
    assert torch.all(p.grad==2**-14)


def test_scratch_grows_and_reuses_for_missing_gradients(fused):
    m,_=fused;a=torch.nn.Parameter(torch.ones(33));b=torch.nn.Parameter(torch.ones(2048))
    opt=m.AdamW([a,b],fused_step=True)
    a.grad=torch.ones_like(a);opt.step();first=opt._analysis_scratch[0]
    b.grad=torch.ones_like(b);opt.step();large=opt._analysis_scratch[0]
    assert large.numel()>first.numel()
    b.grad=None;opt.step();assert opt._analysis_scratch[0] is large
    assert opt.state[b]['step']==1


def test_scaler_accumulation_clip_and_checkpoint(fused):
    m,_=fused;p=torch.nn.Parameter(torch.ones(7));opt=m.AdamW([p],fused_step=True,max_grad_norm=.4)
    scaler=m.GradScaler(init_scale=8,growth_interval=1)
    for _ in range(2):scaler.scale(p.sum()/2).backward()
    grad=p.grad.clone();scaler.step(opt);scaler.update()
    torch.testing.assert_close(p.grad,grad);assert scaler.get_scale()==16
    assert opt.last_grad_norm==pytest.approx(math.sqrt(7),rel=2e-7)
    saved=copy.deepcopy(opt.state_dict());q=torch.nn.Parameter(p.detach().clone());restored=m.AdamW([q])
    restored.load_state_dict(saved)
    assert restored.fused_step and restored.max_grad_norm==.4
    p.grad=torch.full_like(p,3.);q.grad=p.grad.clone();opt.step();restored.step()
    torch.testing.assert_close(p,q,rtol=0,atol=0)
    saved.pop('ruda_step_options');restored.load_state_dict(saved)
    assert not restored.fused_step and restored.max_grad_norm is None


def test_bad_report_poisons_but_does_not_update(fused):
    m,ref=fused;p=torch.nn.Parameter(torch.ones(5));p.grad=torch.ones_like(p)
    opt=m.AdamW([p],fused_step=True)
    ref.training_analyze=lambda gs,inv,work,report,norm:report.fill_(float('nan'))
    with pytest.raises(RuntimeError,match='statistics'):opt.step()
    assert ref.batch_calls==0 and not opt.state
    with pytest.raises(RuntimeError,match='checkpoint'):opt.step()


def test_native_failure_no_successful_step_counter(fused):
    m,ref=fused;p=torch.nn.Parameter(torch.ones(5));p.grad=torch.ones_like(p)
    opt=m.AdamW([p],fused_step=True);opt.step();checkpoint=copy.deepcopy(opt.state_dict());ref.fail=True
    with pytest.raises(RuntimeError,match='batch failure'):opt.step()
    assert opt.state[p]['step']==1
    with pytest.raises(RuntimeError,match='checkpoint'):opt.state_dict()
    ref.fail=False;opt.load_state_dict(checkpoint);opt.step();assert opt.state[p]['step']==2

@pytest.mark.parametrize('kw',[{'fused_step':1},{'fused_step':True,'max_grad_norm':-1},
    {'fused_step':True,'max_grad_norm':float('inf')},{'max_grad_norm':1.},{'fused_step':True,'max_grad_norm':True}])
def test_invalid_options(fused,kw):
    m,_=fused
    with pytest.raises((TypeError,ValueError)):m.AdamW([torch.nn.Parameter(torch.ones(3))],**kw)

@pytest.mark.parametrize('key',['amsgrad','maximize','capturable','differentiable'])
def test_unsupported_group_options_rejected(fused,key):
    m,_=fused
    with pytest.raises(ValueError,match=key):m.AdamW([{'params':[torch.nn.Parameter(torch.ones(3))],key:True}])
