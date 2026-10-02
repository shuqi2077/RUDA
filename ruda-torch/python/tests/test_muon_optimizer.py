"""Real Python optimizer math, NumPy oracle, lifecycle, and checkpoint tests."""
import copy
import importlib
import math
import numpy as np
import pytest
import torch
from torch import nn
from architecture_test_utils import optim, model as hm, NAME, close


def numpy_ns(g,steps=5,eps=1e-7):
    x=np.array(g,dtype=np.float64,copy=True)
    transpose=x.shape[0]>x.shape[1]
    if transpose:x=x.T
    x/=max(np.linalg.norm(x),eps)
    for _ in range(steps):
        a=x@x.T;x=3.4445*x+(-4.775*a+2.0315*(a@a))@x
    return x.T if transpose else x

@pytest.mark.parametrize('shape',[(2,5),(5,2),(3,3)])
@pytest.mark.parametrize('momentum_mode',['sgd','ema'])
@pytest.mark.parametrize('nesterov',[False,True])
@pytest.mark.parametrize('layout',['as_stored','input_output'])
def test_muon_multistep_independent_numpy(shape,momentum_mode,nesterov,layout):
    rng=np.random.default_rng(21);original=rng.normal(size=shape)
    p=nn.Parameter(torch.tensor(original,dtype=torch.double));value=original.copy();buffer=None
    optimizer=optim.Muon([p],lr=.017,momentum=.8,momentum_mode=momentum_mode,
        nesterov=nesterov,matrix_layout=layout,weight_decay=.03)
    for _ in range(4):
        g=rng.normal(size=shape);p.grad=torch.tensor(g,dtype=torch.double);saved=p.grad.clone()
        if momentum_mode=='sgd':
            buffer=g.copy() if buffer is None else .8*buffer+g
            update=g+.8*buffer if nesterov else buffer
        else:
            buffer=.2*g if buffer is None else .8*buffer+.2*g
            update=.2*g+.8*buffer if nesterov else buffer
        update=numpy_ns(update)
        rows,cols=shape if layout=='as_stored' else shape[::-1]
        value=value*(1-.017*.03)-.017*math.sqrt(max(1,rows/cols))*update
        optimizer.step();close(p,torch.tensor(value),atol=1e-12,rtol=1e-11);close(p.grad,saved)
        close(optimizer.state[p]['momentum_buffer'],torch.tensor(buffer),atol=1e-14,rtol=1e-13)
    assert optimizer.state[p]['step']==4

@pytest.mark.parametrize('stable',[False,True])
@pytest.mark.parametrize('steps',[1,3,5])
def test_orthogonalization_normalization_variants(stable,steps):
    g=torch.randn(5,3,dtype=torch.double)
    actual=optim.muon_orthogonalize(g,ns_steps=steps,stable_normalization=stable)
    close(actual,torch.tensor(numpy_ns(g.numpy(),steps)),atol=1e-12,rtol=1e-11)
    close(optim.muon_orthogonalize(torch.zeros_like(g),ns_steps=steps),torch.zeros_like(g))

def test_stable_norm_handles_extreme_finite_gradients():
    g=torch.tensor([[1e30,-2e30],[3e30,4e30]],dtype=torch.float32)
    expected=optim.muon_orthogonalize(g.double()).float()
    close(optim.muon_orthogonalize(g),expected,atol=2e-5,rtol=2e-5)
    assert torch.isfinite(optim.muon_orthogonalize(g)).all()

@pytest.mark.parametrize('adjust',['original','match_rms_adamw'])
def test_lr_scaling_and_explicit_convolution_flatten(adjust):
    p=nn.Parameter(torch.randn(4,2,2));g=torch.randn_like(p)
    with pytest.raises(ValueError):optim.Muon([p])
    optimizer=optim.Muon([p],flatten=True,adjust_lr=adjust,nesterov=False,momentum=0)
    old=p.detach().clone();p.grad=g.clone();optimizer.step()
    factor=1 if adjust=='original' else .2*2
    expected=old-.02*factor*torch.tensor(numpy_ns(g.flatten(1).numpy())).float().reshape_as(p)
    close(p,expected,atol=2e-6,rtol=2e-5)

def test_adamw_group_matches_torch_adamw():
    torch.manual_seed(30)
    p=nn.Parameter(torch.randn(3,4,dtype=torch.double));q=nn.Parameter(p.detach().clone())
    bias=nn.Parameter(torch.randn(4,dtype=torch.double));refbias=nn.Parameter(bias.detach().clone())
    optimizer=optim.MuonAdamW([{'params':[p],'use_muon':True},{'params':[bias],'use_muon':False,'lr':.005,'eps':1e-8,'weight_decay':.04,'betas':(.8,.95)}])
    oracle=torch.optim.AdamW([refbias],lr=.005,eps=1e-8,weight_decay=.04,betas=(.8,.95),foreach=False)
    for _ in range(5):
        p.grad=torch.randn_like(p);bias.grad=torch.randn_like(bias);refbias.grad=bias.grad.clone()
        optimizer.step();oracle.step();close(bias,refbias,atol=1e-13,rtol=1e-12)
    assert optimizer.state[bias]['step']==5

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_skips_entire_group_without_state_or_decay(bad):
    a=nn.Parameter(torch.ones(2,3));b=nn.Parameter(torch.ones(3))
    opt=optim.MuonAdamW([{'params':[a],'use_muon':True},{'params':[b],'use_muon':False}],weight_decay=.1)
    a.grad=torch.ones_like(a);b.grad=torch.ones_like(b);opt.step()
    old_a=a.detach().clone();old_b=b.detach().clone();old=copy.deepcopy(opt.state_dict())
    a.grad.fill_(bad);opt.step();assert opt.last_step_skipped
    close(a,old_a);close(b,old_b)
    for p,saved in zip([a,b],old['state'].values()):
        for k,v in saved.items():
            if isinstance(v,torch.Tensor):close(opt.state[p][k],v)
            else:assert opt.state[p][k]==v


def test_proposed_narrowing_overflow_skips_all():
    p=nn.Parameter(torch.ones(2,2,dtype=torch.float16));q=nn.Parameter(torch.ones(2,dtype=torch.float16))
    opt=optim.MuonAdamW([{'params':[p],'use_muon':True,'lr':1e9},{'params':[q],'use_muon':False,'lr':1e-3}])
    p.grad=torch.eye(2,dtype=p.dtype);q.grad=torch.ones_like(q)
    opt.step();assert opt.last_step_skipped;close(p,torch.ones_like(p));close(q,torch.ones_like(q));assert not opt.state

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_checkpoint_exact_master_precision_and_resume(dtype):
    torch.manual_seed(31);p=nn.Parameter(torch.randn(3,5,dtype=dtype));opt=optim.Muon([p],momentum_mode='ema')
    for _ in range(3):p.grad=torch.randn_like(p);opt.step()
    snapshot=copy.deepcopy(opt.state_dict());q=nn.Parameter(p.detach().clone());other=optim.Muon([q]);other.load_state_dict(snapshot)
    expected_dtype=torch.float64 if dtype==torch.float64 else torch.float32
    assert other.state[q]['master'].dtype==expected_dtype
    close(other.state[q]['master'],opt.state[p]['master'],atol=0,rtol=0)
    for _ in range(3):
        p.grad=torch.randn_like(p);q.grad=p.grad.clone();opt.step();other.step();close(p,q,atol=0,rtol=0)
        close(opt.state[p]['master'],other.state[q]['master'],atol=0,rtol=0)
    # Loading does not alias the checkpoint's mutable buffers.
    other.state[q]['momentum_buffer'].zero_()
    assert snapshot['state'][0]['momentum_buffer'].abs().sum()>0


def test_loss_scale_global_clip_and_mixed_dtype():
    torch.manual_seed(32)
    a=nn.Parameter(torch.randn(3,4));b=nn.Parameter(torch.randn(4,dtype=torch.double))
    aa=nn.Parameter(a.detach().clone());bb=nn.Parameter(b.detach().clone())
    opt=optim.MuonAdamW([{'params':[a],'use_muon':True},{'params':[b],'use_muon':False}],max_grad_norm=.3)
    other=optim.MuonAdamW([{'params':[aa],'use_muon':True},{'params':[bb],'use_muon':False}])
    for _ in range(2):
        g=torch.randn_like(a);h=torch.randn_like(b)
        norm=(g.double().square().sum()+h.square().sum()).sqrt();clip=min(1,.3/norm.item())
        a.grad=g*16;b.grad=h*16;aa.grad=g*clip;bb.grad=h*clip
        old=a.grad.clone();opt.step(loss_scale=16);other.step();close(a,aa,atol=2e-6,rtol=2e-5);close(b,bb,atol=1e-12,rtol=1e-10);close(a.grad,old)
        assert opt.state[a]['master'].dtype==torch.float32
        assert opt.state[b]['master'].dtype==torch.float64


def test_missing_gradient_counters_and_closure_scheduler():
    p=nn.Parameter(torch.randn(2,3));q=nn.Parameter(torch.randn(2,3))
    opt=optim.Muon([p,q]);schedule=torch.optim.lr_scheduler.StepLR(opt,1,gamma=.5)
    opt.step();assert not opt.state and not opt.last_step_skipped
    qold=q.detach().clone()
    def closure():
        opt.zero_grad(set_to_none=True);loss=p.square().sum();loss.backward();return loss
    loss=opt.step(closure);assert isinstance(loss,torch.Tensor);close(q,qold);assert q not in opt.state
    schedule.step();assert opt.param_groups[0]['lr']==.01
    opt.add_param_group({'params':nn.Parameter(torch.randn(3,2))})
    assert len(opt.param_groups)==2


def test_group_selector_tied_embedding_head_biases():
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(13,4);self.hidden=nn.Linear(4,4)
            self.head=nn.Linear(4,13,bias=False);self.head.weight=self.embedding.weight
    mod=Model();opt=optim.MuonAdamW.from_model(mod,muon_modules=[mod],adamw_modules=[mod.head])
    groups={group['use_muon']:{id(p) for p in group['params']} for group in opt.param_groups}
    assert groups[True]=={id(mod.hidden.weight)}
    assert groups[False]=={id(mod.embedding.weight),id(mod.hidden.bias)}
    all_ids=[id(p) for group in opt.param_groups for p in group['params']];assert len(all_ids)==len(set(all_ids))
    with pytest.raises(ValueError):optim.MuonAdamW.from_model(mod,muon_modules=[nn.Linear(4,4)])

@pytest.mark.parametrize('kwargs',[{'momentum':1.},{'momentum':0.},{'dampening':.1},{'lr':-1.},{'eps':0.},
    {'ns_steps':0},{'ns_steps':100},{'ns_steps':True},{'momentum_mode':'bad'}, {'adjust_lr':'bad'},
    {'matrix_layout':'bad'},{'max_grad_norm':0},{'ns_coefficients':(1.,float('nan'),2.)}])
def test_bad_optimizer_configuration(kwargs):
    with pytest.raises((ValueError,TypeError)):optim.Muon([nn.Parameter(torch.ones(2,2))],**kwargs)


def test_reject_sparse_alias_noncontiguous_and_external_mutation():
    storage=torch.randn(20)
    p=nn.Parameter(storage[:6].reshape(2,3));q=nn.Parameter(storage[3:9].reshape(2,3))
    with pytest.raises(ValueError):optim.Muon([p,q])
    # Disjoint same-storage parameters are allowed.
    q=nn.Parameter(storage[6:12].reshape(2,3));opt=optim.Muon([p,q])
    p.grad=p.detach()
    with pytest.raises(ValueError):opt.step()
    p.grad=torch.ones_like(p);q.grad=torch.ones_like(q);opt.step()
    with torch.no_grad():p.add_(1)
    with pytest.raises(ValueError):opt.step()
    sparse_p=nn.Parameter(torch.ones(2,2));other=optim.Muon([sparse_p]);sparse_p.grad=torch.eye(2).to_sparse()
    with pytest.raises(ValueError):other.step()
    noncontig=nn.Parameter(torch.ones(3,2).t())
    with pytest.raises(ValueError):optim.Muon([noncontig])


def test_checkpoint_prevalidation_does_not_replace_valid_state():
    p=nn.Parameter(torch.ones(2,3));opt=optim.Muon([p]);p.grad=torch.randn_like(p);opt.step()
    old=copy.deepcopy(opt.state_dict());bad=copy.deepcopy(old);bad['state'][0]['master']=torch.ones(5)
    with pytest.raises(ValueError):opt.load_state_dict(bad)
    close(opt.state[p]['master'],old['state'][0]['master'])
    with pytest.raises(ValueError):opt.load_state_dict({'state':{},'param_groups':[]})
    opt.param_groups[0]['momentum_mode']='ema'
    with pytest.raises(ValueError):opt.step()


def test_grad_scaler_muon_protocol_cpu_fixture(monkeypatch):
    # Only the native-device argument validator is bypassed; scaler state machine
    # and production Muon numerical updates execute unchanged on real CPU tensors.
    training=importlib.import_module(NAME+'.training')
    monkeypatch.setattr(training,'_tensor',lambda tensor,name:None)
    p=nn.Parameter(torch.ones(2,3));opt=optim.Muon([p]);scaler=training.GradScaler(init_scale=8,growth_interval=1)
    scaler.scale(p.square().sum()).backward();scaler.step(opt);scaler.update();assert scaler.get_scale()==16
    with pytest.raises(RuntimeError):scaler.step(opt)
    opt.zero_grad(set_to_none=True);scaler.scale(p.square().sum()).backward();p.grad[0,0]=float('inf')
    before=p.detach().clone();scaler.step(opt);scaler.update();assert scaler.get_scale()==8;close(p,before)
    clone=training.GradScaler();clone.load_state_dict(scaler.state_dict());assert clone.get_scale()==8
