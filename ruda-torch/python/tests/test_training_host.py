"""CPU reference/contract tests. Explicit shims do not execute Rust/PTX/GPU.

Imports the production Python training module without importing the RUDA device.
All numerical native callbacks below are test-only, never installed by production.
"""
import copy
import importlib.util
from pathlib import Path
import types
import pytest
import torch

PATH=Path(__file__).resolve().parents[1]/'ruda_torch/training.py'

class Reference:
    def __init__(self): self.updates=0; self.checks=0; self.fail=False
    def training_rms_forward(self,x,w,eps):
        r=(x.float().square().mean(-1,keepdim=True)+eps).rsqrt()
        y=x.float()*r
        if w is not None:y=y*w.float()
        return y.to(x.dtype),r.reshape(-1)
    def training_rms_backward(self,x,w,dy,stats,nx,nw):
        r=stats.reshape(*x.shape[:-1],1);a=x.float()*r
        g=dy.float() if w is None else dy.float()*w.float()
        dx=(r*(g-a*(g*a).mean(-1,keepdim=True))).to(x.dtype) if nx else None
        dw=(dy.float()*a).reshape(-1,x.shape[-1]).sum(0).to(w.dtype) if nw else None
        return dx,dw
    def training_layer_forward(self,x,w,b,eps):
        mean=x.float().mean(-1,keepdim=True);var=(x.float()-mean).square().mean(-1,keepdim=True);r=(var+eps).rsqrt()
        y=(x.float()-mean)*r
        if w is not None:y=y*w.float()
        if b is not None:y=y+b.float()
        return y.to(x.dtype),mean.reshape(-1),r.reshape(-1)
    def training_layer_backward(self,x,w,b,dy,mean,rstd,nx,nw,nb):
        width=x.shape[-1];m=mean.reshape(*x.shape[:-1],1);r=rstd.reshape(*x.shape[:-1],1);xh=(x.float()-m)*r
        g=dy.float() if w is None else dy.float()*w.float()
        dx=(r*(g-g.mean(-1,keepdim=True)-xh*(g*xh).mean(-1,keepdim=True))).to(x.dtype) if nx else None
        flat=(dy.float()*xh).reshape(-1,width);dw=flat.sum(0).to(w.dtype) if nw else None
        db=dy.float().reshape(-1,width).sum(0).to(b.dtype) if nb else None
        return dx,dw,db
    def training_silu_forward(self,g,u):return torch.nn.functional.silu(g.float()).to(g.dtype)*u
    def training_silu_backward(self,g,u,dy,ng,nu):
        # Autograd CPU reference rather than reusing the Rust formula.
        with torch.enable_grad():
            x=g.detach().float().requires_grad_();s=torch.nn.functional.silu(x)
            dx=torch.autograd.grad(s,x,(dy*u).float())[0].to(g.dtype) if ng else None
        du=dy*torch.nn.functional.silu(g.float()).to(g.dtype) if nu else None
        return dx,du
    def training_unscale_(self,gs,inv,workspace,found):
        self.checks+=1
        bad=any(not torch.isfinite(g).all() for g in gs)
        for g in gs:g.mul_(inv)
        bad=bad or any(not torch.isfinite(g).all() for g in gs)
        found.fill_(float(bad))
    def training_adamw_(self,p,g,master,m,v,lr,b1,b2,eps,decay,c1,c2):
        if self.fail:raise RuntimeError('explicit host native failure')
        self.updates+=1
        m.mul_(b1).add_(g.float(),alpha=1-b1)
        v.mul_(b2).addcmul_(g.float(),g.float(),value=1-b2)
        master.mul_(1-lr*decay).addcdiv_(m/c1,(v/c2).sqrt()+eps,value=-lr)
        p.copy_(master.to(p.dtype))

@pytest.fixture
def mod(monkeypatch):
    spec=importlib.util.spec_from_file_location('explicit_training_host_test',PATH)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ref=Reference()
    def check(x,name):
        assert isinstance(x,torch.Tensor) and x.device.type=='cpu', 'explicit CPU reference test only'
        if x.dtype not in m._DTYPES or not x.is_contiguous():raise ValueError('unsupported test tensor')
    monkeypatch.setattr(m,'_native',lambda:ref)
    monkeypatch.setattr(m,'_tensor',check)
    return m,ref

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('shape',[(3,17),(2,3,33),(1,),(0,7)])
@pytest.mark.parametrize('weight_kind',['none','same','fp32'])
def test_rms_first_order_against_autograd(mod,dtype,shape,weight_kind):
    m,_=mod;torch.manual_seed(410)
    x=torch.randn(shape,dtype=dtype).requires_grad_()
    w=None if weight_kind=='none' else torch.randn(shape[-1],dtype=dtype if weight_kind=='same' else torch.float32).requires_grad_()
    dy=torch.randn_like(x)
    y=m.rms_norm(x,w,eps=1e-5);y.backward(dy)
    refx=x.detach().float().requires_grad_();refw=None if w is None else w.detach().float().requires_grad_()
    expected=refx*(refx.square().mean(-1,keepdim=True)+1e-5).rsqrt()
    if refw is not None:expected=expected*refw
    expected=expected.to(dtype);expected.backward(dy)
    tol=2e-5 if dtype==torch.float32 else (0.004 if dtype==torch.float16 else 0.04)
    torch.testing.assert_close(y,expected,rtol=tol,atol=tol)
    torch.testing.assert_close(x.grad,refx.grad.to(dtype),rtol=tol,atol=tol)
    if w is not None:torch.testing.assert_close(w.grad,refw.grad.to(w.dtype),rtol=tol,atol=tol)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('affine',['none','same','fp32'])
def test_layer_norm_first_order_against_autograd(mod,dtype,affine):
    m,_=mod;torch.manual_seed(414);shape=(2,3,33)
    x=torch.randn(shape,dtype=dtype).requires_grad_();dy=torch.randn_like(x)
    w=b=None
    if affine!='none':
        adtype=dtype if affine=='same' else torch.float32
        w=torch.randn(shape[-1],dtype=adtype).requires_grad_();b=torch.randn(shape[-1],dtype=adtype).requires_grad_()
    y=m.layer_norm(x,w,b,eps=1e-5);y.backward(dy)
    rx=x.detach().float().requires_grad_();rw=None if w is None else w.detach().float().requires_grad_();rb=None if b is None else b.detach().float().requires_grad_()
    ref=torch.nn.functional.layer_norm(rx,(shape[-1],),rw,rb,1e-5).to(dtype);ref.backward(dy)
    tol=2e-5 if dtype==torch.float32 else (.006 if dtype==torch.float16 else .05)
    torch.testing.assert_close(y,ref,rtol=tol,atol=tol);torch.testing.assert_close(x.grad,rx.grad.to(dtype),rtol=tol,atol=tol)
    if w is not None:
        torch.testing.assert_close(w.grad,rw.grad.to(w.dtype),rtol=tol,atol=tol);torch.testing.assert_close(b.grad,rb.grad.to(b.dtype),rtol=tol,atol=tol)

@pytest.mark.parametrize('which',['layer','norm','gate'])
def test_training_primitives_reject_higher_order(mod,which):
    m,_=mod;x=torch.randn(2,7,requires_grad=True)
    if which=='layer': y=m.layer_norm(x)
    elif which=='norm': y=m.rms_norm(x)
    else: y=m.silu_mul(x,x)
    with pytest.raises(RuntimeError,match='first-order'):torch.autograd.grad(y.sum(),x,create_graph=True)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
def test_gate_gradients_and_optional_inputs(mod,dtype,needs):
    m,_=mod;torch.manual_seed(120)
    g=torch.randn(3,33,dtype=dtype).requires_grad_(needs[0]);u=torch.randn_like(g).requires_grad_(needs[1])
    dy=torch.randn_like(g);y=m.silu_mul(g,u);y.backward(dy)
    rg=g.detach().clone().requires_grad_(needs[0]);ru=u.detach().clone().requires_grad_(needs[1])
    expected=torch.nn.functional.silu(rg.float()).to(dtype)*ru;expected.backward(dy)
    torch.testing.assert_close(y,expected)
    if needs[0]:torch.testing.assert_close(g.grad,rg.grad,atol=.04 if dtype==torch.bfloat16 else .004,rtol=.04 if dtype==torch.bfloat16 else .004)
    if needs[1]:torch.testing.assert_close(u.grad,ru.grad)

@pytest.mark.parametrize('which',['norm','gate'])
def test_grad_accumulation_and_saved_version_checks(mod,which):
    m,_=mod;x=torch.randn(2,7,requires_grad=True);w=torch.randn(7,requires_grad=True)
    fn=lambda:m.rms_norm(x,w) if which=='norm' else m.silu_mul(x,x)
    y=fn();y.sum().backward(retain_graph=True);first=x.grad.clone();y.sum().backward();torch.testing.assert_close(x.grad,2*first)
    y=fn()
    with torch.no_grad():x.add_(1)
    with pytest.raises(RuntimeError,match='modified|inplace'):y.sum().backward()

@pytest.mark.parametrize('which',['norm','gate'])
def test_higher_order_is_explicitly_rejected(mod,which):
    m,_=mod;x=torch.randn(2,7,requires_grad=True)
    y=m.rms_norm(x) if which=='norm' else m.silu_mul(x,x)
    with pytest.raises(RuntimeError,match='first-order'):torch.autograd.grad(y.sum(),x,create_graph=True)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_adamw_fp32_states_reference_and_restore(mod,dtype):
    m,ref=mod;torch.manual_seed(313)
    p=torch.nn.Parameter(torch.randn(31,dtype=dtype));q=torch.nn.Parameter(p.detach().float().clone())
    opt=m.AdamW([p],lr=.005,betas=(.8,.93),eps=1e-6,weight_decay=.03)
    reference=torch.optim.AdamW([q],lr=.005,betas=(.8,.93),eps=1e-6,weight_decay=.03,foreach=False)
    grads=[torch.randn_like(p) for _ in range(8)]
    for i,g in enumerate(grads):
        p.grad=g.clone();q.grad=g.float().clone();opt.step();reference.step()
        torch.testing.assert_close(p,q.detach().to(dtype),atol=2e-6 if dtype==torch.float32 else 0,rtol=2e-6 if dtype==torch.float32 else 0)
        assert opt.state[p]['exp_avg'].dtype==torch.float32
        if i==3:
            checkpoint=copy.deepcopy(opt.state_dict()); saved_param=p.detach().clone()
    restored_p=torch.nn.Parameter(saved_param);restored=m.AdamW([restored_p]);restored.load_state_dict(checkpoint)
    for g in grads[4:]:restored_p.grad=g.clone();restored.step()
    torch.testing.assert_close(p,restored_p,rtol=0,atol=0)
    for k in ('exp_avg','exp_avg_sq'):
        torch.testing.assert_close(opt.state[p][k],restored.state[restored_p][k],rtol=0,atol=0)
    if dtype!=torch.float32:assert restored.state[restored_p]['master_copy'].dtype==torch.float32

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_overflow_is_whole_optimizer_and_no_initial_state(mod,bad,dtype):
    m,ref=mod;a=torch.nn.Parameter(torch.ones(7,dtype=dtype));b=torch.nn.Parameter(torch.ones(5,dtype=dtype))
    opt=m.AdamW([a,b]);a.grad=torch.ones_like(a);b.grad=torch.full_like(b,bad)
    opt.step(loss_scale=8)
    assert opt.last_step_skipped and ref.updates==0 and not opt.state
    assert torch.equal(a,torch.ones_like(a)) and torch.equal(b,torch.ones_like(b))


def test_gradient_accumulation_scale_once_and_missing_grad(mod):
    m,_=mod;a=torch.nn.Parameter(torch.ones(5));b=torch.nn.Parameter(torch.ones(7))
    opt=m.AdamW([a,b],lr=.01);a.grad=torch.full_like(a,16.)
    opt.step(loss_scale=8.)
    assert torch.equal(a.grad,torch.full_like(a,2.))
    assert b not in opt.state and torch.equal(b,torch.ones_like(b))
    opt.zero_grad(set_to_none=True);assert a.grad is None


def test_failure_poison_and_checkpoint_recovery(mod):
    m,ref=mod;p=torch.nn.Parameter(torch.ones(7));opt=m.AdamW([p]);p.grad=torch.ones_like(p);opt.step()
    checkpoint=copy.deepcopy(opt.state_dict());ref.fail=True
    with pytest.raises(RuntimeError,match='host native failure'):opt.step()
    with pytest.raises(RuntimeError,match='checkpoint'):opt.step()
    with pytest.raises(RuntimeError,match='checkpoint'):opt.state_dict()
    ref.fail=False;opt.load_state_dict(checkpoint);p.grad=torch.ones_like(p);opt.step()


def test_all_shape_checks_precede_mutation(mod):
    m,ref=mod;a=torch.nn.Parameter(torch.ones(5));b=torch.nn.Parameter(torch.ones(7))
    opt=m.AdamW([a,b]);a.grad=torch.ones_like(a);b.grad=torch.empty(7,2)[:,0]
    with pytest.raises(ValueError):opt.step()
    assert ref.checks==0 and ref.updates==0


def test_parameter_gradient_alias_rejected(mod):
    m,ref=mod;p=torch.nn.Parameter(torch.ones(7));opt=m.AdamW([p]);p.grad=p.detach()
    with pytest.raises(ValueError,match='overlap'):opt.step()
    assert ref.checks==0

@pytest.mark.parametrize('kw',[{'lr':-1},{'eps':0},{'betas':(.9,1.)},{'lr':float('nan')},{'weight_decay':-1}])
def test_invalid_hyperparameters(mod,kw):
    m,_=mod
    with pytest.raises((ValueError,TypeError)):m.AdamW([torch.nn.Parameter(torch.ones(3))],**kw)


def test_dynamic_scaler_growth_overflow_and_checkpoint(mod):
    m,ref=mod;p=torch.nn.Parameter(torch.ones(7));opt=m.AdamW([p]);s=m.GradScaler(init_scale=8,growth_interval=2)
    for _ in range(2):
        loss=p.sum();s.scale(loss).backward();s.step(opt);s.update();opt.zero_grad()
    assert s.get_scale()==16
    checkpoint=copy.deepcopy(s.state_dict());p.grad=torch.full_like(p,float('inf'))
    s.scale(p.sum());before=p.detach().clone();before_step=opt.state[p]['step']
    s.step(opt);s.update()
    assert s.get_scale()==8 and opt.last_step_skipped
    assert opt.state[p]['step']==before_step and torch.equal(p,before)
    s.load_state_dict(checkpoint);assert s.get_scale()==16


def test_scaler_contract_errors(mod):
    m,_=mod;p=torch.nn.Parameter(torch.ones(3));opt=m.AdamW([p]);s=m.GradScaler(init_scale=8)
    with pytest.raises(RuntimeError):s.update()
    with pytest.raises(RuntimeError):s.step(opt)
    s.scale(p.sum()).backward()
    with pytest.raises(RuntimeError):s.state_dict()
    s.step(opt)
    with pytest.raises(RuntimeError):s.step(opt)
    with pytest.raises(RuntimeError):s.scale(p.sum())
    s.update()
    with pytest.raises(ValueError):s.scale(p.sum().half())
    with pytest.raises(TypeError):s.step(torch.optim.SGD([p],lr=.1))


def test_production_entry_rejects_cpu_without_test_shim():
    spec=importlib.util.spec_from_file_location('training_without_shim',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(ValueError,match='CPU fallback'):m.rms_norm(torch.ones(3))
    with pytest.raises(ValueError,match='CPU fallback'):m.AdamW([torch.nn.Parameter(torch.ones(3))])

@pytest.mark.parametrize('width',[1,7,33,1025,4096])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_mean_before_narrowing_reference(width,dtype):
    x=torch.full((3,width),1000.,dtype=dtype)
    reference=x.float().mean(-1).to(dtype)
    expected=(x.float().sum(-1)/width).to(dtype)
    torch.testing.assert_close(expected,reference,rtol=0,atol=0)
    if dtype==torch.float16 and width>=1025:
        old=x.float().sum(-1).to(dtype)/width
        assert old.isinf().all() and expected.isfinite().all()

def test_checkpoint_load_hooks_explicitly_rejected(mod):
    m,_=mod;p=torch.nn.Parameter(torch.ones(3));o=m.AdamW([p]);state=o.state_dict()
    hook=o.register_load_state_dict_pre_hook(lambda *a:None)
    with pytest.raises(RuntimeError,match='load-state hooks'):o.load_state_dict(state)
    hook.remove();o.load_state_dict(state)

def test_bad_checkpoint_hyperparameters_prechecked(mod):
    m,_=mod;p=torch.nn.Parameter(torch.ones(3));o=m.AdamW([p]);state=o.state_dict();state['param_groups'][0]['lr']=-1
    with pytest.raises(ValueError):o.load_state_dict(state)
    assert o.param_groups[0]['lr']==1e-3

def test_bad_checkpoint_unknown_state_rejected(mod):
    m,_=mod;p=torch.nn.Parameter(torch.ones(3));o=m.AdamW([p]);state=o.state_dict();state['state'][999]={}
    with pytest.raises(ValueError,match='unknown'):o.load_state_dict(state)
