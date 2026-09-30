"""Host numerical/formula and real Python autograd routing checks, not GPU tests."""
import importlib.util
from pathlib import Path
import sys, types
import pytest, torch
from paged_selected_reference import data,dense,analytical,SPEC

def module():
    pkg=types.ModuleType('v32_protocol');pkg.__path__=[];pkg._C=types.SimpleNamespace();sys.modules[pkg.__name__]=pkg
    path=Path(__file__).resolve().parents[1]/'ruda_torch/_paged.py'
    spec=importlib.util.spec_from_file_location(pkg.__name__+'._paged',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class NativeReference:
    """Explicit CPU reference; never a production fallback."""
    def __init__(self,mla=False,spec=SPEC):self.mla=mla;self.spec=spec;self.requests=[]
    def run(self,q,k,v,qp,kp,scale,causal):
        ts=(q,qp,k,kp) if self.mla else (q,k,v)
        return dense(ts,self.mla,scale=scale,causal=causal,spec=self.spec)
    def backward_selected(self,q,k,v,qp,kp,g,scale,causal,needs):
        self.requests.append(tuple(needs));ts=(q,qp,k,kp) if self.mla else (q,k,v)
        return analytical(ts,g,needs,self.mla,scale=scale,causal=causal,spec=self.spec)

CASES=[(False,m) for m in range(1,8)]+[(True,m) for m in range(1,16)]
@pytest.mark.parametrize('mla,mask',CASES)
@pytest.mark.parametrize('causal',[False,True])
def test_all_gradient_selections_match_dense_autograd(mla,mask,causal):
    ts=data(mla);needs=tuple(bool(mask&(1<<i)) for i in range(len(ts)))
    refs=[x.clone().requires_grad_(n) for x,n in zip(ts,needs)]
    inputs=[x.clone().requires_grad_(n) for x,n in zip(ts,needs)]
    expected=dense(refs,mla,causal=causal);go=torch.randn_like(expected)
    expected.backward(go)
    n=NativeReference(mla);m=module();fn=m._PagedMLA if mla else m._PagedGQA
    actual=fn.apply(n,*inputs,.37,causal);actual.backward(go)
    assert n.requests==[needs]
    torch.testing.assert_close(actual,expected)
    for a,b,need in zip(inputs,refs,needs):
        if need:torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-11)
        else:assert a.grad is None

@pytest.mark.parametrize('mla',[False,True])
def test_numerical_gradcheck(mla):
    m=module();n=NativeReference(mla);fn=m._PagedMLA if mla else m._PagedGQA
    ts=[x.requires_grad_() for x in data(mla)]
    assert torch.autograd.gradcheck(lambda *a:fn.apply(n,*a,.37,True),tuple(ts),fast_mode=True)

@pytest.mark.parametrize('mla',[False,True])
def test_frozen_inputs_still_version_checked(mla):
    m=module();n=NativeReference(mla);ts=list(data(mla));ts[0].requires_grad_();fn=m._PagedMLA if mla else m._PagedGQA
    y=fn.apply(n,*ts,.37,True)
    with torch.no_grad():ts[-1].add_(.1)
    with pytest.raises(RuntimeError,match='modified by an inplace'):y.sum().backward()
    assert not n.requests

@pytest.mark.parametrize('mla',[False,True])
def test_higher_derivatives_rejected(mla):
    m=module();n=NativeReference(mla);ts=list(data(mla));ts[0].requires_grad_();fn=m._PagedMLA if mla else m._PagedGQA
    y=fn.apply(n,*ts,.37,True)
    with pytest.raises(RuntimeError,match='first-order'):torch.autograd.grad(y.sum(),ts[0],create_graph=True)

@pytest.mark.parametrize('mla',[False,True])
def test_empty_queries_define_zero_history(mla):
    spec=dict(SPEC,sequence_ids=[],positions=[]);n=NativeReference(mla,spec);m=module();fn=m._PagedMLA if mla else m._PagedGQA
    ts=[x.requires_grad_() for x in data(mla,queries=0)]
    fn.apply(n,*ts,.37,True).sum().backward()
    for t in ts:assert t.grad is not None and torch.count_nonzero(t.grad)==0

def test_kv_alias_contributions_sum():
    m=module();n=NativeReference();q,k,_=data(False);q.requires_grad_();k.requires_grad_()
    qr=q.detach().clone().requires_grad_();kr=k.detach().clone().requires_grad_()
    a=m._PagedGQA.apply(n,q,k,k,.37,True);b=dense((qr,kr,kr));go=torch.randn_like(a)
    a.backward(go);b.backward(go)
    torch.testing.assert_close(k.grad,kr.grad,rtol=1e-10,atol=1e-11)

@pytest.mark.parametrize('frozen',[False,True])
def test_cpu_reference_block_and_checkpoint(frozen):
    # Validate only the independently executed CPU side of the device harness.
    path=Path(__file__).resolve().parents[1]/'examples/train_paged_block.py'
    spec=importlib.util.spec_from_file_location('v32_cpu_block',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.manual_seed(9);a=m.Block(frozen_history=frozen);opt=torch.optim.AdamW(a.parameters(),lr=1e-4)
    x=torch.randn(6,16)*.1;target=torch.randn(6,16)*.1
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss=(a(x)-target).square().mean();loss.backward()
        assert torch.isfinite(loss)
        for name,p in a.named_parameters():
            if frozen and name in ('k.weight','v.weight'):assert p.grad is None
            else:assert p.grad is not None and torch.isfinite(p.grad).all()
        opt.step()
    state=m.cpu_tree({'model':a.state_dict(),'optimizer':opt.state_dict()})
    b=m.Block(frozen_history=frozen);b.load_state_dict(state['model'])
    bopt=torch.optim.AdamW(b.parameters(),lr=1e-4);bopt.load_state_dict(state['optimizer'])
    for model,optim in ((a,opt),(b,bopt)):
        optim.zero_grad(set_to_none=True);(model(x)-target).square().mean().backward();optim.step()
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
