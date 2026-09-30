"""Independent math, Python autograd and source guards. NOT Rust/GPU execution."""
from pathlib import Path
import copy
import torch
import pytest
from paged_selected_reference import data,dense,SPEC
from paged_ordered_reference import inverse_index,ordered_vjp
from test_paged_selected_host import module,NativeReference

CASES=[(False,m) for m in range(1,8)]+[(True,m) for m in range(1,16)]
@pytest.mark.parametrize('mla,mask',CASES)
@pytest.mark.parametrize('causal',[False,True])
def test_owner_reduction_matches_dense_autograd(mla,mask,causal):
    ts=data(mla);needs=tuple(bool(mask&(1<<i)) for i in range(len(ts)))
    refs=[x.clone().requires_grad_(n) for x,n in zip(ts,needs)]
    y=dense(refs,mla,causal=causal);g=torch.linspace(-.4,.6,y.numel(),dtype=y.dtype).reshape_as(y)
    y.backward(g);actual=ordered_vjp(ts,g,needs,mla=mla,spec=SPEC,causal=causal)
    for a,x,n in zip(actual,refs,needs):
        if n:torch.testing.assert_close(a,x.grad,atol=1e-11,rtol=1e-10)
        else:assert a is None

@pytest.mark.parametrize('mla',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_low_precision_final_cast_and_unused_slots(mla,dtype):
    ts=data(mla,dtype=dtype);ref=[x.double().requires_grad_() for x in ts];y=dense(ref,mla)
    g=torch.ones_like(y);y.backward(g)
    result=ordered_vjp(ts,g,[True]*len(ts),mla=mla,spec=SPEC)
    for a,r in zip(result,ref):torch.testing.assert_close(a,r.grad.to(dtype),rtol=2e-5,atol=1e-6)
    for i in ((2,3) if mla else (1,2)):assert torch.count_nonzero(result[i][1])==0

@pytest.mark.parametrize('mla',[False,True])
def test_empty_and_shared_position_gradients(mla):
    spec=dict(SPEC,sequence_ids=[],positions=[]);ts=data(mla,queries=0)
    g=ts[0].new_empty((0,4,5 if mla else 7))
    out=ordered_vjp(ts,g,[True]*len(ts),mla=mla,spec=spec)
    for x in out:assert torch.count_nonzero(x)==0

@pytest.mark.parametrize('mla',[False,True])
def test_unsorted_query_rows(mla):
    spec=copy.deepcopy(SPEC);spec.update(sequence_ids=[1,0,1,0],positions=[3,1,2,4])
    ts=[x.requires_grad_() for x in data(mla)];y=dense(ts,mla,spec=spec);g=torch.ones_like(y);y.backward(g)
    out=ordered_vjp(ts,g,[True]*len(ts),mla=mla,spec=spec)
    for a,x in zip(out,ts):torch.testing.assert_close(a,x.grad,atol=1e-11,rtol=1e-10)

@pytest.mark.parametrize('mla',[False,True])
def test_python_autograd_forwards_ordered_and_needs(mla):
    class N(NativeReference):
        def backward_selected(self,q,k,v,qp,kp,g,scale,causal,needs,ordered=False):
            assert ordered;self.requests.append(tuple(needs));ts=(q,qp,k,kp) if mla else (q,k,v)
            return ordered_vjp(ts,g,needs,mla=mla,spec=self.spec,causal=causal,scale=scale)
    n=N(mla);m=module();fn=m._PagedMLA if mla else m._PagedGQA
    ts=list(data(mla));ts[0].requires_grad_();ts[-1].requires_grad_()
    y=fn.apply(n,*ts,.37,True,True);y.sum().backward()
    assert n.requests==[tuple(t.requires_grad for t in ts)]
    for t in ts:assert (t.grad is not None)==t.requires_grad

@pytest.mark.parametrize('strategy',['fast','',None,True])
def test_bad_strategy_rejected(strategy):
    with pytest.raises(ValueError,match='backward_strategy'):module().PagedAttentionPlan(**SPEC,backward_strategy=strategy)

def test_default_remains_atomic_and_plan_exposes_choice():
    m=module();assert m.PagedAttentionPlan(**SPEC).backward_strategy=='atomic'
    assert m.PagedAttentionPlan(**SPEC,backward_strategy='ordered').backward_strategy=='ordered'

def test_inverse_index_shared_reads_and_size():
    spec=dict(page_size=2,num_pages=4,block_tables=[[2,0],[2,3]],kv_lengths=[3,4],sequence_ids=[1,0,1],positions=[3,2,1])
    x=inverse_index(spec);assert x[3:8]==[0,1,1,3,4]
    e,s,q=x[:3];assert x[e:e+8]==[0,1,0,0,1,0,1,1];assert x[q:]==[1,0,2]
    assert len(x)==3+(4+1)+2*4+(2+1)+3

def test_production_source_guards():
    root=Path(__file__).resolve().parents[3];p=root/'ruDNN/src/paged_attention'
    kernel=(p/'ordered_kernel.rs').read_text();host=(p/'mod.rs').read_text()
    assert 'fetch_add' not in kernel and 'Atomic<' not in kernel
    assert 'dk_acc[j]=fma(probability' in kernel # latent has value AND key derivative
    assert 'ordered_history && (dk.is_some() || dkp.is_some())' in host
    assert 'dst.filter(|_| !ordered_history)' in host
    assert 'No clearing pass is required' in kernel
    assert '64 * 1024 * 1024' in (p/'workspace.rs').read_text()

@pytest.mark.parametrize('d,dv',[(1,3),(33,65),(65,37),(129,129)])
@pytest.mark.parametrize('mla',[False,True])
def test_feature_tails_and_large_logits(d,dv,mla):
    gen=torch.Generator().manual_seed(45);kh=1 if mla else 2;h=4
    def r(*shape):return torch.randn(shape,generator=gen,dtype=torch.double)*.7
    q=r(4,h,d);k=r(4,3,kh,d)
    ts=(q,r(4,h,33),k,r(4,3,1,33)) if mla else (q,k,r(4,3,kh,dv))
    ref=[x.requires_grad_() for x in ts];y=dense(ref,mla,scale=3.1);g=r(*y.shape);y.backward(g)
    out=ordered_vjp(ts,g,[True]*len(ts),mla=mla,spec=SPEC,scale=3.1)
    for a,x in zip(out,ref):torch.testing.assert_close(a,x.grad,rtol=1e-9,atol=1e-10)
