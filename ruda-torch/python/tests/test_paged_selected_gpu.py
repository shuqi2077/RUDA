"""Mandatory device acceptance for selected GQA/MLA backward, no CPU fallback.

Only the reference is CPU. The implementation runs via the real Rust/PTX ABI.
"""
import os
import pytest
import torch
from paged_selected_reference import data,dense,SPEC

DTYPES=os.environ.get('RUDA_PAGED_DTYPES','float32,float16').split(',')
if not DTYPES or len(DTYPES)!=len(set(DTYPES)) or any(x not in ('float32','float16','bfloat16') for x in DTYPES):
    raise ValueError('invalid RUDA_PAGED_DTYPES')

@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU')!='1':pytest.skip('explicit GPU acceptance requires RUDA_REQUIRE_GPU=1')
    import ruda_torch as r
    assert r._paged_backward_available, 'rebuild Rust/C++ selected paged backward API 1'
    assert os.environ.get('RUDA_CUDA_COMPILER')=='ptx' and os.environ.get('RUDA_PTX_VERSION')
    assert r.device_count()==1
    # Real kernel and device round-trip before emitting the execution marker.
    x=torch.tensor([1.25,-2.5]).to('ruda')
    torch.testing.assert_close((x+x).cpu(),torch.tensor([2.5,-5.0]))
    print('RUDA_V32_PAGED_GPU_EXECUTED',flush=True)
    return r

def native_run(plan,ts,mla):
    if mla:return plan.mla(*ts,scale=.37)
    return plan.attention(*ts,scale=.37)

def tol(dtype):return 7e-4 if dtype==torch.float32 else (.007 if dtype==torch.float16 else .04)

@pytest.mark.parametrize('dtype_name',DTYPES)
@pytest.mark.parametrize('mla,mask',[(False,m) for m in range(1,8)]+[(True,m) for m in range(1,16)])
@pytest.mark.parametrize('causal',[False,True])
def test_selective_gradients_and_workspace(backend,dtype_name,mla,mask,causal):
    dtype=getattr(torch,dtype_name);cpu=data(mla,dtype=dtype)
    needs=[bool(mask&(1<<i)) for i in range(len(cpu))]
    ref=[x.detach().float().requires_grad_(n) for x,n in zip(cpu,needs)]
    dev=[x.detach().to('ruda').requires_grad_(n) for x,n in zip(cpu,needs)]
    expected=dense(ref,mla,causal=causal)
    go=torch.linspace(-.3,.7,expected.numel()).reshape_as(expected).to(dtype)
    expected.backward(go.float())
    plan=backend.PagedAttentionPlan(**SPEC)
    out=plan.mla(*dev,scale=.37,causal=causal) if mla else plan.attention(*dev,scale=.37,causal=causal)
    torch.testing.assert_close(out.cpu().float(),expected,rtol=tol(dtype),atol=tol(dtype))
    before=backend.execution_stats();out.backward(go.to('ruda'));backend.synchronize();after=backend.execution_stats()
    assert after['paged_backward_calls']-before['paged_backward_calls']==1
    history=(2,3) if mla else (1,2)
    workspace=0 if dtype==torch.float32 else sum(cpu[i].numel()*4 for i in history if needs[i])
    assert after['paged_backward_history_workspace_bytes_total']-before['paged_backward_history_workspace_bytes_total']==workspace
    count=sum(needs[i] for i in history)
    assert after['kernel_launches']-before['kernel_launches']>=1+count*(1 if dtype==torch.float32 else 2)
    for d,r,n in zip(dev,ref,needs):
        if n:torch.testing.assert_close(d.grad.cpu().float(),r.grad,rtol=tol(dtype),atol=tol(dtype))
        else:assert d.grad is None

@pytest.mark.parametrize('dtype_name',DTYPES)
@pytest.mark.parametrize('mla',[False,True])
def test_empty_queries_zero_history(backend,dtype_name,mla):
    ts=[x.to('ruda').requires_grad_() for x in data(mla,dtype=getattr(torch,dtype_name),queries=0)]
    plan=backend.PagedAttentionPlan(**dict(SPEC,sequence_ids=[],positions=[]))
    out=native_run(plan,ts,mla);out.backward(torch.empty_like(out));backend.synchronize()
    for t in ts:assert t.grad is not None and torch.count_nonzero(t.grad.cpu())==0

@pytest.mark.parametrize('mla',[False,True])
def test_deterministic_query_only_and_reject_history(backend,mla):
    ts=[x.float().to('ruda') for x in data(mla)];ts[0].requires_grad_();plan=backend.PagedAttentionPlan(**SPEC)
    old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        values=[]
        for _ in range(2):
            ts[0].grad=None;y=native_run(plan,ts,mla);y.backward(torch.ones_like(y));values.append(ts[0].grad.cpu())
        assert torch.equal(values[0],values[1])
        ts[2 if mla else 1].requires_grad_()
        y=native_run(plan,ts,mla)
        with pytest.raises(RuntimeError,match='deterministic'):y.backward(torch.ones_like(y))
    finally:torch.use_deterministic_algorithms(old,warn_only=warn)

def test_shared_kv_gradient_addition(backend):
    q,k,_=data(False,dtype=torch.float32);qr=q.requires_grad_();kr=k.requires_grad_()
    qd=q.detach().to('ruda').requires_grad_();kd=k.detach().to('ruda').requires_grad_()
    ref=dense((qr,kr,kr));y=backend.PagedAttentionPlan(**SPEC).attention(qd,kd,kd,scale=.37)
    grad=torch.ones_like(ref);ref.backward(grad);y.backward(grad.to('ruda'))
    torch.testing.assert_close(kd.grad.cpu(),kr.grad,rtol=7e-4,atol=7e-4)

@pytest.mark.parametrize('frozen_history',[False,True])
def test_attention_block_training_checkpoint(backend,frozen_history,tmp_path):
    from pathlib import Path
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'examples/train_paged_block.py'
    spec=importlib.util.spec_from_file_location('v32_train_block',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=m.compare_training(backend,steps=3,frozen_history=frozen_history,checkpoint=tmp_path/'state.pt')
    assert result['steps']==3 and result['checkpoint_next_step_verified']
