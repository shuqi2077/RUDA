"""Real native ordered backward acceptance. No CPU fallback; references only."""
import os
import torch
import pytest
from paged_selected_reference import data,dense,SPEC
from test_paged_selected_gpu import DTYPES,tol

@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU')!='1':pytest.skip('explicit native GPU acceptance required')
    import ruda_torch as r
    assert r._C.paged_backward_api_version>=2 and r._paged_backward_available
    assert os.environ.get('RUDA_CUDA_COMPILER')=='ptx' and os.environ.get('RUDA_PTX_VERSION')
    x=torch.tensor([.25,-1.]).to('ruda');torch.testing.assert_close((x+x).cpu(),torch.tensor([.5,-2.]))
    return r

def call(plan,ts,mla,causal=True):
    return (plan.mla if mla else plan.attention)(*ts,scale=.37,causal=causal)

@pytest.mark.parametrize('dtype_name',DTYPES)
@pytest.mark.parametrize('mla,mask',[(False,m) for m in range(1,8)]+[(True,m) for m in range(1,16)])
@pytest.mark.parametrize('causal',[False,True])
def test_ordered_masks_numerics_and_scratch(backend,dtype_name,mla,mask,causal):
    dtype=getattr(torch,dtype_name);ts=data(mla,dtype=dtype);needs=[bool(mask&(1<<i)) for i in range(len(ts))]
    ref=[x.float().requires_grad_(n) for x,n in zip(ts,needs)];dev=[x.to('ruda').requires_grad_(n) for x,n in zip(ts,needs)]
    expected=dense(ref,mla,causal=causal);go=torch.linspace(-.3,.4,expected.numel()).reshape_as(expected).to(dtype)
    expected.backward(go.float());plan=backend.PagedAttentionPlan(**SPEC,backward_strategy='ordered')
    old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        snapshots=[];reports=[]
        for _ in range(2):
            for x in dev:x.grad=None
            y=call(plan,dev,mla,causal);before=backend.execution_stats();y.backward(go.to('ruda'));backend.synchronize();after=backend.execution_stats()
            snapshots.append([x.grad.detach().cpu() if x.grad is not None else None for x in dev]);reports.append((before,after))
            assert after['paged_backward_history_workspace_bytes_total']==before['paged_backward_history_workspace_bytes_total']
        history=any(needs[i] for i in ((2,3) if mla else (1,2)))
        assert reports[0][1]['paged_backward_ordered_workspace_allocations']-reports[0][0]['paged_backward_ordered_workspace_allocations']==int(history)
        assert reports[1][1]['paged_backward_ordered_workspace_allocations']==reports[1][0]['paged_backward_ordered_workspace_allocations']
        assert reports[0][1]['paged_backward_ordered_calls']-reports[0][0]['paged_backward_ordered_calls']==int(history)
        for a,b,x,n in zip(*snapshots,ref,needs):
            if n:
                assert torch.equal(a,b),'ordered backward changed across repeats'
                torch.testing.assert_close(a.float(),x.grad,atol=tol(dtype),rtol=tol(dtype))
            else:assert a is None and b is None
        print('RUDA_V33_ORDERED_GPU_EXECUTED',flush=True)
    finally:torch.use_deterministic_algorithms(old,warn_only=warn)

@pytest.mark.parametrize('mla',[False,True])
@pytest.mark.parametrize('dtype_name',DTYPES)
def test_empty_query_zero_outputs_no_scratch(backend,mla,dtype_name):
    dev=[x.to('ruda').requires_grad_() for x in data(mla,dtype=getattr(torch,dtype_name),queries=0)]
    plan=backend.PagedAttentionPlan(**dict(SPEC,sequence_ids=[],positions=[]),backward_strategy='ordered')
    y=call(plan,dev,mla);before=backend.execution_stats();y.backward(torch.empty_like(y));backend.synchronize();after=backend.execution_stats()
    assert before['paged_backward_ordered_workspace_allocations']==after['paged_backward_ordered_workspace_allocations']
    for x in dev:assert torch.count_nonzero(x.grad.cpu())==0
    print('RUDA_V33_ORDERED_GPU_EXECUTED',flush=True)

@pytest.mark.parametrize('mla',[False,True])
def test_reused_statistics_do_not_reuse_values(backend,mla):
    plan=backend.PagedAttentionPlan(**SPEC,backward_strategy='ordered');prev=None
    for seed in (2,17,23):
        ts=data(mla,dtype=torch.float32,seed=seed);ref=[x.requires_grad_() for x in ts];dev=[x.detach().to('ruda').requires_grad_() for x in ts]
        y=dense(ref,mla);g=torch.ones_like(y);y.backward(g);z=call(plan,dev,mla);z.backward(g.to('ruda'));backend.synchronize()
        for a,b in zip(dev,ref):torch.testing.assert_close(a.grad.cpu(),b.grad,atol=7e-4,rtol=7e-4)
        count=backend.execution_stats()['paged_backward_ordered_workspace_allocations']
        if prev is not None:assert count==prev
        prev=count
    print('RUDA_V33_ORDERED_GPU_EXECUTED',flush=True)

def test_shared_key_value_input(backend):
    q,k,_=data(False,dtype=torch.float32);qr=q.requires_grad_();kr=k.requires_grad_();qd=q.detach().to('ruda').requires_grad_();kd=k.detach().to('ruda').requires_grad_()
    ref=dense((qr,kr,kr));out=backend.PagedAttentionPlan(**SPEC,backward_strategy='ordered').attention(qd,kd,kd,scale=.37)
    ref.sum().backward();out.sum().backward();torch.testing.assert_close(kd.grad.cpu(),kr.grad,atol=7e-4,rtol=7e-4)
    print('RUDA_V33_ORDERED_GPU_EXECUTED',flush=True)

@pytest.mark.parametrize('frozen',[False,True])
def test_block_training_and_resume(backend,frozen,tmp_path):
    import importlib.util
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'examples/train_paged_block.py';s=importlib.util.spec_from_file_location('ordered_block',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    result=m.compare_training(backend,steps=3,frozen_history=frozen,checkpoint=tmp_path/'state.pt',backward_strategy='ordered')
    assert result['checkpoint_next_step_verified'];print('RUDA_V33_ORDERED_GPU_EXECUTED',flush=True)

@pytest.mark.parametrize('dtype_name',DTYPES)
@pytest.mark.parametrize('mla',[False,True])
@pytest.mark.parametrize('heads,kh,d,dv',[(3,1,33,65),(6,3,65,37),(4,2,129,129)])
def test_feature_tails_and_head_groups(backend,dtype_name,mla,heads,kh,d,dv):
    dtype=getattr(torch,dtype_name);gen=torch.Generator().manual_seed(66)
    def t(*shape):return (torch.randn(shape,generator=gen)*.2).to(dtype)
    q=t(4,heads,d);k=t(4,3,1 if mla else kh,d)
    ts=(q,t(4,heads,33),k,t(4,3,1,33)) if mla else (q,k,t(4,3,kh,dv))
    ref=[x.float().requires_grad_() for x in ts];dev=[x.to('ruda').requires_grad_() for x in ts]
    expected=dense(ref,mla);go=torch.ones_like(expected).to(dtype);expected.backward(go.float())
    y=call(backend.PagedAttentionPlan(**SPEC,backward_strategy='ordered'),dev,mla);y.backward(go.to('ruda'))
    for a,b in zip(dev,ref):torch.testing.assert_close(a.grad.cpu().float(),b.grad,atol=tol(dtype),rtol=tol(dtype))
    print('RUDA_V33_ORDERED_GPU_EXECUTED',flush=True)

def test_changed_heads_rebuild_statistics(backend):
    plan=backend.PagedAttentionPlan(**SPEC,backward_strategy='ordered');before=backend.execution_stats()['paged_backward_ordered_workspace_allocations']
    for expected_allocs,heads in enumerate((4,2),1):
        q,k,v=data(False,dtype=torch.float32);q=q[:,:heads].contiguous();dev=[x.to('ruda').requires_grad_() for x in (q,k,v)]
        y=call(plan,dev,False);y.sum().backward();backend.synchronize()
        assert backend.execution_stats()['paged_backward_ordered_workspace_allocations']-before==expected_allocs
    print('RUDA_V33_ORDERED_GPU_EXECUTED',flush=True)
