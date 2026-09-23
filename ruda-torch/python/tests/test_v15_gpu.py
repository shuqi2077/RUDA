"""Real native RUDA/PTX tests. A required run must fail without a usable GPU."""
import os
import pytest
import torch

@pytest.fixture(scope='module')
def r():
    if os.environ.get('RUDA_REQUIRE_GPU')!='1':
        pytest.skip('explicit hardware run required: RUDA_REQUIRE_GPU=1')
    assert os.environ.get('RUDA_CUDA_COMPILER')=='ptx'
    import ruda_torch
    assert ruda_torch._C.abi_version==9
    return ruda_torch


def inputs(dtype,mla=False):
    from test_v15_host import data
    return data(dtype,mla)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('splits',[2,3,8,32])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('mla',[False,True])
def test_v15_gpu_split_against_cpu_and_unsplit(r,dtype,splits,causal,mla):
    from test_v14_host import dense
    q,k,v,tables,lengths,ids,positions,scale,qp,kp=inputs(dtype,mla)
    expected=dense(q,k,v,tables,lengths,ids,positions,scale,causal,qp,kp).to(dtype)
    qd,kd,vd=(x.to('ruda') for x in (q,k,v));qpd=qp.to('ruda') if mla else None;kpd=kp.to('ruda') if mla else None
    args=dict(page_size=k.shape[1],num_pages=k.shape[0],block_tables=tables,kv_lengths=lengths,sequence_ids=ids,positions=positions)
    split=r.PagedAttentionPlan(**args,splits=splits);single=r.PagedAttentionPlan(**args)
    if mla:
        actual=split.mla(qd,qpd,kd,kpd,scale=scale,causal=causal).cpu()
        baseline=single.mla(qd,qpd,kd,kpd,scale=scale,causal=causal).cpu()
    else:
        actual=split.attention(qd,kd,vd,scale=scale,causal=causal).cpu()
        baseline=single.attention(qd,kd,vd,scale=scale,causal=causal).cpu()
    tol={torch.float32:5e-5,torch.float16:5e-3,torch.bfloat16:3e-2}[dtype]
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
    torch.testing.assert_close(actual,baseline,rtol=tol,atol=tol)


def test_v15_gpu_native_workspace_reused(r):
    q,k,v,tables,lengths,ids,positions,scale,_,_=inputs(torch.float16)
    q,k,v=(x.to('ruda') for x in (q,k,v))
    plan=r.PagedAttentionPlan(page_size=k.shape[1],num_pages=k.shape[0],block_tables=tables,kv_lengths=lengths,sequence_ids=ids,positions=positions,splits=8)
    r.synchronize();before=r.execution_stats();outputs=[]
    for _ in range(4):outputs.append(plan.attention(q,k,v,scale=scale))
    r.synchronize();after=r.execution_stats()
    assert after['paged_workspace_allocations']-before['paged_workspace_allocations']==1
    assert after['paged_workspace_bytes_total']-before['paged_workspace_bytes_total']==plan.workspace_bytes(q.shape[1],v.shape[-1])
    assert after['paged_split_calls']-before['paged_split_calls']==4
    reference=outputs[0].cpu()
    for output in outputs[1:]:torch.testing.assert_close(output.cpu(),reference)
