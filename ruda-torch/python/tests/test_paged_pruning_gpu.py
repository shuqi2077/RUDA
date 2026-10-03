"""Real device acceptance for v34; host tensors are references only."""
import torch
import pytest
from paged_pruning_reference import make_spec,tensors
from paged_selected_reference import dense
from test_paged_ordered_gpu import backend,call
from test_paged_selected_gpu import DTYPES,tol

@pytest.mark.parametrize('dtype_name',DTYPES)
@pytest.mark.parametrize('mla',[False,True])
@pytest.mark.parametrize('mode',['monotonic','duplicate','unsorted','blocked','empty'])
@pytest.mark.parametrize('causal',[True,False])
def test_v34_visibility_numerics_and_reuse(backend,dtype_name,mla,mode,causal):
    dtype=getattr(torch,dtype_name);spec=make_spec(mode,129);ts=tensors(spec,mla,dtype=dtype)
    ref=[x.detach().float().requires_grad_() for x in ts];dev=[x.detach().to('ruda').requires_grad_() for x in ts]
    y=dense(ref,mla,spec=spec,causal=causal);g=torch.linspace(-.3,.4,y.numel()).reshape_as(y).to(dtype)
    y.backward(g.float());plan=backend.PagedAttentionPlan(**spec,backward_strategy='ordered')
    snapshots=[];before=backend.execution_stats()['paged_backward_ordered_workspace_allocations']
    for i in range(2):
        for x in dev:x.grad=None
        z=call(plan,dev,mla,causal);z.backward(g.to('ruda'));backend.synchronize()
        snap=[x.grad.detach().cpu() for x in dev];snapshots.append(snap)
        for a,b in zip(snap,ref):torch.testing.assert_close(a.float(),b.grad,atol=tol(dtype),rtol=tol(dtype))
        count=backend.execution_stats()['paged_backward_ordered_workspace_allocations']
        assert count-before==int(mode!='empty')
    for a,b in zip(*snapshots):assert torch.equal(a,b),'same-device ordered repeat changed'
    for i in ((2,3) if mla else (1,2)):assert torch.count_nonzero(snapshots[0][i][16:])==0
    print('RUDA_V34_PRUNING_GPU_EXECUTED',flush=True)
