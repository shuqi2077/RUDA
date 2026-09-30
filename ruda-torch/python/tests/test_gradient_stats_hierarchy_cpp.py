"""Real C++ validation with recording callbacks. No GPU numerics executed."""
import pytest
import torch
from test_training_cpp import training_bridge
from test_v15_cpp import bridge
from gradient_stats_reference import layout

def make_args(dtype=torch.float32,ns=(32768,1)):
    gs=[torch.empty(n,device='ruda',dtype=dtype) for n in ns]
    rows=sum(max(1,min(1024,(n+31)//32)) for n in ns)
    plan=layout.statistics_plan(rows)
    return gs,torch.empty(rows*3,device='ruda'),torch.empty(plan.scratch_elements,device='ruda'),torch.empty(3,device='ruda')

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('ns',[(0,),(31,33),(32768,),(32768,1),(32768,)*4])
@pytest.mark.parametrize('with_norm',[False,True])
def test_descriptors_and_versions(training_bridge,dtype,ns,with_norm):
    c,state=training_bridge;gs,w,s,r=make_args(dtype,ns)
    versions=[g._version for g in gs]
    with torch.no_grad():c.training_analyze_hierarchical(gs,.125,w,s,r,with_norm)
    assert state.training[-1]==(8,len(gs)+3,(.125,float(with_norm)))
    assert [g._version for g in gs]==versions
    assert w._version==1 and r._version==1 and s._version==int(s.numel()>0)

@pytest.mark.parametrize('kind',['empty_list','scratch_short','scratch_long','scratch_dtype',
  'workspace_short','report_len','report_alias','scratch_alias','same_grad','inverse','cpu','grad','no_grad'])
def test_reject_before_writes(training_bridge,kind):
    c,state=training_bridge;gs,w,s,r=make_args();inverse=1.
    if kind=='empty_list':gs=[]
    if kind=='scratch_short':s=s[:3]
    if kind=='scratch_long':s=torch.empty(9,device='ruda')
    if kind=='scratch_dtype':s=torch.empty(6,device='ruda',dtype=torch.float16)
    if kind=='workspace_short':w=w[:-1]
    if kind=='report_len':r=r[:2]
    if kind=='report_alias':r=w[:3]
    if kind=='scratch_alias':s=w[:6]
    if kind=='same_grad':gs[1]=gs[0][:1]
    if kind=='inverse':inverse=0
    if kind=='cpu':gs[0]=torch.empty(32768)
    if kind=='grad':gs[0].requires_grad_()
    before=len(state.training);versions=[x._version for x in (w,s,r)]
    with torch.set_grad_enabled(kind=='no_grad'),pytest.raises(RuntimeError):
        c.training_analyze_hierarchical(gs,inverse,w,s,r,True)
    assert len(state.training)==before and versions==[x._version for x in (w,s,r)]

def test_failure_marks_scratch_not_grad(training_bridge):
    c,state=training_bridge;gs,w,s,r=make_args();state.training_fail=True
    try:
        with torch.no_grad(),pytest.raises(RuntimeError):c.training_analyze_hierarchical(gs,1.,w,s,r,True)
    finally:state.training_fail=False
    assert all(g._version==0 for g in gs)
    assert all(x._version==1 for x in (w,s,r))

def test_disjoint_views_same_allocation(training_bridge):
    c,state=training_bridge;gs,w,s,r=make_args();buffer=torch.empty(w.numel()+s.numel()+3,device='ruda')
    w2=buffer[:w.numel()];s2=buffer[w.numel():w.numel()+s.numel()];r2=buffer[-3:]
    with torch.no_grad():c.training_analyze_hierarchical(gs,1.,w2,s2,r2,True)
    assert state.training[-1][0]==8
