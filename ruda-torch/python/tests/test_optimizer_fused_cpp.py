"""Actual C++ bridge + explicit recording callback. No numerical GPU execution."""
import pytest
import torch
from test_training_cpp import training_bridge
from test_v15_cpp import bridge

def batch(dtype=torch.float32,n=2):
    ps=[torch.empty(i+17,device='ruda',dtype=dtype) for i in range(n)]
    gs=[torch.empty_like(p) for p in ps]
    masters=[p if dtype==torch.float32 else torch.empty(p.shape,device='ruda') for p in ps]
    first=[torch.empty(p.shape,device='ruda') for p in ps]
    second=[torch.empty(p.shape,device='ruda') for p in ps]
    return ps,gs,masters,first,second,[[.01,.9,.99,1e-8,.01,.1,.01] for _ in ps]

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('with_norm',[False,True])
def test_readonly_analysis_descriptors_versions(training_bridge,dtype,with_norm):
    c,s=training_bridge;gs=[torch.empty(n,device='ruda',dtype=dtype) for n in (0,33,100)]
    work=torch.empty(3*(1+2+4),device='ruda');report=torch.empty(3,device='ruda')
    versions=[g._version for g in gs]
    with torch.no_grad():c.training_analyze(gs,.125,work,report,with_norm)
    assert s.training[-1]==(6,5,(.125,float(with_norm)))
    assert [g._version for g in gs]==versions
    assert work._version==1 and report._version==1

@pytest.mark.parametrize('kind',['length','workspace','report','alias','inverse','cpu','grad'])
def test_analysis_preflight_rejects_before_native(training_bridge,kind):
    c,s=training_bridge;g=torch.empty(33,device='ruda');gs=[g];work=torch.empty(6,device='ruda');report=torch.empty(3,device='ruda');inv=1.
    if kind=='length':gs=[]
    if kind=='workspace':work=torch.empty(5,device='ruda')
    if kind=='report':report=torch.empty(1,device='ruda')
    if kind=='alias':work=g[:6]
    if kind=='inverse':inv=0
    if kind=='cpu':gs=[torch.empty(33)]
    if kind=='grad':g.requires_grad_()
    before=len(s.training)
    with torch.no_grad(),pytest.raises(RuntimeError):c.training_analyze(gs,inv,work,report,True)
    assert len(s.training)==before

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_batch_single_callback_and_unchanged_grads(training_bridge,dtype):
    c,s=training_bridge;args=batch(dtype,5)
    versions=[g._version for g in args[1]];before=len(s.training)
    with torch.no_grad():c.training_adamw_batch_(*args,.125,.5)
    assert len(s.training)==before+1
    op,n,scalars=s.training[-1];assert (op,n,len(scalars))==(7,25,37)
    assert scalars[:2]==(.125,.5)
    assert [g._version for g in args[1]]==versions
    assert all(p._version==1 for p in args[0])

@pytest.mark.parametrize('kind',['length','late_hyper','late_shape','cross_state','cross_gradient','master','grad','clip','inverse','requires_no_grad'])
def test_batch_preflight_all_or_none(training_bridge,kind):
    c,s=training_bridge;args=list(batch());inv=1.;clip=.5
    if kind=='length':args[1]=args[1][:1]
    if kind=='late_hyper':args[-1][-1][3]=0
    if kind=='late_shape':args[3][-1]=torch.empty(100,device='ruda')
    if kind=='cross_state':args[3][1]=args[3][0]  # shape check or alias check rejects
    if kind=='cross_gradient':args[1][0]=args[0][0]
    if kind=='master':args[2][0]=torch.empty_like(args[0][0])
    if kind=='grad':args[1][-1].requires_grad_()
    if kind=='clip':clip=1.1
    if kind=='inverse':inv=float('nan')
    before=len(s.training);versions=[p._version for p in args[0]]
    with torch.set_grad_enabled(kind=='requires_no_grad'),pytest.raises(RuntimeError):
        c.training_adamw_batch_(*args,inv,clip)
    assert len(s.training)==before and [p._version for p in args[0]]==versions


def test_valid_disjoint_views_and_same_shape_cross_alias_rejection(training_bridge):
    c,s=training_bridge;storage=torch.empty(128,device='ruda')
    ps=[storage[:16],storage[16:32]];gs=[storage[32:48],storage[48:64]]
    ms=[storage[64:80],storage[80:96]];vs=[storage[96:112],storage[112:128]]
    h=[[.01,.9,.99,1e-8,.01,.1,.01]]*2
    with torch.no_grad():c.training_adamw_batch_(ps,gs,ps,ms,vs,h,1.,1.)
    before=len(s.training)
    vs[1]=ms[0]
    with torch.no_grad(),pytest.raises(RuntimeError,match='overlap'):c.training_adamw_batch_(ps,gs,ps,ms,vs,h,1.,1.)
    assert len(s.training)==before


def test_native_failure_marks_all_destinations(training_bridge):
    c,s=training_bridge;args=batch();s.training_fail=True
    try:
        with torch.no_grad(),pytest.raises(RuntimeError,match='ABI test failure'):
            c.training_adamw_batch_(*args,1.,1.)
    finally:s.training_fail=False
    assert all(p._version==1 for p in args[0])
    assert all(g._version==0 for g in args[1])
