"""Actual compiled C++ with explicit HOST ABI callbacks; no Rust/PTX/GPU math."""
import itertools
import pytest
import torch
from test_v15_cpp import bridge, _KEEP_ALIVE

@pytest.fixture(scope='module')
def selected(bridge):
    cpp,state,_=bridge
    cpp.initialize_paged_backward(1)
    return cpp,state

def tensors(mla=False,empty=False,dtype=torch.float32):
    q=torch.empty((0 if empty else 2,4,5),device='ruda',dtype=dtype)
    k=torch.empty((3,4,1,5),device='ruda',dtype=dtype)
    v=k if mla else torch.empty((3,4,1,7),device='ruda',dtype=dtype)
    qp=torch.empty((q.shape[0],4,3),device='ruda',dtype=dtype) if mla else None
    kp=torch.empty((3,4,1,3),device='ruda',dtype=dtype) if mla else None
    g=torch.empty((q.shape[0],4,v.shape[-1]),device='ruda',dtype=dtype)
    return q,k,v,qp,kp,g

def plan(cpp,q):
    # Callback inspects metadata only; mathematical validation happens in Rust.
    return cpp.NativePagedPlan(q,[4,3,1,q.shape[0],1,1],[])

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('mla,mask',[(False,m) for m in range(8)]+[(True,m) for m in range(16)])
def test_requested_outputs_only(selected,dtype,mla,mask):
    cpp,state=selected;a=tensors(mla,dtype=dtype);p=plan(cpp,a[0])
    needs=tuple(bool(mask&(1<<i)) for i in range(4 if mla else 3))
    before=state.next;calls=len(state.calls)
    out=p.backward_selected(*a,.3,True,needs)
    assert tuple(x is not None for x in out)==needs
    assert state.next-before==sum(needs) # actual allocator callback, not a source assertion
    if mask:
        assert state.calls[-1]==4
        if mla:expected=(needs[0],needs[2],False,needs[1],needs[3])
        else:expected=(*needs,False,False)
        assert state.last_gradient_descriptors==expected
    else:assert len(state.calls)==calls
    sources=(a[0],a[3],a[1],a[4]) if mla else a[:3]
    for x,src in zip(out,sources):
        if x is not None: assert x.dtype==src.dtype and x.shape==src.shape

@pytest.mark.parametrize('mla',[False,True])
def test_determinism_rejects_only_history(selected,mla):
    cpp,state=selected;a=tensors(mla);p=plan(cpp,a[0]);old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        n=(True,True,False,False) if mla else (True,False,False)
        out=p.backward_selected(*a,.3,True,n);assert out[0] is not None
        n=(False,False,True,False) if mla else (False,True,False)
        calls=len(state.calls);allocs=state.next
        with pytest.raises(RuntimeError,match='deterministic'): p.backward_selected(*a,.3,True,n)
        assert len(state.calls)==calls and state.next==allocs
    finally:torch.use_deterministic_algorithms(old,warn_only=warn)

def test_determinism_warn_only(selected,capfd):
    cpp,state=selected;a=tensors();p=plan(cpp,a[0]);old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True,warn_only=True)
        p.backward_selected(*a,.3,True,[True,True,True])
        assert 'does not have a deterministic implementation' in capfd.readouterr().err
    finally:torch.use_deterministic_algorithms(old,warn_only=warn)

@pytest.mark.parametrize('mla',[False,True])
def test_empty_query_does_not_require_atomic_policy(selected,mla):
    cpp,state=selected;a=tensors(mla,True);p=plan(cpp,a[0]);old=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        out=p.backward_selected(*a,.3,True,[True]*(4 if mla else 3))
        assert state.calls[-1]==4 # Rust must zero history even though q is empty.
    finally:torch.use_deterministic_algorithms(old)

@pytest.mark.parametrize('bad',['mask','shape','dtype','scale','strided','cpu'])
def test_invalid_never_enters_native(selected,bad):
    cpp,state=selected;a=list(tensors());p=plan(cpp,a[0]);needs=[True]*3;scale=.3
    if bad=='mask':needs=[True]
    if bad=='shape':a[5]=torch.empty((2,4,8),device='ruda')
    if bad=='dtype':a[2]=a[2].new_empty(a[2].shape,dtype=torch.float16)
    if bad=='scale':scale=float('nan')
    if bad=='strided':a[0]=a[0].transpose(1,2)
    if bad=='cpu':a[0]=torch.empty_like(a[0],device='cpu')
    calls=len(state.calls);allocs=state.next
    with pytest.raises(RuntimeError):p.backward_selected(*a,scale,True,needs)
    assert len(state.calls)==calls and state.next==allocs

def test_error_propagates(selected):
    cpp,state=selected;a=tensors();p=plan(cpp,a[0]);state.fail=True
    try:
        with pytest.raises(RuntimeError,match='explicit ABI test failure'):p.backward_selected(*a,.3,True,[True]*3)
    finally:state.fail=False
