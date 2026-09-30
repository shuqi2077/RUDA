"""Actual C++ protocol with HOST callbacks; no Rust or PTX numerical work."""
import torch
import pytest
from test_v15_cpp import bridge
from test_paged_selected_cpp import tensors,plan

@pytest.fixture(scope='module')
def ordered(bridge):
    cpp,state,_=bridge;assert cpp.paged_backward_api_version==2;cpp.initialize_paged_backward(2);return cpp,state

@pytest.mark.parametrize('mla,mask',[(False,m) for m in range(8)]+[(True,m) for m in range(16)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_api2_selected_op_and_allocations(ordered,mla,mask,dtype):
    cpp,state=ordered;a=tensors(mla,dtype=dtype);p=plan(cpp,a[0]);needs=[bool(mask&(1<<i)) for i in range(4 if mla else 3)]
    before=state.next;calls=len(state.calls);out=p.backward_selected(*a,.37,True,needs,True)
    assert tuple(x is not None for x in out)==tuple(needs);assert state.next-before==sum(needs)
    if mask:assert state.calls[-1]==5
    else:assert len(state.calls)==calls

@pytest.mark.parametrize('mla',[False,True])
def test_strict_deterministic_accepts_ordered_and_rejects_atomic(ordered,mla):
    cpp,state=ordered;a=tensors(mla);p=plan(cpp,a[0]);old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        out=p.backward_selected(*a,.37,True,[True]*(4 if mla else 3),True);assert state.calls[-1]==5
        with pytest.raises(RuntimeError,match='deterministic'):p.backward_selected(*a,.37,True,[True]*(4 if mla else 3))
    finally:torch.use_deterministic_algorithms(old,warn_only=warn)

def test_ordinary_selected_still_op4(ordered):
    cpp,state=ordered;a=tensors();p=plan(cpp,a[0]);p.backward_selected(*a,.37,True,[True]*3);assert state.calls[-1]==4

@pytest.mark.parametrize('bad',['scale','mask','cpu','dtype','shape'])
def test_invalid_ordered_stops_before_native(ordered,bad):
    cpp,state=ordered;a=list(tensors());p=plan(cpp,a[0]);scale=.37;needs=[True]*3
    if bad=='scale':scale=float('nan')
    elif bad=='mask':needs=[]
    elif bad=='cpu':a[0]=torch.empty(a[0].shape)
    elif bad=='dtype':a[2]=a[2].new_empty(a[2].shape,dtype=torch.float16)
    elif bad=='shape':a[5]=a[5].new_empty((2,4,9))
    before=len(state.calls)
    with pytest.raises(RuntimeError):p.backward_selected(*a,scale,True,needs,True)
    assert len(state.calls)==before

def test_native_error_propagated(ordered):
    cpp,state=ordered;a=tensors();p=plan(cpp,a[0]);state.fail=True
    try:
        with pytest.raises(RuntimeError,match='explicit ABI test failure'):p.backward_selected(*a,.37,True,[True]*3,True)
    finally:state.fail=False
