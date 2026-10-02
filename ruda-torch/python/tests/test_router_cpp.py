"""Actual compiled C++ ABI tests with EXPLICIT host callbacks; no GPU math."""
import ctypes as ct
import pytest
import torch
from test_v15_cpp import bridge, _KEEP_ALIVE


@pytest.fixture(scope='module')
def router_bridge(bridge):
    cpp,state,_=bridge;state.router=[];state.router_fail=False
    class Descriptor(ct.Structure):
        _fields_=[('allocation',ct.c_void_p),('offset',ct.c_size_t),('rank',ct.c_size_t),
                  ('shape',ct.POINTER(ct.c_size_t)),('strides',ct.POINTER(ct.c_size_t)),('dtype',ct.c_uint32)]
    def call(op,ds,n,scoring,norm,scale):
        records=[(tuple(ds[i].shape[j] for j in range(ds[i].rank)),ds[i].dtype) for i in range(n)]
        state.router.append((op,records,scoring,norm,scale));return 17 if state.router_fail else 0
    cb=ct.CFUNCTYPE(ct.c_int,ct.c_uint32,ct.POINTER(Descriptor),ct.c_size_t,ct.c_uint32,ct.c_bool,ct.c_float)(call)
    _KEEP_ALIVE.append(cb);cpp.initialize_router(ct.cast(cb,ct.c_void_p).value)
    return cpp,state


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('itype',[torch.int32,torch.int64])
@pytest.mark.parametrize('mode',[0,1])
def test_real_forward_backward_descriptors(router_bridge,dtype,itype,mode):
    cpp,s=router_bridge;x=torch.empty(3,65,device='ruda',dtype=dtype);ids=torch.empty(3,8,device='ruda',dtype=itype)
    y=cpp.router_weights_forward(x,ids,mode,True,2.5)
    assert y.shape==(3,8) and y.dtype==torch.float32
    version=(x._version,ids._version,y._version)
    dx=cpp.router_weights_backward(x,ids,y,mode,True,2.5)
    assert dx.dtype==dtype and dx.shape==x.shape
    assert version==(x._version,ids._version,y._version)
    assert s.router[-2][0]==0 and s.router[-1][0]==1 and len(s.router[-1][1])==4
    assert s.router[-1][1][1][1]==(4 if itype==torch.int64 else 5)
    assert s.router[-1][1][2][1]==0


@pytest.mark.parametrize('kind',['cpu','float_indices','transposed','wrong_tokens','zero_k','too_many','zero_experts','unknown_mode','scale_zero','scale_inf'])
def test_preflight_failure_never_enters_native(router_bridge,kind):
    cpp,s=router_bridge;x=torch.empty(3,65,device='ruda');ids=torch.empty(3,2,device='ruda',dtype=torch.int64);mode=0;scale=1.
    if kind=='cpu':x=torch.empty(3,65)
    if kind=='float_indices':ids=torch.empty(3,2,device='ruda')
    if kind=='transposed':x=x.t()
    if kind=='wrong_tokens':ids=torch.empty(2,2,device='ruda',dtype=torch.int64)
    if kind=='zero_k':ids=torch.empty(3,0,device='ruda',dtype=torch.int64)
    if kind=='too_many':ids=torch.empty(3,65,device='ruda',dtype=torch.int64)
    if kind=='zero_experts':x=torch.empty(3,0,device='ruda')
    if kind=='unknown_mode':mode=2
    if kind=='scale_zero':scale=0.
    if kind=='scale_inf':scale=float('inf')
    before=len(s.router)
    with pytest.raises(RuntimeError):cpp.router_weights_forward(x,ids,mode,False,scale)
    assert len(s.router)==before


@pytest.mark.parametrize('kind',['dtype','shape','cpu','strides'])
def test_backward_gradient_preflight(router_bridge,kind):
    cpp,s=router_bridge;x=torch.empty(2,17,device='ruda');ids=torch.empty(2,3,device='ruda',dtype=torch.int64)
    grad=torch.empty(2,3,device='ruda')
    if kind=='dtype':grad=torch.empty(2,3,device='ruda',dtype=torch.float16)
    if kind=='shape':grad=torch.empty(2,2,device='ruda')
    if kind=='cpu':grad=torch.empty(2,3)
    if kind=='strides':grad=torch.empty(3,2,device='ruda').t()
    before=len(s.router)
    with pytest.raises(RuntimeError):cpp.router_weights_backward(x,ids,grad,0,True,1.)
    assert len(s.router)==before


def test_native_error_propagates_without_fallback(router_bridge):
    cpp,s=router_bridge;x=torch.empty(2,17,device='ruda');ids=torch.empty(2,3,device='ruda',dtype=torch.int64)
    s.router_fail=True
    try:
        with pytest.raises(RuntimeError,match='ABI test failure'):cpp.router_weights_forward(x,ids,0,False,1.)
    finally:s.router_fail=False


def test_optional_api_version_and_duplicate_init(router_bridge):
    cpp,_=router_bridge
    assert cpp.abi_version==10 and cpp.training_api_version==4 and cpp.graph_api_version==3 and cpp.router_api_version==1
    with pytest.raises(RuntimeError,match='initialization'):cpp.initialize_router(1)
