"""Real compiled C++ protocol tests with an EXPLICIT host callback, not a GPU.
No numerical tensors returned by these recording callbacks are read as results.
"""
import ctypes as ct
import pytest
import torch
from test_v15_cpp import bridge, _KEEP_ALIVE

@pytest.fixture(scope='module')
def training_bridge(bridge):
    cpp,state,_=bridge
    state.training=[];state.training_fail=False
    def command(op,desc,n,scalars,ns):
        state.training.append((op,n,tuple(scalars[i] for i in range(ns))))
        return 17 if state.training_fail else 0
    cb=ct.CFUNCTYPE(ct.c_int,ct.c_uint32,ct.c_void_p,ct.c_size_t,ct.POINTER(ct.c_float),ct.c_size_t)(command)
    _KEEP_ALIVE.append(cb);cpp.initialize_training(ct.cast(cb,ct.c_void_p).value)
    return cpp,state

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('weighted',[False,True])
def test_forward_backward_mixed_weight_descriptor_shapes(training_bridge,dtype,weighted):
    cpp,s=training_bridge;x=torch.empty(3,17,device='ruda',dtype=dtype)
    w=torch.empty(17,device='ruda') if weighted else None
    y,stats=cpp.training_rms_forward(x,w,1e-5)
    assert y.dtype==dtype and stats.dtype==torch.float32 and stats.shape==(3,)
    dx,dw=cpp.training_rms_backward(x,w,y,stats,True,weighted)
    assert dx.shape==x.shape and dx.dtype==dtype
    assert dw is None if not weighted else dw.dtype==torch.float32
    assert s.training[-2][0:2]==(0,4) and s.training[-1][0:2]==(1,6)

@pytest.mark.parametrize('kind',['shape','dtype','strides','eps','cpu'])
def test_invalid_norm_never_calls_native(training_bridge,kind):
    cpp,s=training_bridge;x=torch.empty(3,17,device='ruda');w=torch.empty(17,device='ruda');eps=1e-5
    if kind=='shape':w=torch.empty(16,device='ruda')
    if kind=='dtype':w=torch.empty(17,device='ruda',dtype=torch.float16)
    if kind=='strides':x=x.t()
    if kind=='eps':eps=0
    if kind=='cpu':x=torch.empty(3,17)
    before=len(s.training)
    with pytest.raises(RuntimeError):cpp.training_rms_forward(x,w,eps)
    assert len(s.training)==before

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_layer_norm_protocol_fp32_stats_and_optional_affine(training_bridge,dtype):
    cpp,s=training_bridge;x=torch.empty(2,3,17,device='ruda',dtype=dtype)
    w=torch.empty(17,device='ruda',dtype=torch.float32);b=torch.empty(17,device='ruda',dtype=torch.float32)
    y,mean,rstd=cpp.training_layer_forward(x,w,b,1e-5)
    assert y.shape==x.shape and y.dtype==dtype and mean.dtype==rstd.dtype==torch.float32 and mean.shape==rstd.shape==(6,)
    dx,dw,db=cpp.training_layer_backward(x,w,b,y,mean,rstd,True,True,True)
    assert dx.shape==x.shape and dx.dtype==dtype and dw.dtype==db.dtype==torch.float32
    assert s.training[-2][0:2]==(9,6) and s.training[-1][0:2]==(10,9)

@pytest.mark.parametrize('needs',[(True,True),(False,True),(True,False)])
def test_gate_optional_gradients(training_bridge,needs):
    cpp,s=training_bridge;x=torch.empty(33,device='ruda',dtype=torch.float16)
    y=cpp.training_silu_forward(x,x)
    dg,du=cpp.training_silu_backward(x,x,y,*needs)
    assert (dg is not None,du is not None)==needs
    assert s.training[-1][2]==tuple(map(float,needs))


def test_unscale_bumps_versions_and_checks_workspace(training_bridge):
    cpp,s=training_bridge;g=torch.empty(99,device='ruda');workspace=torch.empty(4,device='ruda');flag=torch.empty((),device='ruda')
    version=g._version
    with torch.no_grad():cpp.training_unscale_([g],.125,workspace,flag)
    assert g._version==version+1 and s.training[-1][0]==4
    before=len(s.training)
    with torch.no_grad(),pytest.raises(RuntimeError,match='length'):cpp.training_unscale_([g],1.,workspace[:3],flag)
    assert len(s.training)==before

@pytest.mark.parametrize('kind',['same_gradient','param_grad','moments','master','negative_lr'])
def test_optimizer_alias_and_scalar_preflight(training_bridge,kind):
    cpp,s=training_bridge
    p=torch.empty(17,device='ruda');g=torch.empty_like(p);m=torch.empty_like(p);v=torch.empty_like(p);master=p;lr=.01
    if kind=='same_gradient':
        # Keep callback-owned allocations alive during Python exception translation.
        workspace=torch.empty(2,device='ruda');found=torch.empty((),device='ruda')
        with torch.no_grad(),pytest.raises(RuntimeError):cpp.training_unscale_([g,g],1.,workspace,found)
        return
    if kind=='param_grad':g=p
    if kind=='moments':v=m
    if kind=='master':master=torch.empty_like(p)
    if kind=='negative_lr':lr=-1
    before=len(s.training)
    with torch.no_grad(),pytest.raises(RuntimeError):cpp.training_adamw_(p,g,master,m,v,lr,.9,.99,1e-8,.01,.1,.01)
    assert len(s.training)==before

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_update_versions_and_fp32_master(training_bridge,dtype):
    cpp,s=training_bridge;p=torch.empty(17,device='ruda',dtype=dtype);g=torch.empty_like(p)
    m=torch.empty(17,device='ruda');v=torch.empty_like(m);master=p if dtype==torch.float32 else torch.empty_like(m)
    ver=p._version
    with torch.no_grad():cpp.training_adamw_(p,g,master,m,v,.01,.9,.99,1e-8,.01,.1,.01)
    assert p._version==ver+1 and m._version==1 and v._version==1


def test_native_failure_and_initialization_contract(training_bridge):
    cpp,s=training_bridge;x=torch.empty(17,device='ruda');s.training_fail=True
    try:
        with pytest.raises(RuntimeError,match='ABI test failure'):cpp.training_silu_forward(x,x)
    finally:s.training_fail=False
    with pytest.raises(RuntimeError,match='initialization'):cpp.initialize_training(1)
    assert cpp.training_api_version==4 and cpp.abi_version==10 and cpp.graph_api_version==2
