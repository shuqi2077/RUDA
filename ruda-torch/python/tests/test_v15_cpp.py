"""Load the actual C++ extension with EXPLICIT host ABI callbacks.

No Rust library, PTX, numerical operator or GPU is executed. Run this file in a
fresh process with RUDA_CPP_TEST_LIBRARY=/absolute/path/to/_C.so. The callbacks
exist only in this test and are never a production compatibility fallback.
"""
import ctypes as ct
import gc
import importlib.util
import os
import types
import pytest
import torch

_KEEP_ALIVE=[]

@pytest.fixture(scope='module')
def bridge():
    path=os.environ.get('RUDA_CPP_TEST_LIBRARY')
    if not path:pytest.skip('explicit compiled C++ test module required')
    spec=importlib.util.spec_from_file_location('_C',path);cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
    state=types.SimpleNamespace(allocations={},plans={},calls=[],next=1,fail=False)
    error=ct.create_string_buffer(b'explicit ABI test failure')
    def callback(args,function):return ct.CFUNCTYPE(ct.c_int,*args)(function)
    def alloc(size,handle,address):
        key=state.next;state.next+=1;buffer=ct.create_string_buffer(max(1,size))
        state.allocations[key]=buffer;handle[0]=key;address[0]=ct.addressof(buffer);return 0
    def free(handle):state.allocations.pop(handle,None);return 0
    def stream(op,stream,obj,flags,result):result[0]=1 if op in (3,7) else 0;return 0
    def paged(op,plan,q,k,v,qp,kp,out,words,nwords,spec,scale,causal):
        state.calls.append(op)
        if state.fail:return 19
        if op==0:
            key=state.next;state.next+=1
            state.plans[key]={'spec':tuple(spec[i] for i in range(6)),'words':tuple(words[i] for i in range(nwords))}
            plan[0]=key
        elif op==1:
            assert plan[0] in state.plans
            # Protocol only. The allocated output is not claimed to contain attention.
        elif op==2:state.plans.pop(plan[0]);plan[0]=None
        return 0
    vp=ct.c_void_p;u32=ct.c_uint32;u64=ct.c_uint64;sz=ct.c_size_t
    callbacks=[
        callback([sz,ct.POINTER(vp),ct.POINTER(u64)],alloc),
        callback([vp],free),
        ct.CFUNCTYPE(vp)(lambda:ct.addressof(error)),
        callback([u32,vp,vp,vp,ct.c_float],lambda *a:19),
        callback([vp,vp,ct.c_bool],lambda *a:19),
        callback([],lambda:0),
        callback([vp,u64],lambda *a:19),
        callback([u32,vp,vp,vp,vp,sz],lambda *a:19),
        callback([vp,vp,vp,vp,ct.c_float,ct.c_float],lambda *a:19),
        callback([vp,vp,vp,vp,vp,vp,ct.c_float],lambda *a:19),
        callback([vp,vp,vp,ct.c_float],lambda *a:19),
        callback([u32,u64,u64,u32,ct.POINTER(u64)],stream),
        callback([u32,ct.POINTER(vp),vp,vp,vp,vp,vp,vp,ct.POINTER(u32),sz,ct.POINTER(u32),ct.c_float,ct.c_bool],paged),
    ]
    _KEEP_ALIVE.extend(callbacks+[error,state,cpp])
    addresses=[ct.cast(x,vp).value for x in callbacks]
    cpp.initialize(addresses)
    torch.utils.rename_privateuse1_backend('ruda')
    torch._register_device_module('ruda',types.SimpleNamespace(is_available=lambda:True,current_device=lambda:0,
        device_count=lambda:1,_is_in_bad_fork=lambda:False))
    return cpp,state,addresses


def tensors():
    return (torch.empty((1,2,3),device='ruda'),torch.empty((1,4,1,3),device='ruda'),torch.empty((1,4,1,5),device='ruda'))


def test_cpp_module_loads_current_abi_and_registrations(bridge):
    cpp,_,_=bridge;assert cpp.abi_version==9
    assert torch._C._dispatch_has_kernel_for_dispatch_key('aten::record_stream','PrivateUse1')

@pytest.mark.parametrize('count',[0,5,7])
def test_cpp_rejects_wrong_header_length(bridge,count):
    cpp,state,_=bridge;q,_,_=tensors();before=len(state.calls)
    with pytest.raises(RuntimeError,match='header'):cpp.NativePagedPlan(q,[1]*count,[])
    assert len(state.calls)==before


def test_cpp_six_word_header_and_plan_lifetime(bridge):
    cpp,state,_=bridge;q,k,v=tensors();before=set(state.plans)
    plan=cpp.NativePagedPlan(q,[4,1,1,1,1,8],[0,0,1,0]);keys=set(state.plans)-before
    assert len(keys)==1;key=keys.pop();assert state.plans[key]['spec'][-1]==8
    out=plan.run(q,k,v,None,None,0.5,True)
    assert tuple(out.shape)==(1,2,5) and out.device.type=='ruda'
    del out,plan;gc.collect();assert key not in state.plans


def test_cpp_native_failure_propagates(bridge):
    cpp,state,_=bridge;q,_,_=tensors();state.fail=True
    try:
        with pytest.raises(RuntimeError,match='explicit ABI test failure'):cpp.NativePagedPlan(q,[4,1,1,1,1,8],[0,0,1,0])
    finally:state.fail=False


def test_cpp_duplicate_initialization_rejected(bridge):
    cpp,_,addresses=bridge
    with pytest.raises(RuntimeError,match='repeated'):cpp.initialize(addresses)


def test_cpp_cpu_tensor_does_not_enter_native_paged(bridge):
    cpp,state,_=bridge;before=len(state.calls)
    with pytest.raises(RuntimeError,match='ruda:0'):cpp.NativePagedPlan(torch.empty(1,2,3),[4,1,1,1,1,8],[0,0,1,0])
    assert len(state.calls)==before
