"""Load real C++ bridge; host-only ABI callbacks do NOT execute Rust/PTX/GPU.

Reuses the established base-ABI test fixture. New callbacks record arguments
and lifecycle, never fabricate numerical output or claim a GPU was executed.
Run in a fresh process with RUDA_CPP_TEST_LIBRARY set to a compiled _C module.
"""
import ctypes as ct
import gc
import pytest
import torch
from test_v15_cpp import bridge, _KEEP_ALIVE

class Node(ct.Structure):
    _fields_=[('op',ct.c_uint32),('a',ct.c_uint32),('b',ct.c_uint32),('scalar',ct.c_float)]

@pytest.fixture(scope='module')
def graph_bridge(bridge):
    cpp,state,_=bridge
    state.graphs={};state.graph_calls=[];state.graph_fail=False;state.current=0
    # Existing base callback reports stream zero. Changing streams here is tested
    # by replacing the base callback in a separate test file/process if required.
    vp=ct.c_void_p;u32=ct.c_uint32;u64=ct.c_uint64;sz=ct.c_size_t
    def command(op,plan,desc,n,ns,count,inputs,flags,result):
        state.graph_calls.append(op)
        if state.graph_fail:return 37
        if op==0:
            key=state.next;state.next+=1
            state.graphs[key]={'count':count,'inputs':inputs,'flags':flags,
                'nodes':[(ns[i].op,ns[i].a,ns[i].b,ns[i].scalar) for i in range(count)],'n':n}
            plan[0]=key;result[0]=max(0,count-1)
        elif op==4:
            state.graphs.pop(plan[0],None);plan[0]=None
        else:
            assert plan[0] in state.graphs
            result[0]=1 if op in (3,6) else 0
        return 0
    cb=ct.CFUNCTYPE(ct.c_int,u32,ct.POINTER(vp),vp,sz,ct.POINTER(Node),sz,sz,u32,ct.POINTER(u64))(command)
    _KEEP_ALIVE.append(cb);cpp.initialize_graph(ct.cast(cb,vp).value)
    return cpp,state

def ts(dtype=torch.float32):
    return [torch.empty((2,17),device='ruda',dtype=dtype) for _ in range(3)]
def build(cpp,tensors=None,tracked=False):
    return cpp.NativeStaticGraph(ts() if tensors is None else tensors,2,[1,0,1],[1.],tracked,False)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_graph_constructor_dtype(graph_bridge,dtype):
    cpp,s=graph_bridge;n=len(s.graphs);g=build(cpp,ts(dtype));assert len(s.graphs)==n+1;g.close();assert len(s.graphs)==n

def test_replay_eager_versions_and_query(graph_bridge):
    cpp,s=graph_bridge;t=ts();g=build(cpp,t);ver=t[-1]._version;before=len(s.graph_calls)
    g.run();g.run(True);assert t[-1]._version==ver+2
    assert s.graph_calls[before:]==[1,5];assert g.query();g.synchronize();g.close();g.close()
    with pytest.raises(RuntimeError,match='closed'):g.run()

def test_snapshot_detects_shape_mutation(graph_bridge):
    cpp,s=graph_bridge;t=ts();g=build(cpp,t);t[0].as_strided_((1,34),(34,1));before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='binding changed'):g.run()
    assert len(s.graph_calls)==before;g.close()

def test_requires_grad_after_build(graph_bridge):
    cpp,s=graph_bridge;t=ts();g=build(cpp,t);t[0].requires_grad_(True);before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='inference'):g.run()
    assert len(s.graph_calls)==before;g.close()

def test_tracked_completion_explicit(graph_bridge):
    cpp,s=graph_bridge;g=build(cpp,tracked=True);assert g.query_completion();g.wait_completion();g.close()
    h=build(cpp)
    with pytest.raises(RuntimeError,match='tracked'):h.query_completion()
    h.close()

def test_destructor_closes(graph_bridge):
    cpp,s=graph_bridge;n=len(s.graphs);g=build(cpp);assert len(s.graphs)==n+1;del g;gc.collect();assert len(s.graphs)==n

def test_native_replay_error_propagates(graph_bridge):
    cpp,s=graph_bridge;g=build(cpp);s.graph_fail=True
    try:
        with pytest.raises(RuntimeError,match='ABI test failure'):g.run()
    finally:s.graph_fail=False;g.close()

def test_close_failure_can_be_retried(graph_bridge):
    cpp,s=graph_bridge;g=build(cpp);s.graph_fail=True
    try:
        with pytest.raises(RuntimeError):g.close()
    finally:s.graph_fail=False
    g.close()

@pytest.mark.parametrize('words,scalars,match',[
    ([999,0,1],[0.],'unsupported'),([1,2,1],[1.],'future'),
    ([14,0,1],[0.],'unary'),([14,0,0],[-0.],'unary'),([1,0,1],[float('nan')],'scalar'),
    ([2,0,1],[1.],'scalar'),([100,0,1],[0.],'epsilon'),([],[],'header')])
def test_invalid_nodes_no_native_call(graph_bridge,words,scalars,match):
    cpp,s=graph_bridge;before=len(s.graph_calls);t=ts()
    # Keep Python callback-backed test allocations alive through exception translation.
    # Production uses Rust free(), not Python ctypes callbacks.
    with pytest.raises(RuntimeError,match=match):cpp.NativeStaticGraph(t,2,words,scalars,False,False)
    assert len(s.graph_calls)==before

def test_cpu_rejected(graph_bridge):
    cpp,s=graph_bridge;before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='ruda:0'):build(cpp,[torch.empty(2,17) for _ in range(3)])
    assert len(s.graph_calls)==before

def test_writable_alias_rejected(graph_bridge):
    cpp,s=graph_bridge;t=ts();t[2]=t[0];before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='distinct storage'):build(cpp,t)
    assert len(s.graph_calls)==before

def test_readonly_alias_allowed(graph_bridge):
    cpp,s=graph_bridge;t=ts();t[1]=t[0];g=build(cpp,t);g.close()

def test_dtype_mismatch(graph_bridge):
    cpp,s=graph_bridge;t=ts();t[1]=torch.empty(2,17,device='ruda',dtype=torch.float16)
    with pytest.raises(RuntimeError,match='promotion'):build(cpp,t)

def test_duplicate_init_rejected(graph_bridge):
    cpp,s=graph_bridge
    with pytest.raises(RuntimeError,match='repeated'):cpp.initialize_graph(1)

def test_inference_mode_output_guard(graph_bridge):
    cpp,s=graph_bridge
    with torch.inference_mode():t=ts();g=build(cpp,t);g.run()
    before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='inference_mode'):g.run()
    assert len(s.graph_calls)==before
    g.close()

def test_multi_node_header(graph_bridge):
    cpp,s=graph_bridge;t=ts()+[torch.empty(17,device='ruda'),torch.empty(2,17,device='ruda')]
    # 3 inputs x,r,w then output sum,y.
    t=[t[0],t[1],t[3],t[2],t[4]]
    g=cpp.NativeStaticGraph(t,3,[1,0,1,100,3,2],[1.,1e-5],True,True)
    meta=list(s.graphs.values())[-1];assert meta['flags']==3 and meta['nodes'][1][:3]==(100,3,2)
    assert g.edge_count==1;g.close()


def test_wrong_stream_rejected_before_native(graph_bridge):
    cpp,s=graph_bridge;g=build(cpp);before=len(s.graph_calls);s.stream=1
    try:
        with pytest.raises(RuntimeError,match='creation stream'):g.run()
        assert len(s.graph_calls)==before
    finally:s.stream=0;g.close()

@pytest.fixture(scope='module')
def frontend(graph_bridge):
    import importlib.util
    from pathlib import Path
    import sys,types
    cpp,state=graph_bridge
    name='explicit_static_graph_cpp_test_package'
    package=types.ModuleType(name);package.__path__=[str(Path(__file__).resolve().parents[1]/'ruda_torch')]
    package._C=cpp;package._graph_available=True;sys.modules[name]=package
    module=__import__(name+'._graph',fromlist=['StaticGraph','GraphOp'])
    return module,state


def test_real_python_wrapper_reuses_outputs(frontend):
    m,s=frontend;t=ts();before=len(s.allocations)
    with m.StaticGraph({'x':t[0],'r':t[1]},[m.GraphOp.add('sum','x','r'),m.GraphOp.silu('y','sum')],outputs=('sum','y')) as plan:
        assert len(s.allocations)==before+2
        a=plan.replay();ptr=a['y'].data_ptr();allocs=len(s.allocations)
        b=plan.run_eager();assert b['y'].data_ptr()==ptr and len(s.allocations)==allocs
        assert plan.info['nodes']==2 and plan.info['workspace_bytes']==2*2*17*4
    with pytest.raises(RuntimeError,match='closed'):plan.replay()
    assert a['y'].data_ptr()==ptr


def test_wrapper_cpu_input_never_falls_back(frontend):
    m,s=frontend;before=len(s.graph_calls)
    with pytest.raises(ValueError,match='native ruda'):m.StaticGraph({'x':torch.empty(1)},[m.GraphOp.silu('y','x')])
    assert len(s.graph_calls)==before


def test_wrapper_unknown_operator_preallocation(frontend):
    m,s=frontend;t=ts();before=(len(s.graph_calls),len(s.allocations))
    with pytest.raises(ValueError,match='unsupported'):m.StaticGraph({'x':t[0]},[m.GraphOp('matmul','y','x')])
    assert (len(s.graph_calls),len(s.allocations))==before


def test_wrapper_missing_capability_explicit(frontend):
    m,s=frontend;import sys
    package=sys.modules[m.__package__];package._graph_available=False
    try:
        with pytest.raises(RuntimeError,match='unavailable'):m.StaticGraph({},[])
    finally:package._graph_available=True

# v24: actual C++ validation against host-only callback allocations. These tests
# do not claim the callback executes any numerical GPU kernel.
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_storage_rounded_silu_mul_code(graph_bridge,dtype):
    cpp,s=graph_bridge;t=ts(dtype)
    g=cpp.NativeStaticGraph(t,2,[101,0,1],[0.],False,False)
    assert list(s.graphs.values())[-1]['nodes']==[(101,0,1,0.)]
    assert cpp.graph_api_version==3
    g.run();g.close()

@pytest.mark.parametrize('scalar',[-0.,1.,float('nan'),float('inf')])
def test_silu_mul_scalar_rejected(graph_bridge,scalar):
    cpp,s=graph_bridge;t=ts();before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='scalar'):
        cpp.NativeStaticGraph(t,2,[101,0,1],[scalar],False,False)
    assert len(s.graph_calls)==before

@pytest.mark.parametrize('infer',[False,True])
def test_exact_reuse_after_last_consumer(graph_bridge,infer):
    cpp,s=graph_bridge;x,a,b,y=[torch.empty(2,17,device='ruda') for _ in range(4)]
    # x -> a -> b -> c(same storage as a) -> y
    g=cpp.NativeStaticGraph([x,a,b,a,y],1,[0,0,0,14,1,1,14,2,2,0,3,3],
                            [0.]*4,False,infer)
    version=a._version;g.run();assert a._version==version+2
    g.run(True);g.close()

@pytest.mark.parametrize('case',['same_node','future_consumer','unused_leaf'])
def test_unsafe_workspace_reuse_rejected(graph_bridge,case):
    cpp,s=graph_bridge;x,a,b,y=[torch.empty(2,17,device='ruda') for _ in range(4)]
    if case=='same_node':
        tensors=[x,a,a];words=[0,0,0,14,1,1]
    elif case=='future_consumer':
        tensors=[x,a,b,a,y];words=[0,0,0,14,1,1,14,2,2,1,1,3]
    else:
        tensors=[x,a,b,a];words=[0,0,0,14,0,0,14,2,2]
    before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='lifetimes'):
        cpp.NativeStaticGraph(tensors,1,words,[0.]*(len(words)//3),False,False)
    assert len(s.graph_calls)==before


def test_shared_disjoint_output_views_still_rejected(graph_bridge):
    cpp,s=graph_bridge;x=torch.empty(2,17,device='ruda');buf=torch.empty(68,device='ruda')
    a=buf.as_strided((2,17),(17,1),0);b=buf.as_strided((2,17),(17,1),34)
    before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='distinct storage'):
        cpp.NativeStaticGraph([x,a,b],1,[0,0,0,14,1,1],[0.,0.],False,False)
    assert len(s.graph_calls)==before


def test_allocation_rebinding_rejected(graph_bridge):
    cpp,s=graph_bridge;t=ts();replacement=torch.empty_like(t[0]);g=build(cpp,t)
    t[0].data=replacement;before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='binding changed'):g.run()
    assert len(s.graph_calls)==before;g.close()


def test_same_shape_offset_change_rejected(graph_bridge):
    cpp,s=graph_bridge;buf=torch.empty(70,device='ruda')
    x=buf.as_strided((2,17),(17,1),1);t=[x,*ts()[:2]];g=build(cpp,t)
    x.as_strided_((2,17),(17,1),2);before=len(s.graph_calls)
    with pytest.raises(RuntimeError,match='binding changed'):g.run()
    assert len(s.graph_calls)==before;g.close()


def test_wrapper_fusion_has_one_allocation_and_node(frontend):
    m,s=frontend;t=ts();before=len(s.allocations)
    with m.StaticGraph({'x':t[0],'up':t[1]},[
        m.GraphOp.silu('unused','up'),m.GraphOp.silu('act','x'),m.GraphOp.mul('y','act','up')],
        optimize=True,reuse_workspace=True) as g:
        assert len(s.allocations)==before+1
        assert g.info['nodes']==1 and g.info['unoptimized_nodes']==3
        assert g.info['fused_activations']==('act',)
        assert g.info['eliminated_outputs']==('unused',)
        assert list(s.graphs.values())[-1]['nodes']==[(101,0,1,0.)]
        g.replay();g.run_eager()

@pytest.mark.parametrize('infer',[False,True])
def test_wrapper_chain_allocations_and_public_pin(frontend,infer):
    m,s=frontend;x=torch.empty(2,17,device='ruda');before=len(s.allocations)
    nodes=[m.GraphOp.silu(f'n{i}','x' if i==0 else f'n{i-1}') for i in range(9)]
    with m.StaticGraph({'x':x},nodes,outputs=('n0','n8'),reuse_workspace=True,
                      infer_dependencies=infer) as g:
        assert len(s.allocations)==before+4
        assert g.info['workspace_allocations']==4
        assert g.info['logical_workspace_bytes']==9*2*17*4
        assert g.info['workspace_bytes']==4*2*17*4
        a=g.replay();ptrs={k:v.data_ptr() for k,v in a.items()}
        assert len(set(ptrs.values()))==2
        for _ in range(3):
            assert {k:v.data_ptr() for k,v in g.replay().items()}==ptrs
        assert len(s.allocations)==before+4
    assert {k:v.data_ptr() for k,v in a.items()}==ptrs


def test_public_activation_disables_fusion(frontend):
    m,s=frontend;t=ts()
    with m.StaticGraph({'x':t[0],'up':t[1]},[
        m.GraphOp.silu('act','x'),m.GraphOp.mul('y','act','up')],
        outputs=('act','y'),optimize=True,reuse_workspace=True) as g:
        assert g.info['nodes']==2 and not g.info['fused_activations']
        assert g.info['workspace_allocations']==2
