"""Real native Rust/PTX graph tests. Explicit GPU run only; no numerical mock.

RUDA_REQUIRE_GPU=1 turns missing hardware/package/compiler into failures. The
standalone v24 validator rejects all skips, empty runs and missing runtime marker.
"""
import os
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch

@pytest.fixture(scope='module')
def r():
    if os.environ.get('RUDA_REQUIRE_GPU')!='1':
        pytest.skip('explicit native GPU execution required')
    assert os.environ.get('RUDA_CUDA_COMPILER')=='ptx'
    import ruda_torch as r
    assert r._C.abi_version==10 and r._graph_available and r._C.graph_api_version==3
    # Runtime marker is emitted only AFTER a real native graph result was read.
    x=torch.tensor([1.,2.]).to('ruda')
    with r.StaticGraph({'x':x},[r.GraphOp.copy('y','x')]) as g:
        before=r.execution_stats()['static_graph_replays']
        torch.testing.assert_close(g.replay()['y'].cpu(),torch.tensor([1.,2.]))
        assert r.execution_stats()['static_graph_replays']==before+1
    print('RUDA_V24_STATIC_GRAPH_RUNTIME abi=10 graph_api=3',flush=True)
    return r

def tol(dtype): return {torch.float32:3e-5,torch.float16:5e-3,torch.bfloat16:4e-2}[dtype]

def check(actual,expected):
    torch.testing.assert_close(actual.cpu(),expected,rtol=tol(expected.dtype),atol=tol(expected.dtype),equal_nan=True)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('width',[7,33,4096])
@pytest.mark.parametrize('infer',[False,True])
def test_residual_rmsnorm(r,dtype,width,infer):
    torch.manual_seed(431)
    x=torch.randn(2,width).to(dtype);res=torch.randn_like(x);w=torch.randn(width).to(dtype)
    xd,rd,wd=(t.to('ruda') for t in (x,res,w))
    nodes=[r.GraphOp.add('sum','x','r'),r.GraphOp.rms_norm('y','sum','w',eps=1e-5)]
    with r.StaticGraph({'x':xd,'r':rd,'w':wd},nodes,outputs=('sum','y'),infer_dependencies=infer) as g:
        rounded=(x.float()+res.float()).to(dtype)
        y=(rounded.float()*torch.rsqrt(rounded.float().square().mean(-1,keepdim=True)+1e-5)*w.float()).to(dtype)
        result=g.replay();ptr=result['y'].data_ptr();check(result['sum'],rounded);check(result['y'],y)
        control=g.run_eager();assert control['y'].data_ptr()==ptr;check(control['y'],y)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('infer',[False,True])
def test_silu_gate_residual(r,dtype,infer):
    torch.manual_seed(43)
    a=torch.randn(3,129).to(dtype);b=torch.randn_like(a);res=torch.randn_like(a)
    ad,bd,rd=(t.to('ruda') for t in (a,b,res))
    # Preserve each existing operator's output cast boundary, not a fused algebraic rewrite.
    gate=(a.float()/(1.+torch.exp(-a.float()))).to(dtype)
    product=(gate.float()*b.float()).to(dtype)
    expected=(product.float()+0.5*res.float()).to(dtype)
    with r.StaticGraph({'a':ad,'b':bd,'r':rd},[
        r.GraphOp.silu('s','a'),r.GraphOp.mul('p','s','b'),r.GraphOp.add('y','p','r',alpha=.5)
    ],infer_dependencies=infer) as g:
        check(g.replay()['y'],expected);check(g.run_eager()['y'],expected)


def test_updated_input_contents_and_no_rebuild(r):
    x=torch.zeros(2,17,device='ruda');res=torch.ones(2,17).to('ruda')
    with r.StaticGraph({'x':x,'r':res},[r.GraphOp.add('y','x','r')]) as g:
        stats=r.execution_stats();ptr=None
        for v in range(4):
            x.copy_(torch.full((2,17),float(v)))
            after_upload=r.execution_stats();y=g.replay()['y']
            assert r.execution_stats()['host_to_device_bytes']==after_upload['host_to_device_bytes']
            if ptr is None:ptr=y.data_ptr()
            assert y.data_ptr()==ptr;check(y,torch.full((2,17),float(v+1)))
        final=r.execution_stats()
        assert final['static_graph_builds']==stats['static_graph_builds']
        assert final['static_graph_replays']==stats['static_graph_replays']+4


def test_inferred_fork_join(r):
    x=torch.tensor([1.,2.,3.]).to('ruda')
    with r.StaticGraph({'x':x},[r.GraphOp.copy('a','x'),r.GraphOp.copy('b','x'),r.GraphOp.add('y','a','b')],infer_dependencies=True) as g:
        check(g.replay()['y'],torch.tensor([2.,4.,6.]));assert g.info['edges']==2


def test_bound_shape_mutation_rejected(r):
    x=torch.ones(2,17).to('ruda')
    with r.StaticGraph({'x':x},[r.GraphOp.copy('y','x')]) as g:
        x.as_strided_((1,34),(34,1))
        with pytest.raises(RuntimeError,match='binding changed'):g.replay()


def test_wrong_stream_rejected(r):
    x=torch.ones(2).to('ruda')
    with r.StaticGraph({'x':x},[r.GraphOp.copy('y','x')]) as g:
        other=r.Stream()
        with r.stream(other):
            with pytest.raises(RuntimeError,match='creation stream'):g.replay()
        check(g.replay()['y'],torch.ones(2))


def test_graph_on_nondefault_stream(r):
    other=r.Stream()
    with r.stream(other):
        x=torch.ones(3).to('ruda')
        with r.StaticGraph({'x':x},[r.GraphOp.copy('y','x')],track_completion=True) as g:
            y=g.replay()['y'];g.wait_completion();assert g.query_completion();check(y,torch.ones(3))


def test_retained_results_after_close(r):
    x=torch.ones(4).to('ruda');g=r.StaticGraph({'x':x},[r.GraphOp.copy('y','x')])
    y=g.replay()['y'];g.close();del x;check(y,torch.ones(4))
    with pytest.raises(RuntimeError,match='closed'):g.replay()


def test_requires_grad_is_rejected(r):
    x=torch.ones(2).to('ruda').requires_grad_(True)
    with pytest.raises(ValueError,match='inference-only'):r.StaticGraph({'x':x},[r.GraphOp.copy('y','x')])


def test_cpu_and_unknown_operator_rejected(r):
    with pytest.raises(ValueError,match='native ruda'):r.StaticGraph({'x':torch.ones(2)},[r.GraphOp.copy('y','x')])
    x=torch.ones(2).to('ruda')
    with pytest.raises(ValueError,match='unsupported'):r.StaticGraph({'x':x},[r.GraphOp('matmul','y','x')])


def test_threaded_replays_are_serialized(r):
    # Fixed input, same default stream in each worker; no overlapping external writes.
    x=torch.ones(2,33).to('ruda')
    with r.StaticGraph({'x':x},[r.GraphOp.silu('y','x')]) as g:
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda _:g.replay(),range(16)))
        g.synchronize();check(g.replay()['y'],torch.nn.functional.silu(torch.ones(2,33)))

# v24 real-device coverage: counts are checked by validate_static_graph.py.
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('width',[1,31,33,257,4096])
@pytest.mark.parametrize('infer',[False,True])
def test_v24_fused_gate_matches_unfused_gpu(r,dtype,width,infer):
    torch.manual_seed(244)
    a=torch.randn(2,width).to(dtype).to('ruda');b=torch.randn(2,width).to(dtype).to('ruda')
    nodes=[r.GraphOp.silu('act','a'),r.GraphOp.mul('y','act','b')]
    with r.StaticGraph({'a':a,'b':b},nodes) as base, r.StaticGraph(
            {'a':a,'b':b},nodes,optimize=True,reuse_workspace=True,
            infer_dependencies=infer) as opt:
        expected=base.replay()['y'].cpu()
        actual=opt.replay()['y'].cpu()
        # FP16/BF16 must preserve the intermediate storage rounding, not merely
        # approximate a mathematically similar all-FP32 compound expression.
        tolerance=3e-6 if dtype==torch.float32 else 0.0
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
        assert opt.info['nodes']==1 and opt.info['workspace_bytes']==base.info['workspace_bytes']//2
        torch.testing.assert_close(opt.run_eager()['y'].cpu(),actual,rtol=0,atol=0,equal_nan=True)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('infer',[False,True])
def test_v24_chain_lifetime_and_repeat(r,dtype,infer):
    torch.manual_seed(245)
    x=torch.randn(2,129).to(dtype).to('ruda')
    nodes=[r.GraphOp.silu(f'n{i}','x' if i==0 else f'n{i-1}') for i in range(9)]
    with r.StaticGraph({'x':x},nodes,outputs=('n0','n8')) as base, r.StaticGraph(
            {'x':x},nodes,outputs=('n0','n8'),reuse_workspace=True,
            infer_dependencies=infer) as opt:
        assert opt.info['workspace_allocations']==4
        pointers=None
        for _ in range(5):
            x.copy_(torch.randn(2,129).to(dtype))
            expected={k:v.cpu() for k,v in base.replay().items()}
            actual=opt.replay()
            if pointers is None:pointers={k:v.data_ptr() for k,v in actual.items()}
            assert pointers=={k:v.data_ptr() for k,v in actual.items()}
            for k in expected:torch.testing.assert_close(actual[k].cpu(),expected[k],rtol=0,atol=0,equal_nan=True)
        held=actual['n0'];saved=held.cpu()
    torch.testing.assert_close(held.cpu(),saved,rtol=0,atol=0,equal_nan=True)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_v24_returned_activation_preserved(r,dtype):
    x=torch.randn(3,33).to(dtype).to('ruda');u=torch.ones(3,33,dtype=dtype).to('ruda')
    nodes=[r.GraphOp.silu('a','x'),r.GraphOp.mul('y','a','u')]
    with r.StaticGraph({'x':x,'u':u},nodes,outputs=('a','y'),optimize=True,reuse_workspace=True) as g:
        y=g.replay();g.synchronize()
        assert not g.info['fused_activations'] and g.info['workspace_allocations']==2
        assert y['a'].data_ptr()!=y['y'].data_ptr()
        torch.testing.assert_close(y['a'].cpu(),y['y'].cpu(),rtol=0,atol=0)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_v24_explicit_fused_node(r,dtype):
    x=torch.tensor([[0.,-0.,-10.,10.,1e-5]],dtype=dtype).to('ruda')
    u=torch.tensor([[1.,1.,-2.,3.,0.5]],dtype=dtype).to('ruda')
    with r.StaticGraph({'x':x,'u':u},[r.GraphOp.silu_mul('y','x','u')]) as fused, r.StaticGraph(
        {'x':x,'u':u},[r.GraphOp.silu('a','x'),r.GraphOp.mul('y','a','u')]) as base:
        a=fused.replay()['y'].cpu();b=base.replay()['y'].cpu()
        torch.testing.assert_close(a,b,rtol=3e-6 if dtype==torch.float32 else 0,atol=0,equal_nan=True)
        assert torch.equal(torch.signbit(a[:,:2]),torch.signbit(b[:,:2]))


def test_v24_invalid_dead_node_not_eliminated(r):
    x=torch.ones(2).to('ruda')
    with pytest.raises(ValueError,match='unsupported'):
        r.StaticGraph({'x':x},[r.GraphOp('unknown','unused','x'),r.GraphOp.copy('y','x')],optimize=True)


def test_v24_nondefault_stream_reuse_and_close(r):
    with r.stream(r.Stream()):
        x=torch.ones(2,33).to('ruda');u=torch.ones_like(x)
        nodes=[r.GraphOp.silu('act','x'),r.GraphOp.mul('p','act','u'),
               r.GraphOp.silu('a','p'),r.GraphOp.silu('b','a'),r.GraphOp.silu('y','b')]
        with r.StaticGraph({'x':x,'u':u},nodes,optimize=True,reuse_workspace=True,
                           infer_dependencies=True,track_completion=True) as g:
            result=g.replay()['y'];g.wait_completion();assert g.query_completion()
            assert torch.isfinite(result.cpu()).all()

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_training_bridge_native_forward_and_same_device_backward(r,dtype):
    """Real native training smoke test; never counted as covered by CPU mirrors."""
    torch.manual_seed(740)
    reference_x=torch.randn(2,17,dtype=dtype,requires_grad=True)
    reference_u=torch.randn_like(reference_x,requires_grad=True)
    reference_w=torch.randn(17,dtype=dtype,requires_grad=True)
    x=reference_x.detach().to('ruda').requires_grad_()
    u=reference_u.detach().to('ruda').requires_grad_()
    w=reference_w.detach().to('ruda').requires_grad_()
    nodes=[r.GraphOp.silu('s','x'),r.GraphOp.mul('p','s','u'),
           r.GraphOp.rms_norm('y','p','w',eps=1e-3)]
    s=(reference_x.float()/(1.+(-reference_x.float()).exp())).to(dtype)
    p=(s.float()*reference_u.float()).to(dtype)
    expected=(p.float()*torch.rsqrt(p.float().square().mean(-1,keepdim=True)+1e-3)
              *reference_w.float()).to(dtype)
    expected.float().sum().backward()
    with r.StaticGraph({'x':x,'u':u,'w':w},nodes,training=True,optimize=True,reuse_workspace=True) as graph:
        first=graph.replay()['y'];second=graph.replay()['y']
        assert first.data_ptr()!=second.data_ptr()
        check(first,expected.detach())
        first.float().sum().backward()
        check(x.grad,reference_x.grad);check(u.grad,reference_u.grad);check(w.grad,reference_w.grad)
        assert graph.info['backward_execution']=='same-device-eager'
