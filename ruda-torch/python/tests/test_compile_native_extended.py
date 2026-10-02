"""API-3 lowering/planning under an explicit CPU native-graph simulator, NOT GPU tests."""
import pytest
import torch
from test_compile_native import simulated, make_call, native, compiler, spec


@pytest.mark.parametrize('name', [n for n in spec.UNARY_CODES if n != 'copy'])
def test_unary_exact_overloads(simulated, name):
    x = torch.linspace(1.01, 1.1, 12).reshape(3,4) if name == 'acosh' else torch.linspace(.1,.8,12).reshape(3,4)
    fn = lambda x: getattr(torch.ops.aten,name).default(x)
    backend, call = make_call(fn,[x],native='required')
    torch.testing.assert_close(call([x]),fn(x))
    assert backend.info['native_replays'] == 1
    backend.close()


@pytest.mark.parametrize('name', ['add','sub','mul','div'])
@pytest.mark.parametrize('scalar', [False,True])
def test_binary_scalar_overloads(simulated,name,scalar):
    x,y = torch.rand(3,4)+.3,torch.rand(3,4)+.3
    fn = (lambda x: getattr(torch,name)(x,2.3)) if scalar else (lambda x,y:getattr(torch,name)(x,y))
    args=[x] if scalar else [x,y]
    backend,call=make_call(fn,args,native='required')
    torch.testing.assert_close(call(args),fn(*args))
    assert backend.info['native_replays']==1


@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_matrix_output_shape_and_storage(simulated,batched,dtype):
    prefix=(2,) if batched else ()
    x,y=torch.randn(*prefix,3,5,dtype=dtype),torch.randn(*prefix,5,7,dtype=dtype)
    fn=torch.bmm if batched else torch.mm
    backend,call=make_call(fn,[x,y],native='required')
    first=call([x,y]); second=call([x+1,y])
    torch.testing.assert_close(first,fn(x,y))
    assert first.shape==(*prefix,3,7) and first.data_ptr()!=second.data_ptr()
    assert backend.info['native_cache_hits']==1


@pytest.mark.parametrize('fn',[lambda x:x.sum((0,2),keepdim=True),
    lambda x:x.mean((-1,),keepdim=True),lambda x:torch.softmax(x,-1),
    lambda x:torch.log_softmax(x,1)])
def test_reductions_and_softmax(simulated,fn):
    x=torch.randn(2,3,4)
    backend,call=make_call(fn,[x],native='required')
    torch.testing.assert_close(call([x]),fn(x))
    assert backend.info['native_replays']==1


def test_attention_core_with_backward_native_regions(simulated):
    def fn(q,k,v):
        scores=torch.bmm(q,k)/2.
        return torch.bmm(torch.softmax(scores,-1),v).tanh()
    args=[torch.randn(2,3,4,requires_grad=True),torch.randn(2,4,5,requires_grad=True),
          torch.randn(2,5,6,requires_grad=True)]
    refs=[x.detach().clone().requires_grad_() for x in args]
    compiled=compiler.compile(fn,device_type='cpu',min_native_ops=1)
    got,expected=compiled(*args),fn(*refs)
    got.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(got,expected)
    for x,r in zip(args,refs):torch.testing.assert_close(x.grad,r.grad)
    assert any(op['target'] == 'aten.bmm.default' and op['execution'] == 'guarded-native-region'
               for g in compiled.info['graphs'] for op in g['operators'])
    compiled.close()


@pytest.mark.parametrize('node,inputs',[
    (spec.GraphOp('mm','y','x','w'),{'x':spec.TensorSpec((3,4),'float32'),'w':spec.TensorSpec((5,2),'float32')}),
    (spec.GraphOp('sum_keepdim','y','x',scalar=8),{'x':spec.TensorSpec((3,4),'float32')}),
    (spec.GraphOp('softmax','y','x',scalar=2),{'x':spec.TensorSpec((3,4),'float32')}),
    (spec.GraphOp('add_scalar','y','x',scalar=float('inf')),{'x':spec.TensorSpec((3,4),'float32')})])
def test_new_invalid_contracts(node,inputs):
    with pytest.raises(ValueError):spec.plan_layout(inputs,[node])


def test_no_rewrite_of_noncontiguous_clone_output(simulated):
    x=torch.randn(3,4).t()
    backend,call=make_call(lambda x:x.clone(),[x])
    got=call([x]); assert got.stride()==x.stride()
    assert not simulated[0]


def test_matmul_and_softmax_manual_training_bridge():
    import importlib
    auto=importlib.import_module('ruda_native_partition_test._graph_autograd')
    x,w=torch.randn(3,4,requires_grad=True),torch.randn(4,5,requires_grad=True)
    layout=spec.plan_layout({'x':spec.TensorSpec((3,4),'float32'),'w':spec.TensorSpec((4,5),'float32')},
        [spec.GraphOp('mm','z','x','w'),spec.GraphOp('softmax','y','z',scalar=1)])
    grad=torch.randn(3,5)
    got=auto.backward_values(layout,[x,w],[grad])
    expected=torch.autograd.grad(torch.softmax(x@w,1),(x,w),grad)
    for g,e in zip(got,expected):torch.testing.assert_close(g,e)


def test_transposed_mm_boundary_is_copied_not_reinterpreted(simulated):
    x,w=torch.randn(4,3).t(),torch.randn(5,4).t()
    assert not x.is_contiguous() and not w.is_contiguous()
    backend,call=make_call(torch.mm,[x,w],native='required')
    torch.testing.assert_close(call([x,w]),x@w)
    assert backend.info['native_replays']==1
    assert all(t.is_contiguous() for t in simulated[0][0].inputs.values())
    backend.close()
