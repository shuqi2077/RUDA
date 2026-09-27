"""Pure planner and CPU reference tests. No production Rust/PTX is executed."""
import importlib
from pathlib import Path
import random
import sys
import types

import pytest
import torch

name='ruda_graph_optimizer_host_test'
package=types.ModuleType(name)
package.__path__=[str(Path(__file__).resolve().parents[1]/'ruda_torch')]
sys.modules[name]=package
m=importlib.import_module(name+'._graph_opt')
T=m.TensorSpec; Op=m.GraphOp


def inputs(dtype='float32',shape=(2,33)):
    return {'x':T(shape,dtype),'u':T(shape,dtype)}


def chain(n=9):
    return [Op.silu('n'+str(i),'x' if i==0 else 'n'+str(i-1)) for i in range(n)]


@pytest.mark.parametrize('dtype',['float32','float16','bfloat16'])
@pytest.mark.parametrize('reuse',[False,True])
def test_fusion_and_dead_branch(dtype,reuse):
    p=m.prepare_plan(inputs(dtype),[Op.copy('unused','x'),Op.silu('a','x'),Op.mul('y','a','u')],
                     optimize=True,reuse_workspace=reuse)
    assert p.layout.words==(101,0,1)
    assert p.original_nodes==3 and len(p.nodes)==1
    assert p.eliminated_outputs==('unused',) and p.fused_activations==('a',)
    assert p.workspace_bytes==2*33*(4 if dtype=='float32' else 2)
    assert p.original_workspace_bytes==3*p.workspace_bytes


def test_defaults_preserve_old_plan():
    nodes=[Op.silu('a','x'),Op.mul('y','a','u')]
    p=m.prepare_plan(inputs(),nodes)
    assert p.nodes==tuple(nodes) and p.storage_roots==tuple(range(4))


@pytest.mark.parametrize('case',['public','fanout','double_read','right_activation'])
def test_fusion_preserves_observable_intermediates(case):
    nodes=[Op.silu('a','x')]
    out=None
    if case=='public':nodes+=[Op.mul('y','a','u')];out=['a','y']
    if case=='fanout':nodes+=[Op.mul('b','a','u'),Op.add('y','b','a')]
    if case=='double_read':nodes+=[Op.mul('y','a','a')]
    if case=='right_activation':nodes+=[Op.mul('y','u','a')]
    p=m.prepare_plan(inputs(),nodes,out,optimize=True)
    assert not p.fused_activations


def test_fusion_does_not_move_after_dependency():
    nodes=[Op.silu('a','x'),Op.copy('u2','u'),Op.mul('y','a','u2')]
    p=m.prepare_plan(inputs(),nodes,optimize=True)
    assert [n.output for n in p.nodes]==['u2','y']
    assert p.layout.words[-3:]==(101,0,2)


def test_two_activations_keep_right_cast():
    p=m.prepare_plan(inputs(),[Op.silu('a','x'),Op.silu('b','u'),Op.mul('y','a','b')],optimize=True)
    assert p.fused_activations==('a',) and [n.kind for n in p.nodes]==['silu','silu_mul']


def test_invalid_dead_node_not_hidden():
    with pytest.raises(ValueError,match='unsupported'):
        m.prepare_plan(inputs(),[Op('bad','unused','x'),Op.copy('y','x')],optimize=True)


@pytest.mark.parametrize('flag',['optimize','reuse_workspace'])
@pytest.mark.parametrize('value',[1,None,'yes'])
def test_options_strict(flag,value):
    with pytest.raises(TypeError):m.prepare_plan(inputs(),chain(),**{flag:value})


@pytest.mark.parametrize('n',[1,2,3,4,9,64,256])
def test_linear_chain_reuses_only_dead_scratch(n):
    p=m.prepare_plan(inputs(),chain(n),reuse_workspace=True)
    assert p.workspace_allocations==min(n,3)
    # Public output is never a previously used allocation.
    assert p.storage_roots[-1]==len(p.layout.names)-1
    # No node may overwrite the input it is concurrently reading.
    for i in range(n):
        out=p.layout.inputs+i;a=p.layout.words[i*3+1]
        assert p.storage_roots[out]!=p.storage_roots[a]


def test_pinned_intermediate_never_reused():
    p=m.prepare_plan(inputs(),chain(12),['n0','n11'],reuse_workspace=True)
    for i in p.layout.output_indices:
        assert p.storage_roots.count(i)==1


def test_no_cross_dtype_or_shape_reuse():
    ins={'x':T((2,),'float32'),'h':T((2,),'float16'),'z':T((1,2),'float32')}
    nodes=[Op.copy('a','x'),Op.copy('b','a'),Op.copy('c','h'),Op.copy('d','c'),Op.copy('e','z'),Op.copy('f','e')]
    p=m.prepare_plan(ins,nodes,['b','d','f'],reuse_workspace=True)
    assert p.workspace_allocations==6


def test_dangling_branch_not_reused_without_pruning():
    p=m.prepare_plan(inputs(),[Op.copy('unused','x')]+chain(5),reuse_workspace=True)
    assert p.storage_roots.count(2)==1


def test_outputs_generator_consumed_once():
    p=m.prepare_plan(inputs(),chain(5),(n for n in ['n0','n4']),optimize=True,reuse_workspace=True)
    assert tuple(p.layout.names[i] for i in p.layout.output_indices)==('n0','n4')


def _reference(node,a,b,dtype):
    x=a.float()
    if node.kind=='copy':return a.clone()
    if node.kind=='silu':return (x/(1+(-x).exp())).to(dtype)
    if node.kind=='silu_mul':
        activation=(x/(1+(-x).exp())).to(dtype)
        return (activation.float()*b.float()).to(dtype)
    if node.kind=='mul':return (x*b.float()).to(dtype)
    if node.kind=='add':return (x+torch.tensor(node.scalar,dtype=torch.float32)*b.float()).to(dtype)
    raise AssertionError(node.kind)


def _execute(plan,data):
    # Deliberately materialize the planned aliases. Reading an overwritten live
    # value makes the differential check fail (not merely a graph-shape test).
    layout=plan.layout
    store={i:data[n].clone() for i,n in enumerate(layout.names[:layout.inputs])}
    for i,node in enumerate(plan.nodes):
        slot=layout.inputs+i
        _,a,b=layout.words[3*i:3*i+3]
        result=_reference(node,store[plan.storage_roots[a]],
                          None if b==m.NO_WEIGHT else store[plan.storage_roots[b]],
                          getattr(torch,layout.specs[slot].dtype))
        store[plan.storage_roots[slot]]=result
    return {layout.names[i]:store[plan.storage_roots[i]].clone() for i in layout.output_indices}


def test_1000_seeded_graphs_numeric_and_live_range_reference():
    gen=random.Random(24001);tg=torch.Generator().manual_seed(24001)
    for case in range(1000):
        dtype=['float32','float16','bfloat16'][case%3]
        ins=inputs(dtype,(2,7));names=list(ins);nodes=[]
        for i in range(gen.randint(3,35)):
            out='n'+str(i);kind=gen.choice(['copy','silu','mul','add'])
            a=gen.choice(names);b=gen.choice(names)
            node=getattr(Op,kind)(out,a,b,alpha=.5) if kind=='add' else (
                Op.mul(out,a,b) if kind=='mul' else getattr(Op,kind)(out,a))
            nodes.append(node);names.append(out)
        outputs=tuple(dict.fromkeys([nodes[-1].output,gen.choice(nodes).output]))
        base=m.prepare_plan(ins,nodes,outputs)
        opt=m.prepare_plan(ins,nodes,outputs,optimize=True,reuse_workspace=True)
        data={n:torch.randn(2,7,generator=tg).mul(.3).to(getattr(torch,dtype)) for n in ins}
        expected=_execute(base,data);actual=_execute(opt,data)
        assert actual.keys()==expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0,equal_nan=True)
        for i in opt.layout.output_indices:assert opt.storage_roots.count(i)==1
        assert opt.workspace_bytes<=base.workspace_bytes


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_rounding_boundary_is_observable(dtype):
    g=torch.Generator().manual_seed(11)
    a=torch.randn(8192,generator=g).to(dtype);b=torch.randn(8192,generator=g).to(dtype)
    fp=a.float()/(1+(-a.float()).exp())
    old=(fp.to(dtype).float()*b.float()).to(dtype)
    naive=(fp*b.float()).to(dtype)
    fused=_reference(Op.silu_mul('y','a','b'),a,b,dtype)
    assert torch.count_nonzero(old!=naive)>0
    torch.testing.assert_close(fused,old,rtol=0,atol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_special_values_roundtrip_reference(dtype):
    a=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),-100.,100.,1e-30],dtype=dtype)
    b=torch.tensor([1.,1.,0.,0.,1.,float('inf'),-1.,2.],dtype=dtype)
    act=_reference(Op.silu('a','x'),a,None,dtype)
    expected=_reference(Op.mul('y','a','u'),act,b,dtype)
    actual=_reference(Op.silu_mul('y','x','u'),a,b,dtype)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
    assert torch.equal(torch.signbit(actual[:2]),torch.signbit(expected[:2]))
