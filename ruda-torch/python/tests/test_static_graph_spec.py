"""Host metadata contract tests; no GPU or production kernel execution."""
import importlib.util
from pathlib import Path
import sys
import pytest

p=Path(__file__).resolve().parents[1]/'ruda_torch/_graph_spec.py'
spec=importlib.util.spec_from_file_location('ruda_graph_spec_under_test',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
T=m.TensorSpec; Op=m.GraphOp; layout=m.plan_layout

def inputs(dtype='float32',shape=(2,17)):
    return {'x':T(shape,dtype),'r':T(shape,dtype),'w':T((shape[-1],),dtype)}

@pytest.mark.parametrize('dtype',['float32','float16','bfloat16'])
@pytest.mark.parametrize('width',[1,7,32,33,4096])
def test_residual_norm_layout(dtype,width):
    p=layout(inputs(dtype,(2,width)),[Op.add('sum','x','r'),Op.rms_norm('y','sum','w',eps=1e-5)],('sum','y'))
    assert p.words==(1,0,1,100,3,2) and p.output_indices==(3,4)
    assert p.workspace_bytes==2*2*width*(4 if dtype=='float32' else 2)

@pytest.mark.parametrize('kind',['copy','silu','mul','add','rms_norm'])
def test_supported_ops(kind):
    op=getattr(Op,kind)('y','x','r') if kind in ('mul','add') else getattr(Op,kind)('y','x')
    p=layout(inputs(),[op]);assert p.names[-1]=='y' and p.output_indices==(3,)

@pytest.mark.parametrize('shape',[(),(0,3),(-1,),(True,),(1,)*9,(2**32,)])
def test_invalid_input_shapes(shape):
    with pytest.raises(ValueError):layout({'x':T(shape,'float32')},[Op.copy('y','x')])

@pytest.mark.parametrize('dtype',['int64','bool','float64','complex64'])
def test_invalid_dtype(dtype):
    with pytest.raises(ValueError):layout({'x':T((2,),dtype)},[Op.copy('y','x')])

@pytest.mark.parametrize('eps',[0.,-1.,float('nan'),float('inf'),1e-100,1e100])
def test_rms_invalid_scalar(eps):
    with pytest.raises(ValueError):layout(inputs(),[Op.rms_norm('y','x','w',eps=eps)])

@pytest.mark.parametrize('nodes',[
    [],[Op.copy('x','x')],[Op.copy('y','missing')],[Op.copy('a','b'),Op.copy('b','x')],
    [Op('mm','y','x','r')],[Op('silu','y','x','r')],[Op('mul','y','x','r',1.)],
    [Op('silu','y','x',None,-0.)],[Op.add('y','x','w')],
    [Op.rms_norm('y','x','r')],[Op.rms_norm('y','x','no_weight')],
])
def test_reject_invalid_nodes(nodes):
    with pytest.raises((TypeError,ValueError)):layout(inputs(),nodes)

@pytest.mark.parametrize('outputs',[[],['x'],['missing'],['y','y'],'y'])
def test_reject_invalid_outputs(outputs):
    with pytest.raises((TypeError,ValueError)):layout(inputs(),[Op.copy('y','x')],outputs)

def test_mixed_dtype_rejected():
    ins=inputs();ins['r']=T((2,17),'float16')
    with pytest.raises(ValueError):layout(ins,[Op.add('y','x','r')])

def test_distinct_subgraphs_can_have_distinct_dtype():
    p=layout({'a':T((2,),'float16'),'b':T((2,),'float32')},[Op.silu('c','a'),Op.silu('d','b')],['c','d'])
    assert p.specs[-2].dtype=='float16' and p.specs[-1].dtype=='float32'

def test_no_output_allocations_in_planner():
    # Scalar constants/layout only, independent of torch/backend initialization.
    assert 'torch' not in m.__dict__

def test_maximum_nodes():
    nodes=[Op.copy(f'n{i}','x' if i==0 else f'n{i-1}') for i in range(256)]
    assert len(layout({'x':T((1,),'float32')},nodes).scalars)==256
    with pytest.raises(ValueError):layout({'x':T((1,),'float32')},nodes+[Op.copy('last','n255')])

def test_no_mutation_of_inputs():
    i=inputs();before=dict(i);layout(i,[Op.silu('y','x')]);assert i==before

def test_scalar_is_actually_fp32():
    p=layout(inputs(),[Op.add('y','x','r',alpha=0.1)])
    assert p.scalars[0]!=0.1 and abs(p.scalars[0]-0.1)<1e-8

def test_bool_scalar_rejected():
    with pytest.raises(TypeError):layout(inputs(),[Op.add('y','x','r',alpha=True)])

def test_no_weight_sentinel():
    assert layout(inputs(),[Op.rms_norm('y','x')]).words[-1]==2**32-1

def test_rms_row_grid_limit():
    with pytest.raises(ValueError,match='row grid'):layout({'x':T((2**28,1),'float32')},[Op.rms_norm('y','x')])
