"""Execute the real mean frontend with a labelled CPU reference native callback."""
import ast
from pathlib import Path
import types
import pytest
import torch

@pytest.fixture
def mean():
    p=Path(__file__).resolve().parents[1]/'ruda_torch/_ops.py';tree=ast.parse(p.read_text());calls=[]
    def execute(op,a,b,out,scalar):
        assert op==107 and a is b
        calls.append((op,a,b,out))
        dims=tuple(i for i,d in enumerate(out.shape) if d==1)
        out.copy_(a.float().mean(dim=dims,keepdim=True).to(out.dtype))
    ns={'torch':torch,'_C':types.SimpleNamespace(execute=execute,storage_mean_api=1),'_dtypes':(torch.float32,torch.float16,torch.bfloat16)}
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_out','_dimension','mean_dim')]
    exec(compile(ast.fix_missing_locations(tree),str(p),'exec'),ns)
    return ns['mean_dim'],calls

@pytest.mark.parametrize('shape,dim',[((),None),((3,4096),[-1]),((2,3,7),[0,2]),((2,0,7),[1]),((0,3),[1]),((3,7),[]),((3,7),None)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('keep',[False,True])
def test_real_frontend_fused_mean(mean,shape,dim,dtype,keep):
    fn,calls=mean;x=torch.full(shape,1000.,dtype=dtype)
    y=fn(x,dim,keep);ref=x.float().mean(dim=dim,keepdim=keep).to(dtype)
    torch.testing.assert_close(y,ref,equal_nan=True)
    assert len(calls)==1 and calls[0][1] is x

@pytest.mark.parametrize('dim',[[1,1],[1,-1],[2],[-3]])
def test_invalid_dimensions_before_dispatch(mean,dim):
    fn,calls=mean
    with pytest.raises((RuntimeError,IndexError)):fn(torch.ones(3,7),dim)
    assert not calls


def test_noncontiguous_and_explicit_dtype(mean):
    fn,calls=mean;x=torch.arange(21,dtype=torch.float16).reshape(3,7).t()
    y=fn(x,[1],False,dtype=torch.float32)
    assert y.dtype==torch.float32
    torch.testing.assert_close(y,x.float().mean(1))
