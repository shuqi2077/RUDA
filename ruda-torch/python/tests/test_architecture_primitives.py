"""Real CPU production-operator tests; no native library or device simulation."""
import copy
import pytest
import torch
from torch import nn
from architecture_test_utils import ops, mhc, close

@pytest.mark.parametrize('size',[1,2,3,8,17])
@pytest.mark.parametrize('k',[0,1,3,99])
def test_topk_stable_oracle(size,k):
    torch.manual_seed(4)
    score=torch.randint(-2,3,(2,3,size)).float().requires_grad_()
    valid=torch.rand(2,3,size)>.3
    values,indices=ops.stable_topk(score,k,valid=valid)
    reference=score.masked_fill(~valid,float('-inf'))
    ids=reference.argsort(dim=-1,descending=True,stable=True)[...,:min(k,size)]
    vals=reference.gather(-1,ids)
    ids=torch.where(valid.gather(-1,ids),ids,torch.full_like(ids,-1))
    close(indices,ids);close(values,vals)
    if k:
        torch.where(indices>=0,values,torch.zeros_like(values)).sum().backward()
        expected=torch.zeros_like(score)
        for b in range(2):
            for t in range(3):
                for i in ids[b,t]:
                    if i>=0:expected[b,t,i]+=1
        close(score.grad,expected)

@pytest.mark.parametrize('size',[1,3,10,101])
def test_topk_external_ties_and_maximum(size):
    score=torch.zeros(2,size);ids=torch.arange(size-1,-1,-1).expand(2,-1)
    values,indices=ops.stable_topk(score,size,indices=ids)
    close(indices,torch.arange(size).expand(2,-1))
    x=torch.randn(size)
    close(ops.maximum_abs(x),x.abs().max())

@pytest.mark.parametrize('empty',[False,True])
def test_gather_sentinel_duplicate_gradient(empty):
    x=torch.randn(2,0 if empty else 4,3,dtype=torch.double,requires_grad=True)
    ids=torch.tensor([[[-1,1,1],[-1,3,0]],[[-1,0,2],[2,2,3]]])
    if empty: ids=torch.full_like(ids,-1)
    y=ops.gather_entries(x,ids)
    expected=torch.zeros(2,2,3,3,dtype=x.dtype)
    for b in range(2):
        for t in range(2):
            for k in range(3):
                if ids[b,t,k]>=0:expected[b,t,k]=x[b,ids[b,t,k]]
    close(y,expected);y.sum().backward()
    grad=torch.zeros_like(x)
    for b in range(2):
        for i in ids[b].flatten():
            if i>=0:grad[b,i]+=1
    close(x.grad,grad)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.bfloat16])
def test_masked_softmax_empty_rows(dtype):
    x=torch.randn(2,3,dtype=dtype,requires_grad=True)
    valid=torch.tensor([[False]*3,[True,False,True]])
    y=ops.masked_softmax(x,valid)
    assert not y[0].any() and y[1,1]==0
    close(y[1].sum(),torch.tensor(1.,dtype=y.dtype))
    y.square().sum().backward();assert torch.isfinite(x.grad).all();assert not x.grad[0].any()

@pytest.mark.parametrize('streams',[1,2,4])
def test_mhc_independent_equations_and_gradients(streams):
    torch.manual_seed(5)
    layer=mhc.MHC(3,streams,sinkhorn_iterations=30,dtype=torch.double)
    x=torch.randn(2,3,streams,3,dtype=torch.double,requires_grad=True)
    flat=x.flatten(-2);normalized=flat/(flat.square().mean(-1,keepdim=True)+layer.eps).sqrt()
    projected=normalized@layer.mapping
    pre=(projected[...,:streams]*layer.alpha[0]+layer.bias[:streams]).sigmoid()
    post=2*(projected[...,streams:2*streams]*layer.alpha[1]+layer.bias[streams:2*streams]).sigmoid()
    residual=(projected[...,2*streams:]*layer.alpha[2]+layer.bias[2*streams:]).reshape(2,3,streams,streams).exp()
    for _ in range(30):
        residual=residual/residual.sum(-2,keepdim=True)
        residual=residual/residual.sum(-1,keepdim=True)
    c=layer.coefficients(x)
    for a,b in zip(c,(pre,post,residual)):close(a,b,atol=1e-12,rtol=1e-12)
    merged,coeff=layer.pre(x)
    y=layer.post(x,merged.square(),coeff)
    oracle=residual@x+post.unsqueeze(-1)*((x*pre.unsqueeze(-1)).sum(-2).square()).unsqueeze(-2)
    close(y,oracle)
    loss=y.square().mean();loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    close(residual.sum(-1),torch.ones_like(residual[...,0]),atol=1e-10,rtol=1e-10)
    close(residual.sum(-2),torch.ones_like(residual[...,0]),atol=1e-8,rtol=1e-8)

def test_sinkhorn_large_logits_and_finite_iterations():
    x=torch.tensor([[1000.,-1000.],[1000.,1000.]],requires_grad=True)
    y=mhc.sinkhorn(x,20)
    assert torch.isfinite(y).all()
    close(y.sum(-1),torch.ones(2))
    assert (y.sum(-2)-1).abs().max()>1e-3  # must not promise exact double stochasticity
    y.square().sum().backward();assert torch.isfinite(x.grad).all()

def test_mhc_gradcheck():
    layer=mhc.MHC(2,2,sinkhorn_iterations=3,dtype=torch.double)
    x=torch.randn(1,2,2,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z:layer(z,nn.SiLU()),(x,),fast_mode=True)

def test_mhc_checkpoint_parameter_and_input_parity():
    a=mhc.MHCResidual(nn.Sequential(nn.Linear(4,5),nn.SiLU(),nn.Linear(5,4)),4,2)
    b=copy.deepcopy(a);b.checkpoint_branch=True
    x=torch.randn(2,3,2,4,requires_grad=True);z=x.detach().clone().requires_grad_()
    y=a(x);v=b(z);close(y,v);y.square().sum().backward();v.square().sum().backward()
    close(x.grad,z.grad)
    for p,q in zip(a.parameters(),b.parameters()):close(p.grad,q.grad)

def test_mhc_sequential_state_dict_and_amp():
    model=mhc.MHCSequential(4,[nn.Linear(4,4),nn.Linear(4,4)],streams=2)
    x=torch.randn(2,3,4,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):y=model(x)
    assert y.dtype==x.dtype;y.sum().backward()
    other=copy.deepcopy(model);other.load_state_dict(model.state_dict());close(model(x),other(x))

@pytest.mark.parametrize('kwargs',[{'width':0},{'streams':0},{'sinkhorn_iterations':0},{'eps':0}])
def test_mhc_bad_config(kwargs):
    with pytest.raises((ValueError,TypeError)):mhc.MHC(**({'width':4}|kwargs))
