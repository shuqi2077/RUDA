"""Execute the production fake-quantizer on CPU, not the Rust packed backend."""
import importlib.util
from pathlib import Path
import copy
import pytest
import torch
spec = importlib.util.spec_from_file_location("learned_quantization_cpu", Path(__file__).resolve().parents[1]/"ruda_torch/quantization.py")
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def test_clipped_input_and_learned_scale_rust_expected_case():
    x=torch.tensor([-10.,-.25,.25,10.],requires_grad=True)
    s=torch.tensor([.5],requires_grad=True)
    y=q.learned_fake_quantize(x,s)
    torch.testing.assert_close(y,torch.tensor([-3.5,-.5,.5,3.5]))
    y.backward(torch.tensor([1.,2.,3.,4.]))
    torch.testing.assert_close(x.grad,torch.tensor([0.,2.,3.,0.]))
    torch.testing.assert_close(s.grad,torch.tensor([21.5]))


@pytest.mark.parametrize("bits",[2,4,8])
@pytest.mark.parametrize("symmetric",[False,True])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_grid_values_and_first_derivative(bits,symmetric,dtype):
    x=torch.tensor([-40.,-1.3,-.2,.2,1.3,40.],dtype=dtype,requires_grad=True)
    s=torch.tensor([.25],requires_grad=True)
    y=q.learned_fake_quantize(x,s,bits=bits,symmetric=symmetric)
    r=x.detach().float()/s.detach();upper=2**(bits-1)-1;lower=-upper if symmetric else -2**(bits-1)
    codes=(r.sign()*(r.abs()+.5).floor()).clamp(lower,upper)
    torch.testing.assert_close(y,(codes*.25).to(dtype))
    y.float().sum().backward();inside=(r>=lower)&(r<=upper)
    torch.testing.assert_close(x.grad,inside.to(dtype))
    torch.testing.assert_close(s.grad,torch.where(inside,codes-r,codes).sum().reshape(1))


def test_partial_multidimensional_blocks_reduce_scale_gradients():
    x=torch.tensor([[.25,.3,4.],[.1,-.7,-5.],[.5,.6,.7]],requires_grad=True)
    s=torch.tensor([[.5,.25],[.4,.2]],requires_grad=True)
    y=q.learned_fake_quantize(x,s,block_shape=(2,2));y.sum().backward()
    expected_y=torch.empty_like(x);expected_grad=torch.empty_like(s)
    for i in range(2):
        for j in range(2):
            sl=(slice(i*2,min(i*2+2,3)),slice(j*2,min(j*2+2,3)))
            r=x.detach()[sl]/s.detach()[i,j];codes=(r.sign()*(r.abs()+.5).floor()).clamp(-7,7)
            expected_y[sl]=codes*s.detach()[i,j]
            expected_grad[i,j]=torch.where((r>=-7)&(r<=7),codes-r,codes).sum()
    torch.testing.assert_close(y,expected_y);torch.testing.assert_close(s.grad,expected_grad)


def test_scale_only_gradient_optimizer_and_checkpoint():
    m=q.LearnedFakeQuantize(torch.tensor([.5]));x=torch.tensor([.25,.7])
    saved=copy.deepcopy(m.state_dict());before=m.scales.detach().clone()
    optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    m(x).square().sum().backward();assert m.scales.grad is not None
    optimizer.step();assert not torch.equal(m.scales,before)
    m.load_state_dict(saved);torch.testing.assert_close(m.scales,before)


def test_floor_has_zero_gradient_and_overflow_stays_finite():
    x=torch.tensor([torch.finfo(torch.float32).max],requires_grad=True)
    s=torch.tensor([0.],requires_grad=True)
    q.learned_fake_quantize(x,s).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(s.grad).all()
    assert s.grad.item()==0 and x.grad.item()==0


def test_aot_autograd_training_matches_eager():
    m=q.LearnedFakeQuantize(torch.tensor([.5]));ref=copy.deepcopy(m)
    x=torch.tensor([-.25,.3,8.],requires_grad=True);xr=x.detach().clone().requires_grad_()
    compiled=torch.compile(m,backend="aot_eager",fullgraph=True)
    a,b=compiled(x),ref(xr);a.sum().backward();b.sum().backward()
    torch.testing.assert_close(a,b);torch.testing.assert_close(x.grad,xr.grad)
    torch.testing.assert_close(m.scales.grad,ref.scales.grad)
    torch._dynamo.reset()


@pytest.mark.parametrize("kwargs",[{'bits':3},{'bits':True},{'block_shape':(0,)},{'block_shape':(1,2)}])
def test_invalid_configuration(kwargs):
    with pytest.raises((ValueError,TypeError)):q.learned_fake_quantize(torch.ones(2),torch.ones(1),**kwargs)


def test_invalid_scale_dtype_and_count():
    with pytest.raises(ValueError):q.learned_fake_quantize(torch.ones(2),torch.ones(1,dtype=torch.float16))
    with pytest.raises(ValueError):q.learned_fake_quantize(torch.ones(2),torch.ones(2))
