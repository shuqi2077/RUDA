"""Real native GPU router acceptance. No CPU runtime substitution or successful skips."""
import os
import pytest
import torch
from router_reference import weights_reference

DTYPES=[getattr(torch,n) for n in os.getenv('RUDA_ROUTER_DTYPES','float32,float16').split(',')]

@pytest.fixture(scope='module')
def backend():
    if os.getenv('RUDA_REQUIRE_GPU')!='1':
        pytest.skip('explicit GPU acceptance requires RUDA_REQUIRE_GPU=1')
    assert os.getenv('RUDA_CUDA_COMPILER')=='ptx' and os.getenv('RUDA_PTX_VERSION')
    import ruda_torch as r
    assert r._router_available and r._C.router_api_version==1 and r._C.abi_version==10
    assert int(r._native.ruda_torch_router_api_version())==1
    before=r.execution_stats()['kernel_launches']
    x=torch.zeros(1,3).to('ruda');ids=torch.tensor([[0,2]]).to('ruda')
    y=r.selected_router_weights(x,ids)
    torch.testing.assert_close(y.cpu(),torch.full((1,2),1/3))
    assert r.execution_stats()['kernel_launches']>before
    print('RUDA_V30_ROUTER_GPU_EXECUTED')
    return r


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('softmax',[False,True])
@pytest.mark.parametrize('norm',[False,True])
@pytest.mark.parametrize('shape',[(1,7,1),(3,65,8),(2,129,64)])
def test_forward_backward(backend,dtype,softmax,norm,shape):
    t,e,k=shape;gen=torch.Generator().manual_seed(719+t+e+k)
    original=torch.randn(t,e,generator=gen).to(dtype)
    ids=torch.stack([torch.randperm(e,generator=gen)[:k] for _ in range(t)])
    grad=torch.randn(t,k,generator=gen)
    x=original.clone().requires_grad_();expected=weights_reference(x,ids,softmax,norm,2.5);expected.backward(grad)
    gpu=original.to('ruda').requires_grad_()
    out=backend.selected_router_weights(gpu,ids.to('ruda'),scoring='softmax' if softmax else 'sigmoid',renormalize=norm,scale=2.5)
    out.backward(grad.to('ruda'));backend.synchronize()
    torch.testing.assert_close(out.cpu(),expected,atol=2e-5,rtol=2e-5)
    tol=5e-5 if dtype==torch.float32 else (.006 if dtype==torch.float16 else .05)
    torch.testing.assert_close(gpu.grad.cpu(),x.grad,atol=tol,rtol=tol)


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('softmax',[False,True])
@pytest.mark.parametrize('norm',[False,True])
def test_repeated_int32_indices(backend,dtype,softmax,norm):
    x=torch.tensor([[.5,1.,-.2,2.]],dtype=dtype,requires_grad=True);ids=torch.tensor([[2,2,0]],dtype=torch.int32)
    g=torch.tensor([[1.,2.,4.]]);ref=weights_reference(x,ids,softmax,norm,1.7);ref.backward(g)
    xd=x.detach().to('ruda').requires_grad_();yd=backend.selected_router_weights(xd,ids.to('ruda'),scoring='softmax' if softmax else 'sigmoid',renormalize=norm,scale=1.7)
    yd.backward(g.to('ruda'))
    torch.testing.assert_close(yd.cpu(),ref,atol=2e-5,rtol=2e-5)
    tol=5e-5 if dtype==torch.float32 else .04
    torch.testing.assert_close(xd.grad.cpu(),x.grad,atol=tol,rtol=tol)


@pytest.mark.parametrize('bad',[-1,17,1<<40])
def test_invalid_indices_are_bounded_nan_rows(backend,bad):
    x=torch.zeros(2,17).to('ruda').requires_grad_();ids=torch.tensor([[bad,0],[1,2]]).to('ruda')
    y=backend.selected_router_weights(x,ids,renormalize=True)
    y.backward(torch.ones(2,2).to('ruda'))
    out=y.cpu();dx=x.grad.cpu()
    assert torch.isnan(out[0]).all() and torch.isnan(dx[0]).all()
    torch.testing.assert_close(out[1],torch.tensor([.5,.5]))
    assert torch.isfinite(dx[1]).all()


def test_empty_tokens(backend):
    x=torch.empty(0,7,device='ruda',requires_grad=True);ids=torch.empty(0,2,device='ruda',dtype=torch.int64)
    y=backend.selected_router_weights(x,ids);y.backward(torch.empty_like(y))
    assert y.shape==(0,2) and x.grad.shape==x.shape


def test_noncontiguous_upstream_gradient(backend):
    x=torch.randn(3,7).to('ruda').requires_grad_();ids=torch.tensor([[1,2],[2,3],[3,4]]).to('ruda')
    gradient=torch.randn(2,3).to('ruda').t()
    y=backend.selected_router_weights(x,ids);y.backward(gradient)
    assert torch.isfinite(x.grad.cpu()).all()


@pytest.mark.parametrize('kind',['logits','indices'])
def test_saved_input_modification_is_rejected(backend,kind):
    x=torch.randn(2,5).to('ruda').requires_grad_();ids=torch.tensor([[0,1],[1,2]]).to('ruda')
    y=backend.selected_router_weights(x,ids)
    with torch.no_grad():
        target=x if kind=='logits' else ids
        target.copy_(target)  # version change without changing shape or validity
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        y.backward(torch.ones(2,2).to('ruda'))


def test_higher_order_rejected(backend):
    x=torch.randn(2,5).to('ruda').requires_grad_();ids=torch.tensor([[0,1],[1,2]]).to('ruda')
    y=backend.selected_router_weights(x,ids)
    with pytest.raises(RuntimeError,match='first-order'):
        torch.autograd.grad(y,x,torch.ones(2,2).to('ruda'),create_graph=True)


@pytest.mark.parametrize('softmax',[False,True])
def test_router_projection_receives_gradient(backend,softmax):
    gen=torch.Generator().manual_seed(501)
    x=torch.randn(3,8,generator=gen,requires_grad=True)
    w=torch.randn(7,8,generator=gen,requires_grad=True)
    ids=torch.tensor([[1,3],[2,4],[0,6]]);g=torch.randn(3,2,generator=gen)
    ref=weights_reference(x@w.t(),ids,softmax,True,2.5);ref.backward(g)
    xd=x.detach().to('ruda').requires_grad_();wd=w.detach().to('ruda').requires_grad_()
    out=backend.selected_router_weights(xd@wd.t(),ids.to('ruda'),scoring='softmax' if softmax else 'sigmoid',renormalize=True,scale=2.5)
    out.backward(g.to('ruda'))
    torch.testing.assert_close(out.cpu(),ref,atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(wd.grad.cpu(),w.grad,atol=3e-4,rtol=3e-4)
    torch.testing.assert_close(xd.grad.cpu(),x.grad,atol=3e-4,rtol=3e-4)


def test_cpu_is_not_a_fallback(backend):
    with pytest.raises(ValueError,match='ruda'):
        backend.selected_router_weights(torch.randn(2,5),torch.tensor([[0],[1]]))
