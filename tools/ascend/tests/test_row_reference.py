"""Independent 32-lane FP32 schedule/math references, NOT generated CCE execution."""
import numpy as np
import pytest
import torch


def reduce_row(x,maximum=False):
    rows,width=x.shape
    assert width%32==0 and 32<=width<=4096
    pieces=x.reshape(rows,width//32,32)
    acc=pieces[:,0].copy()
    for k in range(1,width//32):
        acc=np.maximum(acc,pieces[:,k]) if maximum else np.add(acc,pieces[:,k],dtype=np.float32)
    if maximum:return np.max(acc,axis=-1,keepdims=True)
    return np.sum(acc,axis=-1,keepdims=True,dtype=np.float32)


def softmax(x,log=False):
    shifted=x-reduce_row(x,True)
    exp=np.exp(shifted).astype(np.float32)
    total=reduce_row(exp)
    return shifted-np.log(total).astype(np.float32) if log else exp/total


def rms(x,w):
    mean=reduce_row(x*x)/np.float32(x.shape[1])
    r=np.float32(1)/np.sqrt(mean+np.float32(1e-5))
    return (x*r)*w,r


@pytest.mark.parametrize('rows',[0,1,7,33])
@pytest.mark.parametrize('width',[32,96,256,1024,4096])
def test_row_reduction_finite_reference(rows,width):
    rng=np.random.default_rng(40+rows+width);x=rng.normal(size=(rows,width)).astype(np.float32)
    np.testing.assert_allclose(reduce_row(x),x.astype(np.float64).sum(-1,keepdims=True),rtol=3e-5,atol=3e-5)
    np.testing.assert_array_equal(reduce_row(x,True),x.max(-1,keepdims=True))
    assert reduce_row(x).shape==(rows,1)


@pytest.mark.parametrize('rows,width',[(0,32),(1,32),(3,96),(7,256),(33,4096)])
@pytest.mark.parametrize('log',[False,True])
def test_stable_normalization_reference(rows,width,log):
    rng=np.random.default_rng(width);x=(rng.normal(size=(rows,width))*3+1000).astype(np.float32)
    y=softmax(x,log)
    t=torch.from_numpy(x).double()
    ref=(t.log_softmax(-1) if log else t.softmax(-1)).numpy()
    np.testing.assert_allclose(y,ref,rtol=4e-5,atol=4e-5)
    probability=np.exp(y) if log else y
    np.testing.assert_allclose(probability.sum(-1),np.ones(rows),rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('width',[32,96,256,1024,4096])
@pytest.mark.parametrize('log',[False,True])
def test_softmax_backward_vs_autograd(width,log):
    rng=np.random.default_rng(width+1);x=rng.normal(size=(3,width)).astype(np.float32)
    dy=rng.normal(size=x.shape).astype(np.float32)
    y=softmax(x,log)
    dx=dy-np.exp(y)*reduce_row(dy) if log else y*(dy-reduce_row(y*dy))
    t=torch.tensor(x,dtype=torch.float64,requires_grad=True)
    ref=t.log_softmax(-1) if log else t.softmax(-1)
    (ref*torch.tensor(dy,dtype=torch.float64)).sum().backward()
    np.testing.assert_allclose(dx,t.grad.numpy(),rtol=2e-4,atol=2e-6)


@pytest.mark.parametrize('rows,width',[(0,32),(1,32),(3,96),(7,256),(3,4096)])
def test_rms_stats_shared_weight_and_input_gradient(rows,width):
    rng=np.random.default_rng(width+2);x=rng.normal(size=(rows,width)).astype(np.float32)
    w=rng.normal(size=(width,)).astype(np.float32);dy=rng.normal(size=x.shape).astype(np.float32)
    y,r=rms(x,w)
    g=dy*w
    dx=r*(g-x*(r*r)*(reduce_row(g*x)/np.float32(width)))
    t=torch.tensor(x,dtype=torch.float64,requires_grad=True)
    tw=torch.tensor(w,dtype=torch.float64)
    ref=t*torch.rsqrt(t.square().mean(-1,keepdim=True)+1e-5)*tw
    (ref*torch.tensor(dy,dtype=torch.float64)).sum().backward()
    assert r.shape==(rows,1)
    np.testing.assert_allclose(y,ref.detach().numpy(),rtol=1e-4,atol=2e-6)
    np.testing.assert_allclose(dx,t.grad.numpy(),rtol=1e-4,atol=2e-6)


def test_mean_accumulates_and_divides_before_store():
    x=np.full((3,4096),1000.,np.float32)
    np.testing.assert_array_equal(reduce_row(x)/np.float32(4096),np.full((3,1),1000.,np.float32))


@pytest.mark.parametrize('cores',[1,7,32])
@pytest.mark.parametrize('rows',[0,1,3,31,32,33,1000])
def test_physical_row_assignment_exactly_once(rows,cores):
    assigned=[r for core in range(cores) for r in range(core,rows,cores)]
    assert sorted(assigned)==list(range(rows))
    assert len(set(assigned))==len(assigned)


@pytest.mark.parametrize('width',[32,96,256,4096])
def test_32_lane_semantics_do_not_cross_rows(width):
    rows=7
    x=np.repeat(np.arange(rows,dtype=np.float32)[:,None],width,axis=1)
    np.testing.assert_array_equal(reduce_row(x)[:,0],np.arange(rows,dtype=np.float32)*width)
    np.testing.assert_array_equal(softmax(x),np.full_like(x,1/width))
