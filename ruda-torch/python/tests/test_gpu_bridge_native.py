"""Required hardware tests: absence of the native backend is a failure, not skip.
Run only after building ABI 5 and setting RUDA_CUDA_COMPILER=ptx and a compatible
RUDA_PTX_VERSION. These tests are separate from test_gpu_bridge_host.py.
"""
import os

import pytest
import torch
import ruda_torch

DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def close(actual, expected):
    tol = {torch.float32: 3e-4, torch.float16: 4e-3, torch.bfloat16: 3e-2}[actual.dtype]
    torch.testing.assert_close(actual.cpu(), expected.to(actual.dtype).cpu(), rtol=tol, atol=tol, equal_nan=True)


def unchanged_transfers(before, after):
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert before[key] == after[key], key


@pytest.fixture(scope='module', autouse=True)
def direct_ptx_only():
    assert os.environ.get('RUDA_CUDA_COMPILER') == 'ptx', 'GPU acceptance must run direct PTX'
    assert os.environ.get('RUDA_PTX_VERSION'), 'set the PTX version supported by your driver'
    x = torch.ones(1).to('ruda')
    ruda_torch.synchronize()
    assert x.cpu().item() == 1


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(3,7,5),(17,33,19),(32,32,32),(1,257,63),(0,7,5),(3,0,5)])
@pytest.mark.parametrize('batched', [False,True])
def test_storage_matmul(dtype,shape,batched):
    m,k,n = shape
    gen = torch.Generator().manual_seed(417)
    if batched:
        a = torch.randn(2,k,m,generator=gen).to(dtype).transpose(1,2)
        b = torch.randn(2,n,k,generator=gen).to(dtype).transpose(1,2)
    else:
        a = torch.randn(k,m,generator=gen).to(dtype).t()
        b = torch.randn(n,k,generator=gen).to(dtype).t()
    x,y = a.to('ruda'),b.to('ruda')
    before = ruda_torch.execution_stats()
    result = torch.bmm(x,y) if batched else torch.mm(x,y)
    after = ruda_torch.execution_stats()
    unchanged_transfers(before,after)
    assert after['legacy_fp32_temp_bytes_total'] == before['legacy_fp32_temp_bytes_total']
    if result.numel():
        assert after['rublas_calls']+after['scalar_matmul_calls'] > before['rublas_calls']+before['scalar_matmul_calls']
    close(result,(torch.bmm(a.float(),b.float()) if batched else a.float()@b.float()).to(dtype))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('alpha,beta', [(1.,1.),(.25,-.5),(1.,0.),(0.,1.),(0.,0.)])
@pytest.mark.parametrize('bias_shape', [(),(19,),(1,19),(5,1)])
def test_addmm_epilogue(dtype,alpha,beta,bias_shape):
    gen = torch.Generator().manual_seed(117)
    a,b = torch.randn(5,33,generator=gen).to(dtype),torch.randn(33,19,generator=gen).to(dtype)
    c = torch.randn(bias_shape,generator=gen).to(dtype)
    if beta == 0: c.fill_(float('nan'))
    if alpha == 0: a.fill_(float('nan'))
    x,y,z = a.to('ruda'),b.to('ruda'),c.to('ruda')
    before = ruda_torch.execution_stats()
    result = torch.addmm(z,x,y,alpha=alpha,beta=beta)
    after = ruda_torch.execution_stats()
    unchanged_transfers(before,after)
    assert after['legacy_fp32_temp_bytes_total'] == before['legacy_fp32_temp_bytes_total']
    want_scratch = result.numel()*4 if dtype!=torch.float32 and alpha!=0 and not(alpha==1 and beta==0) else 0
    assert after['addmm_workspace_bytes_total']-before['addmm_workspace_bytes_total'] == want_scratch
    close(result,torch.addmm(c.float(),a.float(),b.float(),alpha=alpha,beta=beta).to(dtype))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('width', [1,7,31,32,33,127,128,257,1025])
@pytest.mark.parametrize('logarithmic', [False,True])
def test_softmax_forward_backward(dtype,width,logarithmic):
    gen = torch.Generator().manual_seed(width)
    a = torch.randn(width,3,generator=gen).t().to(dtype)
    g = torch.randn(3,width,generator=gen).to(dtype)
    x = a.to('ruda').requires_grad_()
    grad = g.to('ruda')
    before = ruda_torch.execution_stats()
    fn = torch.log_softmax if logarithmic else torch.softmax
    y = fn(x,1)
    y.backward(grad)
    after = ruda_torch.execution_stats()
    unchanged_transfers(before,after)
    assert after['legacy_fp32_temp_bytes_total'] == before['legacy_fp32_temp_bytes_total']
    expected = fn(a.float(),1).to(dtype)
    if logarithmic:
        dx = g.float()-expected.float().exp()*g.float().sum(1,keepdim=True)
    else:
        dx = expected.float()*(g.float()-(g.float()*expected.float()).sum(1,keepdim=True))
    close(y,expected)
    close(x.grad,dx.to(dtype))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('special', ['nan','positive_inf','all_negative_inf','masked'])
def test_softmax_nonfinite(dtype,special):
    a = torch.zeros(3,33,dtype=dtype)
    if special=='nan': a[:,31]=float('nan')
    if special=='positive_inf': a[:,32]=float('inf')
    if special=='all_negative_inf': a.fill_(-float('inf'))
    if special=='masked': a[:,1:]=-float('inf')
    close(torch.softmax(a.to('ruda'),1),torch.softmax(a.float(),1).to(dtype))


@pytest.mark.parametrize('dtype_a', DTYPES)
@pytest.mark.parametrize('dtype_b', DTYPES)
@pytest.mark.parametrize('dtype_out', DTYPES)
def test_direct_pointwise_triples(dtype_a,dtype_b,dtype_out):
    a = torch.linspace(-3,3,33).reshape(1,33).to(dtype_a)
    b = torch.tensor([[.25],[1.5],[-2.]]).to(dtype_b)
    x,y = a.to('ruda').expand(3,33),b.to('ruda').expand(3,33)
    out = torch.empty(3,33,device='ruda',dtype=dtype_out)
    before = ruda_torch.execution_stats()
    ruda_torch._C.execute(1,x,y,out,.375)
    after = ruda_torch.execution_stats()
    unchanged_transfers(before,after)
    assert after['direct_pointwise_calls']-before['direct_pointwise_calls']==1
    assert after['legacy_fp32_temp_bytes_total']==before['legacy_fp32_temp_bytes_total']
    close(out,(a.float()+.375*b.float()).to(dtype_out))


@pytest.mark.parametrize('dtype',DTYPES)
def test_linear_backward_storage_path(dtype):
    gen=torch.Generator().manual_seed(913)
    a=torch.randn(3,17,generator=gen).to(dtype)
    w=torch.randn(19,17,generator=gen).to(dtype)
    b=torch.randn(19,generator=gen).to(dtype)
    g=torch.randn(3,19,generator=gen).to(dtype)
    x,weight,bias=[v.to('ruda').requires_grad_() for v in (a,w,b)]
    grad=g.to('ruda')
    out=torch.nn.functional.linear(x,weight,bias)
    out.backward(grad)
    close(out,torch.addmm(b.float(),a.float(),w.float().t()).to(dtype))
    close(x.grad,(g.float()@w.float()).to(dtype))
    close(weight.grad,(g.float().t()@a.float()).to(dtype))
    close(bias.grad,g.float().sum(0).to(dtype))

@pytest.mark.parametrize('dtype',DTYPES)
def test_strided_output_and_expanded_batch(dtype):
    a=torch.arange(2*3*7,dtype=torch.float32).reshape(2,3,7).to(dtype)/17
    b=torch.arange(7*5,dtype=torch.float32).reshape(1,7,5).to(dtype)/23
    x=a.to('ruda');y=b.to('ruda').expand(2,7,5)
    backing=torch.empty((2,5,3),device='ruda',dtype=dtype)
    out=backing.transpose(1,2)
    ruda_torch._C.execute(30,x,y,out,0.)
    close(out,torch.bmm(a.float(),b.float().expand(2,7,5)).to(dtype))


def test_matrix_alias_is_rejected():
    x=torch.eye(4).to('ruda')
    with pytest.raises(RuntimeError):ruda_torch._C.execute(7,x,x,x,0.)


@pytest.mark.parametrize('logarithmic',[False,True])
def test_half_to_float_without_materialized_input(logarithmic):
    a=torch.linspace(-4,4,99).reshape(3,33).half()
    x=a.to('ruda')
    before=ruda_torch.execution_stats()
    fn=torch.ops.aten._log_softmax if logarithmic else torch.ops.aten._softmax
    out=fn(x,-1,True)
    after=ruda_torch.execution_stats()
    assert out.dtype==torch.float32
    assert before['legacy_fp32_temp_bytes_total']==after['legacy_fp32_temp_bytes_total']
    close(out,(torch.log_softmax if logarithmic else torch.softmax)(a.float(),-1))
