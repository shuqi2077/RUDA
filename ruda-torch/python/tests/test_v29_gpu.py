"""v29 real-GPU acceptance: paged autograd and PrivateUse1 autocast."""
import os, pytest, torch
from test_v14_host import dense,inputs,schedule

@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU')!='1':
        pytest.skip('explicit hardware acceptance only: set RUDA_REQUIRE_GPU=1')
    import ruda_torch as r
    assert r._C.abi_version==10
    assert os.environ.get('RUDA_CUDA_COMPILER')=='ptx'
    assert os.environ.get('RUDA_PTX_VERSION')
    torch.ones(1,device='ruda').cpu()
    return r

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_paged_gqa_backward(backend,dtype):
    q,k,v=inputs(dtype);s=schedule();args=[s[x] for x in ('block_tables','kv_lengths','sequence_ids','positions')]
    qr=q.detach().float().requires_grad_();kr=k.detach().float().requires_grad_();vr=v.detach().float().requires_grad_()
    ref=dense(qr,kr,vr,*args,33**-.5,True);go=torch.randn_like(ref);ref.backward(go)
    qd=q.detach().to('ruda').requires_grad_();kd=k.detach().to('ruda').requires_grad_();vd=v.detach().to('ruda').requires_grad_()
    out=backend.PagedAttentionPlan(**s).attention(qd,kd,vd,scale=33**-.5,causal=True)
    out.backward(go.to(dtype).to('ruda'))
    tol=.025 if dtype==torch.float16 else 3e-4
    torch.testing.assert_close(qd.grad.cpu().float(),qr.grad,rtol=tol,atol=tol)
    torch.testing.assert_close(kd.grad.cpu().float(),kr.grad,rtol=tol,atol=tol)
    torch.testing.assert_close(vd.grad.cpu().float(),vr.grad,rtol=tol,atol=tol)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_paged_mla_backward(backend,dtype):
    q,c,_=inputs(dtype,512,512,1,4);s=schedule();g=torch.Generator().manual_seed(81)
    qp=torch.randn((4,4,64),generator=g).to(dtype);kp=torch.randn((6,4,1,64),generator=g).to(dtype)
    args=[s[x] for x in ('block_tables','kv_lengths','sequence_ids','positions')]
    qr=q.detach().float().requires_grad_();cr=c.detach().float().requires_grad_();qpr=qp.detach().float().requires_grad_();kpr=kp.detach().float().requires_grad_()
    ref=dense(qr,cr,cr,*args,192**-.5,True,qpr,kpr);go=torch.randn_like(ref);ref.backward(go)
    qd=q.detach().to('ruda').requires_grad_();cd=c.detach().to('ruda').requires_grad_();qpd=qp.detach().to('ruda').requires_grad_();kpd=kp.detach().to('ruda').requires_grad_()
    out=backend.PagedAttentionPlan(**s).mla(qd,qpd,cd,kpd,scale=192**-.5,causal=True);out.backward(go.to(dtype).to('ruda'))
    tol=.03 if dtype==torch.float16 else 4e-4
    for actual,expected in ((qd.grad,qr.grad),(qpd.grad,qpr.grad),(cd.grad,cr.grad),(kpd.grad,kpr.grad)):
        torch.testing.assert_close(actual.cpu().float(),expected,rtol=tol,atol=tol)

def test_ruda_autocast_dense_projection(backend):
    x=torch.randn((16,32),device='ruda',dtype=torch.float32)
    w=torch.randn((32,24),device='ruda',dtype=torch.float32)
    with torch.autocast(device_type='ruda',dtype=torch.float16):
        y=torch.mm(x,w)
    assert y.dtype==torch.float16
    expected=x.cpu()@w.cpu()
    torch.testing.assert_close(y.cpu().float(),expected,rtol=.02,atol=.02)
