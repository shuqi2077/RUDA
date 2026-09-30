"""Real production GPU training acceptance, never skips missing/unsupported hardware.

Default dtype suite is FP32+FP16. Select BF16 explicitly only on a suitable device
using RUDA_TRAINING_DTYPES=float32,float16,bfloat16. Collection does not load RUDA.
"""
import copy
import ctypes
import os
import pytest
import torch

_NAMES=os.environ.get('RUDA_TRAINING_DTYPES','float32,float16').split(',')
if not _NAMES or len(set(_NAMES))!=len(_NAMES) or any(n not in ('float32','float16','bfloat16') for n in _NAMES):
    raise RuntimeError('invalid RUDA_TRAINING_DTYPES')
DTYPES=[getattr(torch,n) for n in _NAMES]

@pytest.fixture(scope='module')
def r():
    assert os.environ.get('RUDA_CUDA_COMPILER')=='ptx'
    assert os.environ.get('RUDA_PTX_VERSION')
    import ruda_torch as r
    assert r._training_available and r._C.training_api_version==4
    x=torch.tensor([2.,3.]).to('ruda');r.synchronize()
    driver=ctypes.CDLL('libcuda.so.1');kind=ctypes.c_uint()
    driver.cuPointerGetAttribute.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64]
    driver.cuPointerGetAttribute.restype=ctypes.c_int
    assert driver.cuPointerGetAttribute(ctypes.byref(kind),2,x.data_ptr())==0 and kind.value==2
    torch.testing.assert_close((x*x).cpu(),torch.tensor([4.,9.]))
    print('RUDA_V28_TRAINING_GPU_EXECUTED',flush=True)
    return r


def close(a,b,dtype=None):
    dtype=dtype or a.dtype
    tol={torch.float32:5e-4,torch.float16:6e-3,torch.bfloat16:5e-2}[dtype]
    torch.testing.assert_close(a.detach().cpu().float(),b.detach().cpu().float(),atol=tol,rtol=tol,equal_nan=True)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('shape',[(3,17),(2,3,33),(1,4096),(0,7)])
@pytest.mark.parametrize('weights',['none','same','fp32'])
def test_rms_norm_forward_backward(r,dtype,shape,weights):
    torch.manual_seed(712);a=torch.randn(shape,dtype=dtype)
    wh=None if weights=='none' else torch.randn(shape[-1],dtype=dtype if weights=='same' else torch.float32)
    dy=torch.randn_like(a);x=a.to('ruda').requires_grad_();w=None if wh is None else wh.to('ruda').requires_grad_();dg=dy.to('ruda')
    before=r.execution_stats();y=r.rms_norm(x,w,eps=1e-5);y.backward(dg);r.synchronize();after=r.execution_stats()
    for key in ('host_to_device_bytes','device_to_host_bytes'):assert before[key]==after[key]
    xa=a.float().requires_grad_();wa=None if wh is None else wh.float().requires_grad_()
    ref=xa*(xa.square().mean(-1,keepdim=True)+1e-5).rsqrt()
    if wa is not None:ref=ref*wa
    ref=ref.to(dtype);ref.backward(dy)
    close(y,ref);close(x.grad,xa.grad.to(dtype))
    if w is not None:close(w.grad,wa.grad.to(w.dtype),dtype)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('module_kind',['explicit','torch'])
def test_layer_norm_forward_backward(r,dtype,module_kind):
    torch.manual_seed(831);a=torch.randn(3,5,33,dtype=dtype);dy=torch.randn_like(a)
    if module_kind=='explicit':
        layer=r.LayerNorm(33,device='ruda',dtype=torch.float32)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(33).to('ruda'));layer.bias.copy_(torch.randn(33).to('ruda'))
    else:
        layer=torch.nn.LayerNorm(33).to(device='ruda',dtype=dtype)
    x=a.to('ruda').requires_grad_();g=dy.to('ruda')
    before=r.execution_stats();y=layer(x);y.backward(g);r.synchronize();after=r.execution_stats()
    for key in ('host_to_device_bytes','device_to_host_bytes','legacy_fp32_temp_bytes_total'):assert before[key]==after[key]
    rw=layer.weight.detach().cpu().float().requires_grad_();rb=layer.bias.detach().cpu().float().requires_grad_();rx=a.float().requires_grad_()
    ref=torch.nn.functional.layer_norm(rx,(33,),rw,rb,layer.eps).to(dtype);ref.backward(dy)
    close(y,ref);close(x.grad,rx.grad.to(dtype));close(layer.weight.grad,rw.grad.to(layer.weight.dtype),layer.weight.dtype)
    close(layer.bias.grad,rb.grad.to(layer.bias.dtype),layer.bias.dtype)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('needs',[(True,True),(False,True),(True,False)])
def test_gate_forward_backward_optional(r,dtype,needs):
    torch.manual_seed(117);gh=torch.randn(3,33,dtype=dtype);uh=torch.randn_like(gh);dy=torch.randn_like(gh)
    g=gh.to('ruda').requires_grad_(needs[0]);u=uh.to('ruda').requires_grad_(needs[1]);grad=dy.to('ruda')
    before=r.execution_stats();y=r.silu_mul(g,u);y.backward(grad);r.synchronize();after=r.execution_stats()
    for key in ('host_to_device_bytes','device_to_host_bytes'):assert before[key]==after[key]
    a=gh.clone().requires_grad_(needs[0]);b=uh.clone().requires_grad_(needs[1])
    ref=torch.nn.functional.silu(a.float()).to(dtype)*b;ref.backward(dy)
    close(y,ref)
    if needs[0]:close(g.grad,a.grad)
    if needs[1]:close(u.grad,b.grad)

@pytest.mark.parametrize('dtype',DTYPES)
def test_adamw_fp32_master_resume_and_scalar_transfer(r,dtype):
    torch.manual_seed(417);initial=torch.randn(257,dtype=dtype)
    p=torch.nn.Parameter(initial.to('ruda'));q=torch.nn.Parameter(initial.float().clone())
    opt=r.AdamW([p],lr=.005,betas=(.8,.93),eps=1e-6,weight_decay=.03)
    ref=torch.optim.AdamW([q],lr=.005,betas=(.8,.93),eps=1e-6,weight_decay=.03,foreach=False)
    gradients=[torch.randn_like(initial) for _ in range(6)]
    for i,g in enumerate(gradients):
        p.grad=g.to('ruda');q.grad=g.float().clone()
        # Initialization uploads FP32 zero scalars; subsequent iterations must
        # download exactly one flag, not parameter/gradient tensors.
        before=r.execution_stats();opt.step();r.synchronize();after=r.execution_stats();ref.step()
        assert after['device_to_host_bytes']-before['device_to_host_bytes']==4
        close(p,q.to(dtype))
        if i==2:
            state={k:v for k,v in opt.state_dict().items()}
            state=copy.deepcopy(state)
            saved=p.detach().cpu()
    p2=torch.nn.Parameter(saved.to('ruda'));opt2=r.AdamW([p2]);opt2.load_state_dict(state)
    for g in gradients[3:]:p2.grad=g.to('ruda');opt2.step()
    close(p,p2)
    assert opt2.state[p2]['exp_avg'].dtype==torch.float32
    if dtype!=torch.float32:assert opt2.state[p2]['master_copy'].dtype==torch.float32

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('dtype',DTYPES)
def test_overflow_skips_all_parameters_and_steps(r,bad,dtype):
    p=torch.nn.Parameter(torch.ones(33,dtype=dtype).to('ruda'));q=torch.nn.Parameter(torch.ones(31,dtype=dtype).to('ruda'))
    opt=r.AdamW([p,q]);p.grad=torch.ones(33,dtype=dtype).to('ruda');q.grad=torch.full((31,),bad,dtype=dtype).to('ruda')
    opt.step(loss_scale=8)
    assert opt.last_step_skipped and not opt.state
    close(p,torch.ones(33));close(q,torch.ones(31))

@pytest.mark.parametrize('dtype',DTYPES)
def test_mean_finite_despite_half_sum_overflow(r,dtype):
    a=torch.full((3,4096),1000.,dtype=dtype)
    x=a.to('ruda').requires_grad_();before=r.execution_stats();y=x.mean(-1);y.sum().backward()
    after=r.execution_stats();close(y,a.float().mean(-1).to(dtype));close(x.grad,torch.full_like(a,1/4096))
    for key in ('device_to_host_bytes','legacy_fp32_temp_bytes_total'):assert after[key]==before[key]

@pytest.mark.parametrize('shape,axes',[((),None),((2,0,7),[1]),((3,7),[0,1]),((0,3),[-1])])
def test_mean_shape_and_empty_semantics(r,shape,axes):
    a=torch.ones(shape,dtype=torch.float16);close(a.to('ruda').mean(dim=axes),a.float().mean(dim=axes).half())


def test_first_order_only_and_saved_tensor_version(r):
    x=torch.ones(2,17,device='ruda',requires_grad=True)
    with pytest.raises(RuntimeError,match='first-order'):torch.autograd.grad(r.rms_norm(x).sum(),x,create_graph=True)
    y=r.silu_mul(x,x)
    with torch.no_grad():x.add_(1)
    with pytest.raises(RuntimeError,match='modified|inplace'):y.sum().backward()


def test_nondefault_stream_training_lifetimes(r):
    s=r.Stream()
    with r.stream(s):
        x=torch.ones(2,17,device='ruda',requires_grad=True);w=torch.ones(17,device='ruda',requires_grad=True)
        y=r.rms_norm(x,w);y.sum().backward()
        saved=x.grad;del y,x,w
        pressure=[torch.empty(2048,device='ruda') for _ in range(12)]
        s.synchronize()
        assert torch.isfinite(saved.cpu()).all()
    # Native stream IDs are managed by the runtime; Stream has no close().


@pytest.mark.parametrize('dtype',DTYPES)
def test_small_training_loop_and_checkpoint(r,dtype):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.norm=r.RMSNorm(8);self.gate=torch.nn.Linear(8,16);self.up=torch.nn.Linear(8,16);self.down=torch.nn.Linear(16,4)
        def forward(self,x):
            x=self.norm(x);return self.down(r.silu_mul(self.gate(x),self.up(x)))
    torch.manual_seed(44);net=Block().to(device='ruda',dtype=dtype);opt=r.AdamW(net.parameters(),lr=.001)
    scaler=r.GradScaler(init_scale=8,growth_interval=2)
    source=torch.randn(4,8,dtype=dtype).to('ruda');target=torch.randn(4,4).to('ruda')
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        # Two microbatches accumulate before unscale/update. Loss stays FP32.
        for sl in (slice(0,2),slice(2,4)):
            loss=(net(source[sl]).float()-target[sl]).square().mean()/2
            scaler.scale(loss).backward()
        scaler.step(opt);scaler.update()
        assert not opt.last_step_skipped
        for p in net.parameters():assert p.grad is not None and torch.isfinite(p.grad.cpu()).all()
    state=copy.deepcopy(opt.state_dict());scale_state=scaler.state_dict()
    opt.load_state_dict(state);scaler.load_state_dict(scale_state)
    assert all(opt.state[p]['step']==3 for p in net.parameters())
