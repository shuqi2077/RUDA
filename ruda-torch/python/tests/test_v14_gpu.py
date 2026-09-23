"""Real native GPU acceptance; NOT a mock/CPU fallback test suite.
Run separately in fresh processes with RUDA_TORCH_ASYNC=0 and =1.
"""
import gc
import os
import pytest
import torch
from test_v14_host import dense,inputs,schedule

@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU')!='1':
        pytest.skip('explicit hardware acceptance only: set RUDA_REQUIRE_GPU=1')
    # Do not turn failed import, absent driver or backend failure into a skip.
    import ruda_torch
    assert ruda_torch.is_available()
    assert ruda_torch._C.abi_version==9
    assert os.environ.get('RUDA_CUDA_COMPILER')=='ptx'
    assert os.environ.get('RUDA_PTX_VERSION'), 'choose a driver-compatible explicit PTX version'
    torch.zeros(1).to('ruda').cpu()
    return ruda_torch

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('causal',[False,True])
def test_ragged_paged_real_kernel(backend,dtype,causal):
    q,k,v=inputs(dtype);s=schedule();plan=backend.PagedAttentionPlan(**s)
    args=[s[x] for x in ('block_tables','kv_lengths','sequence_ids','positions')]
    expected=dense(q,k,v,*args,33**-.5,causal)
    dev=[x.to('ruda') for x in (q,k,v)]
    for _ in range(3):
        output=plan.attention(*dev,scale=33**-.5,causal=causal).cpu().float()
        torch.testing.assert_close(output,expected,rtol=0.015 if dtype==torch.bfloat16 else 0.003,atol=0.004 if dtype==torch.bfloat16 else 0.001)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_paged_mla_real_kernel(backend,dtype):
    q,c,_=inputs(dtype,512,512,1,4);s=schedule();plan=backend.PagedAttentionPlan(**s)
    g=torch.Generator().manual_seed(44);qp=torch.randn((4,4,64),generator=g).to(dtype);kp=torch.randn((6,4,1,64),generator=g).to(dtype)
    args=[s[x] for x in ('block_tables','kv_lengths','sequence_ids','positions')]
    expected=dense(q,c,c,*args,192**-.5,True,qp,kp)
    output=plan.mla(q.to('ruda'),qp.to('ruda'),c.to('ruda'),kp.to('ruda'),scale=192**-.5).cpu().float()
    torch.testing.assert_close(output,expected,rtol=0.02,atol=0.005)

def test_real_streams_events_and_lifetimes(backend):
    assert os.environ.get('RUDA_TORCH_ASYNC') in ('0','1')
    a,b=backend.Stream(),backend.Stream()
    for i in range(50):
        with backend.stream(a):
            x=torch.full((64,64),i/10.).to('ruda');y=x+x;event=a.record_event()
        with backend.stream(b):
            b.wait_event(event);z=y*2
            backend.record_stream(y,b)
        del x,y;gc.collect()
        # Reuse pressure; event synchronization is checked, not mocked.
        with backend.stream(a):
            scratch=[torch.empty((64,64),device='ruda') for _ in range(8)]
        b.synchronize()
        torch.testing.assert_close(z.cpu(),torch.full((64,64),4*i/10.))
        event.close();del scratch,z
    assert a.query() and b.query()

def test_native_pytorch_guard(backend):
    # Uses c10 DeviceGuard, not only the convenience Python wrappers.
    stream=torch.Stream(device='ruda');event=torch.Event(device='ruda',enable_timing=True)
    with stream:
        x=torch.ones((16,16),device='ruda');y=x+x
        y.record_stream(stream)
        event.record(stream)
    event.synchronize();assert event.query();assert stream.query()
    torch.testing.assert_close(y.cpu(),torch.full((16,16),2.0))

def test_event_elapsed_time(backend):
    stream=backend.Stream();a=backend.Event(enable_timing=True);b=backend.Event(enable_timing=True)
    with backend.stream(stream):
        x=torch.ones((128,128),device='ruda');a.record();x=x+x;b.record()
    b.synchronize();assert a.elapsed_time(b)>=0
    a.close();b.close()
