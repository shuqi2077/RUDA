"""Host math/layout/source contracts; never claim these execute Rust/PTX."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from grouped_backward_reference import tiled_backward,dense_backward

ROOT=Path(__file__).resolve().parents[2]
DTYPES=[torch.float32,torch.float16,torch.bfloat16]
SHAPES=[([1],1,1),([16],16,16),([1,0,17],17,33),([0,0],19,7),([0,3,0,65],35,17),
        ([1,1,1,1],15,31),([17,33,0],33,65),([31,0,1],7,19)]

def quantized(a,d):return torch.from_numpy(a).to(d).float().numpy()

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('lengths,k,n',SHAPES)
def test_tile_transposes_and_output_coverage(dtype,lengths,k,n):
    g=np.random.default_rng(123);m=sum(lengths);e=len(lengths)
    x=quantized(g.normal(0,0.25,(m,k)).astype(np.float32),dtype)
    w=quantized(g.normal(0,0.25,(e,n,k)).astype(np.float32),dtype)
    dy=quantized(g.normal(0,0.25,(m,n)).astype(np.float32),dtype)
    dx,dw=tiled_backward(x,w,dy,lengths);refx,refw=dense_backward(x,w,dy,lengths)
    np.testing.assert_allclose(dx,refx,rtol=1e-5,atol=2e-6)
    np.testing.assert_allclose(dw,refw,rtol=1e-5,atol=2e-6)
    for e,count in enumerate(lengths):
        if count==0:assert np.array_equal(dw[e],np.zeros_like(dw[e]))

@pytest.mark.parametrize('seed',range(16))
def test_random_segment_layouts(seed):
    rng=np.random.default_rng(seed);lengths=rng.integers(0,36,size=rng.integers(1,6)).tolist()
    k=int(rng.integers(1,36));n=int(rng.integers(1,36));m=sum(lengths)
    x=(rng.integers(-16,17,(m,k))/32).astype(np.float32)
    w=(rng.integers(-16,17,(len(lengths),n,k))/32).astype(np.float32)
    dy=(rng.integers(-16,17,(m,n))/32).astype(np.float32)
    got=tiled_backward(x,w,dy,lengths);ref=dense_backward(x,w,dy,lengths)
    for a,b in zip(got,ref):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('seed',range(8))
def test_swiglu_storage_boundaries_match_autograd(dtype,seed):
    gen=torch.Generator().manual_seed(seed)
    g=torch.randn(257,generator=gen).mul(3).to(dtype).requires_grad_()
    up=torch.randn(257,generator=gen).mul(2).to(dtype).requires_grad_()
    dy=torch.randn(257,generator=gen).mul(3).to(dtype)
    gf=g.float();activation=(gf/(1+torch.exp(-gf))).to(dtype)
    y=(activation.float()*up.float()).to(dtype)
    expected=torch.autograd.grad(y,(g,up),dy)
    sigmoid=1/(1+torch.exp(-g.detach().float()))
    candidate_g=((dy.float()*up.detach().float()).to(dtype).float()*sigmoid*(1+g.detach().float()*(1-sigmoid))).to(dtype)
    candidate_up=(dy.float()*activation.detach().float()).to(dtype)
    tol={torch.float32:1e-5,torch.float16:0.004,torch.bfloat16:0.04}[dtype]
    torch.testing.assert_close(candidate_g,expected[0],rtol=tol,atol=tol)
    torch.testing.assert_close(candidate_up,expected[1],rtol=0,atol=0)

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_old_backward_really_differs_without_storage_rounding(dtype):
    gen=torch.Generator().manual_seed(2026)
    g=torch.randn(8192,generator=gen).mul(3).to(dtype).float()
    u=torch.randn(8192,generator=gen).mul(2).to(dtype).float()
    dy=torch.randn(8192,generator=gen).mul(3).to(dtype).float()
    sig=1/(1+torch.exp(-g));activation=(g/(1+torch.exp(-g))).to(dtype).float()
    old_up=(dy*g*sig).to(dtype);new_up=(dy*activation).to(dtype)
    old_g=(dy*u*sig*(1+g*(1-sig))).to(dtype)
    new_g=((dy*u).to(dtype).float()*sig*(1+g*(1-sig))).to(dtype)
    assert torch.count_nonzero(old_up!=new_up)>100
    assert torch.count_nonzero(old_g!=new_g)>100

@pytest.mark.parametrize('dtype',DTYPES)
def test_expert_matrix_reference_autograd(dtype):
    gen=torch.Generator().manual_seed(12);lengths=[2,0,17];m=sum(lengths);e=3;k=7;n=19
    x=torch.randn(m,k,generator=gen).to(dtype).float().requires_grad_()
    w=torch.randn(e,n,k,generator=gen).to(dtype).float().requires_grad_()
    dy=torch.randn(m,n,generator=gen).to(dtype).float()
    ids=torch.repeat_interleave(torch.arange(e),torch.tensor(lengths))
    y=torch.bmm(w[ids],x.unsqueeze(-1)).squeeze(-1)
    dx,dw=torch.autograd.grad(y,(x,w),dy)
    got=tiled_backward(x.detach().numpy(),w.detach().numpy(),dy.numpy(),lengths)
    np.testing.assert_allclose(got[0],dx.numpy(),rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(got[1],dw.numpy(),rtol=2e-5,atol=2e-5)

def test_kernel_contract_no_atomics_or_divergent_early_exit():
    text=(ROOT/'ruBLAS/src/tensor_grouped/backward_tensorcore.rs').read_text()
    assert 'Atomic<' not in text and 'terminate!(' not in text
    assert text.count('cmma::execute')==2
    assert 'let acc = cmma::Matrix::<f32>' in text
    assert 'out: &mut Array<f32>' in text
    assert 'SharedMemory::<f32>::new_aligned(256usize' in text

def test_default_and_strict_capability_policy():
    text=(ROOT/'ruBLAS/src/tensor_grouped/mod.rs').read_text()
    wrapper=text.split('pub unsafe fn grouped_matmul_nt_backward_segmented<R')[1].split('/// Opt-in')[0]
    assert 'GroupedStrategy::Scalar' in wrapper
    before_alloc=text.split('let dinput=empty_device_contiguous_dtype')[0]
    assert 'strategy == GroupedStrategy::TensorCore && !cooperative' in before_alloc
    assert 'props.plane_size_min == 32 && props.plane_size_max == 32' in text
    assert 'matmul.cmma.contains(&cfg)' in text
    assert 'DType::F32);' in text

def test_expert_chain_reuses_public_strategy():
    text=(ROOT/'ruDNN/src/moe/experts.rs').read_text()
    assert text.count('grouped_matmul_nt_backward_segmented_with_strategy(')==3
    assert 'self.backward_with_strategy(grad_output, GroupedStrategy::Scalar)' in text
    assert 'let activated=gate_raw.copy()' not in text
    assert 'kernels::experts::swiglu_out::launch' in text

def test_public_gradients_keep_intermediate_rounding():
    text=(ROOT/'ruDNN/src/moe/kernels/experts.rs').read_text().split('pub(crate) fn swiglu_backward')[1]
    assert 'let intermediate = F::cast_from(dy * u)' in text
    assert 'let silu = F::cast_from(g /' in text
    assert 'f32::cast_from(intermediate)' in text and 'dy * f32::cast_from(silu)' in text

def test_copy_elimination_size_not_peak_memory_claim():
    rows=2048;width=14336;size=rows*width*2
    assert size==56*1024**2
    # Removing a copy eliminates one logical read + write, not a live tensor.
    assert 2*size==112*1024**2
