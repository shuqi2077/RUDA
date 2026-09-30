"""CPU math and Python autograd protocol only; does NOT run Rust/PTX kernels."""
import importlib.util
import pathlib
import sys
import types
import pytest
import torch
from router_reference import weights_reference, analytical_vjp

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_wrapper():
    name = 'ruda_v30_test_only'
    pkg = types.ModuleType(name); pkg.__path__ = []
    native = types.ModuleType(name+'._C')
    native.router_weights_forward = lambda x,i,m,n,s: weights_reference(x,i,m==0,n,s,promote=False)
    native.router_weights_backward = lambda x,i,g,m,n,s: analytical_vjp(x,i,g,m==0,n,s,promote=False)
    pkg._C = native; sys.modules[name] = pkg; sys.modules[name+'._C'] = native
    spec = importlib.util.spec_from_file_location(name+'._router', ROOT/'ruda_torch/_router.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('softmax',[False,True])
@pytest.mark.parametrize('norm',[False,True])
@pytest.mark.parametrize('scale',[.5,2.5])
@pytest.mark.parametrize('shape',[(1,3,1),(3,7,3),(2,65,8),(2,129,64)])
def test_vjp_against_independent_autograd(softmax,norm,scale,shape):
    t,e,k=shape; gen=torch.Generator().manual_seed(803+t+e+k)
    x=torch.randn(t,e,generator=gen,dtype=torch.float64,requires_grad=True)
    # Include repeated selection as well as unique top-k cases.
    ids=torch.randint(e,(t,k),generator=gen); grad=torch.randn(t,k,generator=gen,dtype=torch.float64)
    weights_reference(x,ids,softmax,norm,scale,promote=False).backward(grad)
    actual=analytical_vjp(x.detach(),ids,grad,softmax,norm,scale,promote=False)
    torch.testing.assert_close(actual,x.grad,rtol=1e-11,atol=1e-12)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('softmax',[False,True])
@pytest.mark.parametrize('norm',[False,True])
def test_fp32_scoring_and_final_storage_gradient(dtype,softmax,norm):
    g=torch.Generator().manual_seed(86)
    x=torch.randn(3,65,generator=g).to(dtype).requires_grad_(); ids=torch.randint(65,(3,5),generator=g)
    grad=torch.randn(3,5,generator=g)
    weights=weights_reference(x,ids,softmax,norm,2.5); assert weights.dtype==torch.float32
    weights.backward(grad)
    actual=analytical_vjp(x.detach(),ids,grad,softmax,norm,2.5)
    torch.testing.assert_close(actual,x.grad,rtol=.02 if dtype!=torch.float32 else 1e-5,atol=2e-4 if dtype!=torch.float32 else 2e-7)


@pytest.mark.parametrize('softmax',[False,True])
@pytest.mark.parametrize('norm',[False,True])
def test_custom_function_finite_difference(softmax,norm):
    mod=load_wrapper();x=torch.tensor([[.3,-.7,1.1,.4]],dtype=torch.float64,requires_grad=True)
    ids=torch.tensor([[2,0,2]])
    fn=lambda z:mod._SelectedRouterWeights.apply(z,ids,0 if softmax else 1,norm,1.7)
    assert torch.autograd.gradcheck(fn,(x,),eps=1e-6,atol=2e-6,rtol=2e-5)


def test_unselected_softmax_gradient_is_not_zero():
    x=torch.tensor([[.2,.8,-.1,1.3]],dtype=torch.float64)
    ids=torch.tensor([[3,1]]);g=torch.tensor([[1.,2.]],dtype=torch.float64)
    dx=analytical_vjp(x,ids,g,True,False,2.5,promote=False)
    assert bool((dx[0,[0,2]]<0).all())
    dx_norm=analytical_vjp(x,ids,g,True,True,2.5,promote=False)
    assert torch.equal(dx_norm[0,[0,2]],torch.zeros(2,dtype=x.dtype))


@pytest.mark.parametrize('softmax',[False,True])
def test_one_selected_normalized_slot_has_zero_logit_gradient(softmax):
    x=torch.tensor([[1.,2.,3.]],dtype=torch.float64);ids=torch.tensor([[2]])
    dx=analytical_vjp(x,ids,torch.tensor([[7.]],dtype=torch.float64),softmax,True,2.5,promote=False)
    torch.testing.assert_close(dx,torch.zeros_like(dx),atol=1e-14,rtol=0)


@pytest.mark.parametrize('kind',['logits','indices'])
def test_saved_inputs_detect_in_place_mutation(kind):
    mod=load_wrapper();x=torch.randn(2,4,dtype=torch.float64,requires_grad=True);ids=torch.tensor([[0,2],[1,3]])
    out=mod._SelectedRouterWeights.apply(x,ids,0,False,1.)
    with torch.no_grad():
        (x if kind=='logits' else ids).add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        out.sum().backward()


def test_higher_order_is_rejected_not_silently_wrong():
    mod=load_wrapper();x=torch.randn(1,3,dtype=torch.float64,requires_grad=True)
    y=mod._SelectedRouterWeights.apply(x,torch.tensor([[1,2]]),0,True,1.)
    with pytest.raises(RuntimeError,match='first-order'):
        torch.autograd.grad(y.sum(),x,create_graph=True)


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf'),1e100,1e-100,True,torch.tensor(1.)])
def test_scalar_contract_rejects_invalid(scale):
    with pytest.raises((ValueError,TypeError)):
        load_wrapper()._options('softmax',False,scale)


@pytest.mark.parametrize('mode',['Softmax','unknown',None,1])
def test_unknown_scoring_is_rejected(mode):
    with pytest.raises(ValueError):load_wrapper()._options(mode,False,1.)


def test_public_api_never_accepts_cpu_as_runtime_fallback():
    with pytest.raises(ValueError,match='ruda'):
        load_wrapper().selected_router_weights(torch.randn(2,3),torch.tensor([[0],[1]]))


@pytest.mark.parametrize('softmax',[False,True])
@pytest.mark.parametrize('norm',[False,True])
def test_router_projection_experts_and_dispatch_whole_gradient_chain(softmax,norm):
    """Fixed top-k complete toy MoE mathematical chain, not a device implementation."""
    gen=torch.Generator().manual_seed(993)
    x=torch.randn(3,4,dtype=torch.float64,generator=gen,requires_grad=True)
    router=torch.randn(5,4,dtype=torch.float64,generator=gen,requires_grad=True)
    experts=torch.randn(5,2,4,dtype=torch.float64,generator=gen,requires_grad=True)
    logits=x@router.T;ids=torch.tensor([[3,1],[2,3],[1,0]])
    weights=weights_reference(logits,ids,softmax,norm,1.7,promote=False)
    raw=torch.einsum('th,tkoh->tko',x,experts[ids])
    upstream=torch.randn(3,2,dtype=torch.float64,generator=gen)
    out=(raw*weights[:,:,None]).sum(1); out.backward(upstream)
    with torch.no_grad():
        # combine backward -> selected router VJP -> projection backward
        dw=(raw*upstream[:,None,:]).sum(-1)
        dl=analytical_vjp(logits,ids,dw,softmax,norm,1.7,promote=False)
        expected_router=dl.T@x
        # expert dInput -> dispatch backward is an UNWEIGHTED sum of row gradients
        expert_upstream=upstream[:,None,:]*weights[:,:,None]
        dx_rows=torch.einsum('tko,tkoh->tkh',expert_upstream,experts[ids])
        expected_x=dx_rows.sum(1)+dl@router
    torch.testing.assert_close(router.grad,expected_router,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(x.grad,expected_x,rtol=1e-11,atol=1e-12)


def test_bounds_guard_precedes_offset_and_no_dense_probability_allocation():
    src=(ROOT.parent.parent/'ruDNN/src/moe/kernels/router_training.rs').read_text()
    assert src.count('expert < 0i64 || expert >= experts as i64')==2
    assert 'i64::cast_from(ids[' in src
    assert 'Atomic' not in src and 'Array::new' not in src
    host=(ROOT.parent.parent/'ruDNN/src/moe/router_training.rs').read_text()
    assert 'DType::F32' in host and 'same_execution_queue' in host
