import importlib.util, pathlib, sys, types, torch
ROOT=pathlib.Path(__file__).resolve().parents[1]

def load_paged():
    pkg=types.ModuleType('v29fake');pkg.__path__=[];sys.modules['v29fake']=pkg
    c=types.ModuleType('v29fake._C');sys.modules['v29fake._C']=c
    spec=importlib.util.spec_from_file_location('v29fake._paged',ROOT/'ruda_torch/_paged.py')
    mod=importlib.util.module_from_spec(spec);sys.modules['v29fake._paged']=mod;spec.loader.exec_module(mod);return mod

class Native:
    def run(self,q,k,v,qp,kp,scale,causal):
        if qp is None:return q.sum(-1,keepdim=True).expand(q.shape[0],q.shape[1],v.shape[-1]).clone()
        return (q.sum(-1,keepdim=True)+qp.sum(-1,keepdim=True)).expand(q.shape[0],q.shape[1],v.shape[-1]).clone()
    def backward_selected(self,q,k,v,qp,kp,grad,scale,causal,needs):
        return tuple(g if need else None for g,need in zip(self.backward(q,k,v,qp,kp,grad,scale,causal),needs))
    def backward(self,q,k,v,qp,kp,grad,scale,causal):
        if qp is None:
            return torch.ones_like(q)*3,torch.ones_like(k)*5,torch.ones_like(v)*7
        return torch.ones_like(q)*2,torch.ones_like(qp)*3,torch.ones_like(k)*4,torch.ones_like(kp)*5

def test_gqa_autograd_protocol():
    m=load_paged();n=Native();q=torch.randn(2,3,4,requires_grad=True);k=torch.randn(5,2,1,4,requires_grad=True);v=torch.randn(5,2,1,6,requires_grad=True)
    y=m._PagedGQA.apply(n,q,k,v,.5,True);y.sum().backward()
    assert torch.equal(q.grad,torch.full_like(q,3)) and torch.equal(k.grad,torch.full_like(k,5)) and torch.equal(v.grad,torch.full_like(v,7))

def test_mla_autograd_protocol():
    m=load_paged();n=Native();q=torch.randn(2,3,4,requires_grad=True);qp=torch.randn(2,3,2,requires_grad=True);lat=torch.randn(5,2,1,4,requires_grad=True);kp=torch.randn(5,2,1,2,requires_grad=True)
    y=m._PagedMLA.apply(n,q,qp,lat,kp,.5,True);y.sum().backward()
    assert torch.equal(q.grad,torch.full_like(q,2));assert torch.equal(qp.grad,torch.full_like(qp,3));assert torch.equal(lat.grad,torch.full_like(lat,4));assert torch.equal(kp.grad,torch.full_like(kp,5))

def test_amp_contract_sources():
    cpp=(ROOT/'ruda_torch/csrc/backend.cpp').read_text();init=(ROOT/'ruda_torch/__init__.py').read_text();rust=(ROOT.parent/'src/lib.rs').read_text()
    assert 'TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m)' in cpp
    for op in ('mm','bmm','addmm','linear'): assert f'KERNEL_PRIVATEUSEONE({op}, lower_precision_fp)' in cpp
    assert 'get_amp_supported_dtype' in init and 'torch.float16' in init and 'torch.bfloat16' in init
    assert 'ruda_torch_abi_version() -> u32 { 10 }' in rust
