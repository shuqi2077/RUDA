"""C++/Python dispatch registration audit, NOT numerical GPU validation.

Real production components run on CPU under an operator recorder. A freshly
compiled C++ bridge and production Python registrations are then checked for
matching device dispatch entries. Host ABI callbacks never execute numerics.
"""
import importlib
from pathlib import Path
import sys
import types
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from test_v15_cpp import bridge, _KEEP_ALIVE
from architecture_test_utils import mhc,sparse,optim,model as hm

@pytest.fixture(scope='module')
def device_registrations(bridge):
    cpp,state,_=bridge
    torch.ruda.manual_seed_all=lambda seed: None
    name='ruda_architecture_dispatch_contract'
    package=types.ModuleType(name)
    package.__path__=[str(Path(__file__).resolve().parents[1]/'ruda_torch')]
    package._C=cpp
    package._training_available=False
    package._paged_backward_available=False
    sys.modules[name]=package
    registrations=importlib.import_module(name+'._ops')
    return registrations,state

class Recorder(TorchDispatchMode):
    def __init__(self):super().__init__();self.operations=set()
    def __torch_dispatch__(self,function,types,args=(),kwargs=None):
        self.operations.add(function.name())
        return function(*args,**(kwargs or {}))

@pytest.mark.parametrize('kind',['mhc','csa','hca','hybrid_muon'])
def test_observed_training_operations_have_device_dispatch(device_registrations,kind):
    _,state=device_registrations;before=len(state.allocations)
    torch.manual_seed(56)
    if kind=='mhc':
        module=mhc.MHCSequential(4,[torch.nn.Linear(4,4)],streams=2,sinkhorn_iterations=3)
        x=torch.randn(1,6,4,requires_grad=True)
    elif kind in ('csa','hca'):
        module=(sparse.CSA if kind=='csa' else sparse.HCA)(4,1,compress_ratio=2,window_size=3,
            index_dim=2,rope_dim=2,topk=2,query_chunk_size=4,key_chunk_size=4)
        x=torch.randn(1,6,4,requires_grad=True)
    else:
        module=hm.HybridAttentionLanguageModel(13,4,1,2,streams=2,csa_ratio=2,hca_ratio=4,
            window_size=3,index_dim=2,rope_dim=2,topk=2,sinkhorn_iterations=3,
            query_chunk_size=4,key_chunk_size=4)
        x=torch.randint(0,13,(1,6))
        optimizer=optim.MuonAdamW.from_model(module,muon_modules=list(module.layers),adamw_modules=[module.head],ns_steps=3)
    recorder=Recorder()
    with recorder:
        result=module(x) if kind=='mhc' else module(x,return_aux=True)
        loss=result.square().mean() if kind=='mhc' else result.output.square().mean()+.1*result.indexer_loss
        loss.backward()
        if kind=='hybrid_muon':optimizer.step()
    missing=[]
    for op in sorted(recorder.operations):
        supported=any(torch._C._dispatch_has_kernel_for_dispatch_key(op,key) for key in
            ['PrivateUse1','CompositeImplicitAutograd','CompositeExplicitAutograd','CompositeExplicitAutogradNonFunctional'])
        if not supported:missing.append(op)
    assert not missing,missing
    assert len(state.allocations)==before  # no fake numeric GPU output was used
    print(f'{kind}: {len(recorder.operations)} observed operator schemas have dispatch entries')
