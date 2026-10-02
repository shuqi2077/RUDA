"""Integrated model training and actual PyTorch AOT forward/backward capture.

These run on CPU using production Python modules. native='off' is explicit;
passing does not establish native graph coverage or RUDA GPU availability.
"""
import copy
import io
import pytest
import torch
from torch import nn
from architecture_test_utils import mhc,sparse,optim,model as hm,compiler,close

@pytest.fixture(autouse=True)
def isolation():
    torch._dynamo.reset();torch.manual_seed(42)
    yield
    torch._dynamo.reset()

def model(tied=False):
    return hm.HybridAttentionLanguageModel(19,8,2,2,streams=2,csa_ratio=2,hca_ratio=4,
        window_size=3,topk=2,index_heads=2,index_dim=4,query_rank=4,rope_dim=4,
        query_chunk_size=4,key_chunk_size=4,sinkhorn_iterations=3,tie_embeddings=tied)

def optimizer(m):
    return optim.MuonAdamW.from_model(m,muon_modules=list(m.layers),adamw_modules=[m.embedding,m.head],lr=.003,adamw_lr=.001,ns_steps=3)

@pytest.mark.parametrize('tied',[False,True])
def test_hybrid_multistep_training_checkpoint_resume(tied):
    mod=model(tied);opt=optimizer(mod);tokens=torch.randint(0,19,(2,9));mask=torch.ones_like(tokens,dtype=torch.bool);mask[1,-2:]=False
    original={n:p.detach().clone() for n,p in mod.named_parameters()}
    losses=[]
    for _ in range(3):
        opt.zero_grad(set_to_none=True);result=mod(tokens,valid_mask=mask,return_aux=True)
        loss=hm.next_token_loss(result.output,tokens,valid_mask=mask)+.2*result.indexer_loss
        assert torch.isfinite(loss);loss.backward()
        assert mod.layers[0].attention.indexer.query.weight.grad.abs().sum()>0
        assert mod.layers[0].attention.index_compressor.value.weight.grad.abs().sum()>0
        assert mod.layers[0].attention_connection.mapping.grad.abs().sum()>0
        assert mod.layers[1].attention.compressor.value.weight.grad.abs().sum()>0
        opt.step();assert not opt.last_step_skipped;losses.append(loss.item())
    assert any(not torch.equal(p,original[n]) for n,p in mod.named_parameters())
    # Actual torch serialization/deserialization, not just in-memory alias copies.
    buffer=io.BytesIO();torch.save({'model':mod.state_dict(),'optimizer':opt.state_dict()},buffer);buffer.seek(0)
    checkpoint=torch.load(buffer,weights_only=True);restored=model(tied);restored.load_state_dict(checkpoint['model'])
    resumed=optimizer(restored);resumed.load_state_dict(checkpoint['optimizer'])
    for candidate,stepper in [(mod,opt),(restored,resumed)]:
        stepper.zero_grad(set_to_none=True);r=candidate(tokens,valid_mask=mask,return_aux=True)
        (hm.next_token_loss(r.output,tokens,valid_mask=mask)+.2*r.indexer_loss).backward();stepper.step()
    for p,q in zip(mod.parameters(),restored.parameters()):close(p,q,atol=0,rtol=0)
    if tied:assert restored.head.weight is restored.embedding.weight

@pytest.mark.parametrize('length',[1,2,7])
@pytest.mark.parametrize('allpad',[False,True])
def test_next_token_loss_reference_and_empty(length,allpad):
    logits=torch.randn(2,length,11,dtype=torch.double,requires_grad=True);tokens=torch.randint(0,11,(2,length))
    mask=torch.ones(2,length,dtype=torch.bool)
    if allpad:mask[:]=False
    loss=hm.next_token_loss(logits,tokens,valid_mask=mask)
    if length==1 or allpad:assert loss==0
    else:
        expected=nn.functional.cross_entropy(logits[:,:-1].reshape(-1,11),tokens[:,1:].flatten())
        close(loss,expected)
    loss.backward();assert torch.isfinite(logits.grad).all()


def test_hybrid_lm_causal_logits():
    mod=model().eval();tokens=torch.randint(0,19,(2,9));other=tokens.clone();other[:,5:]=torch.randint(0,19,(2,4))
    with torch.no_grad():close(mod(tokens)[:,:5],mod(other)[:,:5],atol=3e-6,rtol=3e-5)


def test_hybrid_full_model_autocast_training():
    mod=model();opt=optimizer(mod);tokens=torch.randint(0,19,(2,8))
    with torch.autocast('cpu',dtype=torch.bfloat16):result=mod(tokens,return_aux=True)
    loss=hm.next_token_loss(result.output.float(),tokens)+.2*result.indexer_loss
    loss.backward();assert torch.isfinite(loss);opt.step();assert not opt.last_step_skipped

class WithAux(nn.Module):
    def __init__(self,module):super().__init__();self.module=module
    def forward(self,x):return self.module(x,return_aux=True)

@pytest.mark.parametrize('kind',['mhc','csa','hca','hybrid'])
def test_real_aot_captured_forward_backward_and_parameters(kind):
    if kind=='mhc':
        original=mhc.MHCSequential(4,[nn.Linear(4,4)],streams=2,sinkhorn_iterations=3)
        x=torch.randn(1,6,4,requires_grad=True)
    elif kind in ('csa','hca'):
        cls=sparse.CSA if kind=='csa' else sparse.HCA
        original=WithAux(cls(4,1,compress_ratio=2,window_size=3,topk=2,index_dim=2,index_heads=2,
            query_chunk_size=4,key_chunk_size=4,rope_dim=2))
        x=torch.randn(1,6,4,requires_grad=True)
    else:
        original=WithAux(model());x=torch.randint(0,19,(1,6))
    reference=copy.deepcopy(original)
    wrapped=compiler.compile(original,device_type='cpu',native='off',fullgraph=True)
    opt=torch.optim.SGD(original.parameters(),lr=.001,foreach=False)
    refopt=torch.optim.SGD(reference.parameters(),lr=.001,foreach=False)
    try:
        for _ in range(2):
            z=x.detach().clone().requires_grad_(x.requires_grad)
            opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True);x.grad=None
            a=wrapped(x);b=reference(z)
            if kind=='mhc':
                close(a,b,atol=3e-6,rtol=3e-5);la=a.square().mean();lb=b.square().mean()
            else:
                close(a.output,b.output,atol=3e-6,rtol=3e-5);close(a.indexer_loss,b.indexer_loss,atol=3e-6,rtol=3e-5)
                la=a.output.square().mean()+.2*a.indexer_loss;lb=b.output.square().mean()+.2*b.indexer_loss
            la.backward();lb.backward()
            if x.requires_grad:close(x.grad,z.grad,atol=3e-6,rtol=3e-5)
            for p,q in zip(original.parameters(),reference.parameters()):
                if p.grad is None or q.grad is None:assert p.grad is None and q.grad is None
                else:close(p.grad,q.grad,atol=3e-6,rtol=3e-5)
            opt.step();refopt.step()
            for p,q in zip(original.parameters(),reference.parameters()):close(p,q,atol=3e-6,rtol=3e-5)
        info=wrapped.info
        assert info['graphs'],info
        assert any(graph['phase']=='forward' for graph in info['graphs']),info
        assert any(graph['phase']=='backward' for graph in info['graphs']),info
    finally:wrapped.close()


def test_dense_indexer_warmup_updates_no_main_weights_or_moments():
    mod=model();opt=optimizer(mod)
    # Nonzero decay exposes accidental connected-zero main gradients.
    for group in opt.param_groups:group['weight_decay']=.1
    before={n:p.detach().clone() for n,p in mod.named_parameters()}
    result=mod(torch.randint(0,19,(2,9)),return_aux=True,indexer_warmup=True)
    result.indexer_loss.backward()
    for name,p in mod.named_parameters():
        if '.indexer.' in name or '.index_compressor.' in name:
            assert p.grad is not None,name
        else:assert p.grad is None,name
    opt.step()
    for name,p in mod.named_parameters():
        if '.indexer.' not in name and '.index_compressor.' not in name:
            close(p,before[name],atol=0,rtol=0);assert p not in opt.state
    assert not torch.equal(mod.layers[0].attention.indexer.query.weight,before['layers.0.attention.indexer.query.weight'])
