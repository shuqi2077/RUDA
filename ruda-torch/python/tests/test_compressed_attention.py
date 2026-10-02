"""Production CSA/HCA/DSA on CPU; independent dense/block-loop oracles."""
import copy
import pytest
import torch
from torch.nn import functional as F
from architecture_test_utils import sparse as sa, ops, close


def compressor_oracle(module,x,mask):
    b,t,w=x.shape;r=module.ratio;d=module.head_dim
    blocks=[];validity=[]
    for block in range(t//r):
        values=[];gates=[];masks=[]
        if module.overlap and block:
            previous=x[:,(block-1)*r:block*r]
            values.append(F.linear(previous,module.value.weight[:d]))
            gates.append(F.linear(previous,module.gate.weight[:d])+module.position_bias[:,:d])
            masks.append(mask[:,(block-1)*r:block*r])
        current=x[:,block*r:(block+1)*r]
        offset=d if module.overlap else 0
        values.append(F.linear(current,module.value.weight[offset:]))
        gates.append(F.linear(current,module.gate.weight[offset:])+module.position_bias[:,offset:])
        masks.append(mask[:,block*r:(block+1)*r])
        v=torch.cat(values,1);g=torch.cat(gates,1);m=torch.cat(masks,1)
        rows=[]
        for batch in range(b):
            if m[batch].any():
                weights=g[batch,m[batch]].softmax(dim=0)
                value=(weights*v[batch,m[batch]]).sum(0)
                value=value/(value.square().mean()+module.eps).sqrt()*module.norm_weight
            else:value=v[batch].sum(0)*0
            rows.append(value)
        blocks.append(torch.stack(rows));validity.append(m.any(1))
    if not blocks:return x.new_empty((b,0,d)),mask[:,:0]
    return torch.stack(blocks,1),torch.stack(validity,1)

@pytest.mark.parametrize('overlap',[False,True])
@pytest.mark.parametrize('length',[1,3,4,7,13])
def test_compressor_independent_blockloop(overlap,length):
    torch.manual_seed(9)
    mod=sa.LearnedKVCompressor(4,3,3,overlap=overlap,dtype=torch.double)
    with torch.no_grad():mod.position_bias.normal_()
    x=torch.randn(2,length,4,dtype=torch.double,requires_grad=True)
    mask=torch.rand(2,length)>.35;mask[0]=False
    y,valid=mod(x,mask);expected,valid_expected=compressor_oracle(mod,x,mask)
    close(y,expected,atol=1e-12,rtol=1e-12);close(valid,valid_expected)
    y.square().sum().backward();assert torch.isfinite(x.grad).all()
    if length>=3:
        assert mod.value.weight.grad is not None and mod.gate.weight.grad is not None
        assert mod.position_bias.grad is not None and mod.norm_weight.grad is not None

@pytest.mark.parametrize('overlap',[False,True])
@pytest.mark.parametrize('ratio',[2,3,5])
def test_compressor_append_boundaries_storage(overlap,ratio):
    torch.manual_seed(10)
    mod=sa.LearnedKVCompressor(4,3,ratio,overlap=overlap).eval()
    x=torch.randn(2,17,4);mask=torch.rand(2,17)>.25
    state=None;parts=[];masks=[];start=0
    with torch.no_grad():
        expected,expected_valid=mod(x,mask)
        for length in [1,2,6,1,7]:
            y,v,state=mod.append(x[:,start:start+length],mask[:,start:start+length],state)
            parts.append(y);masks.append(v);start+=length
            assert state.tail.shape[1]<ratio and state.previous.shape[1]<=ratio
            assert state.tail.untyped_storage().nbytes()==state.tail.numel()*state.tail.element_size()
            assert state.previous.untyped_storage()._cdata!=x.untyped_storage()._cdata
        close(torch.cat(parts,1),expected,atol=1e-6,rtol=1e-5)
        close(torch.cat(masks,1),expected_valid)

def test_compressor_gradcheck():
    mod=sa.LearnedKVCompressor(2,2,2,overlap=True,dtype=torch.double)
    x=torch.randn(1,4,2,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z:mod(z)[0],(x,),fast_mode=True)

@pytest.mark.parametrize('rope',[0,2,4])
def test_rotary_inverse_and_independent_complex(rope):
    x=torch.randn(2,5,3,6,dtype=torch.double,requires_grad=True)
    positions=torch.arange(5)+11;mod=sa.RotaryEmbedding(rope)
    y=mod(x,positions);close(mod(y,positions,inverse=True),x,atol=1e-12,rtol=1e-12)
    if rope:
        complex_x=torch.view_as_complex(x[...,-rope:].reshape(2,5,3,rope//2,2).contiguous())
        angles=positions.double()[:,None]*mod.frequencies.double()[None,:]
        rot=torch.polar(torch.ones_like(angles),angles)[None,:,None,:]
        reference=torch.view_as_real(complex_x*rot).flatten(-2)
        close(y[...,-rope:],reference,atol=1e-12,rtol=1e-12)
    y.square().sum().backward();assert torch.isfinite(x.grad).all()

@pytest.mark.parametrize('chunks',[(1,1),(3,2),(20,20)])
@pytest.mark.parametrize('topk',[1,3,20])
def test_indexer_dense_score_and_chunked_selection(chunks,topk):
    torch.manual_seed(11)
    mod=sa.DSAIndexer(6,2,4,topk=topk,query_chunk_size=chunks[0],key_chunk_size=chunks[1],rope_dim=2,dtype=torch.double)
    x=torch.randn(2,7,6,dtype=torch.double,requires_grad=True)
    keys=mod.project_keys(x)
    q,w=mod._queries(x)
    expected=(torch.einsum('bthd,bsd->bths',q,keys).relu()*w[...,None]).sum(2)
    close(mod.scores(x,keys),expected)
    allowed=torch.arange(7)[None,None,:]<=torch.arange(7)[None,:,None]
    allowed=allowed.expand(2,-1,-1).clone();allowed[0,:2]=False
    ids=mod.select(x,keys,allowed=allowed)
    reference=expected.masked_fill(~allowed,float('-inf'))
    reference_ids=reference.argsort(dim=-1,descending=True,stable=True)[...,:min(topk,7)]
    reference_ids=torch.where(allowed.gather(-1,reference_ids),reference_ids,torch.full_like(reference_ids,-1))
    close(ids,reference_ids)
    selected=mod.selected_scores(x,keys,ids)
    close(selected,torch.where(ids>=0,expected.gather(-1,ids.clamp_min(0)),torch.zeros_like(selected)))
    teacher=torch.rand_like(selected,requires_grad=True)
    loss=sa.indexer_kl_loss(selected,teacher,ids>=0);loss.backward()
    assert x.grad is None and teacher.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())

def test_indexer_default_causal_and_padding_metadata():
    mod=sa.DSAIndexer(4,2,2,topk=5);x=torch.randn(2,5,4)
    result=mod(x);assert ((result.indices<=torch.arange(5)[None,:,None])|~result.valid).all()
    keyvalid=torch.tensor([[True,False,True,True,False]]*2)
    qvalid=torch.tensor([[False,True,True,True,True]]*2)
    result=mod(x,key_valid=keyvalid,query_valid=qvalid)
    assert (result.indices[:,0]==-1).all()
    for b in range(2):
        selected=result.indices[b][result.valid[b]]
        assert keyvalid[b,selected].all()

@pytest.mark.parametrize('field,value',[
    ('key_valid',torch.ones(2,3)),('query_valid',torch.ones(2,4,dtype=torch.bool)),
    ('key_end_positions',torch.arange(3).float()),('query_positions',torch.arange(4)),
    ('allowed',torch.ones(2,3,3))])
def test_indexer_invalid_metadata(field,value):
    mod=sa.DSAIndexer(4,2,2);x=torch.randn(2,3,4);keys=mod.project_keys(x)
    with pytest.raises(ValueError):mod.select(x,keys,**{field:value})

def test_indexer_empty_keys_and_loss_zero_rows():
    mod=sa.DSAIndexer(4,2,2,external_keys=True)
    x=torch.randn(2,3,4,requires_grad=True);keys=torch.empty(2,0,2)
    out=mod(x,prepared_keys=keys)
    assert out.indices.shape==(2,3,0)
    z=sa.indexer_kl_loss(out.scores,torch.empty(2,3,0));z.backward()
    scores=torch.randn(2,3,4,requires_grad=True);teacher=torch.zeros_like(scores,requires_grad=True)
    loss=sa.indexer_kl_loss(scores,teacher);assert loss==0
    loss.backward();assert not scores.grad.any() and teacher.grad is None

def test_indexer_dense_warmup_loss_oracle_and_detach():
    mod=sa.DSAIndexer(4,2,3,dtype=torch.double);x=torch.randn(2,5,4,dtype=torch.double,requires_grad=True)
    keys=mod.project_keys(x);scores=mod.scores(x,keys)
    teacher=torch.rand(2,5,2,5,dtype=torch.double,requires_grad=True)
    allowed=(torch.arange(5)[None,None,:]<=torch.arange(5)[None,:,None]).expand(2,-1,-1)
    loss=mod.distillation_loss(x,keys,teacher,allowed=allowed)
    target=teacher.detach().sum(-2)*allowed;target=target/target.sum(-1,keepdim=True)
    expected=[]
    for b in range(2):
        for t in range(5):
            q=target[b,t,allowed[b,t]];logp=scores[b,t,allowed[b,t]].log_softmax(0)
            expected.append((q*(q.log()-logp)).sum())
    close(loss,torch.stack(expected).mean());loss.backward()
    assert x.grad is None and teacher.grad is None
    assert mod.query.weight.grad.abs().sum()>0 and mod.key.weight.grad.abs().sum()>0


def make_attention(kind,**kwargs):
    return (sa.CSA if kind=='csa' else sa.HCA)(8,2,compress_ratio=3,window_size=4,
        topk=2,index_dim=4,index_heads=2,query_chunk_size=3,key_chunk_size=2,**kwargs)


def dense_attention_oracle(mod,x,mask):
    b,t,w=x.shape;pos=torch.arange(t)
    latent,q,local=mod._project(x,pos)
    comp,cv=compressor_oracle(mod.compressor,x,mask)
    cp=torch.arange(comp.shape[1])*mod.ratio;comp=mod.rotary(comp,cp)
    index_keys=None
    if mod.indexer is not None:
        index_keys,_=compressor_oracle(mod.index_compressor,x.detach(),mask)
        index_keys=mod.indexer.rotary(index_keys,cp)
        iscores=mod.indexer.scores(x,index_keys,query_latent=latent)
    result=[];teacher_rows=[];score_rows=[]
    for batch in range(b):
        rows=[]
        for tpos in range(t):
            local_ids=[j for j in range(max(0,tpos-mod.window_size+1),tpos+1) if mask[batch,j]]
            compressed_ids=[j for j in range(comp.shape[1]) if cv[batch,j] and (j+1)*mod.ratio-1<=tpos]
            if mod.indexer is not None:
                compressed_ids=sorted(compressed_ids,key=lambda j:(-float(iscores[batch,tpos,j].detach()),j))[:mod.indexer.topk]
            entries=torch.cat((local[batch,local_ids],comp[batch,compressed_ids]),0)
            scores=q[batch,tpos]@entries.T*mod.head_dim**-.5
            if mod.sink is not None:scores=torch.cat((scores,mod.sink[:,None]),-1)
            if not mask[batch,tpos] or (entries.shape[0]==0 and mod.sink is None):
                row=q[batch,tpos]*0
            else:
                prob=scores.softmax(-1)
                if mod.sink is not None:prob=prob[:,:-1]
                row=prob@entries
                if compressed_ids and mod.indexer is not None:
                    teacher_rows.append(prob[:,len(local_ids):].sum(0).detach())
                    score_rows.append(iscores[batch,tpos,compressed_ids])
            rows.append(row)
        result.append(torch.stack(rows))
    result=mod.rotary(torch.stack(result),pos,inverse=True).reshape(b,t,mod.output_groups,-1)
    projected=torch.cat([layer(result[:,:,i]) for i,layer in enumerate(mod.output_down)],-1)
    output=mod.output_up(projected)*mask[...,None]
    losses=[]
    for teacher,scores in zip(teacher_rows,score_rows):
        teacher=teacher/teacher.sum();losses.append((teacher*(teacher.log()-scores.log_softmax(0))).sum())
    loss=torch.stack(losses).mean() if losses else output.sum()*0
    return output,loss

@pytest.mark.parametrize('kind',['csa','hca'])
@pytest.mark.parametrize('sink',[False,True])
@pytest.mark.parametrize('rope',[0,4])
def test_attention_dense_oracle_forward_grad(kind,sink,rope):
    torch.manual_seed(12)
    mod=make_attention(kind,rope_dim=rope,attention_sink=sink,output_groups=2,output_rank=3,dtype=torch.double)
    reference=copy.deepcopy(mod)
    x=torch.randn(2,10,8,dtype=torch.double,requires_grad=True);z=x.detach().clone().requires_grad_()
    mask=torch.rand(2,10)>.3;mask[0,:3]=False
    actual=mod(x,valid_mask=mask,return_aux=True)
    expected,loss=dense_attention_oracle(reference,z,mask)
    close(actual.output,expected,atol=1e-10,rtol=1e-10);close(actual.indexer_loss,loss,atol=1e-10,rtol=1e-10)
    (actual.output.square().mean()+actual.indexer_loss).backward()
    (expected.square().mean()+loss).backward()
    close(x.grad,z.grad,atol=1e-9,rtol=1e-8)
    for (name,p),(_,v) in zip(mod.named_parameters(),reference.named_parameters()):
        if p.grad is None or v.grad is None:assert p.grad is None and v.grad is None,name
        else:close(p.grad,v.grad,atol=1e-9,rtol=1e-8)

@pytest.mark.parametrize('kind',['csa','hca'])
@pytest.mark.parametrize('length',[1,2,3,4,7,12])
def test_attention_causality_future_perturbation(kind,length):
    torch.manual_seed(16);mod=make_attention(kind,rope_dim=4).eval();x=torch.randn(2,13,8)
    with torch.no_grad():
        reference=mod(x)[:,:length]
        close(mod(x[:,:length]),reference,atol=2e-6,rtol=2e-5)
        changed=x.clone();changed[:,length:]=torch.randn_like(changed[:,length:])*100
        close(mod(changed)[:,:length],reference,atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize('kind',['csa','hca'])
@pytest.mark.parametrize('chunks',[[1]*13,[2,3,1,7],[13]])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_attention_cache_exact_chunking_and_raw_memory(kind,chunks,dtype):
    torch.manual_seed(17);mod=make_attention(kind,rope_dim=4,dtype=dtype).eval()
    x=torch.randn(2,13,8,dtype=dtype);mask=torch.rand(2,13)>.2
    with torch.no_grad():
        full=mod(x,valid_mask=mask);cache=None;parts=[];seen=0
        for length in chunks:
            y,cache=mod.forward_cached(x[:,seen:seen+length],cache,valid_mask=mask[:,seen:seen+length])
            parts.append(y);seen+=length
            assert cache.seen==seen and cache.local.shape[1]<=3
            assert cache.compressed.shape[1]==seen//3 and cache.compression.tail.shape[1]<3
            assert cache.local.untyped_storage().nbytes()==cache.local.numel()*cache.local.element_size()
        close(torch.cat(parts,1),full,atol=2e-6,rtol=2e-5)
        assert cache.tensor_bytes>0

@pytest.mark.parametrize('kind',['csa','hca'])
def test_cache_reorder_and_invalidations(kind):
    torch.manual_seed(18);mod=make_attention(kind,rope_dim=4).eval();x=torch.randn(2,9,8)
    with torch.no_grad():
        _,cache=mod.forward_cached(x[:,:5]);new=cache.reorder(torch.tensor([1,0,1]))
        y,end=mod.forward_cached(x[[1,0,1],5:],new)
        close(y,mod(x[[1,0,1]])[:,5:],atol=2e-6,rtol=2e-5)
        with pytest.raises(ValueError):copy.deepcopy(mod).forward_cached(x[:,5:],cache)
        with pytest.raises(ValueError):mod.forward_cached(x[:1,5:],cache)
        with pytest.raises(ValueError):mod.forward_cached(x[:,5:].double(),cache)
        mod.output_up.weight.add_(1)
        with pytest.raises(ValueError):mod.forward_cached(x[:,5:],cache)
    with pytest.raises(RuntimeError):mod.forward_cached(x)
    mod.train()
    with torch.no_grad(),pytest.raises(RuntimeError):mod.forward_cached(x)

@pytest.mark.parametrize('kind',['csa','hca'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
def test_attention_allpad_low_precision_and_gradients(kind,dtype):
    mod=make_attention(kind,rope_dim=4,dtype=dtype)
    x=torch.randn(2,7,8,dtype=dtype,requires_grad=True);mask=torch.zeros(2,7,dtype=torch.bool)
    result=mod(x,valid_mask=mask,return_aux=True)
    assert result.output.dtype==dtype and not result.output.any() and result.indexer_loss==0
    (result.output.square().sum()+result.indexer_loss).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mod.parameters())

@pytest.mark.parametrize('kind',['csa','hca'])
def test_attention_autocast(kind):
    mod=make_attention(kind,rope_dim=4);x=torch.randn(2,9,8,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):result=mod(x,return_aux=True)
    (result.output.float().square().mean()+result.indexer_loss).backward()
    assert torch.isfinite(x.grad).all()

def test_window_one_cache_keeps_no_local_and_single_selection_loss_zero():
    mod=sa.CSA(4,1,compress_ratio=2,window_size=1,topk=1,index_dim=2,query_chunk_size=2).eval()
    x=torch.randn(1,7,4)
    assert mod(x,return_aux=True).indexer_loss.abs()<1e-6
    with torch.no_grad():
        y,cache=mod.forward_cached(x[:,:5]);assert cache.local.shape[1]==0
        z,cache=mod.forward_cached(x[:,5:],cache);close(z,mod(x)[:,5:],atol=1e-6,rtol=1e-5)


def test_csa_dense_warmup_matches_all_key_selection():
    torch.manual_seed(44)
    mod=make_attention('csa',rope_dim=4,dtype=torch.double)
    reference=copy.deepcopy(mod);reference.indexer.topk=100
    x=torch.randn(2,13,8,dtype=torch.double)
    a=mod(x,return_aux=True,indexer_warmup=True);b=reference(x,return_aux=True)
    close(a.output,b.output,atol=1e-12,rtol=1e-11);close(a.indexer_loss,b.indexer_loss,atol=1e-12,rtol=1e-11)
    a.indexer_loss.backward();b.indexer_loss.backward()
    for (name,p),(_,q) in zip(mod.named_parameters(),reference.named_parameters()):
        if p.grad is None or q.grad is None:assert p.grad is None and q.grad is None,name
        else:close(p.grad,q.grad,atol=1e-12,rtol=1e-11)


def test_hca_has_no_synthetic_auxiliary_gradient():
    result=make_attention('hca')(torch.randn(2,7,8,requires_grad=True),return_aux=True)
    assert result.indexer_loss==0 and not result.indexer_loss.requires_grad


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_rotary_frequencies_survive_module_dtype_move(dtype):
    mod=sa.RotaryEmbedding(8,base=12345.)
    before=mod.frequencies.clone();mod.to(dtype=dtype)
    assert mod.frequencies.dtype==torch.float32
    close(mod.frequencies,before,atol=0,rtol=0)
