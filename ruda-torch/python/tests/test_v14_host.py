"""HOST ONLY: real Python plan checks + independent math/protocol references.
These tests do not import/compile the Rust kernels and are not GPU acceptance.
"""
import importlib.util
import itertools
from pathlib import Path
import struct
import sys
import types
import numpy as np
import pytest
import torch

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]/'ruda_torch'

class FakeNative:
    """Explicit protocol double; never executes an operator."""
    def __init__(self): self.current=0;self.next=1;self.events={};self.calls=[];self.plans=[]
    def stream_command(self,op,stream=0,object=0,flags=0):
        self.calls.append((op,stream,object,flags))
        if op==0:return self.current
        if op==1:r=self.next;self.next+=1;return r
        if op==2:r=self.current;self.current=stream;return r
        if op==5:
            object=object or len(self.events)+1;self.events[object]=flags;return object
        if op in (3,7):return 1
        if op==9:self.events.pop(object);return 0
        if op==10:return struct.unpack('<I',struct.pack('<f',2.5))[0]
        return 0
    def NativePagedPlan(self,*args):
        self.plans.append(args)
        return types.SimpleNamespace(run=lambda *args:args)
    def record_stream(self,tensor,id):self.calls.append(('record',tensor,id))

def modules():
    native=FakeNative();pkg=types.ModuleType('_v14_protocol');pkg.__path__=[str(ROOT)];pkg._C=native
    sys.modules[pkg.__name__]=pkg
    loaded=[]
    for name in ('_paged','_streams'):
        spec=importlib.util.spec_from_file_location(pkg.__name__+'.'+name,ROOT/(name+'.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);loaded.append(m)
    return native,*loaded

def schedule():
    return dict(page_size=4,num_pages=6,block_tables=[[3,0],[5],[]],kv_lengths=[7,2,0],
                sequence_ids=[0,1,0,2],positions=[5,1,6,0])

@pytest.mark.parametrize('change',[
 {'page_size':0},{'num_pages':0},{'page_size':True},{'num_pages':2**32},
 {'kv_lengths':[7]},{'positions':[1]},{'sequence_ids':[0,3,0,2]},
 {'positions':[7,1,6,0]},{'block_tables':[[3],[5],[]]},
 {'block_tables':[[3,3],[5],[]]},{'block_tables':[[3,0],[6],[]]},
 {'sequence_ids':[0,-1,0,2]},{'page_size':4.5},
])
def test_plan_rejects(change):
    _,p,_=modules();data=schedule();data.update(change)
    with pytest.raises((ValueError,TypeError)):p.PagedAttentionPlan(**data)

def test_plan_real_python_metadata():
    _,p,_=modules();plan=p.PagedAttentionPlan(**schedule())
    assert plan._spec==(4,6,3,4,2,1)
    assert plan._words[:11]==(0,1,0,2,5,1,6,0,7,2,0)
    assert plan._words[11:]==(3,0,5,2**32-1,2**32-1,2**32-1)

def test_metadata_reused_per_stream_not_per_call():
    n,p,s=modules();plan=p.PagedAttentionPlan(**schedule());q=types.SimpleNamespace(device=types.SimpleNamespace(type='ruda',index=0))
    plan._native(q);plan._native(q);assert len(n.plans)==1
    with s.stream(s.Stream()):plan._native(q);plan._native(q)
    assert len(n.plans)==2

@pytest.mark.parametrize('device',['cpu','cuda','meta'])
def test_no_implicit_device_fallback(device):
    n,p,_=modules();plan=p.PagedAttentionPlan(**schedule())
    with pytest.raises(ValueError):plan._native(types.SimpleNamespace(device=types.SimpleNamespace(type=device,index=0)))
    assert not n.plans

def test_stream_context_restored_on_exception():
    n,_,s=modules();one=s.Stream();two=s.Stream()
    with s.stream(one):
        with pytest.raises(RuntimeError):
            with s.stream(two):raise RuntimeError('test')
        assert s.current_stream().stream_id==one.stream_id
    assert n.current==0

def test_event_wait_and_query_do_not_host_synchronize():
    n,_,s=modules();a=s.Stream();b=s.Stream();e=s.Event().record(a);n.calls.clear()
    b.wait_event(e);e.query();b.query()
    assert all(op not in (4,8,11) for op,*_ in n.calls)
    e.close()

def test_event_timing_and_destruction():
    n,_,s=modules();a=s.Event(enable_timing=True).record();b=s.Event(enable_timing=True).record()
    assert a.elapsed_time(b)==2.5
    a.close();a.close();b.close();assert not n.events

def test_non_timing_event_error():
    _,_,s=modules()
    with pytest.raises(ValueError):s.Event().elapsed_time(s.Event())

# Independent online algorithm in NumPy, compared with dense Torch attention.
def online(q,k,v,tables,lengths,ids,positions,scale,causal,qp=None,kp=None):
    q,k,v=map(lambda x:x.float().numpy(),(q,k,v))
    if qp is not None:qp,kp=qp.float().numpy(),kp.float().numpy()
    result=np.zeros((len(ids),q.shape[1],v.shape[-1]),dtype=np.float32)
    for row,seq in enumerate(ids):
        end=min(lengths[seq],positions[row]+1) if causal else lengths[seq]
        for h in range(q.shape[1]):
            kh=h//(q.shape[1]//k.shape[2]);den=np.float32(0);mx=np.float32(0);acc=result[row,h]
            for t in range(end):
                page=tables[seq][t//k.shape[1]];off=t%k.shape[1]
                score=np.dot(q[row,h],k[page,off,kh])
                if qp is not None:score+=np.dot(qp[row,h],kp[page,off,kh])
                score=np.float32(score*scale);nextmax=max(mx,score) if t else score
                alpha=np.float32(np.exp(mx-nextmax)) if t else np.float32(0)
                beta=np.float32(np.exp(score-nextmax));den=np.float32(den*alpha+beta)
                acc[:]=acc*alpha+v[page,off,kh]*beta;mx=nextmax
            if end:acc[:]/=den
    return torch.from_numpy(result)

def dense(q,k,v,tables,lengths,ids,positions,scale,causal,qp=None,kp=None):
    result=torch.zeros((len(ids),q.shape[1],v.shape[-1]))
    for row,seq in enumerate(ids):
        end=min(lengths[seq],positions[row]+1) if causal else lengths[seq]
        if not end:continue
        kk=torch.stack([k[tables[seq][t//k.shape[1]],t%k.shape[1]] for t in range(end)]).float()
        vv=torch.stack([v[tables[seq][t//v.shape[1]],t%v.shape[1]] for t in range(end)]).float()
        kk=kk.repeat_interleave(q.shape[1]//k.shape[2],dim=1);vv=vv.repeat_interleave(q.shape[1]//k.shape[2],dim=1)
        score=torch.einsum('hd,thd->ht',q[row].float(),kk)
        if qp is not None:
            pp=torch.stack([kp[tables[seq][t//kp.shape[1]],t%kp.shape[1]] for t in range(end)]).float()
            score+=torch.einsum('hd,td->ht',qp[row].float(),pp[:,0])
        result[row]=torch.einsum('ht,thd->hd',torch.softmax(score*scale,dim=-1),vv)
    return result

def inputs(dtype,d=33,dv=17,kh=2,h=4):
    g=torch.Generator().manual_seed(2031+d+dv+kh)
    def rand(*shape):return torch.randn(shape,generator=g).to(dtype)
    return rand(4,h,d),rand(6,4,kh,d),rand(6,4,kh,dv)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('dims',[(8,8),(33,17),(65,127),(128,64)])
@pytest.mark.parametrize('heads',[(4,4),(4,2),(4,1)])
@pytest.mark.parametrize('causal',[False,True])
def test_paged_online_reference(dtype,dims,heads,causal):
    d,dv=dims;h,kh=heads;q,k,v=inputs(dtype,d,dv,kh,h);s=schedule()
    args=[s[x] for x in ('block_tables','kv_lengths','sequence_ids','positions')]
    a=online(q,k,v,*args,d**-.5,causal);b=dense(q,k,v,*args,d**-.5,causal)
    torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
    assert torch.count_nonzero(a[-1])==0

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('rank,posdim',[(33,7),(128,64),(512,64)])
@pytest.mark.parametrize('causal',[False,True])
def test_absorbed_mla_paged_reference(dtype,rank,posdim,causal):
    q,k,_=inputs(dtype,rank,rank,1,4);g=torch.Generator().manual_seed(7)
    qp=torch.randn((4,4,posdim),generator=g).to(dtype);kp=torch.randn((6,4,1,posdim),generator=g).to(dtype)
    args=[schedule()[x] for x in ('block_tables','kv_lengths','sequence_ids','positions')]
    # Explicit model scale, intentionally NOT rank**-.5.
    a=online(q,k,k,*args,192**-.5,causal,qp,kp);b=dense(q,k,k,*args,192**-.5,causal,qp,kp)
    torch.testing.assert_close(a,b,rtol=8e-5,atol=5e-6)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_mla_absorption_algebra(dtype):
    g=torch.Generator().manual_seed(123)
    q=torch.randn(2,4,8,generator=g).to(dtype).float();wk=torch.randn(4,8,16,generator=g).to(dtype).float()
    c=torch.randn(7,16,generator=g).to(dtype).float();wv=torch.randn(4,6,16,generator=g).to(dtype).float()
    expanded_k=torch.einsum('tc,hdc->thd',c,wk);absorbed=torch.einsum('qhd,hdc->qhc',q,wk)
    scores1=torch.einsum('qhd,thd->qht',q,expanded_k);scores2=torch.einsum('qhc,tc->qht',absorbed,c)
    torch.testing.assert_close(scores1,scores2,rtol=1e-4,atol=1e-5)
    prob=torch.softmax(scores1,dim=-1)
    y1=torch.einsum('qht,thd->qhd',prob,torch.einsum('tc,hdc->thd',c,wv))
    y2=torch.einsum('qhc,hdc->qhd',torch.einsum('qht,tc->qhc',prob,c),wv)
    torch.testing.assert_close(y1,y2,rtol=1e-4,atol=1e-5)

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('m,n,k',[(1,1,1),(3,17,33),(15,16,31),(17,65,64),(34,33,17)])
def test_expert_16x16_tile_algebra(dtype,m,n,k):
    g=torch.Generator().manual_seed(m+n+k);x=torch.randn((m,k),generator=g).to(dtype).float()
    w=torch.randn((4,n,k),generator=g).to(dtype).float();offsets=[0,0,1,m,m]
    actual=torch.zeros((m,n));expected=torch.zeros_like(actual)
    for e in range(4):
        lo,hi=offsets[e:e+2];expected[lo:hi]=x[lo:hi]@w[e].T
        for r in range(lo,hi,16):
            for c in range(0,n,16):
                acc=torch.zeros((16,16))
                for kk in range(0,k,16):
                    a=torch.zeros((16,16));b=torch.zeros((16,16))
                    a[:min(16,hi-r),:min(16,k-kk)]=x[r:min(r+16,hi),kk:kk+16]
                    b[:min(16,k-kk),:min(16,n-c)]=w[e,c:c+16,kk:kk+16].T
                    acc+=a@b
                actual[r:min(r+16,hi),c:c+16]=acc[:min(16,hi-r),:min(16,n-c)]
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)

@pytest.mark.parametrize('experts,groups,selected,topk',[(8,2,1,2),(64,8,3,6),(384,1,1,8)])
@pytest.mark.parametrize('top_two',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_group_routing_original_weights(experts,groups,selected,topk,top_two,bias):
    g=torch.Generator().manual_seed(12);logits=torch.randn((3,experts),generator=g);p=logits.sigmoid()
    b=torch.randn((experts,),generator=g)*0.4 if bias else torch.zeros(experts)
    corrected=p+b;per=experts//groups
    group=corrected.reshape(3,groups,per).topk(2 if top_two else 1,dim=-1).values.sum(-1)
    allowed=group.topk(selected,dim=-1).indices;mask=torch.zeros((3,groups),dtype=torch.bool).scatter(1,allowed,True)
    masked=corrected.masked_fill(~mask.repeat_interleave(per,dim=-1),-torch.inf)
    chosen=masked.topk(topk,dim=-1).indices
    ref=p.gather(1,chosen);ref=ref/ref.sum(-1,keepdim=True)*2.827
    for row in range(3):
        gs=[]
        for group_id in range(groups):
            values=sorted(corrected[row,group_id*per:(group_id+1)*per].tolist(),reverse=True)
            gs.append(sum(values[:2 if top_two else 1]))
        selected_set=set(sorted(range(groups),key=lambda i:(-gs[i],i))[:selected])
        ids=sorted((i for i in range(experts) if i//per in selected_set),key=lambda i:(-float(corrected[row,i]),i))[:topk]
        weight=p[row,ids];weight=weight/weight.sum()*2.827
        assert ids==chosen[row].tolist();torch.testing.assert_close(weight,ref[row])


def test_v13_async_not_silently_defaulted():
    native=ROOT.parents[1]/'src'
    text=(native/'lib.rs').read_text()
    assert 'Ok("1") | Ok("true") | Ok("yes")' in text
    assert 'finish_dispatch(&client());' in (native/'matmul.rs').read_text()
    assert 'sync(&client());' not in (native/'matmul.rs').read_text()


def test_async_allocation_does_not_force_sync_when_opted_in():
    text=(ROOT.parents[1]/'src/lib.rs').read_text()
    section=text.split('fn ruda_torch_alloc(',1)[1].split('fn ruda_torch_free(',1)[0]
    assert 'if !async_dispatch_enabled() { sync(&client); }' in section
    assert 'client.get_resource(handle.clone())' in section
