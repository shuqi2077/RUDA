"""Integer scheduling and independent CPU math; NOT Rust/GPU execution."""
from collections import Counter
from pathlib import Path
import random
import torch
import pytest
from paged_pruning_reference import encode,visible_rows,make_spec,tensors,pruned_vjp,BLOCK_ROWS
from paged_ordered_reference import ordered_vjp
from paged_selected_reference import dense

@pytest.mark.parametrize('mode',['monotonic','duplicate','unsorted','blocked','empty'])
@pytest.mark.parametrize('causal',[True,False])
def test_visibility_keeps_original_row_order(mode,causal):
    spec=make_spec(mode,193);index=encode(spec)
    for s in range(len(spec['kv_lengths'])):
        for token in [0,1,15,16,17,31,32,63,64,100,108,109,127,0xffffffff]:
            expected=[r for r,x in enumerate(spec['sequence_ids']) if x==s and (not causal or spec['positions'][r]>=token)]
            assert visible_rows(spec,index,s,token,causal)==expected
            assert visible_rows(spec,index,s,token,causal,False)==expected

@pytest.mark.parametrize('seed',range(12))
def test_randomized_schedule_no_missing_or_reordered_contribution(seed):
    rng=random.Random(seed)
    for _ in range(50):
        spec=make_spec('empty');n=rng.randrange(260)
        spec['sequence_ids']=[rng.randrange(4) for _ in range(n)]
        spec['positions']=[rng.randrange(max(1,spec['kv_lengths'][s])) for s in spec['sequence_ids']]
        index=encode(spec)
        for s in range(4):
            for token in [0,1,31,32,63,108,0xffffffff]:
                for causal in [False,True]:
                    expected=[r for r,x in enumerate(spec['sequence_ids']) if x==s and (not causal or spec['positions'][r]>=token)]
                    assert visible_rows(spec,index,s,token,causal)==expected

@pytest.mark.parametrize('mode',['monotonic','unsorted','blocked','empty'])
def test_budget_and_empty_sequence_links(mode):
    spec=make_spec(mode);index=encode(spec)
    assert encode(spec,len(index))==index
    with pytest.raises(ValueError):encode(spec,len(index)-1)
    with pytest.raises(ValueError):encode(spec,1)
    entries=index[1]-index[0]
    assert all(index[index[0]+i]!=2 for i in range(0,entries,2))
    # Exact header/suffix bound (avoids unnoticed accidental O(Q*history) data).
    q=len(spec['positions']);s=4
    blocks=sum((spec['sequence_ids'].count(i)+BLOCK_ROWS-1)//BLOCK_ROWS for i in range(s))
    assert len(index)==spec['num_pages']+6+4*s+q+entries+blocks

@pytest.mark.parametrize('n',[1,2,31,32,33,65,129])
def test_prefill_exact_row_visit_count(n):
    spec=dict(page_size=n,num_pages=1,block_tables=[[0]],kv_lengths=[n],sequence_ids=[0]*n,positions=list(range(n)))
    index=encode(spec);old=Counter();new=Counter()
    for token in range(n):
        assert visible_rows(spec,index,0,token,True,True,new)==visible_rows(spec,index,0,token,True,False,old)
    assert old['visited_rows']==n*n
    assert new['visited_rows']==n*(n+1)//2

@pytest.mark.parametrize('mla',[False,True])
@pytest.mark.parametrize('mode',['monotonic','duplicate','unsorted','blocked','empty'])
@pytest.mark.parametrize('causal',[True,False])
def test_pruned_vjp_matches_legacy_bits_and_dense_autograd(mla,mode,causal):
    spec=make_spec(mode,35);ts=tensors(spec,mla);needs=[True]*len(ts)
    ref=[x.clone().requires_grad_() for x in ts];y=dense(ref,mla,spec=spec,causal=causal)
    g=torch.linspace(-.4,.6,y.numel(),dtype=torch.double).reshape_as(y);y.backward(g)
    a=pruned_vjp(ts,g,needs,mla=mla,spec=spec,causal=causal)
    b=ordered_vjp(ts,g,needs,mla=mla,spec=spec,causal=causal)
    for x,z,r in zip(a,b,ref):
        assert torch.equal(x,z),'CPU reference accumulation order changed'
        torch.testing.assert_close(x,r.grad,atol=2e-11,rtol=2e-10)

@pytest.mark.parametrize('mla',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_low_precision_shared_page_gradients(mla,dtype):
    spec=make_spec('blocked',67);ts=tensors(spec,mla,dtype=dtype)
    y=dense([x.double() for x in ts],mla,spec=spec);g=torch.ones_like(y)
    a=pruned_vjp(ts,g,[True]*len(ts),mla=mla,spec=spec)
    b=ordered_vjp(ts,g,[True]*len(ts),mla=mla,spec=spec)
    for x,z in zip(a,b):assert torch.equal(x,z)
    for i in ((2,3) if mla else (1,2)):
        assert torch.count_nonzero(a[i][16:])==0 # no-query and unused page


def test_pruning_disabled_and_noncausal_are_full_scans():
    spec=make_spec('blocked',129);i=encode(spec);counts=Counter()
    visible_rows(spec,i,0,100,False,True,counts)
    assert counts['visited_rows']==spec['sequence_ids'].count(0)
    assert counts['binary_probes']==counts['block_probes']==0
    counts=Counter();visible_rows(spec,i,0,100,True,True,counts)
    assert counts['skipped_blocks']>0


def test_production_wiring_and_no_new_atomic_or_fp_formula():
    p=Path(__file__).resolve().parents[3]/'ruDNN/src/paged_attention'
    host=(p/'mod.rs').read_text();kernel=(p/'ordered_kernel.rs').read_text();ws=(p/'ordered_workspace.rs').read_text()
    assert 'history_index::QUERY_BLOCK_ROWS,workspace.query_pruning' in host
    assert 'host: Arc<HostPlan>' in host and 'Arc::ptr_eq' in ws
    assert 'set_query_pruning' in ws and 'query_pruning:true' in ws
    assert 'prune_queries' in kernel and 'block_maxima_base' in kernel
    assert 'fetch_add' not in kernel and 'Atomic<' not in kernel
    assert 'No clearing pass is required' in kernel
