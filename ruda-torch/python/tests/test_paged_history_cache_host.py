"""CPU references and source contracts, NOT production Rust/PTX execution."""
from pathlib import Path
import itertools
import pytest
import torch
from paged_selected_reference import data, dense, SPEC
from paged_history_cache_reference import cached_vjp
from paged_pruning_reference import pruned_vjp

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "ruDNN/src/paged_attention"
MASKS = [(mla, bits) for mla, n in [(False, 3), (True, 4)]
         for bits in itertools.product([False, True], repeat=n) if any(bits)]

@pytest.mark.parametrize('mla,needs', MASKS)
@pytest.mark.parametrize('causal', [False, True])
def test_cached_reference_matches_dense_autograd(mla, needs, causal):
    ts = data(mla)
    refs = [t.clone().requires_grad_(True) for t in ts]
    y = dense(refs, mla, spec=SPEC, causal=causal)
    g = torch.linspace(-.7, .4, y.numel(), dtype=y.dtype).reshape_as(y)
    y.backward(g)
    old, a = cached_vjp(ts, g, needs, mla=mla, spec=SPEC, causal=causal, cached=False)
    new, b = cached_vjp(ts, g, needs, mla=mla, spec=SPEC, causal=causal, cached=True)
    for need, x, z, ref in zip(needs, old, new, refs):
        if need:
            assert torch.equal(x, z), "CPU operation order changed"
            torch.testing.assert_close(z, ref.grad, atol=2e-11, rtol=2e-10)
        else:
            assert x is None and z is None
    assert b.total() <= a.total()
    history = (needs[2] or needs[3]) if mla else (needs[1] or needs[2])
    if not history:
        assert b.total() == 0
    if mla:
        assert b['value_elements'] == 0, "latent must not be loaded twice"
    elif not needs[1]:
        assert b['value_elements'] == 0, "dV needs scores but not V contents"

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mla', [False, True])
def test_precision_and_repeat_use_no_stale_persistent_rows(dtype, mla):
    ts = data(mla, dtype=dtype)
    g = torch.ones((4, 4, 5 if mla else 7), dtype=dtype) * .25
    need = [True] * len(ts)
    old, _ = cached_vjp(ts, g, need, mla=mla, spec=SPEC, cached=False)
    new, _ = cached_vjp(ts, g, need, mla=mla, spec=SPEC, cached=True)
    assert all(torch.equal(x, y) for x, y in zip(old, new))
    changed = [t + .0625 for t in ts]
    a, _ = cached_vjp(changed, g, need, mla=mla, spec=SPEC, cached=False)
    b, _ = cached_vjp(changed, g, need, mla=mla, spec=SPEC, cached=True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert any(not torch.equal(x, y) for x, y in zip(new, b))

@pytest.mark.parametrize('mla', [False, True])
@pytest.mark.parametrize('causal', [False, True])
def test_shared_page_at_different_logical_positions_and_out_of_order_queries(mla, causal):
    spec=dict(page_size=3,num_pages=4,block_tables=[[0,1],[2,0],[]],kv_lengths=[5,6,0],
              sequence_ids=[0,1,0,1,0,2],positions=[4,5,0,3,4,0])
    ts = data(mla, queries=6)
    g = torch.randn((6,4,5 if mla else 7), generator=torch.Generator().manual_seed(35),dtype=torch.float64)
    needs = [True] * len(ts)
    a = pruned_vjp(ts,g,needs,mla=mla,spec=spec,causal=causal)
    b, cnt = cached_vjp(ts,g,needs,mla=mla,spec=spec,causal=causal)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert cnt['key_rows'] <= spec['num_pages'] * spec['page_size'] * (1 if mla else 2)

@pytest.mark.parametrize('mla', [False, True])
def test_unused_page_and_tail_nan_are_never_preloaded(mla):
    # Only physical page 0 offset 0 is live. All other input history is poison.
    spec=dict(page_size=3,num_pages=4,block_tables=[[0],[]],kv_lengths=[1,0],
              sequence_ids=[0,1,0,1],positions=[0,0,0,0])
    ts=list(data(mla))
    for i in ([2,3] if mla else [1,2]):
        t=ts[i]
        live=t[0,0].clone()
        t.fill_(float('nan'));t[0,0].copy_(live)
    g=torch.ones((4,4,5 if mla else 7),dtype=torch.float64)
    out,cnt=cached_vjp(ts,g,[True]*len(ts),mla=mla,spec=spec)
    assert all(torch.isfinite(x).all() for x in out)
    assert cnt['key_rows']==(1 if mla else 2)
    for x in (out[2],out[3]) if mla else (out[1],out[2]):
        assert torch.count_nonzero(x.flatten(start_dim=2)[1:])==0
        assert torch.count_nonzero(x[0,1:])==0

@pytest.mark.parametrize('mla', [False, True])
def test_empty_query_no_history_load(mla):
    spec=dict(SPEC,sequence_ids=[],positions=[])
    ts=data(mla,queries=0);g=torch.empty((0,4,5 if mla else 7),dtype=torch.float64)
    out,cnt=cached_vjp(ts,g,[True]*len(ts),mla=mla,spec=spec)
    assert cnt.total()==0
    assert all(torch.count_nonzero(x)==0 for x in out)

@pytest.mark.parametrize('n', [1,2,17,64])
def test_exact_history_load_count_formula(n):
    # One KV head shared by 4 query heads, all n rows see the same one token.
    ts=(torch.full((n,4,5),.2,dtype=torch.float64),
        torch.full((1,1,1,5),.3,dtype=torch.float64),
        torch.full((1,1,1,7),.4,dtype=torch.float64))
    spec=dict(page_size=1,num_pages=1,block_tables=[[0]],kv_lengths=[1],
              sequence_ids=[0]*n,positions=[0]*n)
    g=torch.ones((n,4,7),dtype=torch.float64)
    _,old=cached_vjp(ts,g,[True]*3,mla=False,spec=spec,cached=False)
    _,new=cached_vjp(ts,g,[True]*3,mla=False,spec=spec,cached=True)
    assert old['key_elements']==n*4*5 and old['value_elements']==n*4*7
    assert new['key_elements']==5 and new['value_elements']==7

def test_production_cache_is_local_lazy_and_preserves_fma_operands():
    code=(MODULE/'ordered_kernel.rs').read_text()
    guard=code.index('if denominator!=0.0')
    load=code.index('if !row_loaded')
    assert guard<load<code.index('row_loaded=true')
    assert 'let mut row_loaded=false;' in code
    assert 'dot=fma(f32::cast_from(q[qrow+d]),key,dot);' in code
    assert 'partial=fma(f32::cast_from(grad[grow+d]),value,partial);' in code
    assert 'if comptime!(mla) { value=cached_key[j]; }' in code
    assert 'if comptime!(!mla && (need_dk || need_dkp))' in code

def test_numerical_accumulation_body_identical_to_v34():
    code=(MODULE/'ordered_kernel.rs').read_text()
    # Exact body retained from the verified baseline, excluding load selection.
    body=code[code.index('                                    if comptime!(need_dk) {'):
              code.index('                                }\n                                head+=1;')]
    import hashlib
    assert hashlib.sha256(body.encode()).hexdigest()=='2a2190c261243467561bdac9cd06690cb6fa58dcb38ed9239a929f3f2e8b87f3'

def test_no_new_device_allocation_or_new_abi():
    workspace=(MODULE/'ordered_workspace.rs').read_text()
    assert 'set_history_row_cache' in workspace
    config=(MODULE/'history_row_cache.rs').read_text()
    assert 'None | Some("0") => Ok(false)' in config
    assert workspace.index('history_row_cache::from_env()')<workspace.index('client.create_from_slice')
    setter=workspace.split('pub fn set_history_row_cache')[1].split('\n')[0]
    assert 'create' not in setter and 'sync' not in setter
    mod=(MODULE/'mod.rs').read_text()
    assert 'workspace.history_row_cache,compact_history,cache_key_slots,cache_value_slots,cache_position_slots' in mod
