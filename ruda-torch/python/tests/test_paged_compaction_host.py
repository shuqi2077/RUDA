"""CPU proofs of page ownership and an independent gradient reference.

These tests do NOT compile or execute Rust, DSL, PTX, or a GPU. The inherited
analytic/dense references verify which pages can receive a contribution. The
address simulator verifies a complete single-writer partition, including NaN
sentinels and missing outputs. It is not imported by any production path.
"""
from pathlib import Path
from itertools import product
import random
import re
import numpy as np
import pytest
import torch
from paged_selected_reference import dense, analytical

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / 'ruDNN/src/paged_attention'


def partition(spec, max_words=None):
    p, size = spec['num_pages'], spec['page_size']
    if p < 1 or size < 1 or (max_words is not None and p > max_words):
        raise ValueError('page/budget bounds')
    used = set()
    for seq in set(spec['sequence_ids']):
        length = spec['kv_lengths'][seq]
        used.update(spec['block_tables'][seq][:(length + size - 1) // size])
    active = sorted(used)
    inactive = sorted(set(range(p)) - used)
    return active + inactive, len(active)


def direct_reachability(spec):
    """Exhaustive TOKEN addressing, independent of partition's ceil(page) rule."""
    return {spec['block_tables'][seq][t // spec['page_size']]
            for seq in spec['sequence_ids'] for t in range(spec['kv_lengths'][seq])}


def schedule_writes(spec, heads, dims, needs, expected=None):
    """Simulate only address ownership, not the numerical gradient kernel."""
    page_ids, active = partition(spec)
    page_size = spec['page_size']
    shape = (spec['num_pages'], page_size, heads)
    values = [np.full((*shape, d), np.nan) if need else None for d, need in zip(dims, needs)]
    counts = [np.zeros((*shape, d), dtype=np.int8) if need else None for d, need in zip(dims, needs)]
    width = max((d for d, need in zip(dims, needs) if need), default=0)
    if not width:
        return values, counts
    # The zero kernel iterates a compacted flat max-width layout. Requested
    # gradients may have different feature widths and independent storage.
    for i in range((spec['num_pages'] - active) * page_size * heads * width):
        col = i % width
        row = i // width
        rows_per_page = page_size * heads
        physical = page_ids[active + row // rows_per_page]
        offset, head = divmod(row % rows_per_page, heads)
        for dst, written, dim in zip(values, counts, dims):
            if dst is not None and col < dim:
                dst[physical, offset, head, col] = 0.
                written[physical, offset, head, col] += 1
    # Existing ordered history kernel: a unique physical (slot, head) owns each
    # requested element. Its numerical computation/order are left unchanged.
    for compressed_slot in range(active * page_size):
        physical = page_ids[compressed_slot // page_size]
        offset = compressed_slot % page_size
        for head in range(heads):
            for j, (dst, written) in enumerate(zip(values, counts)):
                if dst is not None:
                    dst[physical, offset, head] = (expected[j][physical, offset, head] if expected is not None else 1.)
                    written[physical, offset, head] += 1
    return values, counts


SPEC = dict(page_size=3, num_pages=8, block_tables=[[6, 2, 7], [1, 6], [4], []],
            kv_lengths=[5, 6, 2, 0], sequence_ids=[0, 1, 0, 3], positions=[1, 5, 4, 0])


def test_shared_and_unqueried_and_reserved_pages():
    pages, active = partition(SPEC)
    assert pages == [1, 2, 6, 0, 3, 4, 5, 7]
    assert active == 3
    assert set(pages[:active]) == direct_reachability(SPEC)


@pytest.mark.parametrize('mode', ['empty_query', 'empty_history', 'full', 'causal_future', 'only_one_request'])
def test_edge_partition(mode):
    spec = dict(SPEC)
    if mode == 'empty_query':
        spec.update(sequence_ids=[], positions=[])
    elif mode == 'empty_history':
        spec.update(kv_lengths=[0, 0, 0, 0], block_tables=[[], [], [], []])
    elif mode == 'full':
        spec.update(num_pages=2, block_tables=[[1, 0]], kv_lengths=[6], sequence_ids=[0], positions=[0])
    elif mode == 'causal_future':
        spec.update(sequence_ids=[0], positions=[0])
    else:
        spec.update(sequence_ids=[2], positions=[0])
    p, a = partition(spec)
    assert sorted(p) == list(range(spec['num_pages']))
    assert set(p[:a]) == direct_reachability(spec)
    if mode.startswith('empty'): assert a == 0
    if mode == 'full': assert a == 2
    if mode == 'causal_future': assert a == 2, 'future page must remain legal when plan is reused noncausally'


@pytest.mark.parametrize('needs', list(product([False, True], repeat=3)))
@pytest.mark.parametrize('dims', [(1, 7, 3), (5, 1, 7), (33, 65, 9)])
def test_no_missing_duplicate_or_frozen_output_writes(needs, dims):
    values, counts = schedule_writes(SPEC, 2, dims, needs)
    for need, dst, count in zip(needs, values, counts):
        if not need:
            assert dst is None and count is None
        else:
            assert np.all(count == 1)
            assert np.all(np.isfinite(dst))
            p, n = partition(SPEC)
            assert np.all(dst[p[n:]] == 0)


MASKS = [(mla, bits) for mla, n in [(False, 3), (True, 4)]
         for bits in product([False, True], repeat=n) if any(bits)]


@pytest.mark.parametrize('mla,needs', MASKS)
@pytest.mark.parametrize('causal', [True, False])
def test_gradient_support_matches_dense_autograd(mla, needs, causal):
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(36)
    def r(*shape): return torch.randn(shape, dtype=torch.float64, generator=gen) * .2
    q = r(4, 4, 5); k = r(8, 3, 1 if mla else 2, 5)
    ts = (q, r(4, 4, 3), k, r(8, 3, 1, 3)) if mla else (q, k, r(8, 3, 2, 7))
    ref = [x.clone().requires_grad_(True) for x in ts]
    y = dense(ref, mla, spec=SPEC, causal=causal)
    g = r(*y.shape); y.backward(g)
    old = analytical(ts, g, needs, mla, spec=SPEC, causal=causal)
    for need, x, t in zip(needs, old, ref):
        if need: torch.testing.assert_close(x, t.grad, atol=1e-11, rtol=1e-10)
        else: assert x is None
    hgrads = (old[2], None, old[3]) if mla else (old[1], old[2], None)
    hneeds = (needs[2], False, needs[3]) if mla else (needs[1], needs[2], False)
    dims = (5, 5, 3) if mla else (5, 7, 0)
    expected = [None if t is None else t.numpy() for t in hgrads]
    new, writes = schedule_writes(SPEC, k.shape[2], dims, hneeds, expected)
    for dst, count, orig in zip(new, writes, expected):
        if orig is not None:
            np.testing.assert_array_equal(dst, orig)
            assert np.all(count == 1)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('mla', [False, True])
def test_zero_only_storage_cast_and_tail(dtype, mla):
    dims = (5, 5, 3) if mla else (5, 7, 0)
    needs = (True, False, True) if mla else (True, True, False)
    vals, _ = schedule_writes(SPEC, 1 if mla else 2, dims, needs)
    for x in vals:
        if x is not None:
            cast = torch.from_numpy(x).to(dtype)
            assert torch.isfinite(cast).all()
            ids, n = partition(SPEC)
            assert torch.count_nonzero(cast[ids[n:]]) == 0


def test_1000_random_partition_inputs():
    rand = random.Random(3636)
    for _ in range(1000):
        pages, page_size, seqs = rand.randint(1, 64), rand.randint(1, 9), rand.randint(1, 8)
        tables = [rand.sample(range(pages), rand.randint(0, pages)) for _ in range(seqs)]
        lengths = [rand.randint(0, len(t) * page_size) for t in tables]
        ids = [rand.randrange(seqs) for _ in range(rand.randint(0, 20))]
        spec = dict(num_pages=pages, page_size=page_size, block_tables=tables, kv_lengths=lengths, sequence_ids=ids)
        result, active = partition(spec)
        assert sorted(result) == list(range(pages))
        assert len(set(result)) == pages
        assert set(result[:active]) == direct_reachability(spec)
        assert result[:active] == sorted(result[:active])


def test_budget_includes_retained_page_map():
    assert len(partition(SPEC, 8)[0]) * 4 == 32
    with pytest.raises(ValueError): partition(SPEC, 7)
    code = (MODULE / 'ordered_workspace.rs').read_text()
    assert '(WORKSPACE_LIMIT_BYTES-self.bytes)/4' in code
    assert 'self.bytes+=partition.pages.len()*4;' in code
    assert 'if enabled && self.page_partition.is_none()' in code


def test_actual_kernel_contract_and_unchanged_accumulation():
    code = (MODULE / 'ordered_kernel.rs').read_text()
    assert 'if comptime!(compact_history)' in code
    assert 'history_pages[slot/page_size as usize]' in code
    assert 'slot%page_size as usize' in code
    zero = (MODULE / 'history_compaction_kernel.rs').read_text()
    for name in ['q:', 'v:', 'grad:', 'statistics:', 'fetch_add', 'plane_sum']:
        # dv: is an output, not a v: input; match whole identifiers.
        if name.endswith(':'): assert not re.search(r'\b' + re.escape(name), zero)
        else: assert name not in zero
    assert 'if i < elements as usize' in zero
    assert 'if d < key_dim' in zero and 'if d < value_dim' in zero and 'if d < position_dim' in zero
    mod = (MODULE / 'mod.rs').read_text()
    assert 'if launch_history_slots!=0' in mod
    assert 'if inactive_zero_elements!=0' in mod
    assert 'workspace.compact_history && workspace.active_pages<self.host.pages' in mod


def test_constructor_validates_budget_before_upload():
    text = (MODULE / 'ordered_workspace.rs').read_text().split('pub fn set_query_pruning')[0]
    assert text.index('history_compaction::build') < text.index('client.create_from_slice')
    assert text.index('history_compaction::from_env') < text.index('client.create_from_slice')


def test_inherited_invalid_rust_float_fixed():
    text = (MODULE / 'tests_history_row_cache.rs').read_text()
    assert '\n                .37,' not in text
    assert text.count('\n                0.37,') == 2
    new = (MODULE / 'tests_history_compaction.rs').read_text()
    assert not re.search(r'^\s+\.\d+,', new, re.M)


def test_source_schedule_is_not_python_cpu_fallback():
    text = (MODULE / 'mod.rs').read_text()
    assert 'history_compaction_kernel::zero_inactive::launch::<R>' in text
    assert 'ordered_kernel::history_backward::launch::<R>' in text
    cfg = (MODULE / 'history_compaction.rs').read_text()
    assert 'None | Some("0") => Ok(false)' in cfg
    # No tensor values are copied to the host to classify pages.
    ws = (MODULE / 'ordered_workspace.rs').read_text()
    assert 'into_data_sync' not in ws and 'readback' not in ws
