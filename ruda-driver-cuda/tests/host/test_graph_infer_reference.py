"""Independent Python planning reference, NOT production Rust/GPU execution.

Compares interval sweeping + edge reduction with pairwise memory hazards and
boolean closure. Actual Rust has its own 52-test suite; do not conflate these.
"""
from dataclasses import dataclass
import random
import pytest

@dataclass(frozen=True)
class Access:
    allocation: int
    start: int
    end: int
    write: bool

def overlap(a, b):
    return (a.allocation == b.allocation and a.start < a.end and b.start < b.end
            and (a.write or b.write) and a.start < b.end and b.start < a.end)

def closure(parents):
    n = len(parents)
    reach = [[False] * n for _ in range(n)]
    for j, incoming in enumerate(parents):
        for i in incoming:
            reach[i][j] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] |= reach[i][k] and reach[k][j]
    return reach

def oracle(nodes):
    p = [[] for _ in nodes]
    for j in range(len(nodes)):
        for i in range(j):
            if any(overlap(a, b) for a in nodes[i] for b in nodes[j]):
                p[j].append(i)
    return closure(p)

def sweep_model(nodes, budget=16_777_216):
    allocations = {}
    for i, accesses in enumerate(nodes):
        for a in accesses:
            if not 0 <= a.start <= a.end <= (1 << 64) - 1:
                raise ValueError('invalid view')
            if a.start < a.end:
                allocations.setdefault(a.allocation, []).append((i, a))
    hazards = [set() for _ in nodes]
    pairs = 0
    for allocation, ranges in allocations.items():
        merged = []
        for node, a in sorted(ranges, key=lambda item: (item[0], item[1].write, item[1].start, item[1].end)):
            if (merged and merged[-1][0] == node and merged[-1][1].write == a.write
                    and a.start <= merged[-1][1].end):
                last = merged[-1][1]
                merged[-1] = node, Access(allocation, last.start, max(last.end, a.end), last.write)
            else:
                merged.append((node, a))
        # Independent host model uses lists, not production BTreeSets.
        readers, writers = [], []
        for second, b in sorted(merged, key=lambda item: (item[1].start, item[1].end, item[0], item[1].write)):
            readers = [(i, a) for i, a in readers if a.end > b.start]
            writers = [(i, a) for i, a in writers if a.end > b.start]
            for first, a in writers + (readers if b.write else []):
                if first == second:
                    continue
                assert overlap(a, b)
                pairs += 1
                if pairs > budget:
                    raise ValueError('work budget')
                low, high = sorted((first, second))
                hazards[high].add(low)
            (writers if b.write else readers).append((second, b))
    ancestry, parents = [], []
    for node, incoming in enumerate(hazards):
        covered, selected = set(), []
        for parent in sorted(incoming, reverse=True):
            if parent not in covered:
                selected.append(parent)
                covered.add(parent)
                covered.update(ancestry[parent])
        ancestry.append(covered)
        parents.append(sorted(selected))
    return parents, pairs

@pytest.mark.parametrize('seed', range(50))
def test_reference_sweep_reduction_matches_pairwise_oracle(seed):
    rng = random.Random(seed)
    for _ in range(100):
        nodes = []
        for _ in range(rng.randrange(1, 9)):
            row = []
            for _ in range(rng.randrange(5)):
                a, start, length = rng.randrange(3), rng.randrange(24), rng.randrange(12)
                row.append(Access(a, start, start + length, bool(rng.randrange(2))))
            nodes.append(row)
        parents, _ = sweep_model(nodes)
        assert closure(parents) == oracle(nodes)
        reversed_rows = [list(reversed(row)) for row in nodes]
        assert sweep_model(reversed_rows)[0] == parents

def test_reference_disjoint_4096_views_no_overlap_work():
    nodes = [[Access(0, i * 8, i * 8 + 8, True)] for i in range(4096)]
    parents, pairs = sweep_model(nodes, budget=0)
    assert pairs == 0 and all(not p for p in parents)
    assert 4096 * 4095 // 2 > 1_048_576  # v20's per-allocation pair scan.

def test_reference_readers_stay_unordered_but_all_precede_overwrite():
    nodes = [[Access(0, 0, 8, True)], [Access(0, 0, 8, False)],
             [Access(0, 0, 8, False)], [Access(0, 0, 8, True)]]
    assert sweep_model(nodes)[0] == [[], [0], [0], [1, 2]]

def test_reference_readonly_parts_never_promoted_to_writes():
    nodes = [[Access(0, 0, 8, True), Access(0, 8, 16, False)], [Access(0, 8, 16, False)]]
    assert sweep_model(nodes)[0] == [[], []]

def test_reference_real_overlap_budget_still_enforced():
    with pytest.raises(ValueError, match='work budget'):
        sweep_model([[Access(0, 0, 8, True)] for _ in range(4)], budget=1)

def test_reference_maximum_u64_boundaries():
    end = (1 << 64) - 1
    assert sweep_model([[Access(0, end - 2, end - 1, True)], [Access(0, end - 1, end, True)]])[1] == 0
