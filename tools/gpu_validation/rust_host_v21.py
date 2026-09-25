#!/usr/bin/env python3
"""Compile and execute the exact production topology module, without Cargo/GPU.
No substitute implementation is run if rustc is absent. Does not certify full crate/GPU.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
REQUIRED = set(['access_budget_rejected', 'alias_work_is_bounded', 'all_roots_valid', 'allocation_identity_not_equal_offset', 'diamond_has_no_false_branch_edge', 'disjoint_views_allowed', 'dot_contains_requested_edges_only', 'duplicate_parent_rejected', 'edge_budget_enforced', 'empty_rejected', 'empty_views_do_not_overlap', 'exhaustive_small_dags_match_independent_reachability', 'future_parent_rejected', 'invalid_ranges_rejected_even_in_chain', 'maximum_chain_valid', 'maximum_count_checked_before_allocation', 'maximum_parent_rejected_without_indexing', 'missing_join_edge_is_rejected', 'read_only_overlap_is_allowed', 'readonly_input_two_outputs_and_join', 'same_node_alias_is_not_inter_node_race', 'self_edge_rejected', 'transitive_order_permits_overlap', 'unordered_read_after_write_rejected', 'unordered_write_after_read_rejected', 'unordered_writes_rejected', 'unsorted_distinct_parents_are_preserved', 'v21_coalescing_never_bridges_a_gap', 'v21_disjoint_maximum_graph_has_zero_alias_pairs', 'v21_half_open_maximum_address_has_no_overflow', 'v21_inference_access_budget_is_not_hidden_by_coalescing', 'v21_inference_distinct_allocations_and_empty_views', 'v21_inference_empty_rejected', 'v21_inference_invalid_range_rejected', 'v21_inference_is_deterministic_under_argument_reordering', 'v21_inference_matches_pairwise_oracle_on_10000_cases', 'v21_inference_same_node_alias_never_adds_self_edge', 'v21_inference_too_many_nodes_rejected', 'v21_inferred_dense_write_sequence_reduces_to_chain', 'v21_inferred_maximum_disjoint_views_are_all_roots', 'v21_inferred_partial_views_keep_only_actual_hazards', 'v21_inferred_read_after_write_is_preserved', 'v21_inferred_reads_then_overwrite_keep_both_branches', 'v21_inferred_shared_readonly_fork_and_join', 'v21_inferred_write_after_read_is_preserved', 'v21_readonly_parts_are_not_promoted_to_writable', 'v21_real_overlap_work_limit_still_fails_closed', 'v21_same_node_duplicates_and_adjacency_coalesce', 'v21_sweep_acceptance_matches_naive_validator_on_2000_dags', 'v21_sweep_still_rejects_real_conflicts', 'wrong_access_count_rejected', 'wrong_dependency_count_rejected'])
def require_complete(text: str) -> int:
    match = re.search(r'test result: ok\. (\d+) passed; 0 failed; 0 ignored;', text)
    passed = set(re.findall(r'test tests::(\w+) \.\.\. ok', text))
    if not match or int(match.group(1)) < len(REQUIRED) or REQUIRED - passed:
        raise ValueError('incomplete production Rust test result; no skipped/missing case is accepted')
    return int(match.group(1))
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('v21-rust-host'))
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if args.timeout <= 0: parser.error('timeout must be positive')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    report = {'version':'v21', 'production_rust_compiled':False, 'tests_passed':0,
              'full_native_crate_compiled':False, 'gpu_validated':False, 'commands':[], 'errors':[]}
    def save(): (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    compiler = shutil.which('rustc')
    if not compiler:
        report['errors'].append('missing rustc; no production Rust compilation/execution'); save(); return 2
    binary = out/('graph_topology_tests.exe' if os.name=='nt' else 'graph_topology_tests')
    commands=[('version',[compiler,'--version']),
        ('compile',[compiler,'--edition=2024','--test',str(ROOT/'ruda-driver-cuda/src/execution/graph_topology.rs'),'-o',str(binary)]),
        ('test',[str(binary),'--test-threads=1'])]
    try:
        for label, command in commands:
            result=subprocess.run(command,capture_output=True,text=True,timeout=args.timeout,check=False)
            text=result.stdout+result.stderr;(out/f'{label}.log').write_text(text)
            report['commands'].append({'command':command,'returncode':result.returncode})
            if result.returncode: raise RuntimeError(label+' failed')
            if label=='compile': report['production_rust_compiled']=True
            if label=='test': report['tests_passed']=require_complete(text)
            save()
        return 0
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
        report['errors'].append(str(exc));save();print(str(exc),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
