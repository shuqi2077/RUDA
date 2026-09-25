//! Dependency and memory-hazard validation for explicit native graphs.
//! No CUDA or RUDA dependencies: this exact production module is testable with
//! `rustc --edition=2024 --test graph_topology.rs`. It never executes kernels.
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_NODES: usize = 4096;
pub const MAX_EDGES: usize = 65536;
pub const MAX_ACCESSES: usize = 65536;
const MAX_ALIAS_CHECKS: usize = 1_048_576;
// Inference is an opt-in construction step, never replay work. This permits
// one overlapping access per node in a 4096-node chain while remaining bounded.
const MAX_INFERENCE_CHECKS: usize = MAX_NODES * MAX_NODES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    InvalidNodeCount,
    DependencyCount,
    TooManyEdges,
    InvalidParent { node: usize, parent: usize },
    DuplicateParent { node: usize, parent: usize },
    AccessCount,
    TooManyAccesses,
    InvalidRange { node: usize },
    UnorderedAccess { first: usize, second: usize, allocation: usize },
    AliasCheckBudget,
}

/// A half-open byte range in one retained RUDA managed allocation. ReadWrite is
/// deliberately treated as write-capable, including atomics and conditional stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferAccess {
    pub allocation: usize,
    pub start: u64,
    pub end: u64,
    pub writable: bool,
}

#[derive(Debug, Clone)]
pub struct GraphTopology {
    parents: Vec<Vec<usize>>,
    ancestors: Vec<Vec<u64>>,
    total_order: bool,
}

impl GraphTopology {
    /// Nodes are supplied in topological order. Parents must have smaller node
    /// indices. No implicit sort, edge insertion, or removal is performed.
    pub fn new(nodes: usize, parents: Vec<Vec<usize>>) -> Result<Self, TopologyError> {
        if nodes == 0 || nodes > MAX_NODES { return Err(TopologyError::InvalidNodeCount); }
        if parents.len() != nodes { return Err(TopologyError::DependencyCount); }
        let mut edges = 0usize;
        for (node, incoming) in parents.iter().enumerate() {
            edges = edges.checked_add(incoming.len()).ok_or(TopologyError::TooManyEdges)?;
            if edges > MAX_EDGES { return Err(TopologyError::TooManyEdges); }
            let mut seen = [0u64; MAX_NODES / 64];
            for &parent in incoming {
                if parent >= node { return Err(TopologyError::InvalidParent { node, parent }); }
                let mask = 1u64 << (parent % 64);
                if seen[parent / 64] & mask != 0 {
                    return Err(TopologyError::DuplicateParent { node, parent });
                }
                seen[parent / 64] |= mask;
            }
        }
        let words = nodes.div_ceil(64);
        // <= 2 MiB at the 4096-node limit. Used only during construction.
        let mut ancestors = vec![vec![0u64; words]; nodes];
        let mut total_order = true;
        for node in 0..nodes {
            let (prior, current_and_later) = ancestors.split_at_mut(node);
            let current = &mut current_and_later[0];
            for &parent in &parents[node] {
                current[parent / 64] |= 1u64 << (parent % 64);
                for (dst, &src) in current.iter_mut().zip(&prior[parent]) { *dst |= src; }
            }
            if node > 0 && current[(node - 1) / 64] & (1u64 << ((node - 1) % 64)) == 0 {
                total_order = false;
            }
        }
        Ok(Self { parents, ancestors, total_order })
    }

    pub fn chain(nodes: usize) -> Result<Self, TopologyError> {
        if nodes == 0 || nodes > MAX_NODES { return Err(TopologyError::InvalidNodeCount); }
        Self::new(nodes, (0..nodes).map(|i| if i == 0 { vec![] } else { vec![i - 1] }).collect())
    }

    pub fn parents(&self) -> &[Vec<usize>] { &self.parents }
    pub fn into_parents(self) -> Vec<Vec<usize>> { self.parents }

    fn ordered(&self, a: usize, b: usize) -> bool {
        if a == b { return true; }
        let (earlier, later) = if a < b { (a, b) } else { (b, a) };
        self.ancestors[later][earlier / 64] & (1u64 << (earlier % 64)) != 0
    }

    /// Reject unordered overlapping accesses if at least one can write. Shared
    /// read-only inputs and disjoint views may run independently. This uses IR
    /// access declarations and allocation identities, NOT guessed pointer values.
    /// It is not an intra-kernel sanitizer and does not validate hidden pointers.
    pub fn validate_accesses(&self, nodes: &[Vec<BufferAccess>]) -> Result<(), TopologyError> {
        check_accesses(nodes, self.parents.len())?;
        // Fully ordered graphs retain the fast path, including range checks.
        if self.total_order { return Ok(()); }
        visit_conflicts(nodes, MAX_ALIAS_CHECKS, |first, second, allocation| {
            if !self.ordered(first, second) {
                return Err(TopologyError::UnorderedAccess { first, second, allocation });
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Infer a memory dependency DAG from the semantics of the ORIGINAL node
    /// sequence. RAW, WAR and WAW overlaps preserve their original order; read-
    /// only sharing and disjoint views add no edge. This is not stream capture.
    /// No hidden pointers, I/O, host effects or cross-kernel signalling may be
    /// present: those require explicit dependencies via the existing API.
    pub fn infer(nodes: &[Vec<BufferAccess>]) -> Result<Self, TopologyError> {
        let count = nodes.len();
        if count == 0 || count > MAX_NODES { return Err(TopologyError::InvalidNodeCount); }
        check_accesses(nodes, count)?;
        let words = count.div_ceil(64);
        // At most 2 MiB each for hazards and ancestry. No O(accesses^2) matrix.
        let mut hazards = vec![vec![0u64; words]; count];
        visit_conflicts(nodes, MAX_INFERENCE_CHECKS, |a, b, _allocation| {
            let (before, after) = if a < b { (a, b) } else { (b, a) };
            hazards[after][before / 64] |= 1u64 << (before % 64);
            Ok(())
        })?;
        let mut parents: Vec<Vec<usize>> = vec![vec![]; count];
        let mut ancestors = vec![vec![0u64; words]; count];
        let mut edges = 0usize;
        let mut total_order = true;
        for node in 0..count {
            let mut covered = vec![0u64; words];
            // Latest ancestors first: if a parent already reaches a later
            // selected parent, no redundant direct edge is required.
            for word in (0..words).rev() {
                let mut candidates = hazards[node][word] & !covered[word];
                while candidates != 0 {
                    let offset = 63 - candidates.leading_zeros() as usize;
                    let parent = word * 64 + offset;
                    debug_assert!(parent < node);
                    edges = edges.checked_add(1).ok_or(TopologyError::TooManyEdges)?;
                    if edges > MAX_EDGES { return Err(TopologyError::TooManyEdges); }
                    parents[node].push(parent);
                    covered[word] |= 1u64 << offset;
                    for (dst, &src) in covered.iter_mut().zip(&ancestors[parent]) { *dst |= src; }
                    // Skip whole empty words and all ancestors already covered
                    // by this parent, rather than testing every earlier node.
                    candidates = hazards[node][word] & !covered[word];
                }
            }
            parents[node].reverse();
            if node > 0 && covered[(node - 1) / 64] & (1u64 << ((node - 1) % 64)) == 0 {
                total_order = false;
            }
            ancestors[node] = covered;
        }
        Ok(Self { parents, ancestors, total_order })
    }

}

fn check_accesses(nodes: &[Vec<BufferAccess>], count: usize) -> Result<(), TopologyError> {
    if nodes.len() != count { return Err(TopologyError::AccessCount); }
    let mut total = 0usize;
    for (node, accesses) in nodes.iter().enumerate() {
        total = total.checked_add(accesses.len()).ok_or(TopologyError::TooManyAccesses)?;
        if total > MAX_ACCESSES { return Err(TopologyError::TooManyAccesses); }
        if accesses.iter().any(|a| a.start > a.end) {
            return Err(TopologyError::InvalidRange { node });
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct ScanStats {
    coalesced_accesses: usize,
    overlapping_pairs: usize,
}
// (end-exclusive, node index, index in the sorted/coalesced access list).
type ActiveRange = (u64, usize, usize);
fn expire(active: &mut BTreeSet<ActiveRange>, start: u64) {
    while let Some(&(end, _, _)) = active.first() {
        if end > start { break; }
        active.pop_first();
    }
}

/// Sweep byte intervals, not all pairs belonging to an allocation. Active
/// readers and writers are separate, so read/read pairs are never enumerated.
/// Adjacent ranges merge ONLY for the same node and exact access mode; no byte
/// range is widened and no reader is relabelled as a writer.
fn visit_conflicts(
    nodes: &[Vec<BufferAccess>], budget: usize,
    mut visit: impl FnMut(usize, usize, usize) -> Result<(), TopologyError>,
) -> Result<ScanStats, TopologyError> {
    let mut allocations: BTreeMap<usize, Vec<(usize, BufferAccess)>> = BTreeMap::new();
    for (node, accesses) in nodes.iter().enumerate() {
        for &access in accesses {
            if access.start != access.end {
                allocations.entry(access.allocation).or_default().push((node, access));
            }
        }
    }
    let mut stats = ScanStats::default();
    for (allocation, mut ranges) in allocations {
        if !ranges.iter().any(|(_, a)| a.writable) { continue; }
        ranges.sort_unstable_by_key(|(node, a)| (*node, a.writable, a.start, a.end));
        let mut merged: Vec<(usize, BufferAccess)> = Vec::with_capacity(ranges.len());
        for (node, a) in ranges {
            if let Some((last_node, last)) = merged.last_mut() {
                if *last_node == node && last.writable == a.writable && a.start <= last.end {
                    last.end = last.end.max(a.end);
                    continue;
                }
            }
            merged.push((node, a));
        }
        stats.coalesced_accesses += merged.len();
        merged.sort_unstable_by_key(|(node, a)| (a.start, a.end, *node, a.writable));
        let mut readers: BTreeSet<ActiveRange> = BTreeSet::new();
        let mut writers: BTreeSet<ActiveRange> = BTreeSet::new();
        for (index, &(second, b)) in merged.iter().enumerate() {
            expire(&mut readers, b.start);
            expire(&mut writers, b.start);
            let mut emit = |first: usize, prior: usize| -> Result<(), TopologyError> {
                if first == second { return Ok(()); }
                let a = merged[prior].1;
                debug_assert!(a.start < b.end && b.start < a.end);
                stats.overlapping_pairs += 1;
                if stats.overlapping_pairs > budget { return Err(TopologyError::AliasCheckBudget); }
                let (first, second) = if first < second { (first, second) } else { (second, first) };
                visit(first, second, allocation)
            };
            for &(_, first, prior) in &writers { emit(first, prior)?; }
            if b.writable {
                for &(_, first, prior) in &readers { emit(first, prior)?; }
            }
            let key = (b.end, second, index);
            if b.writable { writers.insert(key); } else { readers.insert(key); }
        }
    }
    Ok(stats)
}

/// Graph structure only: no device addresses, scalar contents, or native handles.
pub fn to_dot(parents: &[Vec<usize>]) -> String {
    use std::fmt::Write;
    let mut text = String::from("digraph ruda {\n");
    for node in 0..parents.len() { let _ = writeln!(text, "  n{node} [label=\"kernel {node}\"];"); }
    for (node, incoming) in parents.iter().enumerate() {
        for parent in incoming { let _ = writeln!(text, "  n{parent} -> n{node};"); }
    }
    text.push_str("}\n");
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    fn access(allocation: usize, start: u64, end: u64, writable: bool) -> BufferAccess {
        BufferAccess { allocation, start, end, writable }
    }
    fn roots() -> GraphTopology { GraphTopology::new(2, vec![vec![], vec![]]).unwrap() }
    #[test] fn empty_rejected() { assert!(GraphTopology::chain(0).is_err()); }
    #[test] fn maximum_count_checked_before_allocation() { assert!(GraphTopology::chain(usize::MAX).is_err()); }
    #[test] fn wrong_dependency_count_rejected() { assert!(GraphTopology::new(2, vec![vec![]]).is_err()); }
    #[test] fn self_edge_rejected() { assert!(GraphTopology::new(1, vec![vec![0]]).is_err()); }
    #[test] fn future_parent_rejected() { assert!(GraphTopology::new(2, vec![vec![1], vec![]]).is_err()); }
    #[test] fn maximum_parent_rejected_without_indexing() { assert!(GraphTopology::new(1, vec![vec![usize::MAX]]).is_err()); }
    #[test] fn duplicate_parent_rejected() { assert!(GraphTopology::new(2, vec![vec![], vec![0,0]]).is_err()); }
    #[test] fn unsorted_distinct_parents_are_preserved() {
        let g=GraphTopology::new(3, vec![vec![],vec![],vec![1,0]]).unwrap();
        assert_eq!(g.parents()[2], vec![1,0]);
    }
    #[test] fn all_roots_valid() { assert!(GraphTopology::new(4096, vec![vec![];4096]).is_ok()); }
    #[test] fn maximum_chain_valid() {
        let g=GraphTopology::chain(4096).unwrap(); assert!(g.ordered(0,4095)); assert!(g.total_order);
    }
    #[test] fn edge_budget_enforced() {
        let mut p=vec![vec![];MAX_NODES]; p[MAX_NODES-1]=vec![0;MAX_EDGES+1];
        assert_eq!(GraphTopology::new(MAX_NODES,p).unwrap_err(),TopologyError::TooManyEdges);
    }
    #[test] fn diamond_has_no_false_branch_edge() {
        let g=GraphTopology::new(4,vec![vec![],vec![0],vec![0],vec![1,2]]).unwrap();
        assert!(!g.ordered(1,2)); assert!(g.ordered(0,3)); assert!(g.ordered(3,2));
    }
    #[test] fn read_only_overlap_is_allowed() {
        assert!(roots().validate_accesses(&[vec![access(0,0,32,false)],vec![access(0,0,32,false)]]).is_ok());
    }
    #[test] fn unordered_read_after_write_rejected() {
        assert!(roots().validate_accesses(&[vec![access(0,0,32,true)],vec![access(0,8,24,false)]]).is_err());
    }
    #[test] fn unordered_write_after_read_rejected() {
        assert!(roots().validate_accesses(&[vec![access(0,0,32,false)],vec![access(0,8,24,true)]]).is_err());
    }
    #[test] fn unordered_writes_rejected() {
        assert!(roots().validate_accesses(&[vec![access(0,0,32,true)],vec![access(0,8,24,true)]]).is_err());
    }
    #[test] fn disjoint_views_allowed() {
        assert!(roots().validate_accesses(&[vec![access(0,0,32,true)],vec![access(0,32,64,true)]]).is_ok());
    }
    #[test] fn allocation_identity_not_equal_offset() {
        assert!(roots().validate_accesses(&[vec![access(0,0,32,true)],vec![access(1,0,32,true)]]).is_ok());
    }
    #[test] fn empty_views_do_not_overlap() {
        assert!(roots().validate_accesses(&[vec![access(0,8,8,true)],vec![access(0,0,32,true)]]).is_ok());
    }
    #[test] fn invalid_ranges_rejected_even_in_chain() {
        assert!(GraphTopology::chain(1).unwrap().validate_accesses(&[vec![access(0,9,8,false)]]).is_err());
    }
    #[test] fn wrong_access_count_rejected() { assert!(roots().validate_accesses(&[]).is_err()); }
    #[test] fn access_budget_rejected() {
        assert!(GraphTopology::chain(1).unwrap().validate_accesses(&[vec![access(0,0,0,false);MAX_ACCESSES+1]]).is_err());
    }
    #[test] fn transitive_order_permits_overlap() {
        let g=GraphTopology::new(4,vec![vec![],vec![0],vec![],vec![1]]).unwrap();
        assert!(g.validate_accesses(&[vec![access(0,0,32,true)],vec![],vec![],vec![access(0,0,32,true)]]).is_ok());
    }
    #[test] fn same_node_alias_is_not_inter_node_race() {
        let a=access(0,0,32,true); assert!(roots().validate_accesses(&[vec![a,a],vec![]]).is_ok());
    }
    #[test] fn alias_work_is_bounded() {
        let n=1450;
        let mut parents:Vec<_>=(0..n).map(|i|if i==0 {vec![]} else {vec![i-1]}).collect();
        parents.push(vec![]);
        let g=GraphTopology::new(n+1,parents).unwrap();
        let a=access(0,0,32,true);
        let mut nodes=vec![vec![a];n];
        nodes.push(vec![]);
        assert_eq!(g.validate_accesses(&nodes),Err(TopologyError::AliasCheckBudget));
    }
    #[test] fn readonly_input_two_outputs_and_join() {
        let g=GraphTopology::new(3,vec![vec![],vec![],vec![0,1]]).unwrap();
        let nodes=[vec![access(0,0,32,false),access(1,0,32,true)],
            vec![access(0,0,32,false),access(2,0,32,true)],
            vec![access(1,0,32,false),access(2,0,32,false),access(3,0,32,true)]];
        assert!(g.validate_accesses(&nodes).is_ok());
    }
    #[test] fn missing_join_edge_is_rejected() {
        let g=GraphTopology::new(3,vec![vec![],vec![],vec![0]]).unwrap();
        assert!(g.validate_accesses(&[vec![],vec![access(2,0,32,true)],vec![access(2,0,32,false)]]).is_err());
    }
    #[test] fn dot_contains_requested_edges_only() {
        let dot=to_dot(&[vec![],vec![],vec![0,1]]);
        assert!(dot.contains("n0 -> n2;")); assert!(dot.contains("n1 -> n2;"));
        assert!(!dot.contains("n0 -> n1;")); assert_eq!(dot.matches("label=").count(),3);
    }
    #[test] fn exhaustive_small_dags_match_independent_reachability() {
        // 1024 five-node DAGs. Compare bitset ancestry with boolean transitive closure.
        let n=5; let edges:Vec<_>=(0..n).flat_map(|b|(0..b).map(move |a|(a,b))).collect();
        for mask in 0usize..(1usize<<edges.len()) {
            let mut parents=vec![vec![];n]; let mut reach=vec![vec![false;n];n];
            for (bit,&(a,b)) in edges.iter().enumerate() {
                if mask&(1usize<<bit)!=0 { parents[b].push(a); reach[a][b]=true; }
            }
            for k in 0..n { for a in 0..n { for b in 0..n { let path=reach[a][k]&&reach[k][b]; if path { reach[a][b]=true; } } } }
            let g=GraphTopology::new(n,parents).unwrap();
            for a in 0..n { for b in 0..n { assert_eq!(g.ordered(a,b), a==b||reach[a][b]||reach[b][a]); } }
        }
    }

    #[test] fn v21_disjoint_maximum_graph_has_zero_alias_pairs() {
        let nodes:Vec<_>=(0..MAX_NODES).map(|n|vec![access(0,n as u64*8,n as u64*8+8,true)]).collect();
        let g=GraphTopology::new(MAX_NODES,vec![vec![];MAX_NODES]).unwrap();
        g.validate_accesses(&nodes).unwrap();
        let stats=visit_conflicts(&nodes,MAX_ALIAS_CHECKS,|_,_,_|Ok(())).unwrap();
        assert_eq!(stats.overlapping_pairs,0);
    }
    #[test] fn v21_same_node_duplicates_and_adjacency_coalesce() {
        let nodes=[vec![access(0,0,8,true),access(0,0,8,true),access(0,8,16,true)],
            vec![access(0,0,16,false)]];
        let stats=visit_conflicts(&nodes,MAX_ALIAS_CHECKS,|_,_,_|Ok(())).unwrap();
        assert_eq!(stats.coalesced_accesses,2); assert_eq!(stats.overlapping_pairs,1);
    }
    #[test] fn v21_readonly_parts_are_not_promoted_to_writable() {
        let nodes=[vec![access(0,0,8,true),access(0,8,16,false)],vec![access(0,8,16,false)]];
        roots().validate_accesses(&nodes).unwrap();
    }
    #[test] fn v21_half_open_maximum_address_has_no_overflow() {
        roots().validate_accesses(&[vec![access(1,u64::MAX-2,u64::MAX-1,true)],
            vec![access(1,u64::MAX-1,u64::MAX,true)]]).unwrap();
    }
    #[test] fn v21_sweep_still_rejects_real_conflicts() {
        let nodes=[vec![access(0,16,32,true)],vec![access(0,0,17,false)]];
        assert_eq!(roots().validate_accesses(&nodes).unwrap_err(),
            TopologyError::UnorderedAccess{first:0,second:1,allocation:0});
    }
    #[test] fn v21_inference_empty_rejected() { assert!(GraphTopology::infer(&[]).is_err()); }
    #[test] fn v21_inference_too_many_nodes_rejected() {
        assert!(GraphTopology::infer(&vec![vec![];MAX_NODES+1]).is_err());
    }
    #[test] fn v21_inference_invalid_range_rejected() {
        assert_eq!(GraphTopology::infer(&[vec![access(0,8,7,true)]]).unwrap_err(),TopologyError::InvalidRange{node:0});
    }
    #[test] fn v21_inference_access_budget_is_not_hidden_by_coalescing() {
        assert_eq!(GraphTopology::infer(&[vec![access(0,0,1,true);MAX_ACCESSES+1]]).unwrap_err(),TopologyError::TooManyAccesses);
    }
    #[test] fn v21_inferred_shared_readonly_fork_and_join() {
        let nodes=[vec![access(0,0,32,false),access(1,0,32,true)],
            vec![access(0,0,32,false),access(2,0,32,true)],
            vec![access(1,0,32,false),access(2,0,32,false),access(3,0,32,true)]];
        let g=GraphTopology::infer(&nodes).unwrap();
        assert_eq!(g.parents(),&[vec![],vec![],vec![0,1]]); g.validate_accesses(&nodes).unwrap();
    }
    #[test] fn v21_inferred_write_after_read_is_preserved() {
        let g=GraphTopology::infer(&[vec![access(0,0,8,false)],vec![access(0,0,8,true)]]).unwrap();
        assert_eq!(g.parents(),&[vec![],vec![0]]);
    }
    #[test] fn v21_inferred_read_after_write_is_preserved() {
        let g=GraphTopology::infer(&[vec![access(0,0,8,true)],vec![access(0,0,8,false)]]).unwrap();
        assert_eq!(g.parents(),&[vec![],vec![0]]);
    }
    #[test] fn v21_inferred_reads_then_overwrite_keep_both_branches() {
        let nodes=[vec![access(0,0,8,true)],vec![access(0,0,8,false)],
            vec![access(0,0,8,false)],vec![access(0,0,8,true)]];
        assert_eq!(GraphTopology::infer(&nodes).unwrap().parents(),&[vec![],vec![0],vec![0],vec![1,2]]);
    }
    #[test] fn v21_inferred_partial_views_keep_only_actual_hazards() {
        let nodes=[vec![access(0,0,8,true)],vec![access(0,8,16,true)],
            vec![access(0,4,12,false)],vec![access(0,16,24,true)]];
        assert_eq!(GraphTopology::infer(&nodes).unwrap().parents(),&[vec![],vec![],vec![0,1],vec![]]);
    }
    #[test] fn v21_inference_same_node_alias_never_adds_self_edge() {
        let nodes=[vec![access(0,0,16,false),access(0,0,16,true)],vec![access(0,0,16,false)]];
        assert_eq!(GraphTopology::infer(&nodes).unwrap().parents(),&[vec![],vec![0]]);
    }
    #[test] fn v21_inferred_maximum_disjoint_views_are_all_roots() {
        let nodes:Vec<_>=(0..MAX_NODES).map(|n|vec![access(0,n as u64*8,n as u64*8+8,true)]).collect();
        let g=GraphTopology::infer(&nodes).unwrap(); assert!(g.parents().iter().all(Vec::is_empty));
    }
    #[test] fn v21_inferred_dense_write_sequence_reduces_to_chain() {
        let n=MAX_NODES;
        let nodes=vec![vec![access(0,0,32,true)];n];
        let g=GraphTopology::infer(&nodes).unwrap();
        assert_eq!(g.parents().iter().map(Vec::len).sum::<usize>(),n-1);
        assert_eq!(g.parents()[n-1],vec![n-2]); assert!(g.total_order);
    }
    #[test] fn v21_coalescing_never_bridges_a_gap() {
        let nodes=[vec![access(0,0,8,true),access(0,16,24,true)],vec![access(0,8,16,true)]];
        assert_eq!(GraphTopology::infer(&nodes).unwrap().parents(),&[Vec::<usize>::new(),vec![]]);
    }
    #[test] fn v21_inference_distinct_allocations_and_empty_views() {
        let nodes=[vec![access(0,0,8,true),access(1,4,4,true)],vec![access(1,0,8,true)]];
        assert_eq!(GraphTopology::infer(&nodes).unwrap().parents(),&[Vec::<usize>::new(),vec![]]);
    }
    #[test] fn v21_real_overlap_work_limit_still_fails_closed() {
        let nodes=vec![vec![access(0,0,8,true)];4];
        assert!(matches!(visit_conflicts(&nodes,1,|_,_,_|Ok(())),Err(TopologyError::AliasCheckBudget)));
    }
    #[test] fn v21_inference_is_deterministic_under_argument_reordering() {
        let nodes=vec![vec![access(0,0,4,true),access(1,0,4,false)],
            vec![access(2,0,4,true),access(1,0,4,false)],vec![access(0,0,4,false),access(2,0,4,false)]];
        let mut reversed=nodes.clone(); for n in &mut reversed { n.reverse(); }
        assert_eq!(GraphTopology::infer(&nodes).unwrap().parents(),GraphTopology::infer(&reversed).unwrap().parents());
    }
    // Independent oracle: direct pairwise hazards followed by boolean Floyd-
    // Warshall closure. No sweep, coalescing, ancestry bitsets or edge reduction.
    fn reference_reachability(nodes:&[Vec<BufferAccess>])->Vec<Vec<bool>> {
        let n=nodes.len(); let mut reach=vec![vec![false;n];n];
        for i in 0..n { for j in i+1..n { for a in &nodes[i] { for b in &nodes[j] {
            if a.allocation==b.allocation && a.start!=a.end && b.start!=b.end
                && (a.writable||b.writable) && a.start<b.end && b.start<a.end { reach[i][j]=true; }
        } } } }
        for k in 0..n { for i in 0..n { for j in 0..n {
            if reach[i][k] && reach[k][j] { reach[i][j]=true; }
        } } }
        reach
    }
    #[test] fn v21_inference_matches_pairwise_oracle_on_10000_cases() {
        let mut state=0x7265647563657531u64;
        fn next(s:&mut u64)->u64 { *s ^= *s<<13; *s ^= *s>>7; *s ^= *s<<17; *s }
        for _ in 0..10000 {
            let n=1+(next(&mut state)%8) as usize;
            let nodes:Vec<Vec<_>>=(0..n).map(|_|(0..next(&mut state)%5).map(|_|{
                let alloc=next(&mut state)%3; let start=next(&mut state)%24; let len=next(&mut state)%12;
                access(alloc as usize,start,start+len,next(&mut state)%2!=0)
            }).collect()).collect();
            let expected=reference_reachability(&nodes); let g=GraphTopology::infer(&nodes).unwrap();
            g.validate_accesses(&nodes).unwrap();
            for i in 0..n { for j in i+1..n { assert_eq!(g.ordered(i,j),expected[i][j],"nodes={nodes:?}"); } }
        }
    }
    #[test] fn v21_sweep_acceptance_matches_naive_validator_on_2000_dags() {
        let mut seed=7u64;
        fn next(s:&mut u64)->u64 { *s=s.wrapping_mul(6364136223846793005).wrapping_add(1);*s }
        for _ in 0..2000 {
            let n=6; let mut parents=vec![vec![];n];
            for j in 0..n {for i in 0..j {if next(&mut seed)%3==0 {parents[j].push(i);}}}
            let nodes:Vec<Vec<_>>=(0..n).map(|_|(0..3).map(|_|{
                let a=next(&mut seed)%2;let start=next(&mut seed)%20;let len=next(&mut seed)%8;
                access(a as usize,start,start+len,next(&mut seed)&2!=0)
            }).collect()).collect();
            let g=GraphTopology::new(n,parents).unwrap(); let mut valid=true;
            for i in 0..n {for j in i+1..n {for a in &nodes[i] {for b in &nodes[j] {
                if a.allocation==b.allocation && a.start!=a.end && b.start!=b.end && (a.writable||b.writable)
                    && a.start<b.end && b.start<a.end && !g.ordered(i,j) {valid=false;}
            }}}}
            assert_eq!(g.validate_accesses(&nodes).is_ok(),valid);
        }
    }
}
