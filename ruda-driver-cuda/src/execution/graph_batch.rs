//! Side-effect-free graph batch planning. This file also builds with `rustc --test`.
//! Keep validation separate from the driver commit phase: a late invalid node
//! must not leave earlier nodes updated. Driver failures are NOT rolled back.

pub(crate) const MAX_GRAPH_NODES: usize = 4096;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BatchError<E> {
    InvalidSize,
    OutOfRange(usize),
    Duplicate(usize),
    InvalidNode { index: usize, reason: E },
}

/// Validate all indices before calling even the first node-layout validator.
/// A bounded bitmap avoids hashing and a separate heap allocation for the set.
pub(crate) fn check_indices<E>(node_count: usize, indices: &[usize]) -> Result<(), BatchError<E>> {
    if node_count == 0 || node_count > MAX_GRAPH_NODES
        || indices.is_empty() || indices.len() > node_count {
        return Err(BatchError::InvalidSize);
    }
    let mut seen = [0u64; MAX_GRAPH_NODES / 64];
    for &index in indices {
        if index >= node_count { return Err(BatchError::OutOfRange(index)); }
        let mask = 1u64 << (index % 64);
        if seen[index / 64] & mask != 0 { return Err(BatchError::Duplicate(index)); }
        seen[index / 64] |= mask;
    }
    Ok(())
}

/// Return positions of changed nodes, preserving request order. `validate`
/// must be pure: no upload, native update, allocation mutation or replay.
pub(crate) fn plan_batch<T, E>(node_count: usize, updates: &[(usize, T)],
    mut validate: impl FnMut(usize, &T) -> Result<bool, E>) -> Result<Vec<usize>, BatchError<E>>
{
    let indices: Vec<_> = updates.iter().map(|(index, _)| *index).collect();
    check_indices(node_count, &indices)?;
    let mut changed = Vec::with_capacity(updates.len());
    for (position, (index, value)) in updates.iter().enumerate() {
        if validate(*index, value).map_err(|reason| BatchError::InvalidNode { index: *index, reason })? {
            changed.push(position);
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn empty_batch_rejected() { assert_eq!(check_indices::<()>(1, &[]), Err(BatchError::InvalidSize)); }
    #[test] fn empty_graph_rejected() { assert_eq!(check_indices::<()>(0, &[0]), Err(BatchError::InvalidSize)); }
    #[test] fn graph_limit_is_enforced() { assert_eq!(check_indices::<()>(4097, &[0]), Err(BatchError::InvalidSize)); }
    #[test] fn oversized_batch_rejected() { assert_eq!(check_indices::<()>(1, &[0, 0]), Err(BatchError::InvalidSize)); }
    #[test] fn maximum_index_cannot_overflow_bitmap() { assert_eq!(check_indices::<()>(4096, &[usize::MAX]), Err(BatchError::OutOfRange(usize::MAX))); }
    #[test] fn duplicate_across_bitmap_words_rejected() { assert_eq!(check_indices::<()>(130, &[0, 64, 128, 64]), Err(BatchError::Duplicate(64))); }
    #[test] fn all_4096_nodes_fit() { assert!(check_indices::<()>(4096, &(0..4096).collect::<Vec<_>>()).is_ok()); }
    #[test] fn reverse_order_is_allowed() { assert!(check_indices::<()>(4096, &(0..4096).rev().collect::<Vec<_>>()).is_ok()); }
    #[test] fn request_order_and_changed_positions_are_preserved() {
        let updates = [(3, true), (1, false), (0, true)];
        let mut visited = Vec::new();
        let result = plan_batch(4, &updates, |i, value| { visited.push(i); Ok::<_, ()>(*value) }).unwrap();
        assert_eq!(visited, vec![3, 1, 0]); assert_eq!(result, vec![0, 2]);
    }
    #[test] fn all_unchanged_still_validates_every_node() {
        let mut visited = 0;
        assert!(plan_batch(4, &[(0, ()), (2, ())], |_, _| { visited += 1; Ok::<_, ()>(false) }).unwrap().is_empty());
        assert_eq!(visited, 2);
    }
    #[test] fn duplicate_rejected_before_any_layout_validation() {
        let mut visited = 0;
        let result = plan_batch(2, &[(0, ()), (0, ())], |_, _| { visited += 1; Ok::<_, ()>(true) });
        assert_eq!(result, Err(BatchError::Duplicate(0))); assert_eq!(visited, 0);
    }
    #[test] fn late_bad_index_rejected_before_any_layout_validation() {
        let mut visited = 0;
        let result = plan_batch(2, &[(0, ()), (2, ())], |_, _| { visited += 1; Ok::<_, ()>(true) });
        assert_eq!(result, Err(BatchError::OutOfRange(2))); assert_eq!(visited, 0);
    }
    #[test] fn late_bad_layout_discards_whole_plan() {
        let mut visited = Vec::new();
        let result = plan_batch(3, &[(0, true), (1, false), (2, true)], |i, valid| {
            visited.push(i); if *valid { Ok(true) } else { Err("shape") }
        });
        assert_eq!(result, Err(BatchError::InvalidNode { index: 1, reason: "shape" }));
        assert_eq!(visited, vec![0, 1]);
    }
    #[test] fn sparse_boundary_indices_do_not_alias() {
        for i in 0..4096 { assert!(check_indices::<()>(4096, &[i]).is_ok()); }
        assert!(check_indices::<()>(4096, &[0, 63, 64, 127, 128, 4095]).is_ok());
    }
}
