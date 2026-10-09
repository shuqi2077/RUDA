use super::GraphNode;
use std::collections::{HashMap, HashSet, hash_map::Entry};

/// Check actual reads against their source producer and last-use lifetimes.
pub fn is_valid_execution_order<N: GraphNode>(order: impl Iterator<Item = N> + Clone) -> bool {
    let mut producers = HashMap::new();
    let mut repeated = HashMap::<N::Resource, Vec<usize>>::new();
    for node in order.clone() {
        for resource in node.produced() {
            match producers.entry(resource) {
                Entry::Vacant(entry) => { entry.insert(node.position()); }
                Entry::Occupied(entry) if *entry.get() != node.position() => {
                    repeated.entry(resource).or_insert_with(|| vec![*entry.get()]).push(node.position());
                }
                Entry::Occupied(_) => {}
            }
        }
    }
    for positions in repeated.values_mut() {
        positions.sort_unstable();
        positions.dedup();
    }

    let mut current = HashMap::new();
    let mut dead = HashSet::new();
    let mut last_effect = None;
    for node in order {
        if let Some((first, last)) = node.ordered_range() {
            if last_effect.is_some_and(|previous| previous > first) {
                return false;
            }
            last_effect = Some(last);
        }
        for resource in node.read() {
            let expected = if let Some(positions) = repeated.get(&resource) {
                let before = positions.partition_point(|&position| position < node.position());
                before.checked_sub(1).map(|index| positions[index])
            } else {
                producers.get(&resource).and_then(|&position| (position < node.position()).then_some(position))
            };
            if dead.contains(&resource) || current.get(&resource).copied() != expected {
                return false;
            }
        }
        for resource in node.freed() {
            dead.insert(resource);
        }
        for resource in node.produced() {
            dead.remove(&resource);
            current.insert(resource, node.position());
        }
    }
    true
}
