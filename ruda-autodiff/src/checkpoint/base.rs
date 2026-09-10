use super::{
    retro_forward::RetroForwards,
    state::{BackwardStates, State},
};
use crate::collections::{HashMap, HashSet};
use crate::graph::NodeId;

use alloc::{format, vec, vec::Vec};
use ruda_tensor_config::{autodiff::AutodiffLogLevel, log_autodiff};

#[derive(new, Debug)]
/// Links a [NodeId] to its autodiff graph [NodeRef]
pub(crate) struct NodeTree {
    map: HashMap<NodeId, Vec<NodeId>>,
}

impl NodeTree {
    /// Gives the parents of the node in the autodiff graph
    pub(crate) fn parents(&self, node_id: &NodeId) -> Option<Vec<NodeId>> {
        self.map.get(node_id).cloned()
    }
}

#[derive(new, Debug)]
/// Struct responsible of fetching the output for a node in the autodiff graph during a backward pass
pub struct Checkpointer {
    backward_states: BackwardStates,
    retro_forwards: RetroForwards,
    node_tree: NodeTree,
}

impl Checkpointer {
    /// Gives the output of the given node, by recursively asking parents to compute themselves
    /// or give their pre-computed tensors.
    pub fn retrieve_node_output<T>(&mut self, node_id: NodeId) -> T
    where
        T: Clone + Send + 'static,
    {
        let sorted = self.topological_sort(node_id);
        let num_nodes = sorted.len();
        log_autodiff(AutodiffLogLevel::Basic, move || {
            format!("retrieve_node_output {node_id:?}: {num_nodes} node(s) to compute")
        });

        sorted.into_iter().for_each(|node| {
            log_autodiff(AutodiffLogLevel::Full, move || {
                format!("execute_retro_forward {node:?}")
            });
            self.retro_forwards
                .execute_retro_forward(node, &mut self.backward_states)
        });

        self.backward_states.get_state::<T>(&node_id)
    }

    /// Sorts the ancestors of NodeId in a way such that all parents come before their children
    /// Useful to avoid recursivity later when mutating the states
    ///
    /// The sort on a compute bound state or a memory bound that is already computed is trivial.
    /// The match on State::Computed also serves as a stopping criterion for the sort,
    /// we don't need to look higher than that during traversal.
    fn topological_sort(&self, node_id: NodeId) -> Vec<NodeId> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut pending = vec![(node_id, false)];

        while let Some((node_id, parents_expanded)) = pending.pop() {
            if visited.contains(&node_id) {
                continue;
            }
            if parents_expanded {
                visited.insert(node_id);
                sorted.push(node_id);
                continue;
            }

            match self.backward_states.get_state_ref(&node_id) {
                Some(State::Recompute { .. }) => {
                    pending.push((node_id, true));
                    pending.extend(
                        self.node_tree.parents(&node_id).unwrap()
                            .into_iter().rev().map(|parent| (parent, false)),
                    );
                }
                Some(State::Computed { .. }) => {
                    visited.insert(node_id);
                    sorted.push(node_id);
                }
                None => panic!("Node {node_id:?} is not in the backward_states. "),
            }
        }
        sorted
    }

    /// Checks if checkpointer has been drained adequately. Useful for testing
    pub fn is_empty(&self) -> bool {
        self.backward_states.is_empty() && self.retro_forwards.is_empty()
    }
}
