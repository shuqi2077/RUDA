//! Rooted data-plane collectives. Reduce does not broadcast discarded results.
use std::{collections::HashMap, sync::Arc};
use ruda_tensor::{Backend, TensorMetadata};
use ruda_communication::{Address, Protocol, data_service::TensorDataService};
use crate::{NodeId, global::shared::GlobalCollectiveError};

/// A deterministic rooted tree. None is the centralized/star topology.
pub(super) fn neighbors(node: NodeId, root: NodeId, nodes: &HashMap<NodeId, Address>, arity: Option<u32>)
    -> Result<(Option<NodeId>, Vec<NodeId>), GlobalCollectiveError>
{
    if arity == Some(0) { return Err(GlobalCollectiveError::InvalidTreeArity); }
    if !nodes.contains_key(&root) { return Err(GlobalCollectiveError::CollectiveParamsMismatch); }
    let mut ids: Vec<_> = nodes.keys().copied().filter(|id| *id != root).collect();
    ids.sort();
    ids.insert(0, root);
    let i = ids.iter().position(|id| *id == node)
        .ok_or(GlobalCollectiveError::CollectiveBeforeRegister)?;
    let k = arity.map(|n| n as usize).unwrap_or(ids.len()).max(1);
    let parent = (i != 0).then(|| ids[(i - 1) / k]);
    let first = i.saturating_mul(k).saturating_add(1).min(ids.len());
    let last = first.saturating_add(k).min(ids.len());
    Ok((parent, ids[first..last].to_vec()))
}

pub(super) async fn reduce_sum<B: Backend, P: Protocol>(node: NodeId, root: NodeId,
    nodes: &HashMap<NodeId, Address>, service: &Arc<TensorDataService<B, P>>,
    mut tensor: B::FloatTensorPrimitive, arity: Option<u32>, transfer: u64)
    -> Result<Option<B::FloatTensorPrimitive>, GlobalCollectiveError>
{
    let (parent, children) = neighbors(node, root, nodes, arity)?;
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let device = B::float_device(&tensor);
    for child in children {
        let data = service.download_tensor(nodes[&child].clone(), transfer.into()).await
            .ok_or(GlobalCollectiveError::PeerLost(child))?;
        if data.shape != shape || data.dtype != dtype {
            return Err(GlobalCollectiveError::PeerSentIncoherentTensor);
        }
        tensor = B::float_add(tensor, B::float_from_data(data, &device));
    }
    if parent.is_some() {
        service.expose(tensor, 1, transfer.into()).await;
        Ok(None)
    } else { Ok(Some(tensor)) }
}

pub(super) async fn broadcast<B: Backend, P: Protocol>(node: NodeId, root: NodeId,
    nodes: &HashMap<NodeId, Address>, service: &Arc<TensorDataService<B, P>>,
    tensor: Option<B::FloatTensorPrimitive>, device: &B::Device,
    arity: Option<u32>, transfer: u64) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError>
{
    let (parent, children) = neighbors(node, root, nodes, arity)?;
    let tensor = match parent {
        None => tensor.ok_or(GlobalCollectiveError::BroadcastNoTensor)?,
        Some(parent) => {
            let data = service.download_tensor(nodes[&parent].clone(), transfer.into()).await
                .ok_or(GlobalCollectiveError::PeerLost(parent))?;
            B::float_from_data(data, device)
        }
    };
    if !children.is_empty() {
        service.expose(tensor.clone(), children.len() as u32, transfer.into()).await;
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rooted_trees_cover_every_node_once_for_all_arities_and_roots() {
        for count in 1..12 {
            let nodes = (0..count).map(|i| (NodeId::from(i),format!("ws://127.0.0.1:{}",9000+i).parse().unwrap()))
                .collect::<HashMap<_,Address>>();
            for root in nodes.keys().copied() { for arity in [None,Some(1),Some(2),Some(3),Some(20)] {
                let mut seen = std::collections::HashSet::new();
                let mut stack = vec![root];
                while let Some(node) = stack.pop() {
                    assert!(seen.insert(node),"cycle or duplicate node");
                    let (parent,children) = neighbors(node,root,&nodes,arity).unwrap();
                    assert_eq!(parent.is_none(),node == root);
                    for child in children {
                        assert_eq!(neighbors(child,root,&nodes,arity).unwrap().0,Some(node));
                        stack.push(child);
                    }
                }
                assert_eq!(seen.len(),nodes.len());
            }}
        }
    }
}
