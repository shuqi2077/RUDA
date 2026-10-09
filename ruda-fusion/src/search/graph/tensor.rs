use super::GraphNode;
use ruda_tensor::graph::{FloatOperationIr, NumericOperationIr, OperationIr, TensorId, TensorStatus};
use std::collections::{HashMap, HashSet};

/// One operation with its original source position.
pub struct OperationNode<'a> {
    pub operation: &'a OperationIr,
    pub position: usize,
}

fn ordered_effect(operation: &OperationIr) -> bool {
    match operation {
        OperationIr::Custom(_) | OperationIr::Float(_, FloatOperationIr::Random(_))
        | OperationIr::NumericInt(_, NumericOperationIr::IntRandom(_)) => true,
        #[cfg(feature = "distributed")]
        OperationIr::Distributed(_) => true,
        _ => false,
    }
}

impl GraphNode for OperationNode<'_> {
    type Resource = TensorId;

    fn produced(&self) -> impl Iterator<Item = TensorId> {
        self.operation.outputs().map(|tensor| tensor.id)
    }

    fn read(&self) -> impl Iterator<Item = TensorId> {
        self.operation.inputs().map(|tensor| tensor.id)
    }

    fn freed(&self) -> impl Iterator<Item = TensorId> {
        let drop = matches!(self.operation, OperationIr::Drop(_));
        self.operation.inputs().filter(move |tensor| {
            drop || matches!(tensor.status, TensorStatus::ReadWrite)
        }).map(|tensor| tensor.id)
    }

    fn position(&self) -> usize {
        self.position
    }

    fn ordered_range(&self) -> Option<(usize, usize)> {
        ordered_effect(self.operation).then_some((self.position, self.position))
    }
}

/// Incremental block data flow, retaining original resource-event positions.
#[derive(Clone)]
pub struct TensorFlow {
    produced: HashMap<TensorId, usize>,
    read: HashMap<TensorId, usize>,
    freed: HashSet<TensorId>,
    position: usize,
    effects: Option<(usize, usize)>,
}

impl TensorFlow {
    pub fn new() -> Self {
        Self { produced: HashMap::new(), read: HashMap::new(), freed: HashSet::new(),
            position: usize::MAX, effects: None }
    }

    pub fn contains(&self, resource: TensorId) -> bool {
        self.produced.contains_key(&resource) || self.read.contains_key(&resource)
    }

    pub fn is_empty(&self) -> bool {
        self.produced.is_empty() && self.read.is_empty()
    }

    pub fn register(&mut self, operation: &OperationIr, position: usize) {
        self.position = self.position.min(position);
        if ordered_effect(operation) {
            self.effects = Some(match self.effects {
                Some((first, last)) => (first.min(position), last.max(position)),
                None => (position, position),
            });
        }
        for tensor in operation.inputs() {
            if !self.produced.contains_key(&tensor.id) {
                self.read.entry(tensor.id).and_modify(|last| *last = (*last).max(position))
                    .or_insert(position);
            }
            if matches!(tensor.status, TensorStatus::ReadWrite) || matches!(operation, OperationIr::Drop(_)) {
                self.freed.insert(tensor.id);
            }
        }
        for tensor in operation.outputs() {
            self.produced.entry(tensor.id).and_modify(|first| *first = (*first).min(position))
                .or_insert(position);
            self.read.remove(&tensor.id);
        }
    }
}

impl GraphNode for TensorFlow {
    type Resource = TensorId;

    fn produced(&self) -> impl Iterator<Item = TensorId> {
        self.produced.keys().copied()
    }

    fn read(&self) -> impl Iterator<Item = TensorId> {
        self.read.keys().copied()
    }

    fn freed(&self) -> impl Iterator<Item = TensorId> {
        self.freed.iter().copied()
    }

    fn produces(&self, resource: TensorId) -> bool {
        self.produced.contains_key(&resource)
    }

    fn reads(&self, resource: TensorId) -> bool {
        self.read.contains_key(&resource)
    }

    fn position(&self) -> usize {
        if self.position == usize::MAX { 0 } else { self.position }
    }

    fn read_position(&self, resource: TensorId) -> usize {
        self.read[&resource]
    }

    fn produces_before(&self, resource: TensorId, position: usize) -> bool {
        self.produced.get(&resource).is_some_and(|&produced| produced < position)
    }

    fn ordered_range(&self) -> Option<(usize, usize)> {
        self.effects
    }
}
