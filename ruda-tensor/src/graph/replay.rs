use super::{IrVisitorMut, OperationIr, ScalarIr, TensorId, TensorIr, TensorStatus};
use crate::{ExecutionError, Slice};
use alloc::{collections::{BTreeMap, BTreeSet}, vec, vec::Vec};
use ruda_core::backtrace::BackTrace;
use serde::{Deserialize, Serialize};

/// Runner-issued identifier for an explicitly registered reusable operation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GraphId(pub u64);

/// Ordered relative operations and their complete tensor boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphIr {
    /// Operations with relative tensor IDs, shape dimensions, scalars and slice ranges.
    pub operations: Vec<OperationIr>,
    /// External tensor IDs, in first-use order.
    pub inputs: Vec<TensorId>,
    /// Produced tensors that survive execution, in production order.
    pub outputs: Vec<TensorId>,
}

/// Concrete values for one invocation; intermediate tensor IDs are allocated by the runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBindings {
    /// Relative-to-concrete IDs for every input and surviving output.
    pub tensors: Vec<(TensorId, TensorId)>,
    /// Concrete dimension values indexed by relative dimension ID; ID zero denotes one.
    pub shapes: Vec<usize>,
    /// Scalar operands indexed by their relative placeholder ID.
    pub scalars: Vec<ScalarIr>,
    /// Original slice bounds and steps indexed by their relative placeholder ID.
    pub ranges: Vec<Slice>,
}

impl GraphIr {
    /// Capture concrete operations without executing them, returning the graph and initial bindings.
    pub fn capture(mut operations: Vec<OperationIr>) -> (Self, GraphBindings) {
        let (inputs, outputs) = Self::classify(&operations);
        let mut normalizer = Normalizer {
            tensors: BTreeMap::new(),
            shapes: BTreeMap::from([(1, 0)]),
            bindings: GraphBindings { tensors: Vec::new(), shapes: vec![1], scalars: Vec::new(), ranges: Vec::new() },
        };
        for operation in &mut operations { operation.visit_mut(&mut normalizer); }
        let inputs: Vec<_> = inputs.into_iter().map(|id| {
            let relative = normalizer.tensors[&id];
            normalizer.bindings.tensors.push((relative, id));
            relative
        }).collect();
        let outputs: Vec<_> = outputs.into_iter().map(|id| {
            let relative = normalizer.tensors[&id];
            normalizer.bindings.tensors.push((relative, id));
            relative
        }).collect();
        (Self { operations, inputs, outputs }, normalizer.bindings)
    }

    /// Infer external inputs and surviving outputs without copying the operation sequence.
    pub fn classify(operations: &[OperationIr]) -> (Vec<TensorId>, Vec<TensorId>) {
        let mut referenced = Vec::new();
        let mut referenced_set = BTreeSet::new();
        let mut produced = Vec::new();
        let mut produced_set = BTreeSet::new();
        let mut consumed = BTreeSet::new();
        for operation in operations {
            if let OperationIr::Drop(tensor) = operation { consumed.insert(tensor.id); }
            if !matches!(operation, OperationIr::Init(_)) {
                for tensor in operation.outputs() {
                    if produced_set.insert(tensor.id) { produced.push(tensor.id); }
                }
            }
            for tensor in operation.inputs().chain(operation.outputs()) {
                if referenced_set.insert(tensor.id) { referenced.push(tensor.id); }
                if tensor.status == TensorStatus::ReadWrite { consumed.insert(tensor.id); }
            }
        }
        let inputs = referenced.into_iter().filter(|id| !produced_set.contains(id)).collect();
        let outputs = produced.into_iter().filter(|id| !consumed.contains(id)).collect();
        (inputs, outputs)
    }

    /// Obtain new bindings only when the concrete operations match this graph's relative structure.
    pub fn bindings_for(&self, operations: Vec<OperationIr>) -> Result<GraphBindings, ExecutionError> {
        let (graph, bindings) = Self::capture(operations);
        if &graph != self {
            return Err(ExecutionError::Generic { reason: "graph structure, dtype, status or static options changed".into(), backtrace: BackTrace::capture() });
        }
        Ok(bindings)
    }

    /// Number of operations dispatched by a replay.
    pub fn len(&self) -> usize { self.operations.len() }
    /// Whether replay contains no operations.
    pub fn is_empty(&self) -> bool { self.operations.is_empty() }
}

struct Normalizer {
    tensors: BTreeMap<TensorId, TensorId>,
    shapes: BTreeMap<usize, usize>,
    bindings: GraphBindings,
}
impl IrVisitorMut for Normalizer {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        let next = TensorId::new(self.tensors.len() as u64);
        tensor.id = *self.tensors.entry(tensor.id).or_insert(next);
        for dimension in tensor.shape.iter_mut() {
            let next = self.bindings.shapes.len();
            *dimension = *self.shapes.entry(*dimension).or_insert_with(|| {
                self.bindings.shapes.push(*dimension);
                next
            });
        }
    }
    fn visit_scalar_mut(&mut self, scalar: &mut ScalarIr) {
        let next = self.bindings.scalars.len() as u64;
        self.bindings.scalars.push(*scalar);
        *scalar = ScalarIr::UInt(next);
    }
    fn visit_range_mut(&mut self, range: &mut Slice) {
        let next = self.bindings.ranges.len();
        self.bindings.ranges.push(*range);
        *range = Slice::from(next..next + 1);
    }
}
