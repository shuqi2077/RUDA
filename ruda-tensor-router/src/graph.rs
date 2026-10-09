use alloc::{collections::{BTreeMap, BTreeSet}, format};
use ruda_core::backtrace::BackTrace;
use ruda_tensor::{ExecutionError, Slice};
use ruda_tensor::graph::{GraphBindings, GraphIr, IrVisitorMut, OperationIr, ScalarIr, TensorId, TensorIr};

pub(crate) fn graph_error(reason: impl Into<alloc::string::String>) -> ExecutionError {
    ExecutionError::Generic { reason: reason.into(), backtrace: BackTrace::capture() }
}

pub(crate) struct CachedGraph {
    graph: GraphIr,
    layout: Layout,
}
impl CachedGraph {
    pub(crate) fn new(graph: GraphIr) -> Result<Self, ExecutionError> {
        let (inputs, outputs) = GraphIr::classify(&graph.operations);
        if inputs != graph.inputs || outputs != graph.outputs {
            return Err(graph_error("graph boundary does not match its relative operations"));
        }
        let mut layout = Layout::default();
        for operation in &graph.operations { operation.clone().visit_mut(&mut layout); }
        if layout.invalid { return Err(graph_error("graph has an invalid relative shape, scalar or slice placeholder")); }
        Ok(Self { graph, layout })
    }

    pub(crate) fn replay(
        &self,
        bindings: GraphBindings,
        mut allocate: impl FnMut(&BTreeSet<TensorId>) -> Result<TensorId, ExecutionError>,
        mut execute: impl FnMut(OperationIr),
    ) -> Result<(), ExecutionError> {
        if bindings.shapes.len() < self.layout.shapes || bindings.scalars.len() < self.layout.scalars
            || bindings.ranges.len() < self.layout.ranges {
            return Err(graph_error("graph bindings are missing shape, scalar or slice values"));
        }
        if self.layout.uses_unit && bindings.shapes.first() != Some(&1) {
            return Err(graph_error("relative shape ID zero must bind to one"));
        }
        let expected: BTreeSet<_> = self.graph.inputs.iter().chain(&self.graph.outputs).copied().collect();
        let mut ids = BTreeMap::new();
        let mut reserved = BTreeSet::new();
        for (relative, concrete) in &bindings.tensors {
            if !expected.contains(relative) || ids.insert(*relative, *concrete).is_some() {
                return Err(graph_error(format!("unexpected or duplicate graph tensor binding {relative}")));
            }
            reserved.insert(*concrete);
        }
        if ids.len() != expected.len() { return Err(graph_error("graph bindings are missing boundary tensors")); }
        for &relative in &self.layout.tensors {
            if let alloc::collections::btree_map::Entry::Vacant(entry) = ids.entry(relative) {
                let concrete = allocate(&reserved)?;
                reserved.insert(concrete);
                entry.insert(concrete);
            }
        }
        let mut visitor = Bind { ids: &ids, bindings: &bindings };
        for operation in &self.graph.operations {
            let mut operation = operation.clone();
            operation.visit_mut(&mut visitor);
            execute(operation);
        }
        Ok(())
    }
}

#[derive(Default)]
struct Layout {
    tensors: BTreeSet<TensorId>,
    shapes: usize,
    scalars: usize,
    ranges: usize,
    uses_unit: bool,
    invalid: bool,
}
impl IrVisitorMut for Layout {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        self.tensors.insert(tensor.id);
        for &dimension in tensor.shape.iter() {
            self.uses_unit |= dimension == 0;
            match dimension.checked_add(1) {
                Some(count) => self.shapes = self.shapes.max(count),
                None => self.invalid = true,
            }
        }
    }
    fn visit_scalar_mut(&mut self, scalar: &mut ScalarIr) {
        let index = match *scalar { ScalarIr::UInt(index) => usize::try_from(index).ok(), _ => None };
        match index.and_then(|index| index.checked_add(1)) {
            Some(count) => self.scalars = self.scalars.max(count),
            None => self.invalid = true,
        }
    }
    fn visit_range_mut(&mut self, range: &mut Slice) {
        match usize::try_from(range.start).ok().and_then(|index| index.checked_add(1)) {
            Some(count) => self.ranges = self.ranges.max(count),
            None => self.invalid = true,
        }
    }
}

struct Bind<'a> {
    ids: &'a BTreeMap<TensorId, TensorId>,
    bindings: &'a GraphBindings,
}
impl IrVisitorMut for Bind<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        tensor.id = self.ids[&tensor.id];
        for dimension in tensor.shape.iter_mut() { *dimension = self.bindings.shapes[*dimension]; }
    }
    fn visit_scalar_mut(&mut self, scalar: &mut ScalarIr) {
        let ScalarIr::UInt(index) = *scalar else { unreachable!("registered scalar placeholder"); };
        *scalar = self.bindings.scalars[index as usize];
    }
    fn visit_range_mut(&mut self, range: &mut Slice) {
        *range = self.bindings.ranges[range.start as usize];
    }
}
