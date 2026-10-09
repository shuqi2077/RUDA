use crate::{FuserStatus, NumOperations, OperationFuser, stream::store::ExecutionStrategy};
use ruda_tensor::graph::{OperationIr, TensorId, TensorIr, TensorStatus};
use std::{collections::{HashMap, HashSet}, sync::Arc};

/// A block represents a list of operations, not necessarily in the same order as the execution
/// stream.
///
/// The start and end position of the relative execution stream are tracked in the block alongside
/// the ordering.
pub struct Block<O> {
    builders: Vec<Box<dyn OperationFuser<O>>>,
    operations: Vec<OperationIr>,
    ids: HashSet<TensorId>,
    ordering: Vec<usize>,
    closed_at: Vec<Option<usize>>,
    /// The start position in the relative execution stream.
    pub start_pos: usize,
    /// The end position in the relative execution stream.
    pub end_pos: usize,
}

/// The result of [registering](Block::register) an [operation](OperationIr).
pub enum RegistrationResult {
    /// If the [operation](OperationIr) is correctly registered.
    Accepted,
    /// If the [operation](OperationIr) isn't part of the graph.
    ///
    /// In this case the operation isn't registered.
    NotPartOfTheGraph,
}

/// The optimization found for a [block](Block).
#[derive(Debug, new)]
pub struct BlockOptimization<O> {
    /// The [execution strategy](ExecutionStrategy) to be used to execute the [block](Block).
    pub strategy: ExecutionStrategy<O>,
    /// The ordering of each operation in the relative execution stream.
    pub ordering: Vec<usize>,
}

impl<O: NumOperations> Block<O> {
    /// Create a new block that will be optimized with the provided [optimization builders](OptimizationBuilder).
    pub fn new(builders: &[Box<dyn OperationFuser<O>>]) -> Self {
        Self {
            builders: builders.iter().map(|o| o.clone_dyn()).collect(),
            operations: Vec::new(),
            ids: HashSet::new(),
            ordering: Vec::new(),
            closed_at: vec![None; builders.len()],
            start_pos: usize::MAX,
            end_pos: usize::MIN,
        }
    }

    /// Sort the [blocks](Block) based on the start position.
    pub fn sort(blocks: &mut [Self]) {
        blocks.sort_by_key(|a| a.start_pos);
    }

    /// Optimize the block.
    pub fn optimize(mut self) -> BlockOptimization<O> {
        match find_best_optimization_index(&mut self.builders) {
            BestOptimization::Found { index, score } => {
                let opt = self.builders[index].finish();
                let opt_len = opt.len();
                if opt_len < self.operations.len() {
                    self.ordering.drain(opt_len..);
                }

                let strategy = ExecutionStrategy::Optimization {
                    ordering: Arc::new(self.ordering.clone()),
                    opt,
                    score,
                };
                BlockOptimization::new(strategy, self.ordering)
            }
            BestOptimization::NotFound => {
                let mut settled = self
                    .closed_at
                    .iter()
                    .map(|closed| closed.map(|index| index + 1).unwrap_or(self.operations.len()))
                    .max()
                    .unwrap_or(self.operations.len());
                while let Some(operation) = self.operations.get(settled) {
                    let wanted = self.builders.iter().any(|builder| {
                        let mut probe = builder.clone_dyn();
                        probe.reset();
                        probe.fuse(operation);
                        matches!(probe.status(), FuserStatus::Open) || probe.properties().ready
                    });
                    if wanted {
                        break;
                    }
                    settled += 1;
                }
                self.ordering.truncate(settled);
                let strategy = ExecutionStrategy::Operations {
                    ordering: Arc::new(self.ordering.clone()),
                };
                BlockOptimization::new(strategy, self.ordering)
            }
        }
    }

    /// Returns if the block contains any of the provided [tensors](TensorIr).
    pub fn contains_tensors(&self, tensors: &[&TensorIr]) -> bool {
        for node in tensors {
            if self.ids.contains(&node.id) {
                return true;
            }
        }

        false
    }

    /// Merge the current block with the other one and returns if the operation is successful.
    ///
    /// # Warning
    ///
    /// This will modify the current block even if the other block isn't correctly merged.
    pub fn merge(&mut self, other: &Block<O>) -> bool {
        if !self.can_append(other) {
            return false;
        }
        self.merge_validated(other)
    }

    pub(super) fn merge_validated(&mut self, other: &Block<O>) -> bool {
        let self_ready = self.has_ready_optimization();
        let other_ready = other.has_ready_optimization();

        for (op, pos) in other.operations.iter().zip(&other.ordering) {
            self.register(op, *pos, true);
        }

        // If the merged block can still be improved, keep it — that's the
        // usual lazy-optimization signal.
        if self.still_optimizing() {
            return true;
        }

        // Otherwise, only accept the merge when it *creates* a ready fusion
        // that didn't exist separately. If either side already had a ready
        // fusion before the merge, merging would collapse them and hide one
        // of the fusions — keep the blocks separate instead.
        !self_ready && !other_ready && self.has_ready_optimization()
    }

    /// Whether this append direction preserves production and last-use ordering.
    pub fn can_append(&self, other: &Block<O>) -> bool {
        let operations = self.operations.iter().zip(&self.ordering)
            .chain(other.operations.iter().zip(&other.ordering));
        let mut produced = HashMap::new();
        #[cfg(feature = "distributed")]
        let mut last_collective = None;
        for (operation, &position) in operations.clone() {
            #[cfg(feature = "distributed")]
            if matches!(operation, OperationIr::Distributed(_)) {
                if last_collective.is_some_and(|previous| previous > position) {
                    return false;
                }
                last_collective = Some(position);
            }
            for tensor in operation.outputs() {
                produced.entry(tensor.id).and_modify(|first: &mut usize| {
                    *first = (*first).min(position);
                }).or_insert(position);
            }
        }

        let mut alive = HashSet::new();
        let mut dead = HashSet::new();
        for (operation, &position) in operations {
            for tensor in operation.inputs() {
                if dead.contains(&tensor.id) {
                    return false;
                }
                if !alive.contains(&tensor.id) {
                    if produced.get(&tensor.id).is_some_and(|&first| first < position) {
                        return false;
                    }
                    alive.insert(tensor.id);
                }
            }
            for tensor in operation.inputs() {
                if matches!(tensor.status, TensorStatus::ReadWrite) {
                    alive.remove(&tensor.id);
                    dead.insert(tensor.id);
                }
            }
            for tensor in operation.outputs() {
                dead.remove(&tensor.id);
                alive.insert(tensor.id);
            }
        }
        true
    }

    fn has_ready_optimization(&self) -> bool {
        self.builders.iter().any(|b| b.properties().ready)
    }

    /// Register an [operation](OperationIr) in the current block.
    ///
    /// You need to provide the order of the operation as well as a force flag.
    ///
    /// When the force flag is true, the builder will always accept the operation, otherwise it
    /// might refuse it if the operation [isn't part of the graph](RegistrationResult::NotPartOfTheGraph).
    ///
    /// Forcing is useful to fuse operations that are part of different graphs, but included
    /// in the same optimization.
    pub fn register(
        &mut self,
        operation: &OperationIr,
        order: usize,
        force: bool,
    ) -> RegistrationResult {
        if self.ids.is_empty() {
            self.register_op(operation, order);
            return RegistrationResult::Accepted;
        }
        let mut contains = false;
        for node in operation.nodes() {
            contains = self.ids.contains(&node.id);

            if contains {
                break;
            }
        }

        if !contains && !force {
            return RegistrationResult::NotPartOfTheGraph;
        }

        self.register_op(operation, order);
        RegistrationResult::Accepted
    }

    /// If the block can still be optimized further.
    pub fn still_optimizing(&self) -> bool {
        let mut num_stopped = 0;

        for optimization in self.builders.iter() {
            if let FuserStatus::Closed = optimization.status() {
                num_stopped += 1
            }
        }

        num_stopped < self.builders.len()
    }

    fn register_op(&mut self, operation: &OperationIr, pos: usize) {
        self.operations.push(operation.clone());
        self.ordering.push(pos);

        if pos < self.start_pos {
            self.start_pos = pos;
        }
        if pos + 1 > self.end_pos {
            self.end_pos = pos + 1;
        }

        let index = self.operations.len() - 1;
        for (builder, closed_at) in self.builders.iter_mut().zip(&mut self.closed_at) {
            builder.fuse(operation);
            if closed_at.is_none() && matches!(builder.status(), FuserStatus::Closed) {
                *closed_at = Some(index);
            }
        }

        for node in operation.nodes() {
            self.ids.insert(node.id);
        }
    }
}

impl<O> BlockOptimization<O> {
    /// Maps the ordering of the current block optimization using the given mapping.
    pub fn map_ordering(&mut self, mapping: &[usize]) {
        for i in self.ordering.iter_mut() {
            *i = mapping[*i];
        }
        self.strategy.map_ordering(mapping);
    }
}

impl<O> ExecutionStrategy<O> {
    /// Maps the ordering of the current execution strategy using the given mapping.
    pub fn map_ordering(&mut self, mapping: &[usize]) {
        match self {
            ExecutionStrategy::Optimization { ordering, .. } => {
                for o in Arc::make_mut(ordering).iter_mut() {
                    *o = mapping[*o];
                }
            }
            ExecutionStrategy::Operations { ordering } => {
                for o in Arc::make_mut(ordering).iter_mut() {
                    *o = mapping[*o];
                }
            }
            ExecutionStrategy::Composed(items) => {
                for item in items.iter_mut() {
                    item.map_ordering(mapping);
                }
            }
        }
    }
}

enum BestOptimization {
    NotFound,
    Found { index: usize, score: u64 },
}

fn find_best_optimization_index<O>(
    optimizations: &mut [Box<dyn OperationFuser<O>>],
) -> BestOptimization {
    let mut best_index = BestOptimization::NotFound;
    let mut best_score = 0;

    for (i, optimization) in optimizations.iter().enumerate() {
        let properties = optimization.properties();

        // A score of zero is worse than fusing.
        if properties.ready && properties.score > best_score {
            best_index = BestOptimization::Found {
                index: i,
                score: properties.score,
            };
            best_score = properties.score;
        }
    }

    best_index
}

impl<O> PartialEq for Block<O> {
    fn eq(&self, other: &Self) -> bool {
        // Since the ordering can be seen as operation ids, we can use it to compare
        // blocks.
        let mut sorted_a = self.ordering.clone();
        let mut sorted_b = other.ordering.clone();
        sorted_a.sort();
        sorted_b.sort();

        sorted_a == sorted_b
    }
}

impl<O> core::fmt::Debug for Block<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Block {{ pos: [{:?}, {:?}; {:?}] }}",
            self.start_pos,
            self.end_pos,
            self.ordering.len(),
        ))
    }
}

impl<O> Clone for Block<O> {
    fn clone(&self) -> Self {
        Self {
            builders: self.builders.iter().map(|b| b.clone_dyn()).collect(),
            operations: self.operations.clone(),
            ids: self.ids.clone(),
            ordering: self.ordering.clone(),
            closed_at: self.closed_at.clone(),
            start_pos: self.start_pos,
            end_pos: self.end_pos,
        }
    }
}
