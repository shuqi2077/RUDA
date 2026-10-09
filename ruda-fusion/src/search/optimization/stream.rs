use super::blocks::BlocksOptimizer;
use crate::{
    NumOperations, OperationFuser,
    search::{
        Block, BlockOptimization, RegistrationResult,
        graph::{Dag, GraphNode, OperationNode, TensorFlow, is_valid_execution_order},
        merging::{MergeBlocksResult, merge_blocks_with_guard},
        optimization::blocks::BlocksOptimizerResult,
    },
    stream::{execution::op_kind, store::ExecutionStrategy},
};
use ruda_tensor::graph::{OperationIr, TensorId};
use ruda_tensor_config::{config, fusion::FusionLogLevel, log_fusion};
use std::sync::Arc;

/// Optimize a stream of [operations](OperationIr) using a list of [builders](OptimizationBuilder).
pub struct StreamOptimizer<O> {
    builders: Vec<Box<dyn OperationFuser<O>>>,
    blocks: Vec<Block<O>>,
    length: usize,
    stopped: bool,
    max_blocks: Option<usize>,
}

impl<O: NumOperations> StreamOptimizer<O> {
    /// Create a new stream optimizer.
    pub fn new(builders: Vec<Box<dyn OperationFuser<O>>>) -> Self {
        // Too high and it may break the fusion cache always retriggering explorations.
        let max_blocks = Some(config().fusion.beam_search.max_blocks);
        Self {
            builders,
            blocks: Vec::new(),
            length: 0,
            stopped: false,
            max_blocks,
        }
    }

    /// Register a new [operation](OperationIr) in the optimizer.
    ///
    /// You can use the function [Self::still_optimizing] to know if the operations are actually
    /// being registered.
    pub fn register(&mut self, operation: &OperationIr) {
        if self.stopped {
            let length = self.length;
            log_fusion(FusionLogLevel::Full, || {
                format!(
                    "[stream] {} dropped (optimizer stopped at op {length})",
                    op_kind(operation)
                )
            });
            return;
        }

        if self.blocks.is_empty() {
            self.on_new_block(operation);
            self.length += 1;
            return;
        }

        match self.merge_blocks(operation, false) {
            MergeBlockStep::Full | MergeBlockStep::NoNeed => {}
            MergeBlockStep::Fail | MergeBlockStep::Partial => {
                self.on_dependent_op(operation);
                if !self.stopped {
                    self.length += 1;
                }
                return;
            }
        }

        if let Some(max_blocks) = self.max_blocks {
            if self.register_max_block(operation, max_blocks) {
                self.length += 1;
            } else {
                let length = self.length;
                log_fusion(FusionLogLevel::Medium, || {
                    format!(
                        "[stream] stopped (max_blocks={max_blocks} reached) at op {length} ({})",
                        op_kind(operation)
                    )
                });
                self.stopped = true;
            }
            return;
        }

        let added_count = self.register_inner(operation, false);
        if added_count == 0 {
            self.on_new_block(operation);
        } else {
            self.log_accepted(operation, added_count);
        }

        self.length += 1;
    }

    /// Optimize the current stream on the given [operations](OperationIr).
    ///
    /// # Notes
    ///
    /// The operations provided are the same as the ones used in the [register](Self::register)
    /// method, this simply remove the need for the current type to also keep track of the list of
    /// operations.
    pub fn optimize(&self, operations: &[OperationIr]) -> BlockOptimization<O> {
        let result = BlocksOptimizer::new(self.blocks.clone()).optimize();

        let optimization = match result {
            BlocksOptimizerResult::Full(block_optimization) => block_optimization,
            BlocksOptimizerResult::WithHoles {
                mut strategies,
                mut ordering,
                mut holes,
            } => {
                loop {
                    let mut search = self.new_empty_search();

                    let mut operations_holes = Vec::with_capacity(holes.len());

                    for index in holes.iter() {
                        let op = &operations[*index];
                        operations_holes.push(op.clone());
                        search.register(op);
                    }

                    let mut optimization_of_holes = search.optimize(&operations_holes);

                    optimization_of_holes.map_ordering(&holes);

                    strategies.push(Box::new(optimization_of_holes.strategy));
                    holes.drain(0..optimization_of_holes.ordering.len());
                    ordering.append(&mut optimization_of_holes.ordering);

                    if holes.is_empty() {
                        break;
                    }
                }

                BlockOptimization::new(ExecutionStrategy::Composed(strategies), ordering)
            }
        };
        repair_order(optimization, operations)
    }

    /// Reset the state of the optimizer.
    pub fn reset(&mut self) {
        self.builders.iter_mut().for_each(|b| b.reset());
        self.length = 0;
        self.blocks.clear();
        self.stopped = false;
    }

    /// Returns if some optimizations are still possible within the stream.
    pub fn still_optimizing(&self) -> bool {
        if self.stopped {
            return false;
        }
        if self.blocks.is_empty() {
            return true;
        }

        let mut num_stopped = 0;

        for block in self.blocks.iter() {
            if !block.still_optimizing() {
                num_stopped += 1
            }
        }

        num_stopped < self.blocks.len()
    }

    fn register_max_block(&mut self, operation: &OperationIr, max_blocks: usize) -> bool {
        if max_blocks == 1 {
            // Register in the single block with a force.
            self.register_inner(operation, true);
            return true;
        }
        let added_count = self.register_inner(operation, false);

        if added_count > 0 {
            self.log_accepted(operation, added_count);
            return true;
        }

        if added_count == 0 && self.blocks.len() < max_blocks {
            self.on_new_block(operation);
            return true;
        }

        self.merge_blocks(operation, true);

        if self.blocks.len() >= max_blocks {
            self.stopped = true;
            return false;
        }

        let added_count = self.register_inner(operation, false);

        if added_count == 0 {
            self.on_new_block(operation);
        } else {
            self.log_accepted(operation, added_count);
        }

        true
    }

    fn log_accepted(&self, operation: &OperationIr, added_count: usize) {
        let length = self.length;
        let num_blocks = self.blocks.len();
        log_fusion(FusionLogLevel::Full, || {
            format!(
                "[stream] op {length} {} → accepted in {added_count}/{num_blocks} block(s)",
                op_kind(operation)
            )
        });
    }

    fn register_inner(&mut self, operation: &OperationIr, force: bool) -> usize {
        let mut added_count = 0;
        for block in self.blocks.iter_mut() {
            match block.register(operation, self.length, force) {
                RegistrationResult::Accepted => {
                    added_count += 1;
                }
                RegistrationResult::NotPartOfTheGraph => {}
            }
        }
        added_count
    }

    fn new_empty_search(&self) -> Self {
        Self::new(
            self.builders
                .iter()
                .map(|b| {
                    let mut b = b.clone_dyn();
                    b.reset();
                    b
                })
                .collect(),
        )
    }

    fn merge_blocks(&mut self, operation: &OperationIr, all: bool) -> MergeBlockStep {
        let nodes = operation.nodes();
        let ordered = OperationNode { operation, position: self.length }.ordered_range().is_some();
        let mut block_merges = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            if all || block.contains_tensors(&nodes) || (ordered && block.ordered_range().is_some()) {
                block_merges.push(i);
            }
        }

        if block_merges.len() <= 1 {
            return MergeBlockStep::NoNeed;
        }

        for (index, block) in self.blocks.iter_mut().enumerate() {
            block.seed_constituent(index);
        }
        let guard = Dag::new(&self.blocks).reachability();

        let blocks_to_merge = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, g)| match block_merges.contains(&i) {
                true => Some(g),
                false => None,
            })
            .collect::<Vec<_>>();

        let merged = merge_blocks_with_guard(&blocks_to_merge, false, &guard);

        let mut clear_blocks = || {
            let mut indices = block_merges.to_vec();
            indices.sort();

            for g in indices.into_iter().rev() {
                self.blocks.remove(g);
            }
        };

        match merged {
            MergeBlocksResult::Full(block) => {
                clear_blocks();
                self.blocks.push(block);
                Block::sort(&mut self.blocks);
                MergeBlockStep::Full
            }
            MergeBlocksResult::Partial {
                mut merged,
                mut failed,
            } => {
                clear_blocks();
                self.blocks.append(&mut merged);
                self.blocks.append(&mut failed);
                Block::sort(&mut self.blocks);
                MergeBlockStep::Partial
            }
            MergeBlocksResult::Fail => MergeBlockStep::Fail,
        }
    }

    fn on_new_block(&mut self, operation: &OperationIr) {
        let mut block = Block::new(&self.builders);
        block.register(operation, self.length, true);
        self.blocks.push(block);

        let length = self.length;
        let num_blocks = self.blocks.len();
        log_fusion(FusionLogLevel::Full, || {
            format!(
                "[stream] op {length} {} → new block (total: {num_blocks})",
                op_kind(operation)
            )
        });
    }

    fn on_dependent_op(&mut self, operation: &OperationIr) {
        if let Some(max_blocks) = self.max_blocks
            && self.blocks.len() >= max_blocks
        {
            self.merge_blocks(operation, true);
            if self.blocks.len() >= max_blocks {
                self.stopped = true;
                return;
            }
        }

        let mut block = Block::new(&self.builders);
        block.register(operation, self.length, true);
        self.blocks.push(block);
        if !Dag::new(&self.blocks).is_acyclic() {
            self.blocks.pop();
            self.stopped = true;
        }
    }
}

enum MergeBlockStep {
    Full,
    Partial,
    Fail,
    NoNeed,
}

fn repair_order<O>(optimization: BlockOptimization<O>, operations: &[OperationIr]) -> BlockOptimization<O> {
    if ordering_is_valid(&optimization.ordering, operations) {
        return optimization;
    }
    let strategies = match optimization.strategy {
        ExecutionStrategy::Composed(items) => items,
        _ => return unfused_stream_order(optimization.ordering),
    };
    let mut flattened = Vec::with_capacity(strategies.len());
    flatten_strategies(strategies, &mut flattened);
    let mut chunks = Vec::with_capacity(flattened.len());
    let mut offset = 0;
    for strategy in &flattened {
        let len = strategy_len(strategy);
        chunks.push(Chunk::new(optimization.ordering[offset..offset + len].to_vec(), operations));
        offset += len;
    }
    if let Some(order) = Dag::new(&chunks).topological_order() {
        return assemble(flattened, &chunks, &order, operations);
    }

    let mut split_strategies = Vec::new();
    let mut split_chunks = Vec::new();
    for (strategy, chunk) in flattened.into_iter().zip(chunks) {
        if matches!(*strategy, ExecutionStrategy::Operations { .. }) {
            for position in chunk.positions {
                split_strategies.push(Box::new(ExecutionStrategy::Operations {
                    ordering: Arc::new(vec![position]),
                }));
                split_chunks.push(Chunk::new(vec![position], operations));
            }
        } else {
            split_strategies.push(strategy);
            split_chunks.push(chunk);
        }
    }
    match Dag::new(&split_chunks).topological_order() {
        Some(order) => assemble(split_strategies, &split_chunks, &order, operations),
        None => unfused_stream_order(split_chunks.into_iter().flat_map(|chunk| chunk.positions).collect()),
    }
}

fn flatten_strategies<O>(items: Vec<Box<ExecutionStrategy<O>>>, output: &mut Vec<Box<ExecutionStrategy<O>>>) {
    for item in items {
        if matches!(*item, ExecutionStrategy::Composed(_)) {
            if let ExecutionStrategy::Composed(nested) = *item {
                flatten_strategies(nested, output);
            }
        } else {
            output.push(item);
        }
    }
}

fn assemble<O>(strategies: Vec<Box<ExecutionStrategy<O>>>, chunks: &[Chunk], order: &[usize],
    operations: &[OperationIr]) -> BlockOptimization<O> {
    let mut slots: Vec<_> = strategies.into_iter().map(Some).collect();
    let mut strategies = Vec::with_capacity(order.len());
    let mut ordering = Vec::with_capacity(chunks.iter().map(|chunk| chunk.positions.len()).sum());
    for &index in order {
        strategies.push(slots[index].take().expect("each strategy taken once"));
        ordering.extend_from_slice(&chunks[index].positions);
    }
    if !ordering_is_valid(&ordering, operations) {
        return unfused_stream_order(ordering);
    }
    BlockOptimization::new(ExecutionStrategy::Composed(strategies), ordering)
}

fn unfused_stream_order<O>(mut ordering: Vec<usize>) -> BlockOptimization<O> {
    ordering.sort_unstable();
    BlockOptimization::new(ExecutionStrategy::Operations { ordering: Arc::new(ordering.clone()) }, ordering)
}

fn ordering_is_valid(ordering: &[usize], operations: &[OperationIr]) -> bool {
    is_valid_execution_order(ordering.iter().map(|&position| OperationNode {
        operation: &operations[position], position,
    }))
}

fn strategy_len<O>(strategy: &ExecutionStrategy<O>) -> usize {
    match strategy {
        ExecutionStrategy::Optimization { ordering, .. } | ExecutionStrategy::Operations { ordering } => ordering.len(),
        ExecutionStrategy::Composed(items) => items.iter().map(|item| strategy_len(item)).sum(),
    }
}

struct Chunk {
    positions: Vec<usize>,
    flow: TensorFlow,
}

impl Chunk {
    fn new(positions: Vec<usize>, operations: &[OperationIr]) -> Self {
        let mut flow = TensorFlow::new();
        for &position in &positions {
            flow.register(&operations[position], position);
        }
        Self { positions, flow }
    }
}

impl GraphNode for Chunk {
    type Resource = TensorId;
    fn produced(&self) -> impl Iterator<Item = TensorId> { self.flow.produced() }
    fn read(&self) -> impl Iterator<Item = TensorId> { self.flow.read() }
    fn freed(&self) -> impl Iterator<Item = TensorId> { self.flow.freed() }
    fn produces(&self, resource: TensorId) -> bool { self.flow.produces(resource) }
    fn reads(&self, resource: TensorId) -> bool { self.flow.reads(resource) }
    fn position(&self) -> usize { self.flow.position() }
    fn read_position(&self, resource: TensorId) -> usize { self.flow.read_position(resource) }
    fn produces_before(&self, resource: TensorId, position: usize) -> bool { self.flow.produces_before(resource, position) }
    fn ordered_range(&self) -> Option<(usize, usize)> { self.flow.ordered_range() }
}
