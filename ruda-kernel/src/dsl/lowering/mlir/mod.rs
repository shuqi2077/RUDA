use ruda_core::{kernel::KernelDefinition, launch::ExecutionMode};
use ruda_compiler::optimizer::{Optimizer, OptimizerBuilder};
use crate::dsl::post_processing::{checked_io::CheckedIoProcessor, predicate::PredicateProcessor, saturating::SaturatingArithmeticProcessor};

mod erf_transform;
mod trigonometries_transform;
use erf_transform::ErfTransform;
use trigonometries_transform::{HypotTransform, RhypotTransform};

pub fn optimize(kernel: &KernelDefinition, mode: ExecutionMode) -> Optimizer {
    OptimizerBuilder::default()
            .with_transformer(ErfTransform)
            .with_transformer(HypotTransform)
            .with_transformer(RhypotTransform)
            .with_processor(CheckedIoProcessor::new(
                mode,
                kernel.options.kernel_name.clone(),
            ))
            .with_processor(SaturatingArithmeticProcessor::new(true))
            .with_processor(PredicateProcessor)
            .optimize(kernel.body.clone(), kernel.ruda_dim)
}
