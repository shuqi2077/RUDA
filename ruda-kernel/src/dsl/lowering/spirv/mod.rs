use ruda_core::{kernel::KernelDefinition, launch::ExecutionMode, compiler::WgpuCompilationOptions};
use ruda_compiler::{optimizer::{Optimizer, OptimizerBuilder}, spirv::{SpirvCompiler, SpirvTarget, SpirvLowering, MAX_VECTORIZATION}};
use ruda::runtime::config::{RudaRuntimeConfig, RuntimeConfig, compilation::CompilationLogLevel};
use crate::dsl::post_processing::{checked_io::CheckedIoProcessor, saturating::SaturatingArithmeticProcessor, unroll::UnrollProcessor};
mod bitwise;
mod transformers;
use transformers::{BitwiseTransform, ErfTransform, HypotTransform, IntegerPowerTransform, RhypotTransform};

pub fn compiler<T: SpirvTarget>() -> SpirvCompiler<T> {
    SpirvCompiler::new(SpirvLowering { debug_symbols: debug_symbols_activated, optimize })
}

fn debug_symbols_activated() -> bool {
    matches!(
        RudaRuntimeConfig::get().compilation.logger.level,
        CompilationLogLevel::Full
    )
}

fn optimize(kernel: &KernelDefinition, mode: ExecutionMode, compilation_options: &WgpuCompilationOptions) -> Optimizer {
    OptimizerBuilder::default()
            .with_transformer(IntegerPowerTransform)
            .with_transformer(ErfTransform)
            .with_transformer(BitwiseTransform::new(
                compilation_options.vulkan.supports_arbitrary_bitwise,
            ))
            .with_transformer(HypotTransform)
            .with_transformer(RhypotTransform)
            .with_processor(CheckedIoProcessor::new(
                mode,
                kernel.options.kernel_name.clone(),
            ))
            .with_processor(UnrollProcessor::new(MAX_VECTORIZATION))
            .with_processor(SaturatingArithmeticProcessor::new(true))
            .optimize(kernel.body.clone(), kernel.ruda_dim)
}
