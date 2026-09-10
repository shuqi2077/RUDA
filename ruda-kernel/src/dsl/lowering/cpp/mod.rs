use alloc::{boxed::Box, string::String, vec, vec::Vec};
use ruda_core::{ir::Processor, launch::ExecutionMode};
use ruda_compiler::cpp::shared::{CppCompiler, DialectProcessors};
use super::super::post_processing::{checked_io::CheckedIoProcessor, saturating::SaturatingArithmeticProcessor};

mod cuda;
mod hip;
use cuda::CudaMmaProcessor;
use hip::HipMmaProcessor;

pub type CudaCompiler<M> = CppCompiler<ruda_compiler::cpp::cuda::CudaDialect<M>, CudaProcessors>;
pub type HipCompiler<M> = CppCompiler<ruda_compiler::cpp::hip::HipDialect<M>, HipProcessors>;
#[cfg(feature = "lowering-metal")]
pub type MslCompiler = CppCompiler<ruda_compiler::cpp::metal::MslDialect, MetalProcessors>;

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaProcessors;

impl DialectProcessors for CudaProcessors {
    fn checked_io(mode: ExecutionMode, kernel_name: String) -> Box<dyn Processor> {
        Box::new(CheckedIoProcessor::new(mode, kernel_name))
    }

    fn processors() -> Vec<Box<dyn Processor>> {
        vec![
            Box::new(CudaMmaProcessor),
            Box::new(SaturatingArithmeticProcessor::new(false)),
        ]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HipProcessors;

impl DialectProcessors for HipProcessors {
    fn checked_io(mode: ExecutionMode, kernel_name: String) -> Box<dyn Processor> {
        Box::new(CheckedIoProcessor::new(mode, kernel_name))
    }

    fn processors() -> Vec<Box<dyn Processor>> {
        vec![
            Box::new(HipMmaProcessor),
            Box::new(SaturatingArithmeticProcessor::new(true)),
        ]
    }
}

#[cfg(feature = "lowering-metal")]
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalProcessors;

#[cfg(feature = "lowering-metal")]
impl DialectProcessors for MetalProcessors {
    fn checked_io(mode: ExecutionMode, kernel_name: String) -> Box<dyn Processor> {
        Box::new(CheckedIoProcessor::new(mode, kernel_name))
    }

    fn processors() -> Vec<Box<dyn Processor>> {
        Vec::new()
    }
}

#[cfg(feature = "lowering-metal")]
pub mod metal;
