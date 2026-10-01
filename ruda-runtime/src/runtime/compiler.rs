use crate::runtime::{kernel::{CompiledKernel, KernelMetadata}, server::ExecutionMode};
use ruda_core::ir::StorageType;
pub use ruda_core::compiler::{CompilationError, Compiler};

/// Kernel trait with the `ComputeShader` that will be compiled and cached based on the
/// provided id.
pub trait RudaTask<C: Compiler>: KernelMetadata + Send + Sync {
    /// Export the shared IR when this task originates from a kernel definition.
    /// Source-only tasks do not have an IR representation.
    fn kernel_definition(&self) -> Option<ruda_core::kernel::KernelDefinition> {
        None
    }

    /// Compile a kernel and return the compiled form with an optional non-text representation
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError>;
}
