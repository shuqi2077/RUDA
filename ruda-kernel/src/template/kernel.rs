use super::SourceTemplate;
use alloc::string::ToString;
use crate::dsl::{CompilationError, Compiler, RudaTask, prelude::*};

/// Kernel source to create a [source](SourceTemplate)
pub trait KernelSource: Send + 'static + Sync {
    /// Convert to [source](SourceTemplate)
    fn source(&self) -> SourceTemplate;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;
}

#[derive(new)]
/// Wraps a [kernel source](KernelSource) into a [ruda task](RudaTask).
pub struct SourceKernel<K> {
    kernel_source: K,
    ruda_dim: RudaDim,
}

impl<C: Compiler, K: KernelSource> RudaTask<C> for SourceKernel<K> {
    fn compile(
        &self,
        _compiler: &mut C,
        _options: &C::CompilationOptions,
        _mode: ExecutionMode,
        _address_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let source_template = self.kernel_source.source();
        let source = source_template.complete();

        Ok(CompiledKernel {
            entrypoint_name: "main".to_string(),
            debug_name: Some(core::any::type_name::<K>()),
            source,
            ruda_dim: self.ruda_dim,
            debug_info: None,
            repr: None,
        })
    }
}

impl<K: KernelSource> KernelMetadata for SourceKernel<K> {
    fn id(&self) -> KernelId {
        self.kernel_source.id()
    }

    fn address_type(&self) -> StorageType {
        u32::as_type_native_unchecked().storage_type()
    }
}
