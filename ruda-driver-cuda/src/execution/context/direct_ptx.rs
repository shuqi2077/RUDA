use super::*;
use ruda_compiler::ptx::{PtxCompilationOptions, PtxCompiler, PtxTarget};
use ruda_core::compiler::Compiler;

impl CudaContext {
    pub(super) fn compile_direct_ptx(
        &mut self,
        kernel_id: &KernelId,
        kernel: &dyn RudaTask<CudaCompiler>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
        hash: Option<StableHash>,
        version: (u32, u32),
    ) -> Result<(), LaunchError> {
        let definition = kernel.kernel_definition().ok_or_else(|| CompilationError::UnsupportedInstruction {
            reason: "Direct PTX requires Kernel IR; source-only tasks cannot be compiled by this backend".into(),
            backtrace: BackTrace::capture(),
        })?;
        let representation = PtxCompiler.compile(
            definition,
            &PtxCompilationOptions {
                target: Some(PtxTarget {
                    version,
                    sm: self.arch.version,
                }),
            },
            mode,
            kernel.address_type(),
        )?;
        let shared_mem_bytes = representation.shared_memory_bytes;
        let dynamic_metadata_index = representation.dynamic_metadata_index;
        let max = self.properties.hardware.max_shared_memory_size;
        if shared_mem_bytes > max {
            return Err(ResourceLimitError::SharedMemory {
                requested: shared_mem_bytes,
                max,
                backtrace: BackTrace::capture(),
            }
            .into());
        }
        let compiled = ruda::runtime::kernel::CompiledKernel::<PtxCompiler> {
            entrypoint_name: representation.entrypoint.clone(),
            debug_name: Some(kernel.name()),
            source: representation.source.clone(),
            ruda_dim: representation.ruda_dim,
            repr: Some(representation),
            debug_info: logger
                .compilation_activated()
                .then(|| DebugInformation::new("ptx", kernel_id.clone())),
        };
        logger.log_compilation(&compiled);
        let ptx: Vec<c_char> = CString::new(compiled.source)
            .map_err(|err| CompilationError::Validation {
                reason: format!("Invalid PTX source: {err}"),
                backtrace: BackTrace::capture(),
            })?
            .into_bytes_with_nul()
            .into_iter()
            .map(|byte| byte as c_char)
            .collect();
        if let (Some(cache), Some(hash)) = (&mut self.ptx_cache, hash) {
            if let Err(err) = cache.insert(
                hash,
                PtxCacheEntry {
                    entrypoint_name: compiled.entrypoint_name.clone(),
                    shared_mem_bytes,
                    ptx: ptx.clone(),
                    dynamic_metadata_index,
                },
            ) {
                log::warn!("Unable to save the direct PTX {err:?}");
            }
        }
        self.load_ptx(
            ptx,
            kernel_id.clone(),
            compiled.entrypoint_name,
            compiled.ruda_dim,
            shared_mem_bytes,
            dynamic_metadata_index,
        )?;
        Ok(())
    }
}
