use super::*;
use crate::execution::nvrtc_program::NvrtcProgram;

impl CudaContext {
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn RudaTask<CudaCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let hash = if let Some(cache) = &self.ptx_cache {
            let hash = kernel_id.stable_hash();

            if let Some(entry) = cache.get(&hash) {
                log::trace!("Using PTX cache");

                self.load_ptx(
                    entry.ptx.clone(),
                    kernel_id.clone(),
                    entry.entrypoint_name.clone(),
                    kernel_id.ruda_dim,
                    entry.shared_mem_bytes,
                    entry.dynamic_metadata_index,
                )?;
                return Ok(());
            }
            Some(hash)
        } else {
            None
        };

        log::trace!("Compiling kernel");

        validate_ruda_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        #[cfg(feature = "direct-ptx")]
        if let crate::compiler_backend::CompilerBackend::DirectPtx { version } = self.backend {
            return self.compile_direct_ptx(kernel_id, kernel.as_ref(), mode, logger, hash, version);
        }

        let mut kernel_compiled = kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            kernel.address_type(),
        )?;

        self.validate_shared(&kernel_compiled.repr)?;

        if logger.compilation_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let ruda_dim = kernel_compiled.ruda_dim;
        let arch = if self.arch.version >= 90 {
            format!("--gpu-architecture=sm_{}a", self.arch)
        } else {
            format!("--gpu-architecture=sm_{}", self.arch)
        };

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_path = cccl_include_path();
        let cccl_include_option = format!("--include-path={}", cccl_include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo", "--fmad=false"];
        if cccl_include_path.exists() {
            options.push(&cccl_include_option);
        }

        logger.log_compilation(&kernel_compiled);

        // SAFETY: Calling NVRTC FFI to create, compile, and extract PTX from a program.
        // The `CString` source is null-terminated and outlives the program. On compilation
        // failure, the error log is retrieved and reported before returning.
        // NvrtcProgram destroys the handle on every exit path.
        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::new(kernel_compiled.source.as_str()).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("CUDA source contains an interior NUL: {err}"),
                    backtrace: BackTrace::capture(),
                }
            })?;
            let program =
                NvrtcProgram::new(source).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;
            if cudarc::nvrtc::result::compile_program(program.raw(), &options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program.raw()).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;

                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_string_lossy();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                // Report the exact source submitted to NVRTC, without a second
                // compilation (which could itself fail or have side effects).
                let source = &kernel_compiled.source;
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                })?;
            };
            cudarc::nvrtc::result::get_ptx(program.raw()).map_err(|err| CompilationError::Generic {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?
        };

        let repr = kernel_compiled.repr.unwrap();

        if let Some(cache) = &mut self.ptx_cache {
            let result = cache.insert(
                hash.unwrap(),
                PtxCacheEntry {
                    entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                    shared_mem_bytes: repr.shared_memory_size(),
                    ptx: ptx.clone(),
                    dynamic_metadata_index: None,
                },
            );
            if let Err(err) = result {
                log::warn!("Unable to save the ptx {err:?}");
            }
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            ruda_dim,
            repr.shared_memory_size(),
            None,
        )?;
        Ok(())
    }

    pub(super) fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        ruda_dim: RudaDim,
        shared_mem_bytes: usize,
        dynamic_metadata_index: Option<usize>,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name).unwrap();
        // SAFETY: `ptx` is null-terminated PTX from the selected backend. `func_name` is a
        // null-terminated `CString` matching the kernel entry point in the compiled module.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the PTX: {err:?}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        self.module_names.insert(
            kernel_id.clone(),
            CompiledKernel {
                ruda_dim,
                shared_mem_bytes,
                func,
                dynamic_metadata_index,
            },
        );

        Ok(())
    }

    fn validate_shared(&self, repr: &Option<CudaComputeKernel>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size());
        let max = self.properties.hardware.max_shared_memory_size;
        if let Some(requested) = requested
            && requested > max
        {
            Err(ResourceLimitError::SharedMemory {
                requested,
                max,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}
