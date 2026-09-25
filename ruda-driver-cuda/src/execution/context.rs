use ruda_core::backtrace::BackTrace;
use ruda_compiler::cpp::formatter::format_cpp;
use ruda_compiler::cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};
use ruda::runtime::{
    compiler::CompilationError,
    validation::{validate_ruda_dim, validate_units},
};

use super::storage::gpu::GpuResource;
use crate::{CudaCompiler, execution::stream::Stream};
use crate::{
    CudaComputeKernel,
    install::{cccl_include_path, include_path},
};
use ruda_kernel::dsl::{
    compilation_cache::CompilationCache,
    hash::StableHash,
    server::ResourceLimitError,
    {ir::DeviceProperties, prelude::*},
};
use ruda::runtime::timestamp_profiler::TimestampProfiler;
use ruda::runtime::{compiler::RudaTask, logging::ServerLogger};
use cudarc::driver::DriverError;
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUtensorMap};
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};

use ruda_core::cache::CacheOption;

#[derive(Debug)]
pub(crate) struct CudaContext {
    backend: crate::compiler_backend::CompilerBackend,
    launch_arguments: launch::LaunchArguments,
    pub context: *mut CUctx_st,
    pub module_names: HashMap<KernelId, CompiledKernel>,
    ptx_cache: Option<CompilationCache<StableHash, PtxCacheEntry>>,
    pub timestamps: TimestampProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
}

#[derive(Debug)]
pub struct CompiledKernel {
    ruda_dim: RudaDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
    // Retained with the function; unload only after shutdown synchronization.
    module: cudarc::driver::sys::CUmodule,
    dynamic_metadata_index: Option<usize>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    ptx: Vec<std::ffi::c_char>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dynamic_metadata_index: Option<usize>,
}

impl CudaContext {
    pub fn new(
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        let backend = crate::compiler_backend::CompilerBackend::from_environment()
            .expect("invalid Ruda CUDA compiler configuration");
        #[cfg(feature = "direct-ptx")]
        let compilation_options = {
            let mut options = compilation_options;
            if matches!(backend, crate::compiler_backend::CompilerBackend::DirectPtx { .. }) {
                options.supports_features.grid_constants = true;
            }
            options
        };
        Self {
            backend,
            launch_arguments: launch::LaunchArguments::default(),
            context,
            module_names: HashMap::new(),
            ptx_cache: {
                use ruda::runtime::config::RuntimeConfig;
                let config = ruda::runtime::config::RudaRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "ptx",
                        CacheOption::default().name(backend.cache_namespace(arch.version)).root(root),
                    ))
                } else {
                    None
                }
            },
            arch,
            timestamps: TimestampProfiler::default(),
            compilation_options,
            properties,
        }
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }
}

mod compilation;
#[cfg(feature = "direct-ptx")]
mod direct_ptx;
mod launch;

mod module_config;

impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.module_names.is_empty() { return; }
        // This wait is ONLY at runtime destruction, never per operator.
        // Do not unload executable code while queued kernels may still use it.
        unsafe {
            use cudarc::driver::sys::*;
            let mut previous = std::ptr::null_mut();
            if cuCtxGetCurrent(&mut previous) != CUresult::CUDA_SUCCESS {
                log::warn!("Cannot obtain current context for RUDA module shutdown"); return;
            }
            if cuCtxSetCurrent(self.context) != CUresult::CUDA_SUCCESS {
                log::warn!("Cannot select context for RUDA module shutdown"); return;
            }
            let synchronized = cuCtxSynchronize();
            if synchronized == CUresult::CUDA_SUCCESS {
                for (_, kernel) in self.module_names.drain() {
                    let status = cuModuleUnload(kernel.module);
                    if status != CUresult::CUDA_SUCCESS {
                        log::warn!("RUDA module unload failed: {status:?}");
                    }
                }
            } else {
                // A failed wait does not prove that executable code is idle.
                // Leave cleanup to context teardown rather than risk unloading
                // code still in use, and make the exceptional leak visible.
                log::warn!("RUDA shutdown synchronization failed; modules retained: {synchronized:?}");
            }
            if cuCtxSetCurrent(previous) != CUresult::CUDA_SUCCESS {
                log::warn!("Cannot restore previous context after RUDA module shutdown");
            }
        }
    }
}
