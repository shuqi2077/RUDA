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
