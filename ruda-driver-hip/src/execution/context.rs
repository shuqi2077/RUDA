use super::storage::gpu::GpuResource;
use crate::runtime::HipCompiler;
use crate::{execution::stream::Stream, runtime::HipComputeKernel};
use ruda_core::backtrace::BackTrace;
use ruda_core::cache::CacheOption;
use ruda_core::hash::StableHash;
use ruda_kernel::dsl::{
    compilation_cache::CompilationCache,
    server::ResourceLimitError,
    {ir::DeviceProperties, prelude::*},
};
use ruda_compiler::cpp::formatter::format_cpp;
use ruda_compiler::cpp::shared::CompilationOptions;
use ruda_hip_sys::{HIP_SUCCESS, get_hip_include_path, hiprtcResult_HIPRTC_SUCCESS};
use ruda::runtime::timestamp_profiler::TimestampProfiler;
use ruda::runtime::{
    compiler::CompilationError,
    validation::{validate_ruda_dim, validate_units},
};
use ruda::runtime::{compiler::RudaTask, logging::ServerLogger};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct HipContext {
    pub module_names: HashMap<KernelId, HipCompiledKernel>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<CompilationCache<StableHash, CompilationCacheEntry>>,
}

#[derive(Debug)]
pub struct HipCompiledKernel {
    _module: ruda_hip_sys::hipModule_t,
    func: ruda_hip_sys::hipFunction_t,
    ruda_dim: RudaDim,
    shared_mem_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    binary: Vec<i8>,
}

impl HipContext {
    pub fn new(compilation_options: CompilationOptions, properties: DeviceProperties) -> Self {
        Self {
            module_names: HashMap::new(),
            timestamps: TimestampProfiler::default(),
            compilation_options,
            compilation_cache: {
                use ruda::runtime::config::RuntimeConfig;
                let config = ruda::runtime::config::RudaRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "hip-kernel",
                        CacheOption::default().name("hip-subgroup-v11").root(root),
                    ))
                } else {
                    None
                }
            },
            properties,
        }
    }
}

mod compilation;
mod launch;
