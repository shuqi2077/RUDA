use crate::runtime::config::memory::MemoryConfig;
use crate::runtime::config::streaming::StreamingConfig;

use super::{autotune::AutotuneConfig, compilation::CompilationConfig, profiling::ProfilingConfig};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use ruda_core::config::RuntimeConfig;

/// Static mutex holding the global configuration, initialized as `None`.
static RUDA_GLOBAL_CONFIG: spin::Mutex<Option<Arc<RudaRuntimeConfig>>> = spin::Mutex::new(None);

/// Represents the global configuration for `Ruda`, combining profiling, autotuning, and compilation settings.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RudaRuntimeConfig {
    /// Configuration for profiling `Ruda` operations.
    #[serde(default)]
    pub profiling: ProfilingConfig,

    /// Configuration for autotuning performance parameters.
    #[serde(default)]
    pub autotune: AutotuneConfig,

    /// Configuration for compilation settings.
    #[serde(default)]
    pub compilation: CompilationConfig,

    /// Configuration for streaming settings.
    #[serde(default)]
    pub streaming: StreamingConfig,

    /// Configuration for memory settings.
    #[serde(default)]
    pub memory: MemoryConfig,
}

impl RuntimeConfig for RudaRuntimeConfig {
    fn storage() -> &'static spin::Mutex<Option<Arc<Self>>> {
        &RUDA_GLOBAL_CONFIG
    }

    fn file_names() -> &'static [&'static str] {
        &["ruda.toml", "Ruda.toml"]
    }

    fn section_file_names() -> &'static [(&'static str, &'static str)] {
        &[("ruda-tensor.toml", "runtime"), ("Ruda-Tensor.toml", "runtime")]
    }

    #[cfg(std_io)]
    fn override_from_env(mut self) -> Self {
        use super::compilation::CompilationLogLevel;
        use crate::runtime::config::{
            autotune::{AutotuneLevel, AutotuneLogLevel},
            profiling::ProfilingLogLevel,
        };

        if let Ok(val) = std::env::var("RUDA_DEBUG_LOG") {
            self.compilation.logger.level = CompilationLogLevel::Full;
            self.profiling.logger.level = ProfilingLogLevel::Medium;
            self.autotune.logger.level = AutotuneLogLevel::Full;

            match val.as_str() {
                "stdout" => {
                    self.compilation.logger.stdout = true;
                    self.profiling.logger.stdout = true;
                    self.autotune.logger.stdout = true;
                }
                "stderr" => {
                    self.compilation.logger.stderr = true;
                    self.profiling.logger.stderr = true;
                    self.autotune.logger.stderr = true;
                }
                "1" | "true" => {
                    let file_path = "/tmp/ruda.log";
                    self.compilation.logger.file = Some(file_path.into());
                    self.profiling.logger.file = Some(file_path.into());
                    self.autotune.logger.file = Some(file_path.into());
                }
                "0" | "false" => {
                    self.compilation.logger.level = CompilationLogLevel::Disabled;
                    self.profiling.logger.level = ProfilingLogLevel::Disabled;
                    self.autotune.logger.level = AutotuneLogLevel::Disabled;
                }
                file_path => {
                    self.compilation.logger.file = Some(file_path.into());
                    self.profiling.logger.file = Some(file_path.into());
                    self.autotune.logger.file = Some(file_path.into());
                }
            }
        };

        if let Ok(val) = std::env::var("RUDA_DEBUG_OPTION") {
            match val.as_str() {
                "debug" => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                    self.profiling.logger.level = ProfilingLogLevel::Medium;
                    self.autotune.logger.level = AutotuneLogLevel::Full;
                }
                "debug-full" => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                    self.profiling.logger.level = ProfilingLogLevel::Full;
                    self.autotune.logger.level = AutotuneLogLevel::Full;
                }
                "profile" => {
                    self.profiling.logger.level = ProfilingLogLevel::Basic;
                }
                "profile-medium" => {
                    self.profiling.logger.level = ProfilingLogLevel::Medium;
                }
                "profile-full" => {
                    self.profiling.logger.level = ProfilingLogLevel::Full;
                }
                _ => {}
            }
        };

        if let Ok(val) = std::env::var("RUDA_AUTOTUNE_LEVEL") {
            match val.as_str() {
                "minimal" | "0" => {
                    self.autotune.level = AutotuneLevel::Minimal;
                }
                "balanced" | "1" => {
                    self.autotune.level = AutotuneLevel::Balanced;
                }
                "extensive" | "2" => {
                    self.autotune.level = AutotuneLevel::Extensive;
                }
                "full" | "3" => {
                    self.autotune.level = AutotuneLevel::Full;
                }
                _ => {}
            }
        }

        self
    }
}

#[derive(Clone, Copy, Debug)]
/// How to format ruda type names.
pub enum TypeNameFormatLevel {
    /// No formatting apply, full information is included.
    Full,
    /// Most information is removed for a small formatted name.
    Short,
    /// Balanced info is kept.
    Balanced,
}

/// Format a type name with different options.
pub fn type_name_format(name: &str, level: TypeNameFormatLevel) -> String {
    match level {
        TypeNameFormatLevel::Full => name.to_string(),
        TypeNameFormatLevel::Short => {
            if let Some(val) = name.split("<").next() {
                val.split("::").last().unwrap_or(name).to_string()
            } else {
                name.to_string()
            }
        }
        TypeNameFormatLevel::Balanced => {
            let mut split = name.split("<");
            let before_generic = split.next();
            let after_generic = split.next();

            let before_generic = match before_generic {
                None => return name.to_string(),
                Some(val) => val
                    .split("::")
                    .last()
                    .unwrap_or(val)
                    .trim()
                    .replace(">", "")
                    .to_string(),
            };
            let inside_generic = match after_generic {
                None => return before_generic.to_string(),
                Some(val) => {
                    let mut val = val.to_string();
                    for s in split {
                        val += "<";
                        val += s;
                    }
                    val
                }
            };

            let inside = type_name_list_format(&inside_generic, level);

            format!("{before_generic}{inside}")
        }
    }
}

fn type_name_list_format(name: &str, level: TypeNameFormatLevel) -> String {
    let mut acc = String::new();
    let splits = name.split(", ");

    for a in splits {
        acc += " | ";
        acc += &type_name_format(a, level);
    }

    acc
}

#[cfg(test)]
mod test {
    use super::*;

    #[test_log::test]
    fn test_format_name() {
        let full_name = "ruda_tensor_device::kernel::unary_numeric::unary_numeric::UnaryNumeric<f32, ruda_tensor_device::tensor::base::RudaTensor<_>::copy::Copy, ruda_driver_cuda::runtime::CudaRuntime>";
        let name = type_name_format(full_name, TypeNameFormatLevel::Balanced);

        assert_eq!(name, "UnaryNumeric | f32 | RudaTensor | Copy | CudaRuntime");
    }
}
