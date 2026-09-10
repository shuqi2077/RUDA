use ruda_core::config::RuntimeConfig;
use ruda_core::stub::Arc;

use super::autodiff::AutodiffConfig;
use super::fusion::FusionConfig;

/// Static mutex holding the global Ruda configuration, initialized as `None`.
static RUDA_GLOBAL_CONFIG: spin::Mutex<Option<Arc<RudaTensorConfig>>> = spin::Mutex::new(None);

/// Represents the global configuration for Ruda.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RudaTensorConfig {
    /// Configuration for operation fusion.
    #[serde(default)]
    pub fusion: FusionConfig,

    /// Configuration for autodiff.
    #[serde(default)]
    pub autodiff: AutodiffConfig,
}

impl RuntimeConfig for RudaTensorConfig {
    fn storage() -> &'static spin::Mutex<Option<Arc<Self>>> {
        &RUDA_GLOBAL_CONFIG
    }

    fn file_names() -> &'static [&'static str] {
        &["ruda-tensor.toml", "Ruda-Tensor.toml"]
    }

    // Match ruda-core's `std_io` cfg: only available on platforms where
    // the trait method exists. See ruda-core's build.rs.
    #[cfg(all(
        feature = "std",
        any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos",
            target_os = "android"
        )
    ))]
    fn override_from_env(mut self) -> Self {
        use super::fusion::FusionLogLevel;

        if let Ok(val) = std::env::var("RUDA_FUSION_LOG") {
            let level = match val.to_ascii_lowercase().as_str() {
                "disabled" | "off" | "0" => FusionLogLevel::Disabled,
                "basic" => FusionLogLevel::Basic,
                "medium" => FusionLogLevel::Medium,
                "full" | "1" => FusionLogLevel::Full,
                _ => self.fusion.logger.level,
            };
            self.fusion.logger.level = level;
            // Default to stderr so tests can see the output via `cargo test -- --nocapture`.
            if level != FusionLogLevel::Disabled {
                self.fusion.logger.stderr = true;
            }
        }

        self
    }
}
