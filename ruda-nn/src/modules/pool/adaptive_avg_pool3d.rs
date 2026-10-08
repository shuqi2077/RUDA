use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend, module::adaptive_avg_pool3d},
};

/// Configuration for adaptive volume average pooling.
#[derive(Config, Debug)]
pub struct AdaptiveAvgPool3dConfig {
    /// Requested output depth, height and width.
    pub output_size: [usize; 3],
}

/// Adaptive average pooling of `[batch, channels, depth, height, width]` states.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct AdaptiveAvgPool3d {
    /// Requested output depth, height and width.
    pub output_size: [usize; 3],
}

impl AdaptiveAvgPool3dConfig {
    /// Construct the layer without allocating model parameters.
    pub fn init(&self) -> AdaptiveAvgPool3d {
        AdaptiveAvgPool3d { output_size: self.output_size }
    }
}

impl AdaptiveAvgPool3d {
    /// Reduce a volume to the configured spatial extents on its native backend.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        adaptive_avg_pool3d(input, self.output_size)
    }
}

impl ModuleDisplay for AdaptiveAvgPool3d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new().with_new_line_after_attribute(false).optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add_debug_attribute("output_size", &self.output_size).optional()
    }
}
