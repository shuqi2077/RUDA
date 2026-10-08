use crate::PaddingConfig3d;
use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend, module::avg_pool3d_padded},
};

/// Configuration for average pooling over depth, height and width.
#[derive(Config, Debug)]
pub struct AvgPool3dConfig {
    /// Kernel extents in depth, height and width.
    pub kernel_size: [usize; 3],
    /// Strides, defaulting to the kernel extents.
    #[config(default = "kernel_size")]
    pub strides: [usize; 3],
    /// Symmetric, asymmetric, valid or dynamic same padding.
    #[config(default = "PaddingConfig3d::Valid")]
    pub padding: PaddingConfig3d,
    /// Whether explicit padding contributes to the average denominator.
    #[config(default = "true")]
    pub count_include_pad: bool,
    /// Whether to include partially covered final windows.
    #[config(default = "false")]
    pub ceil_mode: bool,
}

/// Average pooling of native `[batch, channels, depth, height, width]` states.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct AvgPool3d {
    /// Strides in depth, height and width.
    pub stride: [usize; 3],
    /// Kernel extents.
    pub kernel_size: [usize; 3],
    /// Padding policy.
    pub padding: PaddingConfig3d,
    /// Whether explicit padding contributes to the average denominator.
    pub count_include_pad: bool,
    /// Whether to include partially covered final windows.
    pub ceil_mode: bool,
}

impl AvgPool3dConfig {
    /// Construct the pooling layer without allocating model parameters.
    pub fn init(&self) -> AvgPool3d {
        AvgPool3d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            count_include_pad: self.count_include_pad,
            ceil_mode: self.ceil_mode,
        }
    }
}

impl AvgPool3d {
    /// Pool a volume, excluding asymmetric padding when requested.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_, _, depth, height, width] = input.dims();
        let pairs = self.padding.calculate_padding_3d_pairs(
            &[depth, height, width], &self.kernel_size, &self.stride,
        );
        avg_pool3d_padded(input, self.kernel_size, self.stride, pairs,
            self.count_include_pad, self.ceil_mode)
    }
}

impl ModuleDisplay for AvgPool3d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new().with_new_line_after_attribute(false).optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add_debug_attribute("kernel_size", &self.kernel_size)
            .add_debug_attribute("stride", &self.stride)
            .add_debug_attribute("padding", &self.padding)
            .add("count_include_pad", &self.count_include_pad)
            .add("ceil_mode", &self.ceil_mode).optional()
    }
}
