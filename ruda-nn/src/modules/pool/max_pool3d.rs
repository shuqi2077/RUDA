use crate::{PaddingConfig3d, padding::dilated_kernel_size};
use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend, module::max_pool3d_padded},
};

/// Configuration for maximum pooling over depth, height and width.
#[derive(Config, Debug)]
pub struct MaxPool3dConfig {
    /// Kernel extents in depth, height and width.
    pub kernel_size: [usize; 3],
    /// Strides, defaulting to the kernel extents.
    #[config(default = "kernel_size")]
    pub strides: [usize; 3],
    /// Symmetric, asymmetric, valid or dynamic same padding.
    #[config(default = "PaddingConfig3d::Valid")]
    pub padding: PaddingConfig3d,
    /// Spacing between sampled kernel elements.
    #[config(default = "[1, 1, 1]")]
    pub dilation: [usize; 3],
    /// Whether to include partially covered final windows.
    #[config(default = "false")]
    pub ceil_mode: bool,
}

/// Maximum pooling of native `[batch, channels, depth, height, width]` states.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct MaxPool3d {
    /// Strides in depth, height and width.
    pub stride: [usize; 3],
    /// Kernel extents.
    pub kernel_size: [usize; 3],
    /// Padding policy.
    pub padding: PaddingConfig3d,
    /// Kernel dilation.
    pub dilation: [usize; 3],
    /// Whether to include partially covered final windows.
    pub ceil_mode: bool,
}

impl MaxPool3dConfig {
    /// Construct the pooling layer without allocating model parameters.
    pub fn init(&self) -> MaxPool3d {
        MaxPool3d {
            stride: self.strides,
            kernel_size: self.kernel_size,
            padding: self.padding.clone(),
            dilation: self.dilation,
            ceil_mode: self.ceil_mode,
        }
    }
}

impl MaxPool3d {
    /// Reduce a volume using the configured native pooling operations.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_, _, depth, height, width] = input.dims();
        let effective = core::array::from_fn(|axis| {
            dilated_kernel_size(self.kernel_size[axis], self.dilation[axis])
        });
        let pairs = self.padding.calculate_padding_3d_pairs(
            &[depth, height, width], &effective, &self.stride,
        );
        max_pool3d_padded(input, self.kernel_size, self.stride, pairs, self.dilation, self.ceil_mode)
    }
}

impl ModuleDisplay for MaxPool3d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new().with_new_line_after_attribute(false).optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add_debug_attribute("kernel_size", &self.kernel_size)
            .add_debug_attribute("stride", &self.stride)
            .add_debug_attribute("padding", &self.padding)
            .add_debug_attribute("dilation", &self.dilation)
            .add("ceil_mode", &self.ceil_mode).optional()
    }
}
