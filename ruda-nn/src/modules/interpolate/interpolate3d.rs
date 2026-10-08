use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend, module::interpolate3d, ops::InterpolateOptions},
};
use super::InterpolateMode;

/// Configuration for interpolation along depth, height and width.
#[derive(Config, Debug)]
pub struct Interpolate3dConfig {
    /// Explicit output extents, taking precedence over scale factors.
    #[config(default = "None")]
    pub output_size: Option<[usize; 3]>,
    /// Per-axis scale factors when no explicit output extents are supplied.
    #[config(default = "None")]
    pub scale_factor: Option<[f32; 3]>,
    /// Native separable interpolation filter.
    #[config(default = "InterpolateMode::Nearest")]
    pub mode: InterpolateMode,
    /// Corner alignment, matching the existing interpolation modules.
    #[config(default = true)]
    pub align_corners: bool,
}

/// Interpolation of `[batch, channels, depth, height, width]` activations.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Interpolate3d {
    /// Explicit output depth, height and width.
    pub output_size: Option<[usize; 3]>,
    /// Per-axis scale factors when output size is absent.
    pub scale_factor: Option<[f32; 3]>,
    /// Native separable filter.
    pub mode: InterpolateMode,
    /// Whether to align interpolation corners.
    pub align_corners: bool,
}

impl Interpolate3dConfig {
    /// Construct the layer without allocating model parameters.
    pub fn init(self) -> Interpolate3d {
        Interpolate3d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: self.mode,
            align_corners: self.align_corners,
        }
    }
}

impl Interpolate3d {
    /// Resize a volume on its native backend, preserving its gradient graph.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let output = if let Some(size) = self.output_size {
            assert!(size.iter().all(|size| *size > 0), "interpolation output extents must be non-zero");
            size
        } else {
            let factors = self.scale_factor.expect("Either output_size or scale_factor must be provided");
            let [_, _, depth, height, width] = input.dims();
            let sizes = [depth, height, width];
            core::array::from_fn(|axis| {
                let factor = factors[axis];
                assert!(factor.is_finite() && factor > 0.0, "interpolation scale factors must be finite and positive");
                let size = (sizes[axis] as f64) * (factor as f64);
                assert!(size >= 1.0 && size < usize::MAX as f64,
                    "interpolation scale factor produces an empty or overflowing output extent");
                size as usize
            })
        };
        interpolate3d(input, output,
            InterpolateOptions::new(self.mode.clone().into()).with_align_corners(self.align_corners))
    }
}

impl ModuleDisplay for Interpolate3d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new().with_new_line_after_attribute(false).optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add_debug_attribute("mode", &self.mode)
            .add_debug_attribute("output_size", &self.output_size)
            .add_debug_attribute("scale_factor", &self.scale_factor)
            .add("align_corners", &self.align_corners).optional()
    }
}
