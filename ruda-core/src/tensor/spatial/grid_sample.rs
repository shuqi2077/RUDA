use super::InterpolateMode;

/// Padding mode for grid sampling when coordinates are out of bounds.
///
/// Matches PyTorch's `padding_mode` parameter in `grid_sample`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
pub enum GridSamplePaddingMode {
    /// Fill with zeros for out-of-bounds coordinates.
    #[default]
    Zeros,
    /// Clamp coordinates to the border (use nearest edge value).
    Border,
    /// Reflect coordinates at the boundary.
    Reflection,
}

/// Options for grid sampling operations.
#[derive(Debug, Clone)]
pub struct GridSampleOptions {
    /// Interpolation mode (bilinear, nearest, or bicubic).
    pub mode: InterpolateMode,
    /// Padding mode for out-of-bounds coordinates.
    pub padding_mode: GridSamplePaddingMode,
    /// If `true`, grid values of -1 and 1 correspond to the corner pixels.
    /// If `false`, they correspond to the corner points of the corner pixels
    /// (i.e., -1 maps to -0.5 and 1 maps to size - 0.5 in pixel coordinates).
    pub align_corners: bool,
}

impl Default for GridSampleOptions {
    fn default() -> Self {
        Self {
            mode: InterpolateMode::Bilinear,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: false,
        }
    }
}

impl From<InterpolateMode> for GridSampleOptions {
    fn from(value: InterpolateMode) -> Self {
        GridSampleOptions::new(value)
    }
}

impl GridSampleOptions {
    /// Create new grid sample options with the given interpolation mode.
    ///
    /// Uses default values for padding_mode (Zeros) and align_corners (false).
    pub fn new(mode: InterpolateMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set the padding mode.
    pub fn with_padding_mode(mut self, padding_mode: GridSamplePaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }

    /// Set align_corners.
    pub fn with_align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}

