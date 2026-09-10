use super::*;


#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool1dOpIr {
    pub x: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool2dOpIr {
    pub x: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool1dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool2dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dOpIr {
    pub x: TensorIr,
    pub output_size: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dOpIr {
    pub x: TensorIr,
    pub output_size: [usize; 2],
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dOpIr {
    pub x: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesOpIr {
    pub x: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub ceil_mode: bool,
    pub out: TensorIr,
    pub out_indices: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub indices: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool2dOpIr {
    pub x: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[allow(missing_docs)]
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct MaxPool2dWithIndicesOpIr {
    pub x: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub ceil_mode: bool,
    pub out: TensorIr,
    pub out_indices: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool2dWithIndicesBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub indices: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub ceil_mode: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum InterpolateModeIr {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateOptionsIr {
    pub mode: InterpolateModeIr,
    pub align_corners: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateOpIr {
    pub x: TensorIr,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AttentionOptionsIr {
    pub scale: Option<ScalarIr>,
    pub softcap: Option<ScalarIr>,
    pub is_causal: bool,
}

impl From<AttentionOptionsIr> for AttentionModuleOptions {
    fn from(ir: AttentionOptionsIr) -> Self {
        AttentionModuleOptions {
            scale: ir.scale.map(|s| s.elem()),
            softcap: ir.softcap.map(|s| s.elem()),
            is_causal: ir.is_causal,
        }
    }
}

impl From<AttentionModuleOptions> for AttentionOptionsIr {
    fn from(ir: AttentionModuleOptions) -> Self {
        AttentionOptionsIr {
            scale: ir.scale.map(ScalarIr::Float),
            softcap: ir.softcap.map(ScalarIr::Float),
            is_causal: ir.is_causal,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AttentionOpIr {
    pub query: TensorIr,
    pub key: TensorIr,
    pub value: TensorIr,
    pub mask: Option<TensorIr>,
    pub attn_bias: Option<TensorIr>,
    pub options: AttentionOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CtcLossOpIr {
    pub log_probs: TensorIr,
    pub targets: TensorIr,
    pub input_lengths: TensorIr,
    pub target_lengths: TensorIr,
    pub blank: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CtcLossBackwardOpIr {
    pub log_probs: TensorIr,
    pub targets: TensorIr,
    pub input_lengths: TensorIr,
    pub target_lengths: TensorIr,
    pub grad_loss: TensorIr,
    pub blank: usize,
    pub out: TensorIr,
}

impl From<InterpolateModeIr> for InterpolateMode {
    fn from(val: InterpolateModeIr) -> Self {
        match val {
            InterpolateModeIr::Nearest => Self::Nearest,
            InterpolateModeIr::Bilinear => Self::Bilinear,
            InterpolateModeIr::Bicubic => Self::Bicubic,
            InterpolateModeIr::Lanczos3 => Self::Lanczos3,
        }
    }
}

impl From<InterpolateOptionsIr> for InterpolateOptions {
    fn from(val: InterpolateOptionsIr) -> Self {
        Self::new(val.mode.into()).with_align_corners(val.align_corners)
    }
}

impl From<InterpolateMode> for InterpolateModeIr {
    fn from(val: InterpolateMode) -> Self {
        match val {
            InterpolateMode::Nearest => Self::Nearest,
            InterpolateMode::Bilinear => Self::Bilinear,
            InterpolateMode::Bicubic => Self::Bicubic,
            InterpolateMode::Lanczos3 => Self::Lanczos3,
        }
    }
}

impl From<InterpolateOptions> for InterpolateOptionsIr {
    fn from(val: InterpolateOptions) -> Self {
        Self {
            mode: val.mode.into(),
            align_corners: val.align_corners,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum GridSamplePaddingModeIr {
    Zeros,
    Border,
    Reflection,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct GridSampleOptionsIr {
    pub mode: InterpolateModeIr,
    pub padding_mode: GridSamplePaddingModeIr,
    pub align_corners: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct GridSample2dOpIr {
    pub tensor: TensorIr,
    pub grid: TensorIr,
    pub options: GridSampleOptionsIr,
    pub out: TensorIr,
}

impl From<GridSamplePaddingModeIr> for GridSamplePaddingMode {
    fn from(val: GridSamplePaddingModeIr) -> Self {
        match val {
            GridSamplePaddingModeIr::Zeros => Self::Zeros,
            GridSamplePaddingModeIr::Border => Self::Border,
            GridSamplePaddingModeIr::Reflection => Self::Reflection,
        }
    }
}

impl From<GridSamplePaddingMode> for GridSamplePaddingModeIr {
    fn from(val: GridSamplePaddingMode) -> Self {
        match val {
            GridSamplePaddingMode::Zeros => Self::Zeros,
            GridSamplePaddingMode::Border => Self::Border,
            GridSamplePaddingMode::Reflection => Self::Reflection,
        }
    }
}

impl From<GridSampleOptionsIr> for GridSampleOptions {
    fn from(val: GridSampleOptionsIr) -> Self {
        Self {
            mode: val.mode.into(),
            padding_mode: val.padding_mode.into(),
            align_corners: val.align_corners,
        }
    }
}

impl From<GridSampleOptions> for GridSampleOptionsIr {
    fn from(val: GridSampleOptions) -> Self {
        Self {
            mode: val.mode.into(),
            padding_mode: val.padding_mode.into(),
            align_corners: val.align_corners,
        }
    }
}
