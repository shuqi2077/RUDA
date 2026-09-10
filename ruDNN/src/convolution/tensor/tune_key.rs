use ruda_kernel::dsl as kernel_dsl;
use {ruda_core::tensor::DType};
use {ruda_kernel::dsl::AutotuneKey};
use {serde::Deserialize, serde::Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct ConvAutotuneKey {
    pub kernel_size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    pub groups: usize,
    #[autotune(anchor)]
    pub in_channels: usize,
    #[autotune(anchor)]
    pub out_channels: usize,
    pub shape: Vec<usize>,
    #[autotune(anchor)]
    pub batch_size: usize,
    pub has_bias: bool,
    pub dtype: DType,

    pub lhs_shape_align: u8,
    pub lhs_stride_align: u8,
    pub rhs_shape_align: u8,
    pub rhs_stride_align: u8,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct ConvTranspose2dAutotuneKey {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub padding_out: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    #[autotune(anchor)]
    pub in_channels: usize,
    #[autotune(anchor)]
    pub out_channels: usize,
    #[autotune(anchor)]
    pub height: usize,
    #[autotune(anchor)]
    pub width: usize,
    #[autotune(anchor)]
    pub batch_size: usize,
    pub has_bias: bool,
    pub dtype: DType,
}

#[cfg(feature = "tensor-convolution-autotune")]
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(crate) enum ConvTuneKey {
    Conv(ConvAutotuneKey),
}

#[cfg(feature = "tensor-convolution-autotune")]
impl core::fmt::Display for ConvTuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Conv(key) => core::fmt::Debug::fmt(key, f),
        }
    }
}

#[cfg(feature = "tensor-convolution-autotune")]
impl ruda_kernel::dsl::tune::AutotuneKey for ConvTuneKey {}

#[cfg(feature = "tensor-convolution-autotune")]
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(crate) enum ConvTransposeTuneKey {
    ConvTranspose(ConvTranspose2dAutotuneKey),
}

#[cfg(feature = "tensor-convolution-autotune")]
impl core::fmt::Display for ConvTransposeTuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ConvTranspose(key) => core::fmt::Debug::fmt(key, f),
        }
    }
}

#[cfg(feature = "tensor-convolution-autotune")]
impl ruda_kernel::dsl::tune::AutotuneKey for ConvTransposeTuneKey {}
