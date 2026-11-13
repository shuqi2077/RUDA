//! Stable inference configuration values and native runtime entry points.
use super::{
    InferenceError,
    activation::Activation,
    experts::{ExpertActivation, ExpertOptions, RoutingMethod},
    linear::ScaleLayout,
};
/// Scale-factor layouts use the stable public numeric discriminants.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SfLayout {
    Block128x4 = 0,
    Block8x4 = 1,
    Linear = 2,
}
impl From<SfLayout> for ScaleLayout {
    fn from(value: SfLayout) -> Self {
        match value {
            SfLayout::Block128x4 => Self::Block128x4,
            SfLayout::Block8x4 => Self::Block8x4,
            SfLayout::Linear => Self::Linear,
        }
    }
}
impl TryFrom<u32> for SfLayout {
    type Error = InferenceError;
    fn try_from(n: u32) -> Result<Self, Self::Error> {
        match n {
            0 => Ok(Self::Block128x4),
            1 => Ok(Self::Block8x4),
            2 => Ok(Self::Linear),
            _ => Err(InferenceError("unknown scale-factor layout")),
        }
    }
}
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationType {
    Gelu = 0,
    Relu = 1,
    Silu = 2,
    Swiglu = 3,
    Geglu = 4,
    SwigluBias = 5,
    Relu2 = 6,
    SwigluStep = 7,
    GegluTanh = 8,
    Identity = 9,
    Situ = 10,
    InvalidType = 11,
    ClampedRelu2 = 12,
    PowLu = 13,
}
impl TryFrom<u32> for ActivationType {
    type Error = InferenceError;
    fn try_from(n: u32) -> Result<Self, Self::Error> {
        match n {
            0 => Ok(Self::Gelu),
            1 => Ok(Self::Relu),
            2 => Ok(Self::Silu),
            3 => Ok(Self::Swiglu),
            4 => Ok(Self::Geglu),
            5 => Ok(Self::SwigluBias),
            6 => Ok(Self::Relu2),
            7 => Ok(Self::SwigluStep),
            8 => Ok(Self::GegluTanh),
            9 => Ok(Self::Identity),
            10 => Ok(Self::Situ),
            11 => Ok(Self::InvalidType),
            12 => Ok(Self::ClampedRelu2),
            13 => Ok(Self::PowLu),
            _ => Err(InferenceError("unknown expert activation type")),
        }
    }
}
impl ActivationType {
    pub const fn is_gated(self) -> bool {
        matches!(
            self,
            Self::Swiglu
                | Self::Geglu
                | Self::SwigluBias
                | Self::SwigluStep
                | Self::GegluTanh
                | Self::Situ
                | Self::PowLu
        )
    }
    pub fn expert_options(
        self,
        mut options: ExpertOptions,
    ) -> Result<ExpertOptions, InferenceError> {
        options.gated = self.is_gated();
        options.mode = match self {
            Self::SwigluBias => ExpertActivation::SwigluBias,
            Self::SwigluStep => ExpertActivation::SwigluStep,
            Self::Situ => ExpertActivation::Situ,
            Self::ClampedRelu2 => ExpertActivation::ClampedReluSquared,
            Self::PowLu => ExpertActivation::PowLu,
            _ => ExpertActivation::Standard,
        };
        options.activation = match self {
            Self::Gelu | Self::Geglu => Activation::Gelu,
            Self::Relu => Activation::Relu,
            Self::Silu
            | Self::Swiglu
            | Self::SwigluBias
            | Self::SwigluStep
            | Self::Situ
            | Self::PowLu => Activation::Silu,
            Self::Relu2 | Self::ClampedRelu2 => Activation::ReluSquared,
            Self::GegluTanh => Activation::GeluTanh,
            Self::Identity => Activation::Identity,
            Self::InvalidType => {
                return Err(InferenceError(
                    "InvalidType is not an executable activation",
                ));
            }
        };
        Ok(options)
    }
}
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingMethodType {
    Default = 0,
    Renormalize = 1,
    DeepSeekV3 = 2,
    Llama4 = 3,
    RenormalizeNaive = 4,
    TopK = 5,
    SigmoidRenorm = 6,
    MiniMax2 = 7,
    Sigmoid = 8,
    TopKSigmoid = 9,
    Unspecified = 10,
}
impl TryFrom<RoutingMethodType> for RoutingMethod {
    type Error = InferenceError;
    fn try_from(v: RoutingMethodType) -> Result<Self, Self::Error> {
        match v {
            RoutingMethodType::Default => Ok(Self::Default),
            RoutingMethodType::Renormalize => Ok(Self::Renormalize),
            RoutingMethodType::DeepSeekV3 => Ok(Self::DeepSeekV3),
            RoutingMethodType::Llama4 => Ok(Self::Llama4),
            RoutingMethodType::RenormalizeNaive => Ok(Self::RenormalizeNaive),
            RoutingMethodType::TopK => Ok(Self::TopK),
            RoutingMethodType::SigmoidRenorm => Ok(Self::SigmoidRenorm),
            RoutingMethodType::MiniMax2 => Ok(Self::MiniMax2),
            RoutingMethodType::Sigmoid => Ok(Self::Sigmoid),
            RoutingMethodType::TopKSigmoid => Ok(Self::TopKSigmoid),
            RoutingMethodType::Unspecified => Err(InferenceError(
                "Unspecified routing requires an explicit model routing policy",
            )),
        }
    }
}
pub const fn is_gated_activation(activation: ActivationType) -> bool {
    activation.is_gated()
}
/// Return one for zero, otherwise the smallest representable power >= n.
/// Overflow is reported instead of wrapping to zero.
pub fn next_positive_power_of_2(n: usize) -> Result<usize, InferenceError> {
    n.max(1)
        .checked_next_power_of_two()
        .ok_or(InferenceError("power-of-two size overflow"))
}
/// Native RUDA tuning, cache persistence and tracing are shared with ordinary
/// operators; inference does not maintain a second vendor-specific JIT system.
pub use ruda_kernel::dsl::tune as autotune;
