//! Optional backend extension for frozen, directly packed AWQ projections.
use crate::{Backend, tensor::{FloatTensor, IntTensor}};
use core::fmt::{Debug, Display, Formatter};

/// A native backend failure or a violation of the explicitly frozen base contract.
#[derive(Debug)]
pub enum FrozenAwqError<E: Debug> {
    /// Original backend error, without dense fallback or requantization.
    Native(E),
    /// The caller supplied a trainable base scale or bias to a frozen operator.
    TrainableBase,
}
impl<E: Debug> Display for FrozenAwqError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Native(error) => write!(f, "native AWQ: {error:?}"),
            Self::TrainableBase => f.write_str("frozen AWQ scales and bias must not require gradients"),
        }
    }
}
impl<E: Debug> core::error::Error for FrozenAwqError<E> {}

/// Direct AWQ `[K,N/8]` I32 words, `[K/group,N/8]` I32 zero points and
/// `[K/group,N]` FP16/BF16/FP32 scales. The AWQ nibble order and scale-dtype
/// coefficient rounding are preserved. This is not NF4 or a generic QAT STE.
/// Backends implement this explicitly; there is no dequantized default.
pub trait FrozenAwqOps: Backend {
    /// Native validation/launch failure, including autodiff frozen-base validation.
    type AwqError: Debug;

    /// Native packed projection, optionally biased. Output retains input dtype
    /// and leading axes; input storage may differ from original scale storage.
    fn frozen_awq_forward(
        input: FloatTensor<Self>, qweight: IntTensor<Self>, qzeros: IntTensor<Self>,
        scales: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, group_size: usize,
    ) -> Result<FloatTensor<Self>, Self::AwqError>;

    /// Apply the transpose of the same rounded packed coefficients, retaining
    /// gradient dtype/leading axes. No weight/scales/bias gradients are implied.
    fn frozen_awq_input_backward(
        gradient: FloatTensor<Self>, qweight: IntTensor<Self>, qzeros: IntTensor<Self>,
        scales: FloatTensor<Self>, group_size: usize,
    ) -> Result<FloatTensor<Self>, Self::AwqError>;
}
