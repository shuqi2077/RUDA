//! Backend extension for the original RUDA high-nibble-first NF4 format.
use crate::{Backend,FloatDType,tensor::{FloatTensor,IntTensor}};
use core::fmt;

/// Explicit original geometry and execution choices; no quantizer or model family is inferred.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Nf4ProjectionOptions {
    /// Actual logical input width.
    pub input_features:usize,
    /// Actual logical output width, with `[output,input]` packed row-major storage.
    pub output_features:usize,
    /// Original positive even flat block size, including partial final blocks.
    pub block_size:usize,
    /// Maximum decoded output-row tile on the explicitly tiled path.
    pub tile_rows:usize,
    /// Half/BF16 uses the original fused cooperative kernel when true. FP32
    /// remains tiled. Failed fused calls never select the tiled path as recovery.
    pub use_tensor_core:bool,
}

/// Native failure or an unsupported derivative of the explicitly frozen NF4 base.
#[derive(Debug)]
pub enum FrozenNf4Error<E:fmt::Debug> {
    /// Original native backend error.
    Native(E),
    /// Scales, codebook or bias must remain frozen, not learned QAT operands.
    TrainableBase,
    /// Original NF4 API provides first-order input gradients, not their derivatives.
    InputGradientNotDifferentiable,
}
impl<E:fmt::Debug> fmt::Display for FrozenNf4Error<E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Native(error)=>write!(f,"native NF4: {error:?}"),Self::TrainableBase=>f.write_str("NF4 scales/codebook/bias must be frozen"),
            Self::InputGradientNotDifferentiable=>f.write_str("NF4 provides first-order input gradients only")}
    }
}
impl<E:fmt::Debug> core::error::Error for FrozenNf4Error<E> {}

/// Actual U8 byte-packed NF4 with FP32 scales and FP32 codebook. This is not
/// AWQ, bitsandbytes double quantization or a floating fake-quantization STE.
pub trait FrozenNf4Ops:Backend {
    /// Original native or autodiff contract error.
    type Nf4Error:fmt::Debug;
    /// Native packed projection. Output retains incoming floating activation storage.
    fn frozen_nf4_forward(input:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        bias:Option<FloatTensor<Self>>,options:Nf4ProjectionOptions) -> Result<FloatTensor<Self>,Self::Nf4Error>;
    /// Original input VJP: cast incoming gradient to original activation storage,
    /// accumulate FP32 and return that original activation dtype, without base gradients.
    fn frozen_nf4_input_backward(gradient:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        options:Nf4ProjectionOptions,activation_dtype:FloatDType) -> Result<FloatTensor<Self>,Self::Nf4Error>;
}
