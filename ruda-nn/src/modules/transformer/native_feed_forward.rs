use super::{AdaptedFeedForward, AdaptedTransformerBlock, AwqFeedForward, AwqTransformerBlock,
    DenseFeedForward, DenseTransformerBlock, TransformerProjection};
use super::dense::try_residual_branch;
use core::fmt;
use ruda_model::tensor::{NativeSwiGluOps, Tensor};
use crate::attention::{DenseAttentionMask, DenseAttentionOptions};

/// Original projection/collective failure or original native activation failure.
#[derive(Debug)]
pub enum NativeFeedForwardError<E, A> {
    /// The actual projection or communicator error, without changing execution strategy.
    Execution(E),
    /// The actual native activation error, without retrying an unfused substitute.
    Activation(A),
}

impl<E: fmt::Debug, A: fmt::Debug> fmt::Display for NativeFeedForwardError<E, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Execution(error) => write!(f, "native feed-forward execution: {error:?}"),
            Self::Activation(error) => write!(f, "native feed-forward activation: {error:?}"),
        }
    }
}
impl<E: fmt::Debug, A: fmt::Debug> core::error::Error for NativeFeedForwardError<E, A> {}

impl<B: NativeSwiGluOps> DenseFeedForward<B> {
    /// Explicit native SiLU-gate training with original projections and one intermediate dropout.
    /// Ungated and non-SiLU configurations still honor the actual configured activation.
    pub fn try_forward_native<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>, B::SwiGluError> {
        let up = self.up.forward(input.clone());
        let value = if let Some(gate) = &self.gate {
            self.activation.try_forward_gated_native(gate.forward(input), up)?
        } else { self.activation.try_forward_native(up)? };
        Ok(self.down.forward(self.dropout.forward(value)))
    }
}

impl<B: NativeSwiGluOps> AdaptedFeedForward<B> {
    /// Original dense/LoRA projections with the existing selected-gradient native SwiGLU operation.
    /// Adapter scales, input dropout, IDs and intermediate dropout remain unchanged.
    pub fn try_forward_native<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>, B::SwiGluError> {
        let up = self.up.forward(input.clone());
        let value = if let Some(gate) = &self.gate {
            self.activation.try_forward_gated_native(gate.forward(input), up)?
        } else { self.activation.try_forward_native(up)? };
        Ok(self.down.forward(self.dropout.forward(value)))
    }
}

impl<B: NativeSwiGluOps, P: TransformerProjection<B>> AwqFeedForward<B, P> {
    /// Actual caller-loaded dense/LoRA/AWQ/NF4/mixed projection types with native gated activation.
    /// Neither weights nor activation inputs are requantized or merged to select this path.
    pub fn try_forward_native<const D: usize>(&self, input: Tensor<B, D>)
        -> Result<Tensor<B, D>, NativeFeedForwardError<P::Error, B::SwiGluError>> {
        let up = self.up.forward(input.clone()).map_err(NativeFeedForwardError::Execution)?;
        let value = if let Some(gate) = &self.gate {
            let gate = gate.forward(input).map_err(NativeFeedForwardError::Execution)?;
            self.activation.try_forward_gated_native(gate, up)
        } else { self.activation.try_forward_native(up) }.map_err(NativeFeedForwardError::Activation)?;
        self.down.forward(self.dropout.forward(value)).map_err(NativeFeedForwardError::Execution)
    }
}

impl<B: NativeSwiGluOps> DenseTransformerBlock<B> {
    /// Native gated FFN/residual stage, retaining the original pre/post normalization order.
    pub fn try_forward_feed_forward_native<const D: usize>(&self, hidden: Tensor<B, D>)
        -> Result<Tensor<B, D>, B::SwiGluError> {
        try_residual_branch(hidden, &self.feed_forward_norm, &self.residual_dropout, self.norm_first,
            |source| self.feed_forward.try_forward_native(source))
    }

    /// Caller-owned attention/positions followed by the explicitly selected native FFN.
    pub fn try_forward_native_with_positions<F>(&self, input: Tensor<B, 3>, masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions, positions: F) -> Result<Tensor<B, 3>, B::SwiGluError>
        where F: FnOnce(Tensor<B, 4>, Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let hidden = self.forward_attention_with_positions(input, masks, options, positions);
        self.try_forward_feed_forward_native(hidden)
    }
}

impl<B: NativeSwiGluOps> AdaptedTransformerBlock<B> {
    /// Original adapted FFN/residual/norm with native selected gate/up derivatives.
    pub fn try_forward_feed_forward_native(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>, B::SwiGluError> {
        try_residual_branch(hidden, &self.feed_forward_norm, &self.residual_dropout, self.norm_first,
            |source| self.feed_forward.try_forward_native(source))
    }

    /// Actual adapted attention and caller positions, followed by native gated FFN training.
    pub fn try_forward_native_with_positions<F>(&self, input: Tensor<B, 3>, masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions, positions: F) -> Result<Tensor<B, 3>, B::SwiGluError>
        where F: FnOnce(Tensor<B, 4>, Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let hidden = self.forward_attention_with_positions(input, masks, options, positions);
        self.try_forward_feed_forward_native(hidden)
    }
}

impl<B: NativeSwiGluOps, P: TransformerProjection<B>> AwqTransformerBlock<B, P> {
    /// Actual projected FFN/residual/norm stage, independently of the attention projection format.
    pub fn try_forward_feed_forward_native<const D: usize>(&self, hidden: Tensor<B, D>)
        -> Result<Tensor<B, D>, NativeFeedForwardError<P::Error, B::SwiGluError>> {
        try_residual_branch(hidden, &self.feed_forward_norm, &self.residual_dropout, self.norm_first,
            |source| self.feed_forward.try_forward_native(source))
    }

    /// Native FFN training after the unchanged caller-selected projected attention path.
    pub fn try_forward_native_with_positions<F>(&self, input: Tensor<B, 3>, masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions, positions: F)
        -> Result<Tensor<B, 3>, NativeFeedForwardError<P::Error, B::SwiGluError>>
        where F: FnOnce(Tensor<B, 4>, Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let hidden = self.forward_attention_with_positions(input, masks, options, positions)
            .map_err(NativeFeedForwardError::Execution)?;
        self.try_forward_feed_forward_native(hidden)
    }
}
