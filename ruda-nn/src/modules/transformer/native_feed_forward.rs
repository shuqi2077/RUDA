use super::{AdaptedFeedForward, AdaptedTransformerBlock, AdaptedStackLayer, AdaptedTransformerStack,
    AwqFeedForward, AwqTransformerBlock, AwqTransformerStack,
    DenseFeedForward, DenseTransformerBlock, DenseTransformerStack, TransformerProjection};
use super::dense::try_residual_branch;
use core::fmt;
use ruda_model::tensor::{NativeSwiGluOps, Tensor};
use crate::attention::{DenseAttentionMask, DenseAttentionOptions, PackedSequenceLayout,
    PackedAttentionOptions, PackedDocumentAttentionMask};

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
    pub fn try_forward_feed_forward_native<const D: usize>(&self, hidden: Tensor<B, D>) -> Result<Tensor<B, D>, B::SwiGluError> {
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

impl<B: NativeSwiGluOps> DenseTransformerBlock<B> {
    /// Actual packed attention/positions/masks followed by native gate/up activation on flat rows.
    pub fn try_forward_packed_native_with_positions<F>(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>, options: PackedAttentionOptions, positions: F)
        -> Result<Tensor<B, 2>, B::SwiGluError>
        where F: FnOnce(Tensor<B, 3>, Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = match masks {
            Some(masks) => self.forward_packed_attention_masked(input, layout, masks, options, positions),
            None => self.forward_packed_attention(input, layout, options, positions),
        };
        self.try_forward_feed_forward_native(hidden)
    }
}

impl<B: NativeSwiGluOps> AdaptedTransformerBlock<B> {
    /// Packed native FFN training after the original adapted attention and document-local positions.
    pub fn try_forward_packed_native_with_positions<F>(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>, options: PackedAttentionOptions, positions: F)
        -> Result<Tensor<B, 2>, B::SwiGluError>
        where F: FnOnce(Tensor<B, 3>, Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = match masks {
            Some(masks) => self.forward_packed_attention_masked(input, layout, masks, options, positions),
            None => self.forward_packed_attention(input, layout, options, positions),
        };
        self.try_forward_feed_forward_native(hidden)
    }
}

impl<B: NativeSwiGluOps, P: TransformerProjection<B>> AwqTransformerBlock<B, P> {
    /// Original packed projected attention, then native FFN/residual/norm on the actual flat rows.
    pub fn try_forward_packed_native_with_positions<F>(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>, options: PackedAttentionOptions, positions: F)
        -> Result<Tensor<B, 2>, NativeFeedForwardError<P::Error, B::SwiGluError>>
        where F: FnOnce(Tensor<B, 3>, Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = self.forward_packed_attention_with_positions(input, layout, masks, options, positions)
            .map_err(NativeFeedForwardError::Execution)?;
        self.try_forward_feed_forward_native(hidden)
    }
}

impl<B: NativeSwiGluOps> AdaptedStackLayer<B> {
    /// Native gate/up derivatives on the actual dense or adapted layer, without changing its variant.
    pub fn try_forward_native_with_positions<F>(&self, input: Tensor<B, 3>, masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions, positions: F) -> Result<Tensor<B, 3>, B::SwiGluError>
        where F: FnOnce(Tensor<B, 4>, Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match self {
            Self::Dense(block) => block.try_forward_native_with_positions(input, masks, options, positions),
            Self::Adapted(block) => block.try_forward_native_with_positions(input, masks, options, positions),
        }
    }

    /// Original selected/unselected packed layers with native gated activation and actual masks.
    pub fn try_forward_packed_native_with_positions<F>(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>, options: PackedAttentionOptions, positions: F)
        -> Result<Tensor<B, 2>, B::SwiGluError>
        where F: FnOnce(Tensor<B, 3>, Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        match self {
            Self::Dense(block) => block.try_forward_packed_native_with_positions(input, layout, masks, options, positions),
            Self::Adapted(block) => block.try_forward_packed_native_with_positions(input, layout, masks, options, positions),
        }
    }
}

impl<B: NativeSwiGluOps> DenseTransformerStack<B> {
    /// Original block order and caller-owned layer positions, with explicit native FFN activation.
    pub fn try_forward_native_with_positions<F>(&self, mut input: Tensor<B, 3>, masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions, mut positions: F) -> Result<Tensor<B, 3>, B::SwiGluError>
        where F: FnMut(usize, Tensor<B, 4>, Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        for (index, block) in self.blocks.iter().enumerate() {
            input = block.try_forward_native_with_positions(input, masks.clone(), options, |query, key| positions(index, query, key))?;
        }
        Ok(input)
    }

    /// Native FFN over actual flat document rows, without packing/unpacking or cross-document targets.
    pub fn try_forward_packed_native_with_positions<F>(&self, mut input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>, options: PackedAttentionOptions, mut positions: F)
        -> Result<Tensor<B, 2>, B::SwiGluError>
        where F: FnMut(usize, Tensor<B, 3>, Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(input.dims()[0], layout.tokens(), "packed transformer boundaries differ from actual token rows");
        for (index, block) in self.blocks.iter().enumerate() {
            input = block.try_forward_packed_native_with_positions(input, layout, masks, options, |query, key| positions(index, query, key))?;
        }
        Ok(input)
    }
}

impl<B: NativeSwiGluOps> AdaptedTransformerStack<B> {
    /// Actual selected and unselected layers in original order; native paths never add adapters.
    pub fn try_forward_native_with_positions<F>(&self, mut input: Tensor<B, 3>, masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions, mut positions: F) -> Result<Tensor<B, 3>, B::SwiGluError>
        where F: FnMut(usize, Tensor<B, 4>, Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        for (index, layer) in self.layers.iter().enumerate() {
            input = layer.try_forward_native_with_positions(input, masks.clone(), options, |query, key| positions(index, query, key))?;
        }
        Ok(input)
    }

    /// Native gate/up training with original packed layer selection, visibility and projected positions.
    pub fn try_forward_packed_native_with_positions<F>(&self, mut input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>, options: PackedAttentionOptions, mut positions: F)
        -> Result<Tensor<B, 2>, B::SwiGluError>
        where F: FnMut(usize, Tensor<B, 3>, Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(input.dims()[0], layout.tokens(), "packed transformer boundaries differ from actual token rows");
        for (index, layer) in self.layers.iter().enumerate() {
            input = layer.try_forward_packed_native_with_positions(input, layout, masks, options, |query, key| positions(index, query, key))?;
        }
        Ok(input)
    }
}

impl<B: NativeSwiGluOps, P: TransformerProjection<B>> AwqTransformerStack<B, P> {
    /// Every actual mixed-projection block with native FFN activation; real projection errors propagate.
    pub fn try_forward_native_with_positions<F>(&self, mut input: Tensor<B, 3>, masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions, mut positions: F)
        -> Result<Tensor<B, 3>, NativeFeedForwardError<P::Error, B::SwiGluError>>
        where F: FnMut(usize, Tensor<B, 4>, Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        for (index, block) in self.blocks.iter().enumerate() {
            input = block.try_forward_native_with_positions(input, masks.clone(), options, |query, key| positions(index, query, key))?;
        }
        Ok(input)
    }

    /// Original mixed packed/dense weights and document-local attention, with native gate/up activation.
    pub fn try_forward_packed_native_with_positions<F>(&self, mut input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>, options: PackedAttentionOptions, mut positions: F)
        -> Result<Tensor<B, 2>, NativeFeedForwardError<P::Error, B::SwiGluError>>
        where F: FnMut(usize, Tensor<B, 3>, Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        for (index, block) in self.blocks.iter().enumerate() {
            input = block.try_forward_packed_native_with_positions(input, layout, masks, options, |query, key| positions(index, query, key))?;
        }
        Ok(input)
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
