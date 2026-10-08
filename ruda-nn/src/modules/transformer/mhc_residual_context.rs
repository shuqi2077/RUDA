use core::fmt::Debug;
use ruda_model::tensor::{Bool, Int, Tensor, backend::Backend};
use crate::{attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout},
    loss::CausalCrossEntropyConfig, pool::SequencePooling};
use super::{MhcResidualBranchShape, MhcResidualModel, MhcResidualModelSession, MhcResidualModelError,
    MhcResidualTrainingOutput, PackedMhcResidualTrainingOutput, TransformerProjection, SequenceHeadOutput};

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>, H: TransformerProjection<B>> MhcResidualModel<B, P, F, H> {
    pub fn try_forward_hidden_with<R, G>(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>, branch: G)
        -> Result<Tensor<B, 3>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.stack.try_forward_with(self.embed(tokens), valid, branch)
    }

    pub fn try_forward_hidden_with_aux<R, G>(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>,
        indexer_warmup: bool, branch: G) -> Result<CompressedAttentionOutput<B>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.stack.try_forward_with_aux(self.embed(tokens), valid, indexer_warmup, branch)
    }

    pub fn try_forward_with<R: Debug, G>(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>, branch: G)
        -> Result<Tensor<B, 3>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let hidden = self.try_forward_hidden_with(tokens, valid, branch).map_err(MhcResidualModelError::Branch)?;
        self.head.forward(hidden).map_err(MhcResidualModelError::Head)
    }

    pub fn try_forward_with_aux<R: Debug, G>(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>,
        indexer_warmup: bool, branch: G) -> Result<CompressedAttentionOutput<B>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let result = self.try_forward_hidden_with_aux(tokens, valid, indexer_warmup, branch).map_err(MhcResidualModelError::Branch)?;
        Ok(CompressedAttentionOutput { output: self.head.forward(result.output).map_err(MhcResidualModelError::Head)?, indexer_loss: result.indexer_loss })
    }

    pub fn try_forward_sequence_with<R: Debug, G>(&self, tokens: Tensor<B, 2, Int>, valid: Tensor<B, 2, Bool>,
        pooling: SequencePooling, branch: G) -> Result<SequenceHeadOutput<B>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let hidden = self.try_forward_hidden_with(tokens, Some(valid.clone()), branch).map_err(MhcResidualModelError::Branch)?;
        self.head.forward_sequence(hidden, valid, pooling).map_err(MhcResidualModelError::Head)
    }

    /// Existing chunked vocabulary loss, preserving labels, smoothing and the actual branch's derivative contract.
    pub fn try_training_forward_with<R: Debug, G>(&self, tokens: Tensor<B, 2, Int>, labels: Tensor<B, 2, Int>,
        valid: Option<Tensor<B, 2, Bool>>, loss: &CausalCrossEntropyConfig, label_smoothing: f64, indexer_warmup: bool, branch: G)
        -> Result<MhcResidualTrainingOutput<B>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        assert_eq!(tokens.dims(), labels.dims(), "mHC contextual training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "mHC contextual training label device differs");
        let result = self.try_forward_hidden_with_aux(tokens, valid, indexer_warmup, branch).map_err(MhcResidualModelError::Branch)?;
        let causal = loss.try_forward_hidden_with_smoothing(result.output, labels,
            |rows| self.head.forward(rows).map_err(MhcResidualModelError::Head), label_smoothing)?;
        Ok(MhcResidualTrainingOutput { causal, indexer_loss: result.indexer_loss })
    }

    pub fn try_forward_packed_hidden_with<R, G>(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, branch: G) -> Result<Tensor<B, 2>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.stack.try_forward_packed_with(self.embed_packed(tokens, layout), layout, valid, branch)
    }

    pub fn try_forward_packed_with<R: Debug, G>(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, branch: G) -> Result<Tensor<B, 2>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let hidden = self.try_forward_packed_hidden_with(tokens, layout, valid, branch).map_err(MhcResidualModelError::Branch)?;
        self.head.forward(hidden).map_err(MhcResidualModelError::Head)
    }

    pub fn try_forward_packed_hidden_with_aux<R, G>(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool, branch: G) -> Result<PackedCompressedAttentionOutput<B>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.stack.try_forward_packed_with_aux(self.embed_packed(tokens, layout), layout, valid, indexer_warmup, branch)
    }

    pub fn try_forward_packed_with_aux<R: Debug, G>(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool, branch: G)
        -> Result<PackedCompressedAttentionOutput<B>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let result = self.try_forward_packed_hidden_with_aux(tokens, layout, valid, indexer_warmup, branch).map_err(MhcResidualModelError::Branch)?;
        Ok(PackedCompressedAttentionOutput { output: self.head.forward(result.output).map_err(MhcResidualModelError::Head)?,
            document_indexer_losses: result.document_indexer_losses })
    }

    /// Each layer exchanges once even on a rank with no local documents or source tokens.
    pub fn try_packed_training_forward_with<R: Debug, G>(&self, tokens: Tensor<B, 1, Int>, labels: Tensor<B, 1, Int>,
        layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>, loss: &CausalCrossEntropyConfig,
        label_smoothing: f64, indexer_warmup: bool, branch: G)
        -> Result<PackedMhcResidualTrainingOutput<B>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        assert_eq!(tokens.dims(), labels.dims(), "mHC contextual packed training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "mHC contextual packed training label device differs");
        let result = self.try_forward_packed_hidden_with_aux(tokens, layout, valid, indexer_warmup, branch).map_err(MhcResidualModelError::Branch)?;
        let causal = loss.try_forward_packed_hidden_with_smoothing(result.output, labels, layout,
            |rows| self.head.forward(rows).map_err(MhcResidualModelError::Head), label_smoothing)?;
        Ok(PackedMhcResidualTrainingOutput { causal, document_indexer_losses: result.document_indexer_losses })
    }
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>, H: TransformerProjection<B>> MhcResidualModelSession<'a, B, P, F, H> {
    pub fn try_forward_hidden_with<R, G>(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>, branch: G)
        -> Result<Tensor<B, 3>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.stack.try_forward_with(self.model.embed(tokens), valid, branch)
    }

    pub fn try_forward_with<R: Debug, G>(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>, branch: G)
        -> Result<Tensor<B, 3>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let mut pending = self.stack.fork();
        let hidden = pending.try_forward_with(self.model.embed(tokens), valid, branch).map_err(MhcResidualModelError::Branch)?;
        let logits = self.model.head.forward(hidden).map_err(MhcResidualModelError::Head)?;
        self.stack = pending;
        Ok(logits.detach())
    }

    pub fn try_forward_last_with<R: Debug, G>(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>, branch: G)
        -> Result<Tensor<B, 3>, MhcResidualModelError<R, H::Error>>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let mut pending = self.stack.fork();
        let hidden = pending.try_forward_with(self.model.embed(tokens), valid, branch).map_err(MhcResidualModelError::Branch)?;
        let length = hidden.dims()[1];
        let logits = self.model.head.forward(hidden.slice_dim(1, length - 1..length)).map_err(MhcResidualModelError::Head)?;
        self.stack = pending;
        Ok(logits.detach())
    }
}
