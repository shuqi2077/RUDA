use core::fmt;
use ruda_model::{module::Module, tensor::{Bool, Int, Tensor, backend::Backend}};
use crate::{Embedding, attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout},
    loss::{CausalCrossEntropyConfig, CausalLoss}, pool::SequencePooling};
use super::{MhcResidualStack, MhcResidualStackSession, MhcResidualBranch, MhcResidualBranchShape,
    ProjectedTransformerHead, TransformerProjection, TransformerProjectionShape, SequenceHeadOutput};

/// Original branch or output projection failure, without converting it into a numerical fallback.
#[derive(Debug)]
pub enum MhcResidualModelError<F: fmt::Debug, H: fmt::Debug> { Branch(F), Head(H) }
impl<F: fmt::Debug, H: fmt::Debug> fmt::Display for MhcResidualModelError<F, H> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Branch(error) => write!(formatter, "mHC residual branch: {error:?}"),
            Self::Head(error) => write!(formatter, "mHC output projection: {error:?}") }
    }
}
impl<F: fmt::Debug, H: fmt::Debug> core::error::Error for MhcResidualModelError<F, H> {}

/// Actual token table, heterogeneous dense/MoE compressed stack and independent native task head.
#[derive(Module, Debug)]
pub struct MhcResidualModel<B: Backend, P: Module<B>, F: Module<B>, H: Module<B>> {
    pub embedding: Embedding<B>,
    pub stack: MhcResidualStack<B, P, F>,
    pub head: ProjectedTransformerHead<B, H>,
}

#[derive(Debug)]
pub struct MhcResidualTrainingOutput<B: Backend> {
    pub causal: CausalLoss<B>,
    pub indexer_loss: Tensor<B, 1>,
}
#[derive(Debug)]
pub struct PackedMhcResidualTrainingOutput<B: Backend> {
    pub causal: CausalLoss<B>,
    pub document_indexer_losses: Tensor<B, 1>,
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>, H: TransformerProjectionShape<B>> MhcResidualModel<B, P, F, H> {
    pub fn from_parts(embedding: Embedding<B>, stack: MhcResidualStack<B, P, F>, head: ProjectedTransformerHead<B, H>) -> Self {
        let [vocab, width] = embedding.weight.val().dims();
        assert!(vocab > 0 && width > 0, "mHC token table must be nonempty");
        assert_eq!(stack.layers[0].attention.width, width, "mHC embedding/stack width differs");
        assert_eq!(head.projection.dimensions()[0], width, "mHC stack/head input width differs");
        assert!(head.projection.dimensions()[1] > 0, "mHC output classes must be nonempty");
        assert_eq!(embedding.weight.val().device(), stack.layers[0].attention.parts.query_down.device(), "mHC table/stack device differs");
        Self { embedding, stack, head }
    }

    pub(super) fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, length] = tokens.dims();
        assert!(batch > 0 && length > 0, "mHC model token rows must be nonempty");
        assert_eq!(tokens.device(), self.embedding.weight.val().device(), "mHC model token device differs");
        self.embedding.forward(tokens)
    }

    pub(super) fn embed_packed(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        let length = tokens.dims()[0];
        assert_eq!(length, layout.tokens(), "mHC packed token/layout count differs");
        assert_eq!(tokens.device(), self.embedding.weight.val().device(), "mHC packed token device differs");
        let weight = self.embedding.weight.val();
        let width = weight.dims()[1];
        if length == 0 { return Tensor::zeros([0, width], (&weight.device(), weight.dtype())); }
        self.embedding.forward(tokens.reshape([1, length])).reshape([length, width])
    }

    pub fn inference_session(&self) -> MhcResidualModelSession<'_, B, P, F, H> {
        MhcResidualModelSession { model: self, stack: self.stack.inference_session() }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranch<B>, H: TransformerProjection<B>> MhcResidualModel<B, P, F, H> {
    pub fn forward_hidden(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Result<Tensor<B, 3>, F::Error> {
        self.stack.forward(self.embed(tokens), valid)
    }
    pub fn forward_hidden_with_aux(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>, indexer_warmup: bool)
        -> Result<CompressedAttentionOutput<B>, F::Error> {
        self.stack.forward_with_aux(self.embed(tokens), valid, indexer_warmup)
    }
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>)
        -> Result<Tensor<B, 3>, MhcResidualModelError<F::Error, H::Error>> {
        let hidden = self.forward_hidden(tokens, valid).map_err(MhcResidualModelError::Branch)?;
        self.head.forward(hidden).map_err(MhcResidualModelError::Head)
    }
    pub fn forward_with_aux(&self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>, indexer_warmup: bool)
        -> Result<CompressedAttentionOutput<B>, MhcResidualModelError<F::Error, H::Error>> {
        let result = self.forward_hidden_with_aux(tokens, valid, indexer_warmup).map_err(MhcResidualModelError::Branch)?;
        Ok(CompressedAttentionOutput { output: self.head.forward(result.output).map_err(MhcResidualModelError::Head)?, indexer_loss: result.indexer_loss })
    }

    /// Reuse actual native pooling/classification head; labels and pooling policy remain explicit.
    pub fn forward_sequence(&self, tokens: Tensor<B, 2, Int>, valid: Tensor<B, 2, Bool>, pooling: SequencePooling)
        -> Result<SequenceHeadOutput<B>, MhcResidualModelError<F::Error, H::Error>> {
        let hidden = self.forward_hidden(tokens, Some(valid.clone())).map_err(MhcResidualModelError::Branch)?;
        self.head.forward_sequence(hidden, valid, pooling).map_err(MhcResidualModelError::Head)
    }

    pub fn training_forward(&self, tokens: Tensor<B, 2, Int>, labels: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>,
        loss: &CausalCrossEntropyConfig, label_smoothing: f64, indexer_warmup: bool)
        -> Result<MhcResidualTrainingOutput<B>, MhcResidualModelError<F::Error, H::Error>> {
        assert_eq!(tokens.dims(), labels.dims(), "mHC training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "mHC training label device differs");
        let result = self.forward_hidden_with_aux(tokens, valid, indexer_warmup).map_err(MhcResidualModelError::Branch)?;
        let causal = loss.try_forward_hidden_with_smoothing(result.output, labels,
            |rows| self.head.forward(rows).map_err(MhcResidualModelError::Head), label_smoothing)?;
        Ok(MhcResidualTrainingOutput { causal, indexer_loss: result.indexer_loss })
    }

    pub fn forward_packed_hidden(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>)
        -> Result<Tensor<B, 2>, F::Error> {
        self.stack.forward_packed(self.embed_packed(tokens, layout), layout, valid)
    }
    pub fn forward_packed_sequences(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, pooling: SequencePooling)
        -> Result<SequenceHeadOutput<B>, MhcResidualModelError<F::Error, H::Error>> {
        let hidden = self.forward_packed_hidden(tokens, layout, valid.clone()).map_err(MhcResidualModelError::Branch)?;
        self.head.forward_packed_sequences(hidden, layout, valid, pooling).map_err(MhcResidualModelError::Head)
    }
    pub fn forward_packed_with_aux(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool)
        -> Result<PackedCompressedAttentionOutput<B>, MhcResidualModelError<F::Error, H::Error>> {
        let result = self.stack.forward_packed_with_aux(self.embed_packed(tokens, layout), layout, valid, indexer_warmup)
            .map_err(MhcResidualModelError::Branch)?;
        Ok(PackedCompressedAttentionOutput { output: self.head.forward(result.output).map_err(MhcResidualModelError::Head)?,
            document_indexer_losses: result.document_indexer_losses })
    }
    pub fn packed_training_forward(&self, tokens: Tensor<B, 1, Int>, labels: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, loss: &CausalCrossEntropyConfig, label_smoothing: f64, indexer_warmup: bool)
        -> Result<PackedMhcResidualTrainingOutput<B>, MhcResidualModelError<F::Error, H::Error>> {
        assert_eq!(tokens.dims(), labels.dims(), "mHC packed training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "mHC packed training label device differs");
        let result = self.stack.forward_packed_with_aux(self.embed_packed(tokens, layout), layout, valid, indexer_warmup)
            .map_err(MhcResidualModelError::Branch)?;
        let causal = loss.try_forward_packed_hidden_with_smoothing(result.output, labels, layout,
            |rows| self.head.forward(rows).map_err(MhcResidualModelError::Head), label_smoothing)?;
        Ok(PackedMhcResidualTrainingOutput { causal, document_indexer_losses: result.document_indexer_losses })
    }

}

#[derive(Debug)]
pub struct MhcResidualModelSession<'a, B: Backend, P: Module<B>, F: Module<B>, H: Module<B>> {
    pub(super) model: &'a MhcResidualModel<B, P, F, H>,
    pub(super) stack: MhcResidualStackSession<'a, B, P, F>,
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>, H: TransformerProjectionShape<B>> MhcResidualModelSession<'a, B, P, F, H> {
    pub fn position(&self) -> usize { self.stack.position() }
    pub fn tensor_bytes(&self) -> usize { self.stack.tensor_bytes() }
    pub fn clear(&mut self) { self.stack.clear(); }
    pub fn reorder(&mut self, parents: Tensor<B, 1, Int>) { self.stack.reorder(parents); }
    pub fn fork(&self) -> Self { Self { model: self.model, stack: self.stack.fork() } }
    pub fn restore(&mut self, snapshot: Self) {
        assert!(core::ptr::eq(self.model, snapshot.model), "mHC model snapshot belongs to another module/head");
        self.stack.restore(snapshot.stack);
    }
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranch<B>, H: TransformerProjection<B>> MhcResidualModelSession<'a, B, P, F, H> {
    pub fn forward_hidden(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>) -> Result<Tensor<B, 3>, F::Error> {
        self.stack.forward(self.model.embed(tokens), valid)
    }
    pub fn forward(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>)
        -> Result<Tensor<B, 3>, MhcResidualModelError<F::Error, H::Error>> {
        let mut pending = self.stack.fork();
        let hidden = pending.forward(self.model.embed(tokens), valid).map_err(MhcResidualModelError::Branch)?;
        let logits = self.model.head.forward(hidden).map_err(MhcResidualModelError::Head)?;
        self.stack = pending;
        Ok(logits.detach())
    }
    pub fn forward_last(&mut self, tokens: Tensor<B, 2, Int>, valid: Option<Tensor<B, 2, Bool>>)
        -> Result<Tensor<B, 3>, MhcResidualModelError<F::Error, H::Error>> {
        let mut pending = self.stack.fork();
        let hidden = pending.forward(self.model.embed(tokens), valid).map_err(MhcResidualModelError::Branch)?;
        let length = hidden.dims()[1];
        let logits = self.model.head.forward(hidden.slice_dim(1, length - 1..length)).map_err(MhcResidualModelError::Head)?;
        self.stack = pending;
        Ok(logits.detach())
    }
}
