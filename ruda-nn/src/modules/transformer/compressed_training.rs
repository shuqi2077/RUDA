use ruda_model::tensor::{Bool, Int, Tensor, backend::Backend};
use crate::{attention::PackedSequenceLayout, loss::{CausalLanguageModel, PackedCausalLanguageModel, CausalCrossEntropyConfig, CausalLoss}};
use super::HybridAttentionLanguageModel;

/// Unnormalized supervised loss/token count plus the independent layer-summed indexer KL.
/// Callers choose accumulation normalization and any auxiliary coefficient explicitly.
#[derive(Debug)]
pub struct HybridLanguageTrainingOutput<B: Backend> {
    pub causal: CausalLoss<B>,
    pub indexer_loss: Tensor<B, 1>,
}

/// Packed supervised loss and independent per-document layer-summed indexer KL values.
#[derive(Debug)]
pub struct HybridPackedLanguageTrainingOutput<B: Backend> {
    pub causal: CausalLoss<B>,
    pub document_indexer_losses: Tensor<B, 1>,
}

impl<B: Backend> CausalLanguageModel<B> for HybridAttentionLanguageModel<B> {
    fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.backbone.forward(tokens, None)
    }

    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> { self.project_tokens(hidden) }
}

impl<B: Backend> PackedCausalLanguageModel<B> for HybridAttentionLanguageModel<B> {
    fn forward_packed_hidden(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        self.backbone.forward_packed(tokens, layout, None)
    }

    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> { self.project_tokens(hidden) }
}

impl<B: Backend> HybridAttentionLanguageModel<B> {
    /// Reuse the exact existing shifted/ignored-label and bounded-token full-vocabulary
    /// cross entropy. Attention visibility is supplied separately, never inferred from labels.
    pub fn causal_loss(&self, tokens: Tensor<B, 2, Int>, labels: Tensor<B, 2, Int>,
        valid: Option<Tensor<B, 2, Bool>>, loss: &CausalCrossEntropyConfig) -> CausalLoss<B> {
        assert_eq!(tokens.dims(), labels.dims(), "hybrid training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "hybrid training label device differs");
        loss.forward_hidden(self.backbone.forward(tokens, valid), labels, |rows| self.project_tokens(rows))
    }

    /// One actual backbone graph for supervised training plus separately selectable
    /// indexer training. No auxiliary weighting or optimizer policy is installed here.
    pub fn training_forward(&self, tokens: Tensor<B, 2, Int>, labels: Tensor<B, 2, Int>,
        valid: Option<Tensor<B, 2, Bool>>, loss: &CausalCrossEntropyConfig,
        indexer_warmup: bool) -> HybridLanguageTrainingOutput<B> {
        assert_eq!(tokens.dims(), labels.dims(), "hybrid training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "hybrid training label device differs");
        let result = self.backbone.forward_with_aux(tokens, valid, indexer_warmup);
        HybridLanguageTrainingOutput {
            causal: loss.forward_hidden(result.output, labels, |rows| self.project_tokens(rows)),
            indexer_loss: result.indexer_loss,
        }
    }

    /// Supervision shifts within each real document only; vocabulary rows remain
    /// chunked by the original loss configuration and every vocabulary class participates.
    pub fn packed_causal_loss(&self, tokens: Tensor<B, 1, Int>, labels: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, loss: &CausalCrossEntropyConfig) -> CausalLoss<B> {
        assert_eq!(tokens.dims(), labels.dims(), "packed hybrid training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "packed hybrid training label device differs");
        loss.forward_packed_hidden(self.backbone.forward_packed(tokens, layout, valid), labels, layout, |rows| self.project_tokens(rows))
    }

    pub fn packed_training_forward(&self, tokens: Tensor<B, 1, Int>, labels: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, loss: &CausalCrossEntropyConfig,
        indexer_warmup: bool) -> HybridPackedLanguageTrainingOutput<B> {
        assert_eq!(tokens.dims(), labels.dims(), "packed hybrid training token/label geometry differs");
        assert_eq!(tokens.device(), labels.device(), "packed hybrid training label device differs");
        let result = self.backbone.forward_packed_with_aux(tokens, layout, valid, indexer_warmup);
        HybridPackedLanguageTrainingOutput {
            causal: loss.forward_packed_hidden(result.output, labels, layout, |rows| self.project_tokens(rows)),
            document_indexer_losses: result.document_indexer_losses,
        }
    }
}
