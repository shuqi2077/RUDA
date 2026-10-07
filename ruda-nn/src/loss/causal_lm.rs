use ruda_model::{
    config::Config,
    module::Module,
    tensor::{DType, Int, Bool, Tensor, activation::log_softmax, backend::Backend},
};
use crate::attention::PackedSequenceLayout;

/// Caller-built decoder whose attention explicitly honors packed document boundaries.
pub trait PackedCausalLanguageModel<B: Backend>: Module<B> {
    /// Produce `[total_tokens, width]` without attention across document boundaries.
    fn forward_packed_hidden(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2>;
    /// Project actual token rows against the complete vocabulary.
    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// Hidden-state and vocabulary projection contract for decoder-only models.
///
/// Implementations own their architecture, positional encoding and attention
/// masks. This contract does not infer a model family or alter those semantics.
pub trait CausalLanguageModel<B: Backend>: Module<B> {
    /// Produce final normalized hidden states `[batch, sequence, width]`.
    fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3>;
    /// Project token rows against the complete vocabulary.
    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// Exact causal language-model cross entropy with bounded projection chunks.
#[derive(Config, Debug)]
pub struct CausalCrossEntropyConfig {
    /// Maximum token rows in each full-vocabulary projection.
    #[config(default = 32)]
    pub token_chunk_size: usize,
    /// Label value excluded from supervision, not an attention-mask value.
    #[config(default = -100)]
    pub ignore_index: i64,
    /// Pair hidden position `t` with label `t + 1`.
    #[config(default = true)]
    pub shift: bool,
}

/// Unnormalized loss and effective token count for accumulation and DDP.
#[derive(Debug)]
pub struct CausalLoss<B: Backend> {
    /// Sum of supervised token losses in FP32.
    pub loss_sum: Tensor<B, 1>,
    /// Number of non-ignored labels after the optional causal shift.
    pub valid_tokens: Tensor<B, 1, Int>,
}

impl<B: Backend> CausalLoss<B> {
    /// Token mean; an entirely ignored or empty batch has zero loss.
    pub fn mean(&self) -> Tensor<B, 1> {
        self.loss_sum.clone()
            / self
                .valid_tokens
                .clone()
                .float()
                .cast(DType::F32)
                .clamp_min(1)
    }
}

impl CausalCrossEntropyConfig {
    /// Train a packed decoder directly from the native device data batch.
    /// The batch's explicit supervision alignment/sentinel must match this loss.
    #[cfg(feature = "std")]
    pub fn forward_packed_batch<B: Backend, M: PackedCausalLanguageModel<B>>(
        &self, model: &M, batch: ruda_model::data::causal::PackedCausalBatch<B>,
    ) -> CausalLoss<B> {
        use ruda_model::data::causal::CausalTargetAlignment;
        assert_eq!(self.ignore_index, batch.ignore_index, "collator and loss ignore indices differ");
        assert_eq!(self.shift, batch.target_alignment == CausalTargetAlignment::NextToken, "collator and loss target alignment differs");
        let layout = PackedSequenceLayout::new(batch.boundaries, batch.input_ids.dims()[0]);
        self.forward_packed_model(model, batch.input_ids, batch.labels, &layout)
    }

    /// Train a caller's actual padding-aware hidden-state function from a batch.
    ///
    /// The backbone receives the actual visibility mask and reset positions;
    /// it must honor both. Vocabulary projection uses every class in each chunk.
    /// No model-family adapter, label-mask inference or CPU execution is installed.
    #[cfg(feature = "std")]
    pub fn forward_padded_batch<B: Backend>(
        &self, batch: ruda_model::data::causal::PaddedCausalBatch<B>,
        forward_hidden: impl FnOnce(Tensor<B, 2, Int>, Tensor<B, 2, Bool>, Tensor<B, 2, Int>) -> Tensor<B, 3>,
        project: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    ) -> CausalLoss<B> {
        use ruda_model::data::causal::CausalTargetAlignment;
        assert_eq!(self.ignore_index, batch.ignore_index, "collator and loss ignore indices differ");
        assert_eq!(self.shift, batch.target_alignment == CausalTargetAlignment::NextToken, "collator and loss target alignment differs");
        let shape = batch.input_ids.dims();
        assert_eq!(batch.labels.dims(), shape, "batch token/label geometry differs");
        assert_eq!(batch.attention_mask.dims(), shape, "batch visibility geometry differs");
        assert_eq!(batch.position_ids.dims(), shape, "batch position geometry differs");
        let device = batch.input_ids.device();
        assert!(batch.labels.device() == device && batch.attention_mask.device() == device
            && batch.position_ids.device() == device, "causal batch operands must share a device");
        let hidden = forward_hidden(batch.input_ids, batch.attention_mask, batch.position_ids);
        self.forward_hidden(hidden, batch.labels, project)
    }

    /// Train an explicit packed decoder, retaining the full-vocabulary chunk algorithm.
    pub fn forward_packed_model<B: Backend, M: PackedCausalLanguageModel<B>>(
        &self, model: &M, tokens: Tensor<B, 1, Int>, labels: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
    ) -> CausalLoss<B> {
        assert_eq!(tokens.dims(), labels.dims(), "packed tokens and labels differ");
        assert_eq!(tokens.dims()[0], layout.tokens(), "packed model input differs from its document metadata");
        self.forward_packed_hidden(model.forward_packed_hidden(tokens, layout), labels, layout, |rows| model.project(rows))
    }

    /// Use flat hidden states and labels; shifted supervision never crosses documents.
    /// No truncation, padding examples or sampled vocabulary are introduced.
    pub fn forward_packed_hidden<B: Backend>(
        &self, hidden: Tensor<B, 2>, labels: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        project: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    ) -> CausalLoss<B> {
        let [tokens, width] = hidden.dims();
        assert_eq!(tokens, layout.tokens(), "packed hidden states differ from document metadata");
        assert_eq!(labels.dims(), [tokens], "packed hidden and label lengths differ");
        assert_eq!(hidden.device(), labels.device(), "packed hidden and labels must share a device");
        let labels = if self.shift {
            labels.clone().mask_fill(layout.document_starts::<B>(&labels.device()), self.ignore_index)
        } else { labels };
        self.forward_hidden(hidden.reshape([1, tokens, width]), labels.reshape([1, tokens]), project)
    }

    pub fn forward_logits<B: Backend>(
        &self,
        logits: Tensor<B, 3>,
        labels: Tensor<B, 2, Int>,
    ) -> CausalLoss<B> {
        self.forward_hidden(logits, labels, |rows| rows)
    }

    /// Train any decoder implementing the hidden-state/projection contract.
    pub fn forward_model<B: Backend, M: CausalLanguageModel<B>>(
        &self,
        model: &M,
        tokens: Tensor<B, 2, Int>,
        labels: Tensor<B, 2, Int>,
    ) -> CausalLoss<B> {
        assert_eq!(
            tokens.dims(),
            labels.dims(),
            "tokens and labels must have matching shapes"
        );
        self.forward_hidden(model.forward_hidden(tokens), labels, |rows| {
            model.project(rows)
        })
    }

    /// Compute exact loss without constructing `[B, T, vocabulary]` logits.
    ///
    /// The projection can be dense, LoRA or another differentiable module.
    /// Every chunk uses the entire vocabulary; labels other than `ignore_index`
    /// must be valid vocabulary indices. This API does not recompute activations
    /// during backward or promise a bound on total autodiff graph memory.
    pub fn forward_hidden<B: Backend>(
        &self,
        hidden: Tensor<B, 3>,
        labels: Tensor<B, 2, Int>,
        project: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    ) -> CausalLoss<B> {
        assert!(
            self.token_chunk_size > 0,
            "token_chunk_size must be positive"
        );
        let [batch, sequence, width] = hidden.dims();
        assert_eq!(
            labels.dims(),
            [batch, sequence],
            "hidden and label shapes must match"
        );
        assert_eq!(
            hidden.device(),
            labels.device(),
            "hidden and labels must share a device"
        );
        let length = if self.shift {
            sequence.saturating_sub(1)
        } else {
            sequence
        };
        let count = batch.checked_mul(length).expect("token count overflow");
        if count == 0 {
            let mask = Tensor::<B, 3, Bool>::zeros(hidden.dims(), &hidden.device()).bool_not();
            return CausalLoss {
                loss_sum: hidden.cast(DType::F32).mask_fill(mask, 0).sum(),
                valid_tokens: Tensor::zeros([1], &labels.device()),
            };
        }
        let (hidden, labels) = if self.shift && sequence > 0 {
            (
                hidden.slice([0..batch, 0..length, 0..width]),
                labels.slice([0..batch, 1..sequence]),
            )
        } else {
            (hidden, labels)
        };
        let hidden = hidden.reshape([count, width]);
        let labels = labels.reshape([count]);
        let ignored = labels.clone().equal_elem(self.ignore_index);
        let valid_tokens = ignored.clone().bool_not().int().sum();
        let targets = labels.mask_fill(ignored.clone(), 0);
        let mut loss_sum = Tensor::zeros([1], (&hidden.device(), DType::F32));
        for start in (0..count).step_by(self.token_chunk_size) {
            let end = start.saturating_add(self.token_chunk_size).min(count);
            let logits = project(hidden.clone().slice([start..end, 0..width]));
            assert_eq!(
                logits.dims()[0],
                end - start,
                "projection must preserve token rows"
            );
            assert!(
                logits.dims()[1] > 0,
                "projection vocabulary must be nonempty"
            );
            let selected = log_softmax(logits.cast(DType::F32), 1)
                .gather(
                    1,
                    targets
                        .clone()
                        .slice([start..end])
                        .reshape([end - start, 1]),
                )
                .reshape([end - start]);
            loss_sum = loss_sum
                - selected
                    .mask_fill(ignored.clone().slice([start..end]), 0)
                    .sum();
        }
        CausalLoss {
            loss_sum,
            valid_tokens,
        }
    }
}

#[cfg(test)]
mod tests;
