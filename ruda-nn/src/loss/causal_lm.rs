use ruda_model::{
    config::Config,
    module::Module,
    tensor::{DType, Int, Tensor, activation::log_softmax, backend::Backend},
};

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
            return CausalLoss {
                loss_sum: Tensor::zeros([1], (&hidden.device(), DType::F32)),
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
