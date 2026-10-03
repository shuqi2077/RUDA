use crate::{Dropout, DropoutConfig, Linear, LinearConfig};
use ruda_model::{
    config::Config,
    module::{Initializer, Module},
    tensor::{Tensor, backend::Backend},
};

/// Configuration for an adapter on an existing dense linear projection.
#[derive(Config, Debug)]
pub struct LoRALinearConfig {
    /// Adapter rank, independent of the model family.
    pub rank: usize,
    /// Adapter multiplier is `alpha / rank`.
    pub alpha: f64,
    /// Dropout applies only to the adapter input.
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Frozen base projection plus a trainable low-rank residual.
///
/// `base(x) + (alpha / rank) * B(A(dropout(x)))`. Parameter records retain
/// the base and both adapters; no model-specific parameter naming is required.
#[derive(Module, Debug)]
pub struct LoRALinear<B: Backend> {
    /// Original projection, including its bias and parameter mappers.
    pub base: Linear<B>,
    /// Projection from input width to adapter rank.
    pub adapter_a: Linear<B>,
    /// Projection from adapter rank to output width, initialized to zero.
    pub adapter_b: Linear<B>,
    /// Adapter input dropout.
    pub dropout: Dropout,
    /// Explicit adapter multiplier.
    pub scale: f64,
}

impl LoRALinearConfig {
    /// Attach an adapter without changing the base weights or their IDs.
    pub fn init<B: Backend>(&self, base: Linear<B>) -> LoRALinear<B> {
        assert!(self.rank > 0, "LoRA rank must be positive");
        assert!(self.alpha.is_finite(), "LoRA alpha must be finite");
        assert!(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "LoRA dropout must be in [0, 1)"
        );
        let [input, output] = base.weight.val().dims();
        let device = base.weight.val().device();
        let dtype = base.weight.val().dtype();
        let mut adapter_a = LinearConfig::new(input, self.rank)
            .with_bias(false)
            .init(&device);
        let mut adapter_b = LinearConfig::new(self.rank, output)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(&device);
        adapter_a.weight = adapter_a.weight.map(|value| value.cast(dtype));
        adapter_b.weight = adapter_b.weight.map(|value| value.cast(dtype));
        LoRALinear {
            base: base.no_grad(),
            adapter_a,
            adapter_b,
            dropout: DropoutConfig::new(self.dropout).init(),
            scale: self.alpha / self.rank as f64,
        }
    }
}

impl<B: Backend> LoRALinear<B> {
    /// Project an input with any supported leading dimensions.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let update = self
            .adapter_b
            .forward(self.adapter_a.forward(self.dropout.forward(input.clone())));
        self.base.forward(input) + update.mul_scalar(self.scale)
    }

    /// Consume the adapter and merge its weights into a frozen dense layer.
    ///
    /// This is the dropout-free projection used for inference. It is not an
    /// optimizer-state conversion and cannot be used to resume adapter training.
    pub fn merge(self) -> Linear<B> {
        let update = self
            .adapter_a
            .weight
            .val()
            .matmul(self.adapter_b.weight.val())
            .mul_scalar(self.scale)
            .detach();
        let mut base = self.base;
        base.weight = base
            .weight
            .map(|weight| (weight + update).detach().set_require_grad(false));
        base
    }
}

#[cfg(test)]
mod tests;
