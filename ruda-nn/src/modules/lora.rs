use crate::{Dropout, DropoutConfig, Linear, LinearConfig};
use ruda_model::{
    config::Config,
    module::{Initializer, Module},
    tensor::{Tensor, DType, backend::Backend},
};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

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
        let dtype = base.weight.val().dtype();
        self.init_with_options(base, dtype, false)
    }

    /// Select adapter storage independently from the frozen dense base.
    /// `use_rslora` explicitly selects alpha/sqrt(rank) instead of alpha/rank.
    /// FP32 adapters on FP16/BF16 bases retain FP32 trainable leaves and gradients;
    /// the returned activation retains the base projection's storage dtype.
    pub fn init_with_options<B: Backend>(&self, base: Linear<B>, adapter_dtype: DType, use_rslora: bool) -> LoRALinear<B> {
        assert!(self.rank > 0, "LoRA rank must be positive");
        assert!(self.alpha.is_finite(), "LoRA alpha must be finite");
        assert!(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "LoRA dropout must be in [0, 1)"
        );
        let [input, output] = base.weight.val().dims();
        let device = base.weight.val().device();
        assert!(adapter_dtype.is_float(), "adapter storage must be floating");
        let mut adapter_a = LinearConfig::new(input, self.rank)
            .with_bias(false)
            .init(&device);
        let mut adapter_b = LinearConfig::new(self.rank, output)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(&device);
        adapter_a.weight = adapter_a
            .weight
            .map(|value| value.cast(adapter_dtype).detach().require_grad());
        adapter_b.weight = adapter_b
            .weight
            .map(|value| value.cast(adapter_dtype).detach().require_grad());
        self.from_adapters(base, adapter_a, adapter_b, use_rslora)
    }

    /// Attach explicitly loaded A/B matrices, preserving all supplied IDs and
    /// storage dtypes. Does not initialize, replace or requantize their values.
    /// Both adapters must be bias-free and on the base device, and trainable
    /// when the selected backend enables autodiff.
    pub fn from_adapters<B: Backend>(
        &self, base: Linear<B>, adapter_a: Linear<B>, adapter_b: Linear<B>, use_rslora: bool,
    ) -> LoRALinear<B> {
        assert!(self.rank > 0 && self.alpha.is_finite(), "invalid adapter rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout), "LoRA dropout must be in [0, 1)");
        let weight = base.weight.val();
        let [input, output] = weight.dims();
        let a = adapter_a.weight.val();
        let b = adapter_b.weight.val();
        assert_eq!(a.dims(), [input, self.rank], "adapter A dimensions differ");
        assert_eq!(b.dims(), [self.rank, output], "adapter B dimensions differ");
        assert!(adapter_a.bias.is_none() && adapter_b.bias.is_none(), "LoRA adapters must be bias-free");
        assert!(a.device() == weight.device() && b.device() == weight.device(), "adapter/base devices differ");
        assert!(!B::ad_enabled(&weight.device()) || (a.is_require_grad() && b.is_require_grad()), "loaded adapters must be trainable with autodiff enabled");
        let denominator = if use_rslora { (self.rank as f64).sqrt() } else { self.rank as f64 };
        LoRALinear { base: base.no_grad(), adapter_a, adapter_b,
            dropout: DropoutConfig::new(self.dropout).init(), scale: self.alpha / denominator }
    }
}

impl<B: Backend> LoRALinear<B> {
    /// Project an input with any supported leading dimensions.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let adapted = self.dropout.forward(input.clone().cast(self.adapter_a.weight.val().dtype()));
        let hidden = self.adapter_a.forward(adapted).cast(self.adapter_b.weight.val().dtype());
        let update = self.adapter_b.forward(hidden).mul_scalar(self.scale);
        let base = self.base.forward(input);
        let dtype = base.dtype();
        base + update.cast(dtype)
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
            .cast(self.adapter_b.weight.val().dtype())
            .matmul(self.adapter_b.weight.val())
            .mul_scalar(self.scale)
            .detach();
        let mut base = self.base;
        base.weight = base
            .weight
            .map(|weight| {
                let dtype = weight.dtype();
                let merged = if dtype == update.dtype() { weight + update } else {
                    let work = if dtype == DType::F64 || update.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
                    (weight.cast(work) + update.cast(work)).cast(dtype)
                };
                merged.detach().set_require_grad(false)
            });
        base
    }
}

#[cfg(test)]
mod tests;
